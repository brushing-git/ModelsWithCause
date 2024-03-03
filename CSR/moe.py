import torch
import copy
import torch.nn as nn
import torch.nn.functional as F

class Expert(nn.Module):
    def __init__(self, dim_model: int, dim_ffn: int, dropout=0.1) -> None:
        super().__init__()

        self.fc1 = nn.Linear(dim_model, dim_ffn)
        self.fc2 = nn.Linear(dim_ffn, dim_model)
        self.dropout = nn.Dropout(dropout)
        self.activation = nn.ReLU()
    
    def forward(self, x) -> torch.tensor:
        out = self.fc2(self.activation(self.fc1(x)))
        out = self.dropout(out)
        return out

class NoisyTopKRouter(nn.Module):
    def __init__(self, dim_model: int, n_experts: int, top_k: int) -> None:
        super(NoisyTopKRouter, self).__init__()
        self.top_k = top_k
        self.topkroute_linear = nn.Linear(dim_model, n_experts)
        self.noise_linear = nn.Linear(dim_model, n_experts)
    
    def forward(self, x) -> tuple:
        logits = self.topkroute_linear(x)
        noise_logits = self.noise_linear(x)

        noise = torch.randn_like(logits) * F.softplus(noise_logits)
        noisy_logits = logits + noise

        top_k_logits, indices = noise_logits.topk(self.top_k, dim=-1)
        zeros = torch.full_like(noisy_logits, float('-inf'))

        sparse_logits = zeros.scatter(-1, indices, top_k_logits)
        out = F.softmax(sparse_logits, dim=-1)
        return out, indices

class SparseMoE(nn.Module):
    def __init__(self, n_experts: int, dim_model: int, 
                 dim_ffn: int, top_k: int, dropout=0.1) -> None:
        super().__init__()

        self.experts = nn.ModuleList([Expert(dim_model, dim_ffn, dropout=dropout) for _ in range(n_experts)])
        self.router = NoisyTopKRouter(dim_model, n_experts, top_k)
        self.top_k = top_k
    
    def forward(self, x) -> torch.tensor:
        gating_out, indices = self.router(x)
        final_out = torch.zeros_like(x)

        # Reshape for batch processing
        flat_x = x.view(-1, x.size(-1))
        flat_gating_out = gating_out.view(-1, gating_out.size(-1))

        # Process experts in parallel
        for i, expert in enumerate(self.experts):
            # Mask
            expert_mask = (indices == i).any(dim=-1)
            flat_mask = expert_mask.view(-1)

            if flat_mask.any():
                expert_in = flat_x[flat_mask]
                expert_out = expert(expert_in)

                # Gating scores
                gating_scores = flat_gating_out[flat_mask, i].unsqueeze(1)
                weighted_out = expert_out * gating_scores

                # Update the final out
                final_out[expert_mask] += weighted_out.squeeze(1)
        
        return final_out

class MoEDecoderLayer(nn.Module):
    def __init__(self, dim_model: int, 
                 n_heads: int, 
                 n_experts: int, 
                 top_k: int,
                 dropout: float = 0.1,
                 ffn: float = 2048) -> None:
        super().__init__()

        self.sa = nn.MultiheadAttention(dim_model, n_heads, dropout=dropout)
        self.ma = nn.MultiheadAttention(dim_model, n_heads, dropout=dropout)
        self.moe = SparseMoE(n_experts, dim_model, ffn, top_k, dropout)
        self.norm1 = nn.LayerNorm(dim_model)
        self.norm2 = nn.LayerNorm(dim_model)
        self.norm3 = nn.LayerNorm(dim_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)
    
    def forward(self, tgt,
                memory,
                tgt_mask=None,
                memory_mask=None,
                tgt_key_padding_mask=None,
                memory_key_padding_mask=None) -> torch.tensor:
        
        x = tgt
        x = self.norm1(x + self._sa_block(x, tgt_mask, tgt_key_padding_mask))
        x = self.norm2(x + self._mha_block(x, memory, memory_mask, memory_key_padding_mask))
        x = self.norm3(x + self._moe_block(x))
        
        return x
    
    def _sa_block(self, x,
                  attn_mask,
                  key_padding_mask) -> torch.tensor:
        out = self.sa(x, x, x,
                    attn_mask=attn_mask,
                    key_padding_mask=key_padding_mask,
                    is_causal=False,
                    need_weights=False)[0]
        out = self.dropout1(out)
        return out
    
    def _mha_block(self, x,
                   mem,
                   attn_mask,
                   key_padding_mask) -> torch.tensor:
        out = self.ma(x, mem, mem,
                      attn_mask=attn_mask,
                      key_padding_mask=key_padding_mask,
                      is_causal=False,
                      need_weights=False)[0]
        out = self.dropout2(out)
        return out
    
    def _moe_block(self, x) -> torch.tensor:
        out = self.moe(x)
        out = self.dropout3(out)
        return out

class MoEDecoder(nn.Module):
    def __init__(self, decoder_layer, 
                 num_layers: int,
                 norm=None) -> None:
        super().__init__()

        self.n_layers = num_layers
        self.layers = nn.ModuleList([copy.deepcopy(decoder_layer) for _ in range(self.n_layers)])
        self.norm = norm
    
    def forward(self, tgt, 
                memory, 
                tgt_mask=None, 
                memory_mask=None, 
                tgt_key_padding_mask=None,
                memory_key_padding_mask=None) -> torch.tensor:
        out = tgt

        for mod in self.layers:
            out = mod(out, memory,
                      tgt_mask=tgt_mask,
                      tgt_key_padding_mask=tgt_key_padding_mask,
                      memory_key_padding_mask=memory_key_padding_mask)
        
        if self.norm is not None:
            out = self.norm(out)

        return out