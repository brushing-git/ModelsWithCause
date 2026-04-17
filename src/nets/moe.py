import torch
import copy
import torch.nn as nn
import torch.nn.functional as F

class Expert(nn.Module):
    def __init__(
            self, 
            dim_model: int, 
            dim_ffn: int, 
            dropout: float = 0.1
    ) -> None:
        """
        Expert class for mixture of experts. It is a dim_ffn hidden network MLP.

        Attributes
        fc1 : nn.Linear : input fully connected layer to hidden
        fc2 : nn.Linear : output fully connected layer from hidden
        dropout : nn.Dropout : dropout in case it is needed
        activation : nn.ReLU : ReLU activation
        """
        super().__init__()

        self.fc1 = nn.Linear(dim_model, dim_ffn)
        self.fc2 = nn.Linear(dim_ffn, dim_model)
        self.dropout = nn.Dropout(dropout)
        self.activation = nn.ReLU()
    
    def forward(
            self, 
            x: torch.Tensor
    ) -> torch.Tensor:
        """
        Forward pass through an expert.

        Params
        x : torch.tensor : input tensor

        Returns
        out : torch.tensor : the output tensor
        """
        out = self.fc2(self.activation(self.fc1(x)))
        out = self.dropout(out)
        return out

class NoisyTopKRouter(nn.Module):
    def __init__(
            self, 
            dim_model: int, 
            n_experts: int, 
            top_k: int
    ) -> None:
        """
        A Noisy Top K router as described in https://arxiv.org/abs/1701.06538

        Attributes
        top_k : int : the number of outputs on the network to keep; all other are 0
        topkroute_linear : nn.Linear : the gating weights W_g
        noise_linear : nn.Linear : the noise weights W_noise (used to apply the degree of noise)
        """
        super(NoisyTopKRouter, self).__init__()

        # Top_k
        self.top_k = top_k

        # W_g
        self.topkroute_linear = nn.Linear(dim_model, n_experts)

        # W_noise
        self.noise_linear = nn.Linear(dim_model, n_experts)
    
    def forward(
            self, 
            x: torch.Tensor
    ) -> tuple:
        """
        Forward pass that takes the top_k gate values.

        We take the top k outputs from the gate:

        H(x)i = (x @ W_g)_i + epsilon * Softplus((x @ W_noise)_i) 

        where epsilon ~ N(0,1)

        Params
        x : torch.tensor : input tensor

        Returns
        out : torch.tensor : tensor of output gating
        indices : torch.tensor : the indices of the top_k gates to apply
        """
        # Get x @ W_g
        logits = self.topkroute_linear(x)

        # Get x @ W_noise
        noise_logits = self.noise_linear(x)

        # Compute epsilon * Softplus(x @ W_noise)
        noise = torch.randn_like(logits) * F.softplus(noise_logits)

        # Add both functions together
        noisy_logits = logits + noise
        
        # Take the top_k logits and indices
        top_k_logits, indices = noisy_logits.topk(self.top_k, dim=-1)

        # Create sparse logits, i.e. 0 everywhere except on the top_k
        zeros = torch.full_like(noisy_logits, float('-inf'))

        # Scatter the correct logits
        sparse_logits = zeros.scatter(-1, indices, top_k_logits)

        # Apply softmax
        out = F.softmax(sparse_logits, dim=-1)

        return out, indices

class SparseMoE(nn.Module):
    def __init__(
            self, 
            n_experts: int, 
            dim_model: int, 
            dim_ffn: int, 
            top_k: int, 
            dropout: float = 0.1
    ) -> None:
        """
        Creates a switch transformer MLP layer.

        Attributes
        experts : nn.ModuleList : a module list of the Expert (1 hidden lyr MLP) class
        router : NoisyTopKRouter : the sparse noisy router
        top_k : int : the top_k members to select from the router
        """
        super().__init__()

        # Experts
        self.experts = nn.ModuleList([Expert(dim_model, dim_ffn, dropout=dropout) for _ in range(n_experts)])

        # Router
        self.router = NoisyTopKRouter(dim_model, n_experts, top_k)

        # Top_k
        self.top_k = top_k
    
    def forward(
            self, 
            x: torch.Tensor
    ) -> torch.Tensor:
        """
        Forward pass on the switch MLP lyr

        Params
        x : torch.tensor : input tensor

        Returns
        final_out : torch.tensor : the aggregated output tensor from the MLP
        """
        # Get the dimensions
        batch_size, seq_len, dim = x.shape

        # Push through the router
        gating_out, indices = self.router(x)

        # Get the final out tensor
        final_out = torch.zeros_like(x)

        # Reshape for batch processing
        flat_x = x.reshape(-1, dim)
        flat_gating_out = gating_out.reshape(-1, gating_out.shape[-1])

        # Process experts in parallel
        for i, expert in enumerate(self.experts):
            # Mask
            expert_mask = (indices == i).any(dim=-1)
            flat_mask = expert_mask.reshape(-1)

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
    def __init__(
            self, 
            dim_model: int, 
            n_heads: int, 
            n_experts: int, 
            top_k: int,
            dropout: float = 0.1,
            ffn: float = 2048
    ) -> None:
        """
        A switch transformer decoder layer.

        Attributes
        sa : nn.MultiheadAttention : self-attention on input
        ma : nn.MultiheadAttention : cross-attention on input
        moe : SparseMoE : sparse MoE MLP layer
        norm1 : nn.LayerNorm : normalization after sa
        norm2 : nn.LayerNorm : normalization after ma
        norm3 : nn.LayerNorm : normalization after moe
        dropout1-3 : nn.Dropout : application of dropout after sa, ma, moe
        """
        super().__init__()

        # Self-attention
        self.sa = nn.MultiheadAttention(
            dim_model, n_heads, dropout=dropout, batch_first=True
        )

        # Cross-attention
        self.ma = nn.MultiheadAttention(
            dim_model, n_heads, dropout=dropout, batch_first=True
        )

        # Sparse MoE MLP
        self.moe = SparseMoE(
            n_experts, dim_model, ffn, top_k, dropout
        )

        # Norms
        self.norm1 = nn.LayerNorm(dim_model)
        self.norm2 = nn.LayerNorm(dim_model)
        self.norm3 = nn.LayerNorm(dim_model)

        # Dropout
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)
    
    def forward(
            self, 
            tgt: torch.Tensor,
            memory: torch.Tensor,
            tgt_mask: torch.Tensor = None,
            memory_mask: torch.Tensor = None,
            tgt_key_padding_mask: torch.Tensor = None,
            memory_key_padding_mask: torch.Tensor = None
    ) -> torch.Tensor:
        """
        Standard decoder layer algorithm:

        1. x = norm1(x + self_attention(x))
        2. x = norm2(x + cross_attention(x, encoded))
        3. x = norm3(x + mlp(x))

        Params
        tgt : torch.tensor : the target embedding of the input tensor
        memory : torch.tensor : the encoded (as needed) cross-attention embedding
        tgt_mask : torch.tensor : mask for the tgt
        memory_mask : torch.tensor : mask for the encoded
        tgt_key_padding_mask : torch.tensor : padding for tgt
        memory_key_padding_mask : torch.tensor : padding for the encoded

        Returns
        x : torch.tensor : output of the decoder layer
        """
        # Set x to tgt
        x = tgt

        # Apply self-attention
        x = self.norm1(x + self._sa_block(x, tgt_mask, tgt_key_padding_mask))

        # Apply cross-attention
        x = self.norm2(x + self._mha_block(x, memory, memory_mask, memory_key_padding_mask))

        # Apply MLP
        x = self.norm3(x + self._moe_block(x))
        
        return x
    
    def _sa_block(
            self, 
            x: torch.Tensor,
            attn_mask: torch.Tensor,
            key_padding_mask: torch.Tensor
    ) -> torch.Tensor:
        """
        Applies self-attention

        Params
        x : torch.tensor : embedding space tensor of input
        attn_mask : torch.tensor : mask for input
        key_padding_mask : torch.tensor : padding mask

        Returns
        out : torch.tensor : output tensor after attention and dropout
        """
        # Apply self attention
        out = self.sa(
            x, 
            x, 
            x,
            attn_mask=attn_mask,
            key_padding_mask=key_padding_mask,
            is_causal=False,
            need_weights=False
        )[0]

        # Apply dropout
        out = self.dropout1(out)

        return out
    
    def _mha_block(
            self, 
            x: torch.Tensor,
            mem: torch.Tensor,
            attn_mask: torch.Tensor,
            key_padding_mask: torch.Tensor
    ) -> torch.Tensor:
        """
        Applies cross attention.

        Params
        x : torch.tensor : embedding space tensor of input
        mem : torch.tensor : embedding space tensor of encoded
        attn_mask : torch.tensor : mask for inputs
        key_padding_mask : torch.tensor : padding for inputs

        Returns
        out : torch.tensor : output tensor after attention and dropout
        """
        # Apply cross attention
        out = self.ma(
            x, 
            mem, 
            mem,
            attn_mask=attn_mask,
            key_padding_mask=key_padding_mask,
            is_causal=False,
            need_weights=False
        )[0]

        # Apply dropout
        out = self.dropout2(out)

        return out
    
    def _moe_block(
            self, 
            x: torch.Tensor
    ) -> torch.Tensor:
        """
        Applies MoE MLP.

        Params
        x : torch.tensor : embedding space tensor of input

        Returns
        out : torch.tensor : output of mlp and dropout
        """
        # Apply moe block
        out = self.moe(x)

        # Apply dropout
        out = self.dropout3(out)

        return out

class MoEDecoder(nn.Module):
    def __init__(
            self, 
            decoder_layer: nn.Module, 
            num_layers: int,
            norm: nn.Module = None
    ) -> None:
        """
        Decoder for Switch Transformer

        Attributes
        n_layers : int : the number of layers
        layers : nn.ModuleList : list of decoder layers
        norm : nn.Module : final output norm
        """
        super().__init__()

        # Set the attributes
        self.n_layers = num_layers
        self.layers = nn.ModuleList([copy.deepcopy(decoder_layer) for _ in range(self.n_layers)])
        self.norm = norm
    
    def forward(
            self, 
            tgt: torch.Tensor, 
            memory: torch.Tensor, 
            tgt_mask: torch.Tensor = None, 
            memory_mask: torch.Tensor = None, 
            tgt_key_padding_mask: torch.Tensor = None,
            memory_key_padding_mask: torch.Tensor = None
    ) -> torch.Tensor:
        """
        Forward pass through the decoder.

        Params
        tgt : torch.tensor : embedding input tensor
        memory : torch.tensor : embedding encoded tensor
        tgt_mask : torch.tensor : the tgt tensor mask
        memory_mask : torch.tensor : the encoded tensor mask
        tgt_key_padding_mask : torch.tensor : padding for tgt
        memory_key_padding_mask : torch.tensor : padding for encoded

        Returns
        out : torch.tensor : output tensor after running through layers
        """
        out = tgt

        for mod in self.layers:
            out = mod(
                out, 
                memory,
                tgt_mask=tgt_mask,
                memory_mask=memory_mask,
                tgt_key_padding_mask=tgt_key_padding_mask,
                memory_key_padding_mask=memory_key_padding_mask
            )
        
        if self.norm is not None:
            out = self.norm(out)

        return out