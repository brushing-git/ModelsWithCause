import torch
import torch.nn as nn
import CSR.utils as ut
from torch.distributions import Categorical
from torch.nn.functional import softmax
from CSR.moe import MoEDecoderLayer, MoEDecoder
from math import log, sqrt
from tqdm import tqdm

class NADE(nn.Module):
    def __init__(self, in_dim: int, hidden_dim: int, cat=2, optimizer=torch.optim.Adam) -> None:
        super(NADE, self).__init__()
        self.D = in_dim
        self.H = hidden_dim
        self.C = cat
        self.activation = nn.Sigmoid()
        self.logprob = nn.LogSoftmax(dim=1)
        self.optimizer = optimizer
        self.device = ut.set_device()

        self.params = nn.ParameterDict({
            'V': nn.Parameter(torch.randn(self.H, self.D, self.C)),
            'b': nn.Parameter(torch.zeros(self.D, self.C)),
            'W': nn.Parameter(torch.randn(self.H, self.D)),
            'c': nn.Parameter(torch.zeros(1, self.H))
        })

        nn.init.xavier_normal_(self.params['V'])
        nn.init.xavier_normal_(self.params['W'])
    
    def forward(self, x):
        a_d = self.params['c'].expand(x.size(0), -1)
        y_hat, p_hat = self._estimate_logits(a_d, x)
        return y_hat, p_hat

    def _estimate_logits(self, a_d, x, sample=False) -> tuple:
        if sample:
            assert (x is None), "No input for sampling as first time"

        y_hat = []
        x_hat = []
        p_hat = []

        for d in range(self.D):
            h_d = self.activation(a_d)
            logits = h_d @ self.params['V'][:,d,:] + self.params['b'][d,:]
            p_d = self.logprob(logits)
            y_hat.append(logits)
            p_hat.append(p_d)

            if sample:
                x = Categorical(probs=p_d).sample().type(torch.float)
                x_hat.append(x)
                a_d = a_d + x @ self.params['W'][:,d:d+1].t()
            else:
                a_d = a_d + x[:,d:d+1] @ self.params['W'][:,d:d+1].t()
        
        y_hat = torch.cat(y_hat, 1)
        p_hat = torch.cat(p_hat, 1)

        if sample:
            x_hat = torch.cat(x_hat)
            x_hat = x_hat.type(torch.long)
            return y_hat, p_hat, x_hat
        else:
            return y_hat, p_hat
    
    def _compute_loss(self, y_hat, y, loss_fn) -> torch.Tensor:
        total_loss = 0
        for i in range(self.D):
            start_idx = i*self.C
            end_idx = start_idx+self.C
            predictions = y_hat[:,start_idx:end_idx]
            true_features = y[:,i]
            total_loss += loss_fn(predictions, true_features)
        
        loss = total_loss / self.D

        return loss

    def _train_step(self, x, y, optim_fn, loss_fn) -> float:
        y_hat, _ = self.forward(x)
        loss = self._compute_loss(y_hat, y, loss_fn)
        optim_fn.zero_grad()
        loss.backward()
        optim_fn.step()
        loss = loss.detach()

        return loss.item()
    
    def _eval(self, te_loader, loss_fn) -> tuple:
        val_loss = []
        self.eval()
        
        for x, y in iter(te_loader):
            x, y = x.to(self.device), y.to(self.device)

            y_hat, _ = self.forward(x)
            loss = self._compute_loss(y_hat, y, loss_fn)
            loss = loss.detach()
            val_loss.append(loss.item())
        
        te_loss = sum(val_loss) / len(val_loss)

        return te_loss
    
    def sample(self, n=1) -> list:
        a_d = self.params['c'].expand(n, -1)
        _, _, x_hat = self._estimate_logits(a_d, x=None, sample=True)
        return x_hat.tolist()
    
    def estimate_prob(self, x, y) -> list:
        _, p_hat = self.forward(x)
        return p_hat.tolist()
    
    def parameter_count(self):
        total_params = sum(p.numel() for p in self.parameters())
        print(f"Number of parameters: {total_params}")    

    def fit(self, tr_loader, te_loader, epochs: int, lr: float, step_size=50) -> dict:
        self.to(self.device)
        optimizer = self.optimizer(self.parameters(), lr=lr)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=0.5)
        loss_fn = nn.CrossEntropyLoss()

        hist = {'tr_loss': [], 'te_loss': [], 'te_auc': [], 'current_lr': []}

        for step in range(epochs):
            self.train()
            tr_loss = []

            for x, y in tqdm(tr_loader):
                x, y = x.to(self.device), y.to(self.device)

                loss = self._train_step(x, y, optimizer, loss_fn)
                tr_loss.append(loss)
            
            te_loss = self._eval(te_loader, loss_fn)
            tr_avg_loss = sum(tr_loss) / len(tr_loss)
            current_lr = scheduler.get_last_lr()[0]

            s = ('Metrics for epoch {} are:\n tr_loss: {}, ' 
              'te_loss: {}, te_auc: {}, current_lr: {}').format(step,
                                                        tr_avg_loss,
                                                        te_loss,
                                                        0.0,
                                                        current_lr)
            print(s)

            hist['tr_loss'].append(tr_avg_loss)
            hist['te_loss'].append(te_loss)
            hist['current_lr'].append(current_lr)

            scheduler.step()

        return hist

class PositionalEncoding(nn.Module):
    def __init__(self, dim_model: int, dropout_p: float, max_len: int, device) -> None:
        super().__init__()

        self.dropout = nn.Dropout(dropout_p)

        pos_encoding = torch.zeros(max_len, dim_model)
        positions_list = torch.arange(0, max_len, dtype=torch.float).view(-1,1)

        # 1000^(2i/dim_model)
        division_term = torch.exp(torch.arange(0, dim_model, 2).float()
                                  * (-log(10000.0)) / dim_model)
        
        # PE(pos, 2i) = sin(pos/1000^(2i/dim_model))
        pos_encoding[:,0::2] = torch.sin(positions_list * division_term)

        # PE(pos, 2i + 1) = cos(pos/1000^(2i/dim_model))
        pos_encoding[:,1::2] = torch.cos(positions_list * division_term)

        # Saving buffer
        self.pos_encoding = pos_encoding.unsqueeze(0).transpose(0,1).to(device)
    
    def forward(self, token_embedding: torch.tensor) -> torch.tensor:
        return self.dropout(token_embedding + self.pos_encoding[:token_embedding.size(0),:])

class Transformer(nn.Module):
    def __init__(self, n_tokens: int, dim_model: int, n_heads: int, 
                 n_encoder_lyrs: int, n_decoder_lyrs: int, dropout_p: float,
                 ffn=2048, optimizer=torch.optim.Adam, SOS_token=6, EOS_token=7) -> None:
        super().__init__()

        self.dim_model = dim_model
        self.optimizer = optimizer
        self.SOS_token = SOS_token
        self.EOS_token = EOS_token
        self.logprob = nn.LogSoftmax(dim=1)
        self.device = ut.set_device()

        # Layers
        self.positional_encoder = PositionalEncoding(
            dim_model=dim_model, dropout_p=dropout_p, max_len=5000, device=self.device
        )

        self.embedding = nn.Embedding(n_tokens, dim_model)
        self.transformer = nn.Transformer(
            d_model=dim_model,
            nhead=n_heads,
            num_encoder_layers=n_encoder_lyrs,
            num_decoder_layers=n_decoder_lyrs,
            dim_feedforward=ffn,
            dropout=dropout_p
        )
        self.out = nn.Linear(dim_model, n_tokens)

    def forward(self, src, tgt, tgt_mask=None, src_pad_mask=None, 
                tgt_pad_mask=None) -> torch.tensor:
        src = self.embedding(src) * sqrt(self.dim_model)
        tgt = self.embedding(tgt) * sqrt(self.dim_model)
        src = self.positional_encoder(src)
        tgt = self.positional_encoder(tgt)

        src = src.permute(1,0,2)
        tgt = tgt.permute(1,0,2)

        transformer_out = self.transformer(src, 
                                           tgt, 
                                           tgt_mask=tgt_mask,
                                           src_key_padding_mask=src_pad_mask, 
                                           tgt_key_padding_mask=tgt_pad_mask)
        out = self.out(transformer_out)

        return out
    
    def _append_SOS_EOS(self, x) -> torch.tensor:
        x = x.type(torch.long)
        sos = torch.full((x.size(0),1), self.SOS_token)
        eos = torch.full((x.size(0),1), self.EOS_token)
        x = torch.cat((sos, x, eos), dim=1)

        return x
    
    def _get_tgt_mask(self, size: int) -> torch.tensor:
        mask = torch.tril(torch.ones(size, size) == 1)
        mask = mask.float()
        mask = mask.masked_fill(mask == 0, float('-inf'))
        mask = mask.masked_fill(mask == 1, float(0.0))

        return mask
    
    def _create_pad_mask(self, matrix: torch.tensor, pad_token: int) -> torch.tensor:
        return (matrix == pad_token)
    
    def _train_step(self, x, y, optim_fn, loss_fn) -> float:
        y_input = y[:,:-1] # shift the target
        y_expected = y[:,1:] # target prediction

        # Mask the other letters
        sequence_length = y_input.size(1)
        tgt_mask = self._get_tgt_mask(sequence_length).to(self.device)

        y_hat = self.forward(x, y_input, tgt_mask)
        y_hat = y_hat.permute(1,2,0)

        loss = loss_fn(y_hat, y_expected)
        optim_fn.zero_grad()
        loss.backward()
        optim_fn.step()
        loss = loss.detach()

        return loss.item()
    
    def _eval(self, te_loader, loss_fn) -> tuple:
        val_loss = []
        self.eval()

        for x, y in iter(te_loader):
            x, y = self._append_SOS_EOS(x), self._append_SOS_EOS(y) # append the SOS and EOS tokens
            x, y = x.to(self.device), y.to(self.device)

            # Shift the tgt and mask
            y_input = y[:,:-1]
            y_expected = y[:,1:]
            sequence_length = y_input.size(1)
            tgt_mask = self._get_tgt_mask(sequence_length).to(self.device)

            y_hat = self.forward(x, y_input, tgt_mask)
            y_hat = y_hat.permute(1,2,0)

            loss = loss_fn(y_hat, y_expected)
            loss = loss.detach()
            val_loss.append(loss.item())
        
        te_loss = sum(val_loss) / len(val_loss)

        return te_loss
    
    def sample(self, x, max_length: int) -> list:
        self.to(self.device)
        self.eval()

        x = self._append_SOS_EOS(x)
        x = x.to(self.device)
        y_input = torch.tensor([[self.SOS_token]], dtype=torch.long, device=self.device)
        num_tokens = len(x[0])

        for _ in range(max_length):
            tgt_mask = self._get_tgt_mask(y_input.size(1)).to(self.device)
            
            y_hat = self.forward(x, y_input, tgt_mask)

            next_item = y_hat.topk(1)[1].view(-1)[-1].item()
            next_item = torch.tensor([[next_item]], device=self.device)

            y_input = torch.cat((y_input, next_item), dim=1)

            if next_item.view(-1).item() == self.EOS_token:
                break

        return y_input.view(-1).tolist()
    
    def estimate_prob(self, x, y) -> torch.tensor:
        self.to(self.device)
        self.eval()

        with torch.no_grad():
            x, y = self._append_SOS_EOS(x), self._append_SOS_EOS(y)
            x, y = x.to(self.device), y.to(self.device)

            # Shift the tgt and mask
            y_input = y[:,:-1]
            y_expected = y[:,1:]
            sequence_length = y_input.size(1)
            tgt_mask = self._get_tgt_mask(sequence_length).to(self.device)

            y_hat = self.forward(x, y_input, tgt_mask)
            y_hat = y_hat.permute(1,2,0)
            ps = self.logprob(y_hat)

            # Get the correct labels and sum the right log probs
            target_sequence = y[:, 1:]
            p_hat = torch.gather(ps, 1, target_sequence.unsqueeze(1)).squeeze(1)

        return p_hat

    def parameter_count(self):
        total_params = sum(p.numel() for p in self.parameters())
        print(f"Number of parameters: {total_params}")

    def fit(self, tr_loader, te_loader, epochs: int, lr: float, step_size=50) -> dict:
        self.to(self.device)
        optimizer = self.optimizer(self.parameters(), lr=lr)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=0.5)
        loss_fn = nn.CrossEntropyLoss()

        hist = {'tr_loss': [], 'te_loss': [], 'te_auc': [], 'current_lr': []}

        for step in range(epochs):
            self.train()
            tr_loss = []

            for x, y in tqdm(tr_loader):
                x, y = self._append_SOS_EOS(x), self._append_SOS_EOS(y) # append the SOS and EOS tokens
                x, y = x.to(self.device), y.to(self.device)

                loss = self._train_step(x, y, optimizer, loss_fn)
                tr_loss.append(loss)
            
            te_loss = self._eval(te_loader, loss_fn)
            tr_avg_loss = sum(tr_loss) / len(tr_loss)
            current_lr = scheduler.get_last_lr()[0]

            s = ('Metrics for epoch {} are:\n tr_loss: {}, ' 
              'te_loss: {}, te_auc: {}, current_lr: {}').format(step,
                                                        tr_avg_loss,
                                                        te_loss,
                                                        0.0,
                                                        current_lr)
            print(s)

            hist['tr_loss'].append(tr_avg_loss)
            hist['te_loss'].append(te_loss)
            hist['current_lr'].append(current_lr)

            scheduler.step()

        return hist

class DecoderTransformer(Transformer):
    def __init__(self, n_tokens: int, dim_model: int, n_heads: int, 
                 n_decoder_lyrs: int, dropout_p: float, ffn=2048,
                 optimizer=torch.optim.Adam, SOS_token=6, EOS_token=7) -> None:
        super().__init__(n_tokens=n_tokens, dim_model=dim_model, n_heads=n_heads,
                        n_encoder_lyrs=1, n_decoder_lyrs=n_decoder_lyrs,
                        dropout_p=dropout_p, optimizer=optimizer, ffn=ffn,
                        SOS_token=SOS_token, EOS_token=EOS_token)
        
        decoder_lyr = nn.TransformerDecoderLayer(d_model=dim_model, 
                                                 nhead=n_heads, 
                                                 dim_feedforward=ffn, 
                                                 dropout=dropout_p)
        decoder_norm = nn.LayerNorm(dim_model)
        self.decoder = nn.TransformerDecoder(decoder_layer=decoder_lyr,
                                                  num_layers=n_decoder_lyrs,
                                                  norm=decoder_norm)
    
    def forward(self, src, tgt, tgt_mask=None, src_pad_mask=None, 
                tgt_pad_mask=None) -> torch.tensor:
        src = self.embedding(src) * sqrt(self.dim_model)
        tgt = self.embedding(tgt) * sqrt(self.dim_model)
        src = self.positional_encoder(src)
        tgt = self.positional_encoder(tgt)

        src = src.permute(1,0,2)
        tgt = tgt.permute(1,0,2)

        decoder_out = self.decoder(tgt, 
                                   src, 
                                   tgt_mask=tgt_mask, 
                                   tgt_key_padding_mask=tgt_pad_mask, 
                                   memory_key_padding_mask=src_pad_mask)
        out = self.out(decoder_out)

        return out

class MoEDecoderTransformer(DecoderTransformer):
    def __init__(self, n_tokens: int, dim_model: int, n_heads: int, 
                 n_decoder_lyrs: int, dropout_p: float, n_experts: int, 
                 top_k: int, ffn=2048, optimizer=torch.optim.Adam, 
                 SOS_token=6, EOS_token=7) -> None:
        super().__init__(n_tokens=n_tokens, dim_model=dim_model, n_heads=n_heads,
                        n_decoder_lyrs=n_decoder_lyrs, dropout_p=dropout_p, ffn=ffn, 
                        optimizer=optimizer, SOS_token=SOS_token, EOS_token=EOS_token)
        
        decoder_lyr = MoEDecoderLayer(dim_model=dim_model, n_heads=n_heads, 
                                      n_experts=n_experts, top_k=top_k, dropout=dropout_p,
                                      ffn=ffn)
        decoder_norm = nn.LayerNorm(dim_model)
        self.decoder = MoEDecoder(decoder_layer=decoder_lyr,
                                  num_layers=n_decoder_lyrs,
                                  norm=decoder_norm)