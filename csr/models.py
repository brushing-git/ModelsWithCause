import torch
import gc
import torch.nn as nn
import csr.utils as ut
from abc import abstractmethod
from torch.distributions import Categorical
from torch.nn.functional import softmax
from torch.optim.lr_scheduler import CosineAnnealingLR
from csr.moe import MoEDecoderLayer, MoEDecoder
from math import log, sqrt
from tqdm import tqdm

class NADE(nn.Module):
    def __init__(
            self, 
            in_dim: int, 
            hidden_dim: int, 
            cat=2, 
            optimizer=torch.optim.Adam
    ) -> None:
        """
        A Neural Autogregressive Distribution Estimator class as implemented in https://arxiv.org/abs/1605.02226.

        Basic architecture:

            p(x_next | x_previous:start ) = V_next @ h_d + b_next
            h_d = sigmoid(a_d + W_next:next+1 @ x_next:next+1)
            a_d = c

        Attributes
        self.D : int : the input dimension, which is the length of the input tokens
        self.H : int : the size of the hidden layer for the masked autoencoder
        self.C : int : the range of the output categorical random variable
        self.activation : torch.nn.Sigmoid : the hidden activation function
        self.logprob : torch.nn.LogSoftmax : the log softmax for computing output probabilities
        self.optimizer : torch.optim.Optimizer : an optimizer for training
        self.device : torch.device : a device, set automatically by utilities
        self.params : torch.nn.ParameterDict : a dictionary of the parameters, consisting of
            V: the output weight matrix (H, D, C)
            b: the output bias matrix (D, C)
            W: the hidden weight matrix (H, D)
            c: the hidden bias matrix (1, H)
        """
        super(NADE, self).__init__()
        self.D = in_dim
        self.H = hidden_dim
        self.C = cat
        self.activation = nn.Sigmoid()
        self.logprob = nn.LogSoftmax(dim=1)
        self.optimizer = optimizer
        self.device = ut.set_device()
        
        # Create the parameter dictionary
        self.params = nn.ParameterDict({
            'V': nn.Parameter(torch.randn(self.H, self.D, self.C)),
            'b': nn.Parameter(torch.zeros(self.D, self.C)),
            'W': nn.Parameter(torch.randn(self.H, self.D)),
            'c': nn.Parameter(torch.zeros(1, self.H))
        })

        # Initialize the weight matrices with a xavier normal
        nn.init.xavier_normal_(self.params['V'])
        nn.init.xavier_normal_(self.params['W'])
    
    def forward(
            self, 
            x: torch.tensor
    ) -> tuple:
        # Initialize the initial input
        a_d = self.params['c'].expand(x.shape[0], -1)
        
        # Compute the predictions and probabilities
        y_hat, p_hat = self._estimate_logits(a_d, x)

        return y_hat, p_hat

    def _estimate_logits(
            self, 
            a_d: torch.tensor, 
            x: torch.tensor, 
            sample=False
    ) -> tuple:
        """
        Estimates the logits, log probabilities, and if sampling is required, returns a sample.

        Params
        a_d : torch.tensor : the initial input built from the hidden input biases
        x : torch.tensor : a single batch tensor for the input sequence
        sample : bool : governs whether we return a sample or not

        Returns
        y_hat : torch.tensor : the output logits with dim (length, categories)
        p_hat : torch.tensor : the output log probabilities with dim (length, categories)
        x_hat : torch.tensor : an output sample with dim (1, variable length)
        """
        # Check to make sure we have an input for sampling
        if sample:
            assert (x is None), "No input for sampling as first time"

        # Storage for the outputs
        y_hat = []
        x_hat = []
        p_hat = []

        # Set sequence length
        seq_len = self.D if sample else x.shape[1]

        # Loop through the sequence and compute the masked autoencoder
        for d in range(seq_len):
            # Apply the activation
            h_d = self.activation(a_d)

            # Get the output logits and probabilities
            logits = h_d @ self.params['V'][:,d,:] + self.params['b'][d,:]
            p_d = self.logprob(logits)

            # Add them to the output list
            y_hat.append(logits)
            p_hat.append(p_d)
            
            # Update the hidden state
            if sample:
                # Sample and add sample
                x = Categorical(probs=p_d).sample().type(torch.float)
                x_hat.append(x)

                # Update the hidden state
                a_d = a_d + x @ self.params['W'][:,d:d+1].t()
            else:
                # Update the hidden state
                a_d = a_d + x[:,d:d+1] @ self.params['W'][:,d:d+1].t()
        
        # Concatenate along the first dimension
        y_hat = torch.cat(y_hat, 1)
        p_hat = torch.cat(p_hat, 1)

        if sample:
            # Create the output sample
            x_hat = torch.cat(x_hat)
            x_hat = x_hat.type(torch.long)
            return y_hat, p_hat, x_hat
        else:
            return y_hat, p_hat
    
    def _compute_loss(
            self, 
            y_hat: torch.tensor, 
            y: torch.tensor, 
            loss_fn: torch.nn.CrossEntropyLoss
    ) -> torch.Tensor:
        """
        Computes the loss.

        Params
        y_hat : torch.tensor : the model logits
        y : torch.tensor : the original sequence and hence targets
        loss_fn : torch.nn.CrossEntropyLoss : the CrossEntropyLoss

        Returns
        loss : torch.tensor : the loss computed from the sequence
        """
        # Initialize the loss
        total_loss = 0

        # Loop through and compute the losses
        for i in range(self.D):
            # Chunk for the corresponding logits on this sequence
            start_idx = i*self.C
            end_idx = start_idx+self.C

            # Get the logit predictions
            predictions = y_hat[:,start_idx:end_idx]

            # Get the true label
            true_features = y[:,i]

            # Compute the loss
            total_loss += loss_fn(predictions, true_features)
        
        # Average the loss
        loss = total_loss / self.D

        return loss

    def _train_step(
            self, 
            x: torch.tensor, 
            y: torch.tensor, 
            optim_fn: torch.optim.Optimizer, 
            loss_fn: torch.nn.CrossEntropyLoss
    ) -> float:
        """
        Computes the train step for the model by computing loss and backpropagating it.

        Params
        x : torch.tensor : tensor for the input sequence
        y : torch.tensor : the same tensor
        optim_fn : torch.optim.Optimizer : an optimizer to apply SGD
        loss_fn : torch.nn.CrossEntropyLoss : the CrossEntropy Loss

        Returns
        loss_item : float : the total loss on the train step
        """
        # Forward pass
        y_hat, _ = self.forward(x)

        # Compute loss
        loss = self._compute_loss(y_hat, y, loss_fn)

        # Backpropagate
        optim_fn.zero_grad()
        loss.backward()

        # Step the optimizer
        optim_fn.step()

        # Grab the loss and detach
        loss_item = loss.detach().item()

        return loss_item
    
    def _eval(
            self, 
            te_loader: torch.utils.data.DataLoader, 
            loss_fn: torch.nn.CrossEntropyLoss
    ) -> float:
        """
        Semi-private method for evaluating a model on a validation set.

        Params
        te_loader : DataLoader : a data loader
        loss_fn : torch.nn.CrossEntropyLoss : the CrossEntropyLoss function

        Returns
        te_loss : float : the validation loss
        """
        # Validation loss and evaluation
        val_loss = []
        self.eval()
        
        # Apply no grad
        with torch.no_grad():
            # Loop through the loader
            for x, y in iter(te_loader):
                # Move to device
                x, y = x.to(self.device), y.to(self.device)
            
                # Forward pass for logits
                y_hat, _ = self.forward(x)

                # Compute loss
                loss = self._compute_loss(y_hat, y, loss_fn)

                # Detach and append item
                loss = loss.detach()
                val_loss.append(loss.item())
        
            # Average the loss
            te_loss = sum(val_loss) / len(val_loss)

        return te_loss
    
    def sample(
            self, 
            n: int = 1
    ) -> list:
        """
        Samples the model and returns the sample.

        Params
        n : int : the length of the sample

        Returns
        x_hat : list : a sampled list
        """
        # Expand the initial biases by the length
        a_d = self.params['c'].expand(n, -1)

        # Sample based on the logits
        _, _, x_hat = self._estimate_logits(a_d, x=None, sample=True)

        return x_hat.tolist()
    
    def estimate_prob(
            self, 
            x: torch.tensor, 
            y: torch.tensor
    ) -> list:
        """
        Gets the estimated probabilites from the input.

        Params
        x : torch.tensor : the input tensor
        y : torch.tensor : a dummy for congruence with other class methods

        Returns
        p_hat : list : the probabilities across the assigned tokens
        """
        # Move to device and set to evaluation
        self.to(self.device)
        self.eval()
        x = x.to(self.device)

        # Compute the probabilities
        _, p_hat = self.forward(x)

        return p_hat.tolist()
    
    def parameter_count(self) -> None:
        """
        Prints the number of parameters.

        Params:
        None

        Returns:
        None
        """
        total_params = sum(p.numel() for p in self.parameters())
        print(f"Number of parameters: {total_params}")    

    def fit(
            self, 
            tr_loader: torch.utils.data.DataLoader, 
            te_loader: torch.utils.data.DataLoader, 
            epochs: int, 
            lr: float, 
            step_size: int = 50
    ) -> dict:
        """
        Fits the model to a training data set and evaluates it.

        Params
        tr_loader : DataLoader : the training data loader
        te_loader : DataLoader : the validation data loader
        epochs : int : number of training loops
        lr : float : the learning rate
        step_size : int : the number of steps between halving the learning rate

        Returns
        hist : dict : a dictionary of the learning statistics
        """
        # Set to device
        self.to(self.device)

        # Initialize the optimizer
        optimizer = self.optimizer(self.parameters(), lr=lr)

        # Apply the Step Scheduler
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=0.5)

        # Set the loss_fn as CrossEntropyLoss
        loss_fn = nn.CrossEntropyLoss()

        # Set up the history
        hist = {'tr_loss': [], 'te_loss': [], 'te_auc': [], 'current_lr': []}

        # Loop and train the model
        for step in range(epochs):
            # Set to train
            self.train()
            tr_loss = []

            # Go through the training data
            for x, y in tqdm(tr_loader):
                x, y = x.to(self.device), y.to(self.device)

                # Compute the loss
                loss = self._train_step(x, y, optimizer, loss_fn)

                # Save the loss
                tr_loss.append(loss)
            
            # Evaluate the test statistics
            te_loss = self._eval(te_loader, loss_fn)
            tr_avg_loss = sum(tr_loss) / len(tr_loss)
            current_lr = scheduler.get_last_lr()[0]

            # Print and save the training statistiscs
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

            # Step the scheduler
            scheduler.step()
        
        # Garbage cleanup
        ut.clear_cache()
        gc.collect()

        return hist

class PositionalEncoding(nn.Module):
    def __init__(
            self, 
            dim_model: int, 
            dropout_p: float, 
            max_len: int, 
            device: torch.device
    ) -> None:
        """
        Sinusoidal positional encoder for injecting order information into Transformer processing.

        Attributes
        dropout : torch.nn.Dropout : Dropout layer
        pos_encoding : torch.tensor : tensor that is (max_len, dim_model)
        """
        super().__init__()

        # Initialize dropout
        self.dropout = nn.Dropout(dropout_p)

        # Initialize encoding
        pos_encoding = torch.zeros(max_len, dim_model)
        positions_list = torch.arange(0, max_len, dtype=torch.float).view(-1,1)

        # 1000^(2i/dim_model)
        division_term = torch.exp(torch.arange(0, dim_model, 2).float()
                                  * (-log(10000.0)) / dim_model)
        
        # PE(pos, 2i) = sin(pos/1000^(2i/dim_model))
        pos_encoding[:,0::2] = torch.sin(positions_list * division_term)

        # PE(pos, 2i + 1) = cos(pos/1000^(2i/dim_model))
        pos_encoding[:,1::2] = torch.cos(positions_list * division_term)

        # Saving posititional encoding
        self.pos_encoding = pos_encoding.unsqueeze(0).transpose(0,1).to(device)
    
    def forward(
            self, 
            token_embedding: torch.tensor
    ) -> torch.tensor:
        """
        Applies the positional encoder on input.

        Params
        token_embedding : torch.tensor : the embedding of the tokens (max_len, dim_model)

        Returns
        token_embedding : torch.tensor : token_embedding after adding in positional information and applying dropout
        """
        return self.dropout(token_embedding + self.pos_encoding[:token_embedding.size(0),:])

class GenerativeModel(nn.Module):
    def __init__(
            self,
            n_tokens: int,
            dim_model: int,
            dropout_p: float,
            optimizer: torch.optim.Optimizer = torch.optim.Adam,
            SOS_token: int = 6,
            EOS_token: int = 7
    ) -> None:
        """
        Base class for the transformer based models.

        Attributes
        n_tokens : int : the vocabulary size
        dim_model : int : the dimension of the embedding space
        dropout_p : float : value between 0.0 and 1.0 for dropout probability
        optimizer : torch.optim.Optimizer : the optimizer for the fit method
        SOS_token : int : the start of sequence token
        EOS_token : int : the end of sequence token
        """
        super().__init__()

        # Set parameters
        self.dim_model = dim_model
        self.optimizer = optimizer
        self.SOS_token = SOS_token
        self.EOS_token = EOS_token
        self.logprob = nn.LogSoftmax(dim=-1)
        self.device = ut.set_device()

        # Positional Encoder
        self.positional_encoder = PositionalEncoding(
            dim_model=dim_model, dropout_p=dropout_p, max_len=5000, device=self.device
        )

        # Embedding
        self.embedding = nn.Embedding(n_tokens, dim_model)
    
    @abstractmethod
    def forward(self) -> None:
        pass

    def _append_SOS_EOS(
            self, 
            x: torch.tensor
    ) -> torch.tensor:
        """
        Adds the start of sequence and end of sequence tokens to an input tensor.

        Params
        x : torch.tensor : the input tensor for input

        Returns
        x : torch.tensor : the input tensor with the SOS and EOS tokens
        """
        # Convert to a long tensor
        x = x.type(torch.long)

        # Add the SOS and EOS tokens
        sos = torch.full((x.shape[0],1), self.SOS_token)
        eos = torch.full((x.shape[0],1), self.EOS_token)

        # Concatenate the sequence
        x = torch.cat((sos, x, eos), dim=1)

        return x
    
    def _get_tgt_mask(
            self, 
            size: int
    ) -> torch.tensor:
        """
        Creates the causal mask for both src and tgt.

        Params
        size : int : the size of the input tensor

        Returns
        mask : torch.tensor : the mask, i.e. a lower triangle matrix
        """
        # Create lower triangle matrix
        mask = torch.tril(torch.ones(size, size) == 1).float()

        # Fill the zero entries with -inf
        mask = mask.masked_fill(mask == 0, float('-inf'))

        # Fill the 1.0s with zeros
        mask = mask.masked_fill(mask == 1, float(0.0))

        return mask
    
    def _create_pad_mask(
            self, 
            matrix: torch.tensor, 
            pad_token: int
    ) -> torch.tensor:
        """
        Adds padding as needed for the sequence.

        Params
        matrix : torch.tensor : the matrix to pad
        pad_token : int : the padding token to add

        Returns
        matrix : torch.tensor : the padded matrix
        """
        return (matrix == pad_token)
    
    def _train_step(
            self, 
            x: torch.tensor, 
            y: torch.tensor, 
            optim_fn: torch.optim.Optimizer, 
            loss_fn: torch.nn.CrossEntropyLoss
    ) -> float:
        """
        Train step for processing an input and computing loss.

        Params
        x : torch.tensor : the input tensor
        y : torch.tensor : the target tensor (a duplicate of the input tensor)
        optim_fn : torch.optim.Optimizer : optimizer such as Adam
        loss_fn : torch.nn.CrossEntropy : loss function expecting logits as input

        Returns
        loss_item : float : the loss on this particular batch
        """
        # Shift the target and shift the prediction to avoid SOS and EOS
        y_input = y[:,:-1]
        y_expected = y[:,1:].contiguous()

        # Mask the other letters
        sequence_length = y_input.shape[1]
        tgt_mask = self._get_tgt_mask(sequence_length).to(self.device)

        # Forward pass through the model, batch first
        y_hat = self.forward(x, y_input, tgt_mask).contiguous()

        # Apply loss function
        loss = loss_fn(y_hat.view(-1, y_hat.shape[-1]), y_expected.view(-1))

        # Backpropagate
        optim_fn.zero_grad()
        loss.backward()

        # Step the optimizer
        optim_fn.step()

        # Detach loss
        loss_item = loss.detach().item()

        # Garbage clean up
        del loss, y_hat, x, y

        return loss_item
    
    def _eval(
            self, 
            te_loader: torch.utils.data.DataLoader, 
            loss_fn: torch.nn.CrossEntropyLoss
    ) -> float:
        """
        Private evaluation for testing validation loss.

        Params
        te_loader : torch.utils.data.DataLoader : the validation data loader
        loss_fn : torch.nn.CrossEntropyLoss : the cross entropy loss function
        """
        # Create val list
        val_loss = []

        # Set to eval
        self.eval()

        # Apply torch no grad
        with torch.no_grad():
            # Loop through loader
            for x, y in iter(te_loader):
                # Append SOS and EOS tokens
                x, y = self._append_SOS_EOS(x), self._append_SOS_EOS(y)

                # Set to device
                x, y = x.to(self.device), y.to(self.device)

                # Shift the target, shape (B, S)
                y_input = y[:,:-1]
                y_expected = y[:,1:].contiguous()

                # Apply the target mask
                sequence_length = y_input.shape[1]
                tgt_mask = self._get_tgt_mask(sequence_length).to(self.device)

                # Forward pass, shape (B, S, T)
                y_hat = self.forward(x, y_input, tgt_mask).contiguous()

                # Compute loss
                loss = loss_fn(y_hat.view(-1, y_hat.shape[-1]), y_expected.view(-1))
                loss = loss.detach()
                val_loss.append(loss.item())
        
            te_loss = sum(val_loss) / len(val_loss)

        return te_loss
    
    def sample(
            self, 
            x: torch.tensor, 
            max_length: int
    ) -> list:
        """
        For sampling from the model, using a multinomial.

        Params
        x : torch.tensor : the input tensor to start sampling
        max_length : int : the maximum length to sample

        Returns
        y_input : torch.tensor : the sample from the model
        """
        # Move to device and set to evaluate
        self.to(self.device)
        self.eval()

        # Append the SOS and EOS token
        x = self._append_SOS_EOS(x)
        x = x.to(self.device)

        # Start the tensor with the SOS Token
        y_input = torch.tensor([[self.SOS_token]], dtype=torch.long, device=self.device)

        # The number of tokens
        num_tokens = len(x[0])

        # Sample with no gradient
        with torch.no_grad():
            # Loop through max_length
            for _ in range(max_length):
                # Apply target mask
                tgt_mask = self._get_tgt_mask(y_input.shape[1]).to(self.device)

                # Get the output logits
                output = self.forward(x, y_input, tgt_mask)

                # Convert to softmax probabilities
                probs = nn.functional.softmax(output[:,-1,:], dim=-1)

                # Sample from multinomial
                next_item = torch.multinomial(probs, 1).item()

                # Convert to a tensor
                next_item = torch.tensor([[next_item]], device=self.device)

                # Concatenate
                y_input = torch.cat((y_input, next_item), dim=1)

                # Break if we get the last token
                if next_item.view(-1).item() == self.EOS_token:
                    break

        return y_input.view(-1).tolist()
    
    def estimate_prob(
            self, 
            x: torch.tensor, 
            y: torch.tensor
    ) -> torch.tensor:
        """
        Estimates probabilities from the sequence.

        Params
        x : torch.tensor : the input tensor
        y : torch.tensor : copy of the input tensor

        Returns
        p_hat : torch.tensor : tensor of probabilities
        """
        # Move to device and set to eval
        self.to(self.device)
        self.eval()

        with torch.no_grad():
            # Add the SOS and EOS tokens
            x, y = self._append_SOS_EOS(x), self._append_SOS_EOS(y)

            # Add to device
            x, y = x.to(self.device), y.to(self.device)

            # Shift the target, ignoring the last and the first token
            y_input = y[:,:-1]
            y_expected = y[:,1:]

            # Get the sequence length
            sequence_length = y_input.shape[1]

            # Apply the target mask
            tgt_mask = self._get_tgt_mask(sequence_length).to(self.device)

            # Get the logits
            y_hat = self.forward(x, y_input, tgt_mask)

            # Apply the log probability function
            ps = self.logprob(y_hat)

            # Get the correct labels
            target_sequence = y[:, 1:-1]

            # Sum the target probabilities, here the last dimension (2)
            p_hat = torch.gather(ps, 2, target_sequence.unsqueeze(1)).squeeze(1)

        return p_hat.tolist()

    def parameter_count(self):
        """
        Prints the number of parameters.

        Params:
        None

        Returns:
        None
        """
        total_params = sum(p.numel() for p in self.parameters())
        print(f"Number of parameters: {total_params}")

    def fit(
            self, 
            tr_loader: torch.utils.data.DataLoader, 
            te_loader: torch.utils.data.DataLoader, 
            epochs: int, 
            lr: float, 
            step_size: int = 50
    ) -> dict:
        """
        Fit method for training the Generative Model.

        Params
        tr_loader : DataLoader : the training data loader
        te_loader : DataLoader : the validation data loader
        epochs : int : number of training loops
        lr : float : the learning rate
        step_size : int : the number of steps between halving the learning rate

        Returns
        hist : dict : a dictionary of the learning statistics
        """
        # Move to device
        self.to(self.device)

        # Initialize the optimizer
        optimizer = self.optimizer(self.parameters(), lr=lr)

        # Set a CosineAnnealing Scheduler
        scheduler = CosineAnnealingLR(optimizer=optimizer, T_max=epochs)

        # Set the loss function to CrossEntropyLoss
        loss_fn = nn.CrossEntropyLoss()

        # Dictionary for storing statistics
        hist = {'tr_loss': [], 'te_loss': [], 'te_auc': [], 'current_lr': []}

        for step in range(epochs):
            # Set to train
            self.train()

            # Set training loss list
            tr_loss = []

            # Go through the batches
            for x, y in tqdm(tr_loader):
                # Append the SOS and EOS tokens
                x, y = self._append_SOS_EOS(x), self._append_SOS_EOS(y)

                # Move to device
                x, y = x.to(self.device), y.to(self.device)

                # Perform the train step
                loss = self._train_step(x, y, optimizer, loss_fn)

                # Append the loss
                tr_loss.append(loss)
            
            # Evaluate on the validation data
            te_loss = self._eval(te_loader, loss_fn)

            # Get average training loss
            tr_avg_loss = sum(tr_loss) / len(tr_loss)

            # Get the current learning rate
            current_lr = scheduler.get_last_lr()[0]

            # Print and store the metrics
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

            # Step the optimizer
            scheduler.step()

            # Garbage cleanup
            ut.clear_cache()
            gc.collect()

        return hist

class Transformer(GenerativeModel):
    def __init__(
            self, 
            n_tokens: int, 
            dim_model: int, 
            n_heads: int, 
            n_encoder_lyrs: int, 
            n_decoder_lyrs: int, 
            dropout_p: float,
            ffn: int = 2048, 
            optimizer: torch.optim.Optimizer = torch.optim.Adam, 
            SOS_token: int = 6, 
            EOS_token: int = 7
    ) -> None:
        """
        Transformer model with encoder and decoder layers. Utilizes the pytorch transformer model as base.

        Attributes
        n_tokens : int : the vocabulary size
        dim_model : int : the dimension of the embedding space
        n_heads : int : number of heads in multiheaded attention; must evenly divide into dim_model
        n_encoder_lyrs : int : number of encoder layers
        n_decoder_lyrs : int : number of decoder layers
        dropout_p : float : value between 0.0 and 1.0 for dropout probability
        ffn : int : the width of the positionwise mlps
        optimizer : torch.optim.Optimizer : the optimizer for the fit method
        SOS_token : int : the start of sequence token
        EOS_token : int : the end of sequence token
        """
        super().__init__(
            n_tokens,
            dim_model,
            dropout_p,
            optimizer=optimizer,
            SOS_token=SOS_token,
            EOS_token=EOS_token
        )
        # Transformer
        self.transformer = nn.Transformer(
            d_model=dim_model,
            nhead=n_heads,
            num_encoder_layers=n_encoder_lyrs,
            num_decoder_layers=n_decoder_lyrs,
            dim_feedforward=ffn,
            dropout=dropout_p,
            batch_first=True
        )

        # Output layer
        self.out = nn.Linear(dim_model, n_tokens)

    def forward(
            self, 
            src: torch.tensor, 
            tgt: torch.tensor, 
            tgt_mask: torch.tensor = None, 
            src_pad_mask: torch.tensor = None, 
            tgt_pad_mask: torch.tensor = None
    ) -> torch.tensor:
        """
        Implements a forward pass with the transformer block.

        Params
        src : torch.tensor : the source tensor to compare with the target
        tgt : torch.tensor : the target tensor to compare with the source
        tgt_mask : torch.tensor : the target mask
        src_pad_mask : torch.tensor : the padded mask if needed
        tgt_pad_mask : the torch.tensor : the padded mask if needed

        Returns
        out : torch.tensor : the output of the transformer block
        """
        # Embed
        src = self.embedding(src) * sqrt(self.dim_model)
        tgt = self.embedding(tgt) * sqrt(self.dim_model)

        # Apply positional encoding
        src = self.positional_encoder(src)
        tgt = self.positional_encoder(tgt)

        # Move through transformer block
        transformer_out = self.transformer(
            src, 
            tgt, 
            tgt_mask=tgt_mask,
            src_key_padding_mask=src_pad_mask, 
            tgt_key_padding_mask=tgt_pad_mask
        )

        # Apply fc layer
        out = self.out(transformer_out)

        return out

class DecoderTransformer(GenerativeModel):
    def __init__(
            self, 
            n_tokens: int, 
            dim_model: int, 
            n_heads: int, 
            n_decoder_lyrs: int, 
            dropout_p: float, 
            ffn: int = 2048,
            optimizer: torch.optim.Optimizer = torch.optim.Adam, 
            SOS_token: int = 6, 
            EOS_token: int = 7
    ) -> None:
        """
        Decoder transformer model. Utilizes the pytorch decoder model as base.

        Attributes
        n_tokens : int : the vocabulary size
        dim_model : int : the dimension of the embedding space
        n_heads : int : number of heads in multiheaded attention; must evenly divide into dim_model
        n_decoder_lyrs : int : number of decoder layers
        dropout_p : float : value between 0.0 and 1.0 for dropout probability
        ffn : int : the width of the positionwise mlps
        optimizer : torch.optim.Optimizer : the optimizer for the fit method
        SOS_token : int : the start of sequence token
        EOS_token : int : the end of sequence token
        """
        super().__init__(
            n_tokens,
            dim_model,
            dropout_p,
            optimizer=optimizer,
            SOS_token=SOS_token,
            EOS_token=EOS_token
        )
        
        # Decoder layer
        decoder_lyr = nn.TransformerDecoderLayer(
            d_model=dim_model, 
            nhead=n_heads, 
            dim_feedforward=ffn, 
            dropout=dropout_p,
            batch_first=True
        )

        # Decoder renormalization
        decoder_norm = nn.LayerNorm(dim_model)

        # The decoder block
        self.decoder = nn.TransformerDecoder(
            decoder_layer=decoder_lyr,
            num_layers=n_decoder_lyrs,
            norm=decoder_norm
        )

        # Output layer
        self.out = nn.Linear(dim_model, n_tokens)
    
    def forward(
            self, 
            src: torch.tensor, 
            tgt: torch.tensor, 
            tgt_mask: torch.tensor = None, 
            src_pad_mask: torch.tensor = None, 
            tgt_pad_mask: torch.tensor = None
    ) -> torch.tensor:
        """
        Implements a forward pass with the decoder block.

        Params
        src : torch.tensor : the source tensor to compare with the target
        tgt : torch.tensor : the target tensor to compare with the source
        tgt_mask : torch.tensor : the target mask
        src_pad_mask : torch.tensor : the padded mask if needed
        tgt_pad_mask : the torch.tensor : the padded mask if needed

        Returns
        out : torch.tensor : the output of the transformer block
        """
        # Embed
        src = self.embedding(src) * sqrt(self.dim_model)
        tgt = self.embedding(tgt) * sqrt(self.dim_model)

        # Apply positional encoder
        src = self.positional_encoder(src)
        tgt = self.positional_encoder(tgt)

        # Push through decoder block
        decoder_out = self.decoder(
            tgt, 
            src, 
            tgt_mask=tgt_mask, 
            tgt_key_padding_mask=tgt_pad_mask, 
            memory_key_padding_mask=src_pad_mask
        )

        # Apply fc layer
        out = self.out(decoder_out)

        return out

class MoEDecoderTransformer(GenerativeModel):
    def __init__(
            self, 
            n_tokens: int, 
            dim_model: int, 
            n_heads: int, 
            n_decoder_lyrs: int, 
            dropout_p: float, 
            n_experts: int, 
            top_k: int, 
            ffn: int = 2048, 
            optimizer: torch.optim.Optimizer = torch.optim.Adam, 
            SOS_token: int = 6, 
            EOS_token: int = 7
    ) -> None:
        """
        Switch transformer model. Utilizes custom switch transformer architecture.

        Attributes
        n_tokens : int : the vocabulary size
        dim_model : int : the dimension of the embedding space
        n_heads : int : number of heads in multiheaded attention; must evenly divide into dim_model
        n_decoder_lyrs : int : number of decoder layers
        dropout_p : float : value between 0.0 and 1.0 for dropout probability
        n_experts : int : the number of experts per mlp block
        top_k : int : the number of experts to query
        ffn : int : the width of the positionwise mlps
        optimizer : torch.optim.Optimizer : the optimizer for the fit method
        SOS_token : int : the start of sequence token
        EOS_token : int : the end of sequence token
        """
        super().__init__(
            n_tokens,
            dim_model,
            dropout_p,
            optimizer=optimizer,
            SOS_token=SOS_token,
            EOS_token=EOS_token
        )
        
        # Decoder layer
        decoder_lyr = MoEDecoderLayer(
            dim_model=dim_model, 
            n_heads=n_heads, 
            n_experts=n_experts, 
            top_k=top_k, 
            dropout=dropout_p,
            ffn=ffn
        )

        # Decoder normalization
        decoder_norm = nn.LayerNorm(dim_model)

        # Decoder block
        self.decoder = MoEDecoder(
            decoder_layer=decoder_lyr,
            num_layers=n_decoder_lyrs,
            norm=decoder_norm
        )

        # Output layer
        self.out = nn.Linear(dim_model, n_tokens)
        
    def forward(
            self, 
            src: torch.tensor, 
            tgt: torch.tensor, 
            tgt_mask: torch.tensor = None, 
            src_pad_mask: torch.tensor = None, 
            tgt_pad_mask: torch.tensor = None
    ) -> torch.tensor:
        """
        Implements a forward pass with the decoder block.

        Params
        src : torch.tensor : the source tensor to compare with the target
        tgt : torch.tensor : the target tensor to compare with the source
        tgt_mask : torch.tensor : the target mask
        src_pad_mask : torch.tensor : the padded mask if needed
        tgt_pad_mask : the torch.tensor : the padded mask if needed

        Returns
        out : torch.tensor : the output of the transformer block
        """
        # Embed
        src = self.embedding(src) * sqrt(self.dim_model)
        tgt = self.embedding(tgt) * sqrt(self.dim_model)

        # Apply positional encoder
        src = self.positional_encoder(src)
        tgt = self.positional_encoder(tgt)

        # Push through decoder block
        decoder_out = self.decoder(
            tgt, 
            src, 
            tgt_mask=tgt_mask, 
            tgt_key_padding_mask=tgt_pad_mask, 
            memory_key_padding_mask=src_pad_mask
        )

        # Apply fc layer
        out = self.out(decoder_out)

        return out