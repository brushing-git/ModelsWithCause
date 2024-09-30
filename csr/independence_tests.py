import torch
import numpy as np
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm

def compute_log_probs(
        model: torch.nn.Module,
        x_seq: torch.tensor,
        y_seq: torch.tensor,
    ) -> torch.tensor:
    """
    Compute the log probabilities of the next token given the input tokens.

    Args:

    model : torch.nn.Module : a language model
    x_seq : torch.tensor : the input sequence with shape (batch_size, src_seq_len)
    y_seq : torch.tensor : the target sequence with shape (batch_size, tgt_seq_len)

    Returns:

    log_probs : torch.tensor : the log probabilities with shape (batch_size, vocab_size)
    """
    # Set device
    device = model.device
    model.to(device)

    # Set to eval
    model.eval()

    # Get the logit outputs
    with torch.no_grad():
        # Append SOS and EOS tokens
        x_seq, y_seq = model._append_SOS_EOS(x_seq), model._append_SOS_EOS(y_seq)
        x_seq, y_seq = x_seq.to(device), y_seq.to(device)

        # Shift the target to create the input and expected outputs
        y_input = y_seq[:, :-1] # ignore last token
        y_expected = y_seq[:, 1:] # ignore the first token
        seq_len = y_input.shape[1]

        # Create target mask
        tgt_mask = model._get_tgt_mask(seq_len).to(device)

        # Forward pass through model
        y_hat = model.forward(x_seq, y_input, tgt_mask) # shape (tgt_seq_len, batch_size, vocab_size)
        y_hat = y_hat.permute(1, 0, 2) # shape (batch_size, tgt_seq_len, vocab_size)

        # Get the logits
        last_logits = y_hat[:,-1,:] # shape (batch_size, vocab_size)

        # Compute log probs        
        log_probs = F.log_softmax(last_logits, dim=-1) # shape (batch_size, vocab_size)
    
    # Move back to device
    log_probs = log_probs.to(torch.device("cpu"))
    return log_probs

def js_divergence(
        log_probs_p: torch.tensor,
        log_probs_q: torch.tensor
) -> torch.tensor:
    """
    Compute the Jensen-Shanon divergence between two log probability distributions.

    Args:

    log_probs_p : torch.tensor : log probs from distribution P shape (batch_size, vocab_size)
    log_probs_q : torch.tensor : log probs from distribution Q shape (batch_size, vocab_size)

    Returns:

    js_div : torch.tensor : Jensen-Shannon divergence between P and Q (batch_size,)
    """
    # Convert to probabilities
    probs_p = torch.exp(log_probs_p)
    probs_q = torch.exp(log_probs_q)

    # Compute the average distribution
    m = 0.5 * (probs_p + probs_q)

    # Add a small value and move back to logarithm
    m_log = torch.log((m + 1e-12))

    # Compute the KL-divergence per batch shape (batch_size,)
    kl_pm = torch.sum(probs_p * (log_probs_p - m_log), dim=1)
    kl_qm = torch.sum(probs_q * (log_probs_q - m_log), dim=1)

    # Compute Jensen-Shannon divergence
    js_div = 0.5 * (kl_pm + kl_qm)
    return js_div

def test_markov_property(
        model: torch.nn.Module,
        sequences: np.ndarray,
        batch_size: int = 64
) -> np.ndarray:
    """
    Test the Markov property for some given sequences.

    Args:

    model : torch.nn.Module : a language model of the transformer variety
    sequences : torch.tensor : a batch of sequences to be processed shape (n_samples, seq_len)

    Returns:

    divergences : np.ndarray : the divergences for each sequence for full and reduced shape (n_samples, seq_len)
    """
    # Set n_samples, seq_len
    n_samples, seq_len = sequences.shape[0], sequences.shape[1]

    # Divergences shape (n_samples, seq_len)
    divergences = torch.zeros((n_samples, seq_len))

    # Batchify the data
    dataset = TensorDataset(torch.tensor(sequences, dtype=torch.long))
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

    # Loop through and compute probabilities
    for batch_indx, (x,) in enumerate(tqdm(loader)):
        # Get the current batch_size
        batch_size_current = x.shape[0]

        # Get initial token
        initial_tokens = x[:,0].unsqueeze(1) # shape (batch_size, 1)

        # Loop through the items in the sequence
        # Ignore the first token
        for i in range(1, seq_len):
            # Prepare the input
            input_full = x[:,:i] # shape (batch_size, i)

            # Reduce context
            if i == 1:
                input_reduced = initial_tokens # shape (batch_size, 1)
            else:
                immediate_predecessor = x[:,i-1].unsqueeze(1) # shape (batch_size, 1)
                input_reduced = torch.cat([initial_tokens, immediate_predecessor], dim=1) # shape (batch_size, 2)
            
            # Prepare the src and tgt
            x_seq_full = input_full.clone()
            y_seq_full = input_full.clone()
            x_seq_reduced = input_reduced.clone()
            y_seq_reduced = input_reduced.clone()

            # Compute log probabilities
            log_probs_full = compute_log_probs(model, x_seq_full, y_seq_full)
            log_probs_reduced = compute_log_probs(model, x_seq_reduced, y_seq_reduced)

            # Compute the Jensen-Shannon divergence
            divergence = js_divergence(log_probs_full, log_probs_reduced)

            # Store in tensor
            start_indx = batch_indx * batch_size
            end_indx = start_indx + batch_size_current
            divergences[start_indx:end_indx,i] = divergence
    
    return divergences.numpy()