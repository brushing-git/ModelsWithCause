"""
intervention.py

We use this to generate intervention datasets. The goal is to produce two datasets.

The first is a conventional markov_chain.py dataset.

The second is dataset with interventions indicated by the number '6'.  Each 6 
corresponds to restarting the markov chain.

Each intervention dataset will have random interventions that occurs throughout the data 
set of variable length.  Datasets can be generated with all possible interventions or a 
subset of interventions.

The key idea behind an intervention is the factorized distribution:

p(v | do(x)) = \prod p(v_{i} | pa_{i}) | X = x

In the case of the Markov Chains what this means is that we alter the probability of a sequence 
by deleting the probability given in the usual sequence with the aforementioned intervention.

For example, if p(v_{1}, v_{2}, ..., v_{n}) = p(v_{1}) \prod_{i=2} p(v_{i} | v_{i-1}), we say 
p(v_{1}, v_{2}, ..., v_{i-1}, v_{i+1}, ..., v_{n} | do(v_{i})) = 
                    p(v_{1}) \prod_{j=2}^{i-1} p(v_{j} | v_{j-1}) \prod_{j=i+1}^{n} p(v_{j} | v_{j-1})

We will test this by changing the probability of the generated sequences in the intervention dataset to 
reflect the truncated factorizations.  Put simply, the truncated factorizations should induce different 
inductive regularities in the data.
"""
import numpy as np
from tqdm import tqdm
from multiprocessing import Pool
from src.data.markov_chain import generate_recurrent_matrix

def generate_intervention_sample(
        start: int, 
        p_init: float, 
        mat: np.ndarray, 
        seq_len: int
) -> tuple[np.ndarray, float]:
    """
    ps is: log p(v_{1}, ..., v_{i-1}, v_{i+1}, ..., v_{n} | do(v_{i} = v))

    Returns
    seq : np.ndarray : sequence involving the intervention
    ps : float : the calculated probability of the sequence in log probabilities
    """
    state = start
    seq = np.array([state])
    rvs = mat.shape[0]

    intervention_idx = np.random.choice(np.arange(seq_len)[2:-2])
    intervention_symb = np.array([rvs])
    ps = np.log(p_init)

    i = len(seq)
    while i < seq_len+1:
        if i < intervention_idx-1 or i > intervention_idx:
            # Execute as in the generate_sample function
            next_state = np.random.choice(rvs, 1, p=mat[state])
            # Calculate the probability
            ps += np.log(mat[state, next_state[0]])
            # Stack and set next state
            seq = np.hstack((seq, next_state))
            state = next_state[0]
        else:
            # Do the intervention 
            next_state = np.random.choice(rvs, 1)

            # Insert the intervention symbol
            seq = np.hstack((seq, intervention_symb))
            
            # Insert the intervention
            seq = np.hstack((seq, next_state))

            # Set the next state
            state = next_state[0]
        
        i = len(seq)
    
    return seq, ps

def generate_normal_sample(
        start: int, 
        p_init: float, 
        mat: np.ndarray, 
        seq_len: int
) -> tuple[np.ndarray, float]:
    """
    ps is log p(v_{1}, ...., v_{n})

    Returns
    seq = np.ndarray : sequence with intervention at the end
    ps : float : the calculated probability of the sequence in log probabilities
    """
    state = start
    seq = np.array([state])
    rvs = mat.shape[0]

    ps = np.log(p_init)

    for _ in range(seq_len-1):
        next_state = np.random.choice(rvs, 1, p=mat[state])
        # Calculate the probability
        ps += np.log(mat[state, next_state[0]])
        # Stack and set next state
        seq = np.hstack((seq, next_state))
        state = next_state[0]

    # Add the last internvetion symbol
    next_state = np.array([rvs])
    seq = np.hstack((seq, next_state))
    
    return seq, ps

def sample_worker(args):
    rvs, markov_mats, prior, seq_len, intervention = args
    start = np.random.choice(rvs, p=prior)
    mat = markov_mats[start]
    p_init = prior[start]

    # Do the intervention or not
    if intervention:
        sample, ps = generate_intervention_sample(start=start, p_init=p_init, mat=mat, seq_len=seq_len-1)
        return [sample, ps]
    else:
        sample, ps = generate_normal_sample(start=start, p_init=p_init, mat=mat, seq_len=seq_len-1)
        return [sample, ps]

def build_intervention_big_dataset(
        rvs: int, 
        tr_size: int, 
        seq_len: int, 
        intervention_p: float = 0.8, 
        num_processes: int =8
) -> tuple[np.ndarray, np.ndarray, tuple[np.ndarray, np.ndarray]]:
    """
    Builds a data set, associated probabilities, and the generating stochastic matrices.

    Returns
    seq : np.ndarray : dataset of sequences with interventions and non-intervention samples
    ps : np.ndarray : associated probabilities for each sample in seq
    (prior, markov_mat) : tuple : the prior probabilities and generating markov matrices
    """
    # Compute the sizes
    tr_int_size = int(tr_size * intervention_p)
    tr_norm_size = tr_size - tr_int_size
    
    # Build the transition matrices
    markov_mats = []
    for _ in range(rvs):
        mat = generate_recurrent_matrix(rvs)
        markov_mats.append(mat)
    
    # Generate the prior
    prior = np.random.rand(rvs)
    prior = prior / prior.sum()

    # Prepare tuple of arguments for each intervention process
    pool_args = [(rvs, markov_mats, prior, seq_len, True) for _ in range(tr_int_size)]

    # Create a pool of processes and create the intervention data
    with Pool(processes=num_processes) as pool:
        int_data = list(tqdm(pool.imap(sample_worker, pool_args), total=tr_int_size))
    
    # Capture the sequences and probabilities
    int_seq = [sublist[0] for sublist in int_data]
    int_ps = [sublist[1] for sublist in int_data]

    int_seq = np.array(int_seq)
    int_ps = np.expand_dims(np.array(int_ps), axis=1)

    # Prepare tuple of arguments for each normal process
    pool_args = [(rvs, markov_mats, prior, seq_len, False) for _ in range(tr_norm_size)]

    # Create a pool of processes and create the normal data
    with Pool(processes=num_processes) as pool:
        norm_data = list(tqdm(pool.imap(sample_worker, pool_args), total=tr_norm_size))

    # Capture the sequences and probabilities
    norm_seq = [sublist[0] for sublist in norm_data]
    norm_ps = [sublist[1] for sublist in norm_data]

    norm_seq = np.array(norm_seq)
    norm_ps = np.expand_dims(np.array(norm_ps), axis=1)

    # Combine the data
    seq = np.vstack((int_seq, norm_seq))
    ps = np.vstack((int_ps, norm_ps))

    return seq, ps, (prior, markov_mats)

if __name__ == '__main__':
    print('Do not run this.')