import numpy as np
from tqdm import tqdm
from multiprocessing import Pool

def generate_recurrent_matrix(n: int) -> np.ndarray:
    A = np.zeros((n,n))
    while np.any(A == 0.0):
        A = np.random.rand(n,n)
        A = A / A.sum(axis=1, keepdims=True)

    return A

def generate_sample(
        start: int, 
        mat: np.ndarray, 
        seq_len: int, 
        p_init: float
) -> np.ndarray:
    state = start
    seq = np.array([state])
    rvs = mat.shape[0]

    # Probability to be saved
    ps = np.log(p_init)
    for _ in range(seq_len):
        next_state = np.random.choice(rvs, 1, p=mat[state])
        # Calculate the probability
        ps += np.log(mat[state, next_state[0]])
        seq = np.hstack((seq, next_state))
        state = next_state[0]
    
    return seq, ps

def build_markov_dataset(
        rvs: int, 
        tr_size: int, 
        seq_len: int
) -> np.ndarray:
    # Build the transition matrices
    markov_mats = []
    for _ in range(rvs):
        mat = generate_recurrent_matrix(rvs)
        markov_mats.append(mat)
    
    # Generate the prior
    prior = np.random.rand(rvs)
    prior = prior / prior.sum()

    # Generate samples
    data = []
    for _ in tqdm(range(tr_size)):
        start = np.random.choice(rvs, p=prior)
        mat = markov_mats[start]
        sample, _ = generate_sample(start=start, mat=mat, seq_len=seq_len-1)
        data.append(sample)
    
    data = np.array(data)

    return data

def sample_worker(args):
    rvs, markov_mats, prior, seq_len = args
    start = np.random.choice(rvs, p=prior)
    p_init = prior[start]
    mat = markov_mats[start]
    seq, ps = generate_sample(start=start, mat=mat, seq_len=seq_len-1, p_init=p_init)
    return [seq, ps]

def build_markov_big_dataset(
        rvs: int, 
        tr_size: int, 
        seq_len: int, 
        num_processes: int = 8
) -> np.ndarray:
    # Build the transition matrices
    markov_mats = []
    for _ in range(rvs):
        mat = generate_recurrent_matrix(rvs)
        markov_mats.append(mat)
    
    # Generate the prior
    prior = np.random.rand(rvs)
    prior = prior / prior.sum()

    # Prepare tuple of arguments for each process
    pool_args = [(rvs, markov_mats, prior, seq_len) for _ in range(tr_size)]

    # Create a pool of processes
    with Pool(processes=num_processes) as pool:
        data = list(tqdm(pool.imap(sample_worker, pool_args), total=tr_size))
    
    # Capture the sequences and probabilities
    seq = [sublist[0] for sublist in data]
    ps = [sublist[1] for sublist in data]

    seq = np.array(seq)
    ps = np.expand_dims(np.array(ps), axis=1)

    return seq, ps, (prior, markov_mats)

if __name__ == '__main__':
    print('Do not run this.')
