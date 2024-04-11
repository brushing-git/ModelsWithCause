import numpy as np
from tqdm import tqdm

def generate_recurrent_matrix(n: int) -> np.ndarray:
    A = np.zeros((n,n))
    while np.any(A == 0.0):
        A = np.random.rand(n,n)
        A = A / A.sum(axis=1, keepdims=True)

    return A

def generate_sample(start: int, mat: np.ndarray, seq_len: int) -> np.ndarray:
    state = start
    seq = np.array([state])
    rvs = mat.shape[0]
    for _ in range(seq_len):
        next_state = np.random.choice(rvs, 1, p=mat[state])
        seq = np.hstack((seq, next_state))
        state = next_state[0]
    
    return seq

def build_markov_dataset(rvs: int, tr_size: int, seq_len: int) -> np.ndarray:
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
        sample = generate_sample(start=start, mat=mat, seq_len=seq_len-1)
        data.append(sample)
    
    data = np.array(data)

    return data