import numpy as np
from tqdm import tqdm
from multiprocessing import Pool

def generate_recurrent_matrix(n: int) -> np.ndarray:
    """
    The matrix conditional on the previous two outputs
    """
    A = np.zeros((n*n,n))
    while np.any(A == 0.0):
        A = np.random.rand(n*n,n)
        A = A / A.sum(axis=1, keepdims=True)

    return A

def generate_sample(
        start: list[int],
        mat: np.ndarray, 
        seq_len: int, 
        p_init: float
) -> np.ndarray:
    """
    Assumes the previous states are ordered from most recent to most distant, 
    i.e. idx = 0 is the last state and idx = -1 is the most distant state.
    """
    prev_states = start.copy()
    seq = np.array(prev_states)
    rvs = mat.shape[1]

    # Probability to be saved
    ps = np.log(p_init)

    for _ in range(seq_len):
        # Match the index in a n-variable case
        idx = sum(
            [ state * rvs**i for i, state in enumerate(prev_states[::-1]) ]
        )

        # Get the next state
        next_state = np.random.choice(rvs, 1, p=mat[idx])

        # Calculate the probability
        ps += np.log(mat[idx, next_state[0]])
        seq = np.hstack((seq, next_state))

        # Update the previous states
        prev_states.pop()
        prev_states.append(next_state[0])
    
    return seq, ps

def generate_starting_sequence(
        rvs: int,
        prior: np.ndarray
) -> tuple[int, list[int]]:
    # Create the start encoding
    matrix_idx = np.random.choice(rvs*rvs, p=prior)
    idx = matrix_idx
    start = []

    for _ in range(2):
        start.append(idx % rvs)
        idx //= rvs
    start = start[::-1]

    return matrix_idx, start

def build_higher_markov_dataset(
        rvs: int, 
        tr_size: int, 
        seq_len: int
) -> np.ndarray:
    # Build the transition matrices (rvs x rvs)
    markov_mats = []
    for _ in range(rvs*rvs):
        mat = generate_recurrent_matrix(rvs)
        markov_mats.append(mat)
    
    # Generate the joint prior
    prior = np.random.rand(rvs*rvs)
    prior = prior / prior.sum()

    # Generate samples
    data = []
    for _ in tqdm(range(tr_size)):
        matrix_idx, start = generate_starting_sequence(rvs, prior)
        mat = markov_mats[matrix_idx]
        p_init = prior[matrix_idx]
        sample, _ = generate_sample(start=start, mat=mat, seq_len=seq_len-2, p_init=p_init)
        data.append(sample)
    
    data = np.array(data)

    return data

def sample_worker(args):
    rvs, markov_mats, prior, seq_len = args
    matrix_idx, start = generate_starting_sequence(rvs, prior)
    p_init = prior[matrix_idx]
    mat = markov_mats[matrix_idx]
    seq, ps = generate_sample(start=start, mat=mat, seq_len=seq_len-2, p_init=p_init)
    return [seq, ps]

def build_higher_markov_big_dataset(
        rvs: int, 
        tr_size: int, 
        seq_len: int, 
        num_processes: int = 8
) -> np.ndarray:
    # Build the transition matrices
    markov_mats = []
    for _ in range(rvs*rvs):
        mat = generate_recurrent_matrix(rvs)
        markov_mats.append(mat)
    
    # Generate the prior
    prior = np.random.rand(rvs*rvs)
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
