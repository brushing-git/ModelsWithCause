import numpy as np

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
    for _ in range(tr_size):
        start = np.random.choice(rvs, p=prior)
        mat = markov_mats[start]
        sample = generate_sample(start=start, mat=mat, seq_len=seq_len-1)
        data.append(sample)
    
    data = np.array(data)

    return data

########################## SMALL SEQUENCE ##########################
# generate dataset with parameters:
rvs = 6
tr_size = 50000 # fifty thousand
seq_len = 10
dataset = build_markov_dataset(rvs, tr_size, seq_len)

# print parameters
f = open('DATASETS/markov_chain-dice-small-parameters.txt', 'w')
param = "number of possible states:" + str(rvs) + "\n"
param += "training size: " + str(tr_size) + "\n"
param += "sequence length: " + str(seq_len) + "\n"
param += "procedure:\n\t1. generate " + str(rvs) + " " + str(rvs) + "x"  + str(rvs) + " recurrent matrix of floats between (but exlcuding) 0 and 1.\n\t"
param += "2. select a starting matrix based on random distrubution of the possible states.\n\t"
param += "3. using this matrix, generate a sample in which sequences with the same leading state and transition counts are generated in the same fashion.\n\t"
param += "4. repeat."
f.write(param)
f.close()

np.savetxt("DATASETS/markov_chain-dice-small-training.txt", dataset, delimiter="", newline=",", fmt='%d')

from humanReadable import *
translateCSV(dataset, 0, "DATASETS/markov_chain-dice-small-readable.csv")

# generate dataset with parameters:
rvs = 2
tr_size = 50000 # fifty thousand
seq_len = 10
dataset = build_markov_dataset(rvs, tr_size, seq_len)

# print parameters
f = open('DATASETS/markov_chain-coin-small-parameters.txt', 'w')
param = "number of possible states:" + str(rvs) + "\n"
param += "training size: " + str(tr_size) + "\n"
param += "sequence length: " + str(seq_len) + "\n"
param += "procedure:\n\t1. generate " + str(rvs) + " " + str(rvs) + "x"  + str(rvs) + " recurrent matrix of floats between (but exlcuding) 0 and 1.\n\t"
param += "2. select a starting matrix based on random distrubution of the possible states.\n\t"
param += "3. using this matrix, generate a sample in which sequences with the same leading state and transition counts are generated in the same fashion.\n\t"
param += "4. repeat."
f.write(param)
f.close()

np.savetxt("DATASETS/markov_chain-coin-small-training.txt", dataset, delimiter="", newline=",", fmt='%d')

from humanReadable import *
translateCSV(dataset, 0, "DATASETS/markov_chain-coin-small-readable.csv")

########################## MEDIUM SEQUENCE ##########################
# generate dataset with parameters:
rvs = 6
tr_size = 50000 # fifty thousand
seq_len = 100
dataset = build_markov_dataset(rvs, tr_size, seq_len)

# print parameters
f = open('DATASETS/markov_chain-dice-medium-parameters.txt', 'w')
param = "number of possible states:" + str(rvs) + "\n"
param += "training size: " + str(tr_size) + "\n"
param += "sequence length: " + str(seq_len) + "\n"
param += "procedure:\n\t1. generate " + str(rvs) + " " + str(rvs) + "x"  + str(rvs) + " recurrent matrix of floats between (but exlcuding) 0 and 1.\n\t"
param += "2. select a starting matrix based on random distrubution of the possible states.\n\t"
param += "3. using this matrix, generate a sample in which sequences with the same leading state and transition counts are generated in the same fashion.\n\t"
param += "4. repeat."
f.write(param)
f.close()

np.savetxt("DATASETS/markov_chain-dice-medium-training.txt", dataset, delimiter="", newline=",", fmt='%d')

from humanReadable import *
translateCSV(dataset, 0, "DATASETS/markov_chain-dice-medium-readable.csv")

# generate dataset with parameters:
rvs = 2
tr_size = 50000 # fifty thousand
seq_len = 100
dataset = build_markov_dataset(rvs, tr_size, seq_len)

# print parameters
f = open('DATASETS/markov_chain-coin-medium-parameters.txt', 'w')
param = "number of possible states:" + str(rvs) + "\n"
param += "training size: " + str(tr_size) + "\n"
param += "sequence length: " + str(seq_len) + "\n"
param += "procedure:\n\t1. generate " + str(rvs) + " " + str(rvs) + "x"  + str(rvs) + " recurrent matrix of floats between (but exlcuding) 0 and 1.\n\t"
param += "2. select a starting matrix based on random distrubution of the possible states.\n\t"
param += "3. using this matrix, generate a sample in which sequences with the same leading state and transition counts are generated in the same fashion.\n\t"
param += "4. repeat."
f.write(param)
f.close()

np.savetxt("DATASETS/markov_chain-coin-medium-training.txt", dataset, delimiter="", newline=",", fmt='%d')

from humanReadable import *
translateCSV(dataset, 0, "DATASETS/markov_chain-coin-medium-readable.csv")

########################## LARGE SEQUENCE ##########################
# generate dataset with parameters:
rvs = 6
tr_size = 50000 # fifty thousand
seq_len = 500
dataset = build_markov_dataset(rvs, tr_size, seq_len)

# print parameters
f = open('DATASETS/markov_chain-dice-large-parameters.txt', 'w')
param = "number of possible states:" + str(rvs) + "\n"
param += "training size: " + str(tr_size) + "\n"
param += "sequence length: " + str(seq_len) + "\n"
param += "procedure:\n\t1. generate " + str(rvs) + " " + str(rvs) + "x"  + str(rvs) + " recurrent matrix of floats between (but exlcuding) 0 and 1.\n\t"
param += "2. select a starting matrix based on random distrubution of the possible states.\n\t"
param += "3. using this matrix, generate a sample in which sequences with the same leading state and transition counts are generated in the same fashion.\n\t"
param += "4. repeat."
f.write(param)
f.close()

np.savetxt("DATASETS/markov_chain-dice-large-training.txt", dataset, delimiter="", newline=",", fmt='%d')

from humanReadable import *
translateCSV(dataset, 0, "DATASETS/markov_chain-dice-large-readable.csv")

# generate dataset with parameters:
rvs = 2
tr_size = 50000 # fifty thousand
seq_len = 500
dataset = build_markov_dataset(rvs, tr_size, seq_len)

# print parameters
f = open('DATASETS/markov_chain-coin-large-parameters.txt', 'w')
param = "number of possible states:" + str(rvs) + "\n"
param += "training size: " + str(tr_size) + "\n"
param += "sequence length: " + str(seq_len) + "\n"
param += "procedure:\n\t1. generate " + str(rvs) + " " + str(rvs) + "x"  + str(rvs) + " recurrent matrix of floats between (but exlcuding) 0 and 1.\n\t"
param += "2. select a starting matrix based on random distrubution of the possible states.\n\t"
param += "3. using this matrix, generate a sample in which sequences with the same leading state and transition counts are generated in the same fashion.\n\t"
param += "4. repeat."
f.write(param)
f.close()

np.savetxt("DATASETS/markov_chain-coin-large-training.txt", dataset, delimiter="", newline=",", fmt='%d')

from humanReadable import *
translateCSV(dataset, 0, "DATASETS/markov_chain-coin-large-readable.csv")
