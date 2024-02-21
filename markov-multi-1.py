# EXCHANGE-DATASET-1

# Colin Roberson, Bruce Rushing

import numpy as np
from random_functions import *

np.random.seed(252)

# Build the data set
TRSIZE = 1000 # 1 million
RVL = 10 # size

# 6-sided die
sides = np.array([0, 1, 2, 3, 4, 5])
# "consider the previous n rolls to calculate next roll"
consider_n = 5
# procedure: consider starting letter and count transitions between letters; sequences with the same transition counts have equal probability

# each starting array corresponds to an n x n matrix
starting_all = np.array([rand_arrays(len(sides), len(sides)), rand_arrays(len(sides), len(sides)), rand_arrays(len(sides), len(sides)), rand_arrays(len(sides), len(sides)), rand_arrays(len(sides), len(sides)), rand_arrays(len(sides), len(sides))])

training_d = np.zeros((TRSIZE, RVL))

# initial state:
training_d[0][0] = 2

for i in range(TRSIZE):
    # observe previous consider_n states to determine starting value and transition count
    backstep = min(consider_n, i)
    # starting value in consider_n-length sequence
    starting = training_d[i - backstep][0]
    # transitition matrix for starting value state
    prob_arr = starting_all[starting]
    # grab consider_n-length sequence
    prev_seq = training_d[i - backstep, i]
    transitions = np.array((len(prev_seq), 2))
    for j in range(len(prev_seq) - 1):
        transitions[j] = (prev_seq[j], prev_seq[j + 1])
    next_value = -1
    for j in range(len(transitions)):
        next_value

training_d = training_d.astype(np.single)

# print parameters
f = open('markov-multi-1-parameters.txt', 'w')
f.write('')
f.close()

np.savetxt("markov-multi-1-training.txt", training_d, delimiter="", newline=",", fmt='%d')