# EXCHANGE-DATASET-1

# Colin Roberson, Bruce Rushing
# Editted by Catherine Park

import numpy as np
from random_functions import *

from humanReadable import *

np.random.seed(252)

# Build smaller data set
TRSIZE = 100 # 5 million
RVL = 10 # size

# 6-sided die
sides = np.array([0, 1, 2, 3, 4, 5])
# "consider the previous n rolls to calculate next roll"
consider_n = 8
# procedure: consider starting letter and count transitions between letters; sequences with the same transition counts have equal probability

# each starting array corresponds to an n x n matrix
starting_all = np.array([rand_arrays(len(sides), len(sides)), rand_arrays(len(sides), len(sides)), rand_arrays(len(sides), len(sides)), rand_arrays(len(sides), len(sides)), rand_arrays(len(sides), len(sides)), rand_arrays(len(sides), len(sides))])

training_d = np.zeros(TRSIZE * RVL, int)

# initial sequence probabilities:
first_n = np.array([np.random.choice(sides) for _ in range(consider_n)])
# print(first_n)

for i in range(TRSIZE * RVL):
    # observe previous n states to determine starting value and transition count
    working_sum = 0
    # get starting value of n-length sequence
    if i >= consider_n: 
        starting_value = training_d[i - consider_n]
        # print(starting_value)
        # get corresponding matrix
        starting_arr = starting_all[starting_value]
        # print(starting_arr)
        # count transitions (transition from x to y corresponds to index [x, y]
        for j in range(i - consider_n, i - 1):
            # use some math to combine these values such that they have equal weight in influencing the subsequent value...
            working_sum += starting_arr[training_d[j], training_d[j + 1]]
        threshold = consider_n / len(sides)
        training_d[i] = int(working_sum / threshold)
    else: 
        training_d[i] = first_n[i]

training_d = training_d.astype(np.single)

translateCSV(training_d, 1, "markov-multi-data-readable.cvs")