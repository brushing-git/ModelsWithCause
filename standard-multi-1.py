import numpy as np
from random_functions import *

np.random.seed(9)

TRSIZE = 100 # 1 million
RVL = 10

die_biases = rand_arrays_ONE(6, 6)
prior = rand_array_ONE(6)

training_d = np.zeros((TRSIZE, RVL))
for i in range(TRSIZE):
    # cycle 3 coins
    arr = np.random.multinomial(1, prior, 1)
    switch = int(str(np.where(arr == 1)[1])[1])
    # transition to {switch} die which has its own biases
    trial = np.random.multinomial(1, die_biases[switch], RVL)
    index = np.zeros(RVL)
    for j in range(RVL):
        index[j] = int(str(np.where(trial[j] == 1)[0])[1])
    training_d[i,:] = index

training_d = training_d.astype(np.single)

np.savetxt("standard-multi-1-training.txt", training_d, delimiter="", newline=",", fmt='%d')

# print parameters
f = open('standard-multi-1-parameters.txt', 'w')
f.write()
f.close()