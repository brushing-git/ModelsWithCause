import numpy as np
from random_functions import *

np.random.seed(9)

TRSIZE = 10000 # 1ten thousand
RVL = 10

die_biases = rand_arrays_ONE(6, 6)
prior = rand_array_ONE(6)

training_d = np.zeros((TRSIZE, RVL))
for i in range(TRSIZE):
    arr = np.random.multinomial(1, prior, 1)
    switch = int(str(np.where(arr == 1)[1])[1])
    # transition to {switch} die which has its own biases
    trial = np.random.multinomial(1, die_biases[switch], RVL)
    index = np.zeros(RVL)
    for j in range(RVL):
        index[j] = int(str(np.where(trial[j] == 1)[0])[1])
    training_d[i,:] = index

training_d = training_d.astype(np.single)

# print parameters
f = open('standard-multi-1-parameters.txt', 'w')
param = "number of dice: " + str(len(prior)) + "\n"
param += "probability distribution of dice (respectively): " + str(prior) + "\n"
param += "biases of each die:\n" + str(die_biases) + "\n"
param += "procedure:\n"
param += "\t1. choose die based on probabilities\n\t2. roll die 10 times\n\t3. repeat"
f.write(param)
f.close()

np.savetxt("standard-multi-1-training.txt", training_d, delimiter="", newline=",", fmt='%d')

from humanReadable import *
translateCSV(training_d, 1, "standard-multi-data-readable.csv")