import numpy as np
from random_functions import *

np.random.seed(7)

TRSIZE = 10000 # 1 million
RVL = 10

coin_biases = rand_array(3)
prior = rand_array_ONE(3)

training_d = np.zeros((TRSIZE, RVL))
for i in range(TRSIZE):
    # cycle 3 coins
    arr = np.random.multinomial(1, prior, 1)
    switch = int(str(np.where(arr == 1)[1])[1])
    training_d[i,:] = np.random.binomial(n=1, p=coin_biases[switch], size=RVL)

training_d = training_d.astype(np.single)

# print parameters
f = open('DATASETS/standard-binary-1-parameters.txt', 'w')
param = "number of coins: " + str(len(prior)) + "\n"
param += "probability distribution of coins (respectively): " + str(prior) + "\n"
param += "biases of coins (respectively): " +  str(coin_biases) +  "\n"
param += "procedure:\n"
param += "\t1. choose a coin based on probabilities.\n\t2. flip the coin 10 times.\n\t3. repeat"
f.write(param)
f.close()

np.savetxt("DATASETS/standard-binary-1-training.txt", training_d, delimiter="", newline=",", fmt='%d')

from humanReadable import *
translateCSV(training_d, 0, "DATASETS/standard-binary-data-readable.csv")
