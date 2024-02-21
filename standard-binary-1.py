import numpy as np
from random_functions import *

np.random.seed(7)

TRSIZE = 1000000 # 1 million
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
f = open('standard-binary-1-parameters.txt', 'w')
f.write("number of coins: %d\n", len(prior))
f.write("biases of coins (respectively): ", coin_biases, "\n")
f.write("transition probabilities (respectively): ", prior, "\n")
f.write("procedure:\n")
f.close()

np.savetxt("standard-binary-1-training.txt", training_d, delimiter="", newline=",", fmt='%d')

from humanReadable import *
translateCSV(training_d, 0, "standard-binary-data-readable.csv")
