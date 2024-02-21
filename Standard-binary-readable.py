import numpy as np
from random_functions import *

from humanReadable import *

np.random.seed(7)

TRSIZE = 80 # smaller dataset
RVL = 10

coin_biases = rand_array(3)
prior = rand_array_ONE(3)

training_d = np.zeros((TRSIZE, RVL))
for i in range(TRSIZE):
    # cycle 3 coins
    arr = np.random.multinomial(1, prior, 1)
    switch = int(np.where(arr == 1)[1])
    training_d[i,:] = np.random.binomial(n=1, p=coin_biases[switch], size=RVL)

training_d = training_d.astype(np.single)

translateCSV(training_d, 0, "standard-binary-data-readable.csv")