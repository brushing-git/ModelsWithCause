# MARKOV-BINARY-1 in Readable Format

# Colin Roberson, Bruce Rushing
# Editted by Catherine Park

import numpy as np
from humanReadable import *

np.random.seed(69)

# Build the smaller data set
TRSIZE = 80 # 100
RVL = 10 # size (~number of trials)

coins = np.array([0, 1, 2, 3])
biases = np.array([.35, .5, .7, .55])

# transition probabilities:
# [ from coin 1 to [itself, coin 2, coin 3, coin 4], from coin 2 to [ ... ] ... ]
transition = np.array([[0, .2, .3, .5],[.4, 0, .3, .3],[.2, .35, 0, .45],[.15, .25, .6, 0]])
# COIN CANNOT REPEAT STATE

training_d = np.zeros((TRSIZE, RVL))

# initial state
coin_state: int = 0 

for i in range(TRSIZE):
    # switch depending on current coin state and transition probabilities table
    arr=transition[coin_state]
    coin_state = np.random.choice(coins, p=arr)
    training_d[i,:] = np.random.binomial(n=1,p=biases[coin_state],size=RVL)

training_d = training_d.astype(np.single)
# call the function to translate the binary into coin flips (HEADS/TAILS)
translateCSV(training_d, 0, "markov-binary-data-readable.csv")

