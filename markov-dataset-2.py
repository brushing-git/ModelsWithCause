# MARKOV-DATASET-2

# Colin Roberson, Bruce Rushing

import numpy as np

np.random.seed(34)

# Build the data set
TRSIZE = 1000000 # 1 million
TESIZE = 100000 # 1 hundred thousand
RVL = 10 # size (~number of times trials are run)

coins = np.array([0, 1, 2, 3])
biases = np.array([.4, .6, .88, .12])

# transition probabilities:
# [ from coin 1 to [itself, coin 2, coin 3, coin 4], from coin 2 to [ ... ] ... ]
transition = np.array([[0, .15, .35, .5],[.3, 0, .32, .38],[.14, .56, 0, .3],[.03, .67, .3, 0]])
# COIN WILL NOT REPEAT STATE

training_d = np.zeros((TRSIZE, RVL))

# initial state: coin 1 (index 0)
coin_state: int = 0

for i in range(TRSIZE):
    # switch depending on current coin state and transition probabilities table
    # arr=transition[coin_state]
    coin_state = np.random.choice(coins, p=transition[coin_state])
    training_d[i,:] = np.random.binomial(n=1,p=biases[coin_state],size=RVL)

training_d = training_d.astype(np.single)
        
coin_state: int = 0

testing_d = np.zeros((TESIZE, RVL))
for i in range(TESIZE):
    # same as training_d
    # switch depending on current coin state and transition probabilities table
    arr=transition[coin_state]
    coin_state = np.random.choice(coins, p=arr)
    testing_d[i,:] = np.random.binomial(n=1,p=biases[coin_state],size=RVL)

testing_d = testing_d.astype(np.single)

np.savetxt("markov-dataset-2-training.txt", training_d, delimiter="", fmt='%d')
np.savetxt("markov-dataset-2-testing.txt", testing_d, delimiter="", fmt='%d')