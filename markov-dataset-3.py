# MARKOV-DATASET-3

# Colin Roberson, Bruce Rushing

import numpy as np

np.random.seed(323)

# Build the data set
TRSIZE = 5000000 # 5 million
TESIZE = 1000000 # 1 million
RVL = 10 # size (~number of times probability is run)

coins = np.array([0, 1, 2, 3])
biases = np.array([.43, .79, .24, .56])

# transition probabilities:
# [ from coin 1 to [itself, coin 2, coin 3, coin 4], from coin 2 to [ ... ] ... ]
transition = np.array([[.1, .2, .3, .4],[.2, 2, .3, .3],[.1, .35, .1, .45],[.15, .25, .3, .3]])
# COIN CAN REPEAT STATE

training_d = np.zeros((TRSIZE, RVL))

# initial state: coin 1 (index 0)
coin_state = 0 

for i in range(TRSIZE):
    # switch depending on current coin state and transition probabilities table
    arr=transition[coin_state]
    coin_state = np.random.choice(coins, p=arr)
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

np.savetxt("markov-dataset-3-training.txt", training_d, delimiter="", newline="", fmt='%d')
np.savetxt("markov-dataset-3-testing.txt", testing_d, delimiter="", newline="", fmt='%d')