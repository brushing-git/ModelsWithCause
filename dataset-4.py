# DATASET-4

# Bruce Rushing 
# edited by Colin Roberson

import os
path = os.getcwd()
# os.chdir(path + '/myVRAIresearch/pytorch-generative')

import numpy as np

np.random.seed(126)

# Build the data set
TRSIZE = 1000000 # 1 million
TESIZE = 100000 # 1 hundred thousand
RVL = 10 # size (~number of times probability is run)

coins = np.array([0, 1, 2, 3])
biases = np.array([.21435, .35278375, .673285, .500992])

# transition probabilities:
# [ from coin 1 to [itself, coin 2, coin 3, coin 4], from coin 2 to [ ... ] ... ]
transition = np.array([[.251, .252, .253, .244],[.01, .08, .02, .89],[.2, .3, .05, .45],[.35, .35, .15, .15]])
# COIN CAN REPEAT STATE

# ask about HMMlearn package

training_d = np.zeros((TRSIZE, RVL))

# initial state: coin 1 (index 0)
coin_state = 0 

for i in range(TRSIZE):
    # switch depending on current/previous coin state and transition probabilities table
    if (coin_state == 0):
        # select new state based on transition probabilities
        coin_state = np.random.choice(coins, 1, p=transition[0])
    elif (coin_state == 1):
        coin_state = np.random.choice(coins, 1, p=transition[1])
    elif (coin_state == 2):
        coin_state = np.random.choice(coins, 1, p=transition[2])
    else:
        coin_state = np.random.choice(coins, 1, p=transition[3])
    training_d[i,:] = np.random.binomial(n=1,p=biases[coin_state],size=RVL)

training_d = training_d.astype(np.single)
print(training_d.dtype)
        
testing_d = np.zeros((TESIZE, RVL))
for i in range(TESIZE):
    # same as training_d
    # switch depending on current/previous coin state and transition probabilities table
    if (coin_state == 0):
        coin_state = np.random.choice(coins, 1, p=transition[0])
    elif (coin_state == 1):
        coin_state = np.random.choice(coins, 1, p=transition[1])
    elif (coin_state == 2):
        coin_state = np.random.choice(coins, 1, p=transition[2])
    else:
        coin_state = np.random.choice(coins, 1, p=transition[3])
    training_d[i,:] = np.random.binomial(n=1,p=biases[coin_state],size=RVL)

testing_d = testing_d.astype(np.single)
print(testing_d.dtype)

print(training_d.shape)
print(testing_d.shape)

print(training_d)
print(testing_d)
