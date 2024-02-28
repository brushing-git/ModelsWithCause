import numpy as np

np.random.seed(69)

# Build the data set
TRSIZE = 10000 # 1 million
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

# print parameters
f = open('markov-binary-1-parameters.txt', 'w')
param = "number of coins:" + str(len(coins)) + "\n"
param += "biases of coins (respectively): " + str(biases) + "\n"
param += "transition probabilities for each coin (respectively):\n" + str(transition) + "\n"
param += "procedure:\n\t1. set initial coin state to one of " + str(len(coins)) + " coins.\n"
param += "\t2. obtain set of probabilities based on previous coin state. each coin has a unique set of probabilities to transition to another coin. in this sample, the coin state cannot repeat itself.\n"
param += "\t3. transition to another coin based on this set.\n\t4. perform 10 trials (flips) of the new coin based on its unique bias.\n\t5. repeat."
f.write(param)
f.close()

np.savetxt("markov-binary-1-training.txt", training_d, delimiter="", newline=",", fmt='%d')

from humanReadable import *
translateCSV(training_d, 0, "markov-binary-data-readable.csv")
