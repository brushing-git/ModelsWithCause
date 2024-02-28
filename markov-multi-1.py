import numpy as np
from random_functions import *

np.random.seed(252)

# Build the data set
TRSIZE = 10000 # 1 million
RVL = 10 # size

# number of dice
dice = 4
# sides per die
sides = 6
# biases for sides of each dice
# used only to generate 10 rolls once dice_state is determined via transition probabilities
biases = rand_arrays_ONE(6, 4)
# "consider the previous n elements to calculate next roll"
consider_n = 4

# probabilities for transition to element x given previous consider_n-length sequence ...
transition_matrix = rand_arrays_ONE(dice, dice)

training_d = np.zeros((TRSIZE, RVL))

# track previous consider_n-length sequence
previous_states = np.zeros(consider_n)

ratio = (float) (dice / (consider_n - 1))

for i in range(TRSIZE):
    if i < consider_n:
        dice_state = np.random.choice(dice)
        previous_states[i] = dice_state
    else:
        arr = np.zeros(consider_n - 1)
        for j in range(consider_n - 1):
            arr[j] = transition_matrix[int(previous_states[j])][int(previous_states[j + 1])]

        #
        arr2 = np.zeros(dice)
        for j in range(dice):
            arr2[j] = np.random.choice(arr)
        sum = np.sum(arr2)
        if sum == 0: sum == 1
        for j in range(dice):
            arr2[j] /= sum
        #
        
        choose_dice = np.random.multinomial(1, arr2, 1)
        dice_state = int(str(np.where(choose_dice == 1)[0])[1])

        for j in range(consider_n - 1):
            previous_states[j] = previous_states[j + 1]
        previous_states[consider_n - 1] = dice_state
        

    # current dice_state computed
    trial = np.random.multinomial(1, biases[dice_state], RVL)
    index = np.zeros(RVL)
    for j in range(RVL):
        index[j] = int(str(np.where(trial[j] == 1)[0])[1])
    training_d[i,:] = index

training_d = training_d.astype(np.single)

# print parameters
f = open('markov-multi-1-parameters.txt', 'w')
param = "number of dice: " +  str(dice) + "\n"
param += "biases of each die:\n" + str(biases) + "\n"
param += "transition probabilities:\n" + str(transition_matrix) + "\n"
param += "procedure: see additional documents\n"
f.write(param)
f.close()

np.savetxt("markov-multi-1-training.txt", training_d, delimiter="", newline=",", fmt='%d')

from humanReadable import *
translateCSV(training_d, 1, "markov-multi-data-readable.cvs")