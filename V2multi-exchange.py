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

transition_matrix = rand_arrays_ONE(dice, dice)

training_d = np.zeros((TRSIZE, RVL))

# track previous consider_n-length sequence
previous_states = np.zeros(consider_n, int)

for i in range(TRSIZE):
    if i < consider_n:
        dice_state = np.random.choice(dice)
        previous_states[i] = dice_state
    else:
        sum = 0
        for j in range (1, consider_n):
            sum += transition_matrix[previous_states[j - 1]][previous_states[j]]
        sum *= (consider_n / dice)
        dice_state = int(round(sum))
        for j in range(consider_n - 1):
            previous_states[j] = previous_states[j + 1]
        previous_states[consider_n - 1] = dice_state
    # current dice_state computed, now roll RVL times
    trial = np.random.multinomial(1, biases[dice_state], RVL)
    index = np.zeros(RVL)
    for j in range(RVL):
        index[j] = int(str(np.where(trial[j] == 1)[0])[1])
    training_d[i,:] = index

training_d = training_d.astype(np.single)

# print parameters
f = open('V2multi-exchange-1-parameters.txt', 'w')
param = "number of dice: " +  str(dice) + "\n"
param += "biases of each die:\n" + str(biases) + "\n"
param += "transition probability matrix:\n" + str(transition_matrix) + "\n"
param += "procedure:\n\t1. randomly choose the first " + str(consider_n) + " dice states.\n\t2. add the unique probabilities from the transition matrix based on the transition count (i.e. a transition from x to y corresponds to a value at transition_matrix[x][y]\n\t"
param += "3. multiply this sum by the ratio between " + str(consider_n) + " and " + str(dice) + ".\n\t"
param += "4. round this value to the nearest whole number; this is the dice state.\n\t"
param += "5. roll the chosen die " + str(RVL) + " times.\n\t6. repeat."
f.write(param)
f.close()

np.savetxt("V2multi-exchange-1-training.txt", training_d, delimiter="", newline=",", fmt='%d')

from humanReadable import *
translateCSV(training_d, 1, "V2multi-exchange-data-readable.cvs")