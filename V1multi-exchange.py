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

# if there are x dice states, then there are x^2 unique transitions
# for a sequence of length consider_n, there can be consider_n - 1 transitions
# 

# generate array of possible sequences of length consider_n for each starting dice state

training_d = np.zeros((TRSIZE, RVL))

# track previous consider_n-length sequence
previous_states = np.zeros(consider_n)

for i in range(TRSIZE):
    if i < consider_n:
        dice_state = np.random.choice(dice)
        previous_states[i] = dice_state
    else:
        transition_count = 0
        for j in range(1, consider_n):
            if previous_states[j - 1] != previous_states[j]:
                # count transitions
                transition_count += 1
        current_state = previous_states[0]
        for count in range(transition_count):
            current_state = (current_state + count) % dice
        dice_state = int(current_state)
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
f = open('V1multi-exchange-1-parameters.txt', 'w')
param = "number of dice: " +  str(dice) + "\n"
param += "biases of each die:\n" + str(biases) + "\n"
param += "procedure:\n\t1. randomly choose the first " + str(consider_n) + " dice states.\n\t2. count the transitions between different states.\n\t"
param += "3. use this transition count to cycle through the " + str(dice) + " until the count is exhausted.\n\t4. roll the chosen die " + str(RVL) + " times.\n\t5. repeat."
f.write(param)
f.close()

np.savetxt("V1multi-exchange-1-training.txt", training_d, delimiter="", newline=",", fmt='%d')

from humanReadable import *
translateCSV(training_d, 1, "V1multi-exchange-data-readable.cvs")