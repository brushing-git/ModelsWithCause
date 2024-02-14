# EXCHANGE-DATASET-1

# Colin Roberson, Bruce Rushing

import numpy as np

np.random.seed(252)

# Build the data set
TRSIZE = 1000 # 5 million
TESIZE = 200 # 1 million
RVL = 10 # size (~number of times probability is run)

# 6-sided die
sides = np.array([0, 1, 2, 3, 4, 5])
# "consider the previous n rolls to calculate next roll"
consider_n = 8
# procedure: consider starting letter and count transitions between letters; sequences with the same transition counts have equal probability

# rand_arrayONE(size L)
# generates an array of length L of floats [0,1) whose values sum to 1
def rand_arrayONE(L):
    arr = np.array([np.random.random() for _ in range(L)])
    sum = np.sum(arr)
    if sum == 0: sum = 1
    for i in range(len(arr)):
        arr[i] /= sum
    return arr

# rand_array()
# generates array of length L of floats [0.0, 1.0)
def rand_array(L):
    return np.array([np.random.random() for _ in range(L)])

# rand_array2()
# generates len(sides) arrays using rand_array
def rand_array2():
    return np.array([rand_array(len(sides)) for _ in range(len(sides))])

# each starting array corresponds to an n x n matrix
starting_all = np.array([rand_array2(), rand_array2(), rand_array2(), rand_array2(), rand_array2(), rand_array2()])

training_d = np.zeros(TRSIZE * RVL, int)

# initial sequence probabilities:
first_n = np.array([np.random.choice(sides) for _ in range(consider_n)])
# print(first_n)

for i in range(TRSIZE * RVL):
    # observe previous n states to determine starting value and transition count
    working_sum = 0
    # get starting value of n-length sequence
    if i >= consider_n: 
        starting_value = training_d[i - consider_n]
        # print(starting_value)
        # get corresponding matrix
        starting_arr = starting_all[starting_value]
        # print(starting_arr)
        # count transitions (transition from x to y corresponds to index [x, y]
        for j in range(i - consider_n, i - 1):
            # use some math to combine these values such that they have equal weight in influencing the subsequent value...
            working_sum += starting_arr[training_d[j], training_d[j + 1]]
        threshold = consider_n / len(sides)
        training_d[i] = int(working_sum / threshold)
    else: 
        training_d[i] = first_n[i]

training_d = training_d.astype(np.single)       

testing_d = np.zeros(TESIZE * RVL, int)

# initial sequence probabilities:
first_n = np.array([np.random.choice(sides) for _ in range(consider_n)])
# print(first_n)

for i in range(TESIZE * RVL):
    # observe previous n states to determine starting value and transition count
    working_sum = 0
    # get starting value of n-length sequence
    if i >= consider_n: 
        starting_value = testing_d[i - consider_n]
        # get corresponding matrix
        starting_arr = starting_all[starting_value]
        # count transitions (transition from x to y corresponds to index [x, y]
        for j in range(i - consider_n, i - 1):
            # use some math to combine these values such that they have equal weight in influencing the subsequent value...
            working_sum += starting_arr[testing_d[j], testing_d[j + 1]]
        threshold = consider_n / len(sides)
        testing_d[i] = int(working_sum / threshold)
    else: 
        testing_d[i] = first_n[i]

testing_d = testing_d.astype(np.single)

np.savetxt("exchange-dataset-1-training.txt", training_d, delimiter="", newline="", fmt='%d')
np.savetxt("exchange-dataset-1-testing.txt", testing_d, delimiter="", newline="", fmt='%d')