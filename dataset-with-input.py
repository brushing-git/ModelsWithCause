# takes user input for coin bias, how many coins, and creates text files for each dataset

# Bruce Rushing 
# edited by Catherine Park

import os
os.chdir('/Users/catherinepark/pytorch-generative') # may need to change this depending on folder/env


import numpy as np

np.random.seed(69)

# Build the data set
TRSIZE = 5000000
TESIZE = 1000000
RVL = 10

# ----------
# example, given 7 coins, each with a different probability of being picked
# p1 = 0 # no chance of heads/all tails
# p2 = 0.2 # 20%                   
# p3 = 0.4 # 40%                   -|
# p4 = 0.5 # 50%, standard coinflip | more likely to pick these coins
# p5 = 0.6 # 60%                   -|
# p6 = 0.8 # 80%                   
# p7 = 1 # all heads
#  ---------
coins = input("Enter the number of coins to choose from: ")
tracker = 0
coin_prob_arr = {0}; # no need to have seperate variables
while tracker < coins:
    temp = input("Enter the bias for coin " + str(tracker + 1) + ": ")
    coin_prob_arr.add(temp)
    tracker+=1

prior = input("What will be the bias for choosing coins? Enter a value range [0 - 1]: ")
datasets_l, datasets_h = input("How many datasets should be produced? Enter a value for the length and height range: ")

training_d = np.zeros((TRSIZE, RVL))
which_coins = np.random.binomial(n=(coins+1), p=prior, size=(datasets_l,datasets_h)) 
# this chooses which coin to use ^
# for each possibility, generate dataset with the coin at index [x] from ^ set
for i in which_coins:
    for k in range(TRSIZE):
        training_d[i,:] = np.random.binomial(n=1, p=coin_prob_arr[i], size=RVL)
    training_d = training_d.astype(np.single)
    filestring = "dataset_" + str(i + 1) + ".txt"
    np.savetxt(filestring, training_d, delimiter=", ")
print("Datasets have been stored, call on them using the given format \"dataset_#.txt\"")