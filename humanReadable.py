# Catherine Park
import csv

# Take in a csv file and makes in human readable, complete with headers
# input: 
# dataSet - numpy array, the data set to translate 
# indicator - int, whether it is coin flips (0) or dice rolls (1),
# output - string, name of file outputted, must include ".csv" in given input
# bias - double/float, the given bias of the coin/dice to be used in the header
def translateCSV(dataSet, indicator, output, bias):
    arr = []
    header = ""
    if (indicator):
        arr = [" One", " Two", " Three", " Four", " Five", " Six"]
        header = ["** This Dataset represents dice rolls taken from a dice with a bias of " + str(bias), 
         " for 6, and utilizes Markov chain and Partial exchangability techniqies" ,
         " to ensure that the probability of the values are dependant on the ",
         "previous results rather than pure randomness. This data was created with", 
         " the purpose of training AI to be able to predict the successive sequence",
         " will be given the current sequence. **"]
    else:
        arr = [" Tails", " Heads"]
        header = ["** This Dataset represents coin flips taken from a coin with a bias of " + str(bias),
                  " for heads, and utilizes Markov chain and Partial exchangability techniqies" ,
                " to ensure that the probability of the heads/tails are dependant on the" ,
                " previous results rather than pure randomness. This data was created with",
                " the purpose of training AI to be able to predict the successive sequence",
                " will be given the current sequence. **"]

    with open(output, 'w',newline='') as file:
        writer = csv.writer(file)
        writer.writerow(header)
        for row in dataSet:
            int_row = row.astype(int)
            string_list = []
            for point in int_row:
                string_list.append(arr[point])
            writer.writerow(string_list)
        
