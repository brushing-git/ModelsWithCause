# Catherine Park
import csv

# Take in a csv file and makes in human readable, complete with headers
# input: 
# dataSet - numpy array, the data set to translate 
# indicator - int, whether it is coin flips (0) or dice rolls (1),
# output - string, name of file outputted, must include ".csv" in given input
# bias - double/float, the given bias of the coin/dice to be used in the header
def translateCSV(dataSet, indicator, output):
    arr = []
    header = ""
    if (indicator):
        arr = [" One", " Two", " Three", " Four", " Five", " Six"]
    else:
        arr = [" Tails", " Heads"]

    with open(output, 'w',newline='') as file:
        writer = csv.writer(file)
        for row in dataSet:
            int_row = row.astype(int)
            string_list = []
            for point in int_row:
                string_list.append(arr[point])
            writer.writerow(string_list)
        
