import numpy as np
import time
from CSR.markov_chain import build_markov_big_dataset

def main():
    data = build_markov_big_dataset(6, 50000, 100, num_processes=8)
    fn = 'test_data.txt'
    print('Saving file.')
    # Convert each row to a single string without commas between numbers
    t0 = time.time()
    formatted_data = ["".join(map(str, row)) for row in data.astype(int)]

    # Now save it with rows delimited by a comma
    with open(fn, 'w') as f:
        f.write(','.join(formatted_data))
    
    t1 = time.time()
    print(f'The time it took was {t1-t0}')

if __name__ == '__main__':
    main()