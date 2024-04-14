import numpy as np
import time

def construct_count_matrix(variables: int, sequence: np.ndarray) -> np.ndarray:
    count_mat = np.zeros((variables, variables), dtype=int)

    for i in range(1, sequence.shape[0]):
        var = sequence[i]
        prev_var = sequence[i-1]
        count_mat[prev_var, var] += 1
    
    return count_mat

def permute_sequence(variables: int, sequence: np.ndarray) -> np.ndarray:
    count_mat = construct_count_matrix(variables, sequence)

    new_sequence = [sequence[0]]

    while np.any(count_mat):
        # Construct the outbound count
        current_var = new_sequence[-1]
        print(f'The count matrix is\n{count_mat}')
        total_mat = np.sum(count_mat, axis=1)
        print(f'The total matrix is\n{total_mat}')
        node_current_total = total_mat[current_var]

        # Get the options for the next step
        options = []
        for i, total in enumerate(total_mat):
            difference = np.absolute(node_current_total - total)
            if difference <= 1 and total > 0 and count_mat[current_var, i] > 0:
                options.append(i)
        
        # Choose the next step
        print(f'The options are\n{options}')
        next_var = np.random.choice(options)
        new_sequence.append(next_var)
        print(f'The new sequence is\n{new_sequence}')
        count_mat[current_var, next_var] -= 1
        time.sleep(10.0)
    
    return new_sequence