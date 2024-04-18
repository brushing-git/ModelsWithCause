import numpy as np
from math import ceil

def construct_count_matrix(variables: int, sequence: np.ndarray) -> np.ndarray:
    count_mat = np.zeros((variables, variables), dtype=int)

    for i in range(1, sequence.shape[0]):
        var = sequence[i]
        prev_var = sequence[i-1]
        count_mat[prev_var, var] += 1
    
    return count_mat

def generate_sequence(variables: int, current_sequence: list, count_matrix: np.ndarray, max_length: int) -> list:
    # Base case
    if len(current_sequence) == max_length:
        return current_sequence[:]
    
    # Set the current state
    current_state = current_sequence[-1]

    # Set the options based on where we are in the sequence
    if len(current_sequence) <= ceil(max_length*0.05):
        # Set options to a random permutation
        options = np.random.permutation(np.arange(variables))
    else:
        # Set options to a the count with highest value
        options = np.argsort(-count_matrix[current_state, :])

    # Loop through the options trying the next states
    for next_state in options:
        if count_matrix[current_state, next_state] > 0:
            # Add the next sequence
            current_sequence.append(next_state)
            # Subtract the count_matrix
            count_matrix[current_state, next_state] -= 1

            # Recurse
            result = generate_sequence(variables=variables,
                                              current_sequence=current_sequence,
                                              count_matrix=count_matrix,
                                              max_length=max_length)
            
            # Return the first valid result
            if result is not None:
                return result
            
            # Backtrack
            current_sequence.pop()
            count_matrix[current_state, next_state] += 1
    
    # Return None if none were found
    return None

def build_permutations(variables: int, sequence: np.ndarray, n_perms: int) -> np.ndarray:
    # Convert the count matrix
    sequence = sequence.astype(int)

    permutations = []

    # Loop through and generate random permutations
    for _ in range(n_perms):
        # Rebuild the count matrix and start sequence
        count_mat = construct_count_matrix(variables, sequence)
        start = [sequence[0]]

        # Build the permutation
        new_perm = generate_sequence(variables, start, count_mat, sequence.shape[0])
        permutations.append(new_perm)
    
    # Convert to numpy array of float
    permutations = np.array(permutations).astype(float)

    return permutations