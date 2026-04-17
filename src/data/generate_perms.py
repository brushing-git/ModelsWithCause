import numpy as np
from math import ceil

def construct_count_matrix(variables: int, sequence: np.ndarray) -> np.ndarray:
    count_mat = np.zeros((variables, variables), dtype=int)

    for i in range(1, sequence.shape[0]):
        var = sequence[i]
        prev_var = sequence[i-1]
        count_mat[prev_var, var] += 1
    
    return count_mat

def generate_sequence(
        variables: int, 
        current_sequence: list, 
        count_matrix: np.ndarray, 
        max_length: int
) -> list:
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

def build_permutations(
        variables: int, 
        sequence: np.ndarray, 
        n_perms: int,
        max_attempts_per_perm: int = 100
) -> np.ndarray:
    # Convert the count matrix
    sequence = sequence.astype(int)

    permutations = []

    # Track unique permutations to avoid duplicates
    seen_permutations = set()
    seen_permutations.add(tuple(sequence))

    total_attempts = 0
    max_total_attempts = n_perms * max_attempts_per_perm

    while len(permutations) < n_perms and total_attempts < max_total_attempts:
        total_attempts += 1

        # Rebuild the count matrix and start sequence
        count_mat = construct_count_matrix(variables, sequence)
        start = [sequence[0]]

        # Build the permutation
        new_perm = generate_sequence(variables, start, count_mat, sequence.shape[0])

        # Check if we got a valid permutation
        if new_perm is None:
            continue

        # Check if it is unique
        perm_tuple = tuple(new_perm)
        if perm_tuple in seen_permutations:
            continue

        seen_permutations.add(perm_tuple)
        permutations.append(new_perm)
    
    if len(permutations) < n_perms:
        raise ValueError(
            f'Could only generate {len(permutations)} unique permutation '
            f'after {total_attempts} attempts. '
            f'Thes equence may not have enough valid Markov-equivalent permutations.'
        )
    
    # Convert to numpy array of float
    permutations = np.array(permutations).astype(float)

    return permutations

def construct_count_matrix_order2(variables: int, sequence: np.ndarray) -> np.ndarray:
    """
    Construct a 3D transition count matrix for a 2nd-order Markov process.
    
    For a 2nd-order process, we count triplets: how many times state k follows
    the pair of states (i, j). This gives us count_mat[i, j, k].
    
    Args:
        variables: Number of distinct states/variables in the sequence
        sequence: 1D numpy array of state indices (integers 0 to variables-1)
    
    Returns:
        3D numpy array of shape (variables, variables, variables) where
        count_mat[i, j, k] = number of times state k follows the pair (i, j)
    """
    count_mat = np.zeros((variables, variables, variables), dtype=int)
    
    # We need at least 3 elements to have a 2nd-order transition
    for i in range(2, sequence.shape[0]):
        prev_prev = sequence[i - 2]
        prev = sequence[i - 1]
        curr = sequence[i]
        count_mat[prev_prev, prev, curr] += 1
    
    return count_mat

def generate_sequence_order2(
        variables: int,
        current_sequence: list[int],
        count_matrix: np.ndarray,
        max_length: int
) -> list[int]:
    """
    Recursively generate a sequence that preserves 2nd-order Markov statistics.
    
    Uses backtracking search to find a valid sequence. At each step, the algorithm:
    1. Looks at the last TWO elements to determine the current 2nd-order state
    2. Finds all valid next states (those with remaining transition counts)
    3. Tries each option, backtracking if it leads to a dead end
    
    The early-sequence randomization (first 5% of positions) helps explore
    different valid permutations rather than always finding the same one.
    
    Args:
        variables: Number of distinct states
        current_sequence: Partially built sequence (modified in place during search)
        count_matrix: 3D count matrix (modified in place during search)
        max_length: Target length of the final sequence
    
    Returns:
        Complete sequence as a list if successful, None if no valid completion exists
    """
    # Base case: we've reached the target length
    if len(current_sequence) == max_length:
        return current_sequence[:]
    
    # Safety check: need at least 2 elements to determine 2nd-order state
    if len(current_sequence) < 2:
        raise ValueError(
            "Sequence must be initialized with at least 2 elements for 2nd-order process"
        )
    
    # The current 2nd-order state is the pair of last two elements
    prev_prev = current_sequence[-2]
    prev = current_sequence[-1]
    
    # Determine search order for next states
    # Early in the sequence: randomize to explore different permutations
    # Later: prioritize states with highest remaining counts (greedy heuristic)
    if len(current_sequence) <= ceil(max_length * 0.05) + 1:  # +1 because we start with 2 elements
        options = np.random.permutation(np.arange(variables))
    else:
        # Sort by descending count - try most common transitions first
        options = np.argsort(-count_matrix[prev_prev, prev, :])
    
    # Try each possible next state
    for next_state in options:
        if count_matrix[prev_prev, prev, next_state] > 0:
            # Tentatively add this state
            current_sequence.append(next_state)
            count_matrix[prev_prev, prev, next_state] -= 1
            
            # Recurse to try completing the sequence
            result = generate_sequence_order2(
                variables=variables,
                current_sequence=current_sequence,
                count_matrix=count_matrix,
                max_length=max_length
            )
            
            # If we found a valid completion, return it
            if result is not None:
                return result
            
            # Backtrack: undo our choice and try the next option
            current_sequence.pop()
            count_matrix[prev_prev, prev, next_state] += 1
    
    # No valid next state found - this path is a dead end
    return None

def build_permutations_order2(
        variables: int,
        sequence: np.ndarray,
        n_perms: int,
        max_attempts_per_perm: int = 100
) -> np.ndarray:
    """
    Generate multiple unique 2nd-order Markov-exchangeable permutations.
    
    Two sequences are 2nd-order Markov-exchangeable if they have:
    - The same first two elements (initial 2nd-order state)
    - The same 3D transition count matrix
    
    This means the sequences have identical 2nd-order statistical properties
    but potentially different orderings of transitions.
    
    Args:
        variables: Number of distinct states in the sequence
        sequence: Original sequence to generate permutations of
        n_perms: Number of unique permutations to generate
        max_attempts_per_perm: Maximum attempts per permutation before giving up
    
    Returns:
        2D numpy array of shape (n_perms, len(sequence)) containing the permutations
    
    Raises:
        ValueError: If sequence is too short for 2nd-order analysis (< 3 elements)
        ValueError: If unable to generate enough unique permutations
    
    Example:
        >>> seq = np.array([0, 1, 2, 1, 2, 0, 1, 2])
        >>> perms = build_permutations_order2(variables=3, sequence=seq, n_perms=5)
        >>> # Each row in perms has the same 2nd-order transition counts as seq
    """
    sequence = sequence.astype(int)
    
    # Validate sequence length
    if len(sequence) < 3:
        raise ValueError(
            f"Sequence must have at least 3 elements for 2nd-order Markov analysis, "
            f"got {len(sequence)}"
        )
    
    permutations = []
    
    # Track seen permutations to ensure uniqueness
    seen_permutations = set()
    seen_permutations.add(tuple(sequence))
    
    total_attempts = 0
    max_total_attempts = n_perms * max_attempts_per_perm
    
    while len(permutations) < n_perms and total_attempts < max_total_attempts:
        total_attempts += 1
        
        # Rebuild count matrix fresh for each attempt
        count_mat = construct_count_matrix_order2(variables, sequence)
        
        # Initialize with the first TWO elements (the initial 2nd-order state)
        # This ensures all permutations start from the same 2nd-order state
        start = [int(sequence[0]), int(sequence[1])]
        
        # Attempt to generate a valid permutation
        new_perm = generate_sequence_order2(
            variables=variables,
            current_sequence=start,
            count_matrix=count_mat,
            max_length=sequence.shape[0]
        )
        
        # Skip if generation failed
        if new_perm is None:
            continue
        
        # Skip if we've seen this permutation before
        perm_tuple = tuple(new_perm)
        if perm_tuple in seen_permutations:
            continue
        
        seen_permutations.add(perm_tuple)
        permutations.append(new_perm)
    
    if len(permutations) < n_perms:
        raise ValueError(
            f"Could only generate {len(permutations)} unique permutations "
            f"after {total_attempts} attempts. "
            f"The sequence may not have enough valid 2nd-order Markov-equivalent permutations."
        )
    
    return np.array(permutations).astype(float)