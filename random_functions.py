import numpy as np

# rand_array_ONE(size L)
# generates an array of length L of floats [0,1) whose values sum to 1
def rand_array_ONE(L):
    np.random.seed(69)
    arr = np.array([np.random.random() for _ in range(L)])
    sum = np.sum(arr)
    if sum == 0: sum = 1
    for i in range(len(arr)):
        arr[i] /= sum
    return arr

# rand_array(size L)
# generates array of length L of floats [0.0, 1.0)
def rand_array(L):
    np.random.seed(69)
    return np.array([np.random.random() for _ in range(L)])

# rand_arrays_ONE(size N, size L)
# generates N arrays of length L using rand_array_ONE (values sum to 1)
def rand_arrays_ONE(L, N):
    return np.array([rand_array_ONE(L) for _ in range(N)])

# rand_arrays(size N, size L)
# generates N arrays of length L using rand_array
def rand_arrays(L, N):
    return np.array([rand_array(L) for _ in range(N)])