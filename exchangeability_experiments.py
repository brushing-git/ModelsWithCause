"""
N_SAMPLES should be set to one of these values corresponding to the effect size:

effect size = 0.05, sample size = 6388
effect size = 0.1, sample size = 1598
effect size = 0.25, sample size = 257
effect size = 1.0, sample size = 18

The null samples ideally should be set to one of these values correesponding to the effect size:
effect size = 0.05, sample size = 251370
effect size = 0.1, sample size = 62844
effect size = 0.25, sample size = 10056
effect size = 1.0, sample size = 630
"""

import torch
import numpy as np
import concurrent.futures
import os
from tqdm import tqdm
from CSR.datasets import load_data
from CSR.generate_perms import build_permutations, construct_count_matrix
from CSR.models import NADE, Transformer, DecoderTransformer, MoEDecoderTransformer

# Set seeds
torch.manual_seed(0)
np.random.seed(123)
rng = np.random.default_rng(123)

# Data parameters
FN = 'markov_chain-dice-100-training.txt'
MARKOV = True # Set this parameter to generate permutations of Markov Exchangeable sequences
TEXTLENGTH = 100
CAT = 6
SOS_TOKEN = 6 # This needs to be set depending on the type of dataset
EOS_TOKEN = 7 # This needs to be set depending on the type of dataset

# Model parameters
MODELS = ['NADE', 'Transformer', 'DecoderTransformer', 'MOE']
MODEL = MODELS[2]
MODEL_PATH = 'DecoderTransformer_10-4-4096.pt'

# Experimental Data Parameters
N_SAMPLES = 1598 # This is for the permutation data, we aim for 80% power at effect size 0.1
N_PERMUTATIONS = 5 # Keep this low otherwise it will take forever to build the dataset
DATASETS_NAMES = ['SE-coin', 'SE-dice', 'ME-coin', 'ME-dice']
DATA_NAME = DATASETS_NAMES[3] # Set the index to the right name

# Functions
def backtrack_search(variables: int, 
                     sequence: np.ndarray, 
                     n_perms: int, 
                     experiment_data: np.ndarray, 
                     idx: int) -> None:
    # Do the backtracking search
    permutations = build_permutations(variables=variables, 
                                      sequence=sequence, 
                                      n_perms=n_perms)
    # Store the original sequence as the first element in dim=1
    experiment_data[idx,0,:] = sequence[:]
    # Store the remainder as the following elements in in dim=1
    experiment_data[idx,1:,:] = permutations[:,:]

def exchangeable_permutations(sequence: np.ndarray,
                              n_perms: int,
                              experiment_data: np.ndarray,
                              idx: int) -> None:
    # Build the permutations
    permutations = set()

    while len(permutations) < n_perms:
        perm = tuple(rng.permutation(sequence))
        permutations.add(perm)
    
    perm_array = np.array([list(perm) for perm in permutations]).astype(float)
    # Store the original sequence as the first element in dim=1
    experiment_data[idx,0,:] = sequence[:]
    # Store the remainder as the following elements in in dim=1
    experiment_data[idx,1:,:] = perm_array[:,:]

def build_experiment_data(dataset: np.ndarray, 
                          n_samples: int, 
                          n_permutations: int, 
                          variables: int,
                          markov: bool = True) -> np.ndarray:
    # Sample randomly from dataset
    data_indices = np.arange(dataset.shape[0])
    rng.shuffle(data_indices)

    # Build the permutations, we will store them in a (n_samples, n_permutations+1, sequence_length) array
    length = dataset.shape[1]
    experiment_data = np.zeros((n_samples, n_permutations+1, length))

    # Index counter for experiment_data
    i = 0

    with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
        # Loop through the data and try to do backtrack search
        for idx in data_indices:
            print(f'Trying sample {i}.')
            if i < n_samples:
                # Set the sequence
                sequence = dataset[idx,:]

                # Do the Markov permutations if markov == True else do normal permutations
                if markov:
                    # Put the backtracking search to executor
                    future = executor.submit(backtrack_search, 
                                            variables, 
                                            sequence, 
                                            n_permutations,
                                            experiment_data,
                                            i)
                    try:
                        # Try to see if we can get a timely result
                        result = future.result(timeout=30)
                        i += 1
                    except concurrent.futures.TimeoutError:
                        print(f'Search timed out for index {idx}. Moving to the next sample.')
                    except Exception as e:
                        print(f'An error occurred: {e}')
                else:
                    # Compute the permutations for exchangeability
                    exchangeable_permutations(sequence=sequence,
                                              n_perms=n_permutations,
                                              experiment_data=experiment_data,
                                              idx=i)
                    i += 1
            else:
                break
    
    return experiment_data

def test_model(model, exp_dataset) -> tuple:
    """
    Loops through an experimental dataset and computes the difference between the first sequence and its 
    permutations assigned log probabilities.
    """

    results_perm = np.zeros((exp_dataset.shape[0], exp_dataset.shape[1]-1))
    results_null = np.zeros((exp_dataset.shape[0], exp_dataset.shape[1]-1))

    transform = torch.from_numpy
    indxs = [i for i in range(exp_dataset.shape[0])]
    for i in tqdm(range(exp_dataset.shape[0])):
        # Set target sequence and permutations
        target_sequence = exp_dataset[i,0,:]
        permutations = exp_dataset[i,1:,:]

        # Convert target sequence
        target_sequence_x = transform(target_sequence).type(torch.float).unsqueeze(0)
        target_sequence_y = transform(target_sequence).type(torch.long).unsqueeze(0)

        # Register the original log probabilities
        target_log_prob = model.estimate_prob(target_sequence_x, target_sequence_y)
        target_log_prob = sum(target_log_prob[0])

        # Loop through the permutations and store the difference in log probabilities
        for j in range(permutations.shape[0]):
            # Set the sequence and transform it
            perm_sequence = permutations[j,:]
            perm_sequence_x = transform(perm_sequence).type(torch.float).unsqueeze(0)
            perm_sequence_y = transform(perm_sequence).type(torch.long).unsqueeze(0)

            # Get the perm log probabilities
            perm_log_prob = model.estimate_prob(perm_sequence_x, perm_sequence_y)
            perm_log_prob = sum(perm_log_prob[0])

            # Take the difference
            difference_log_prob = target_log_prob - perm_log_prob

            # Store the log probability
            results_perm[i,j] = difference_log_prob
        
        # Loop through and build some null data to test
        random_indxs = indxs.copy()
        random_indxs.remove(i)
        random_indxs = rng.choice(random_indxs, exp_dataset.shape[1]-1, replace=False)
        
        for j in range(random_indxs.shape[0]):
            # Set the sequence and transform it
            alt_sequence = exp_dataset[j,0,:]
            alt_sequence_x = transform(alt_sequence).type(torch.float).unsqueeze(0)
            alt_sequence_y = transform(alt_sequence).type(torch.long).unsqueeze(0)

            # Get the perm log probabilities
            alt_log_prob = model.estimate_prob(alt_sequence_x, alt_sequence_y)
            alt_log_prob = sum(alt_log_prob[0])

            # Take the absolute value of the difference
            difference_log_prob = target_log_prob - alt_log_prob

            # Store the log probability
            results_null[i,j] = difference_log_prob
    
    return results_perm, results_null

def build_model(model_name: str, file_path: str):
    """
    Builds the appropriate model based on the model_name keyword.
    """

    if model_name == 'NADE':
        model = NADE(in_dim=TEXTLENGTH, hidden_dim=16, cat=CAT)
        model.load_state_dict(torch.load(file_path, map_location=model.device))
        return model
    elif model_name == 'Transformer':
        model = Transformer(n_tokens=CAT+2, 
                            dim_model=TEXTLENGTH, 
                            n_heads=5, 
                            n_encoder_lyrs=2,
                            n_decoder_lyrs=8,
                            dropout_p=0.0,
                            ffn=2048,
                            SOS_token=SOS_TOKEN,
                            EOS_token=EOS_TOKEN)
        model.load_state_dict(torch.load(file_path, map_location=model.device))
        return model
    elif model_name == 'DecoderTransformer':
        model = DecoderTransformer(n_tokens=CAT+2,
                                   dim_model=TEXTLENGTH,
                                   n_heads=10,
                                   n_decoder_lyrs=4,
                                   dropout_p=0.0,
                                   ffn=4096,
                                   SOS_token=SOS_TOKEN,
                                   EOS_token=EOS_TOKEN)
        model.load_state_dict(torch.load(file_path, map_location=model.device))
        return model
    elif model_name == 'MOE':
        model = MoEDecoderTransformer(n_tokens=CAT+2,
                                      dim_model=TEXTLENGTH,
                                      n_heads=10,
                                      n_decoder_lyrs=4,
                                      dropout_p=0.0,
                                      n_experts=5,
                                      top_k=2,
                                      ffn=4096,
                                      SOS_token=SOS_TOKEN,
                                      EOS_token=EOS_TOKEN)
        model.load_state_dict(torch.load(file_path, map_location=model.device))
        return model
    else:
        raise Exception('Incorrect model name.  Must be one of NADE, Transformer, DecoderTransformer, MOE.')
    
def main():
    # Set the directory to save the experiment
    new_dir = "Experiment" + MODEL + DATA_NAME + str(TEXTLENGTH)
    path = os.getcwd()
    new_path = os.path.join(path, new_dir)
    os.makedirs(new_path, exist_ok=True)

    # Load and create the dataset
    print('Loading the data.')
    dataset = load_data(FN, TEXTLENGTH)
    print('Building the experiment dataset.')
    exp_data = build_experiment_data(dataset=dataset, 
                                     n_samples=N_SAMPLES, 
                                     n_permutations=N_PERMUTATIONS, 
                                     variables=CAT,
                                     markov=MARKOV)
    
    # Save the dataset
    print('Saving the experimental data.')
    fn = DATA_NAME + str(TEXTLENGTH) + 'samples-' + str(N_SAMPLES) + 'perms-' + str(N_PERMUTATIONS)
    flat_data = exp_data.reshape(-1, exp_data.shape[2]) # flatten out for saving
    save_path = os.path.join(new_path, fn + '.csv')
    np.savetxt(save_path, flat_data, delimiter=',')

    # Save the parameters for recall (will need to reshape since it is flattened)
    save_path = os.path.join(new_path, fn + 'parameters.txt')
    with open(save_path, 'w') as param_file:
        param_file.write(f'{exp_data.shape[0]},{exp_data.shape[1]},{exp_data.shape[2]}')
        param_file.write('\n# The dimensions are in the order: m, n, p')        

    # Build the model
    print('Build the model.')
    model_path = MODEL_PATH
    model = build_model(MODEL, model_path)

    # Test the model
    print('Testing the model.')
    results_perm, results_null = test_model(model=model, exp_dataset=exp_data)

    # Save the perm results
    print('Saving the results.')
    fn = MODEL + '-' + DATA_NAME + str(TEXTLENGTH) + '-permutation_results'
    save_path = os.path.join(new_path, fn + '.csv')
    np.savetxt(save_path, results_perm, delimiter=',')

    # Save the null results
    fn = MODEL + '-' + DATA_NAME + str(TEXTLENGTH) + '-null_results'
    save_path = os.path.join(new_path, fn + '.csv')
    np.savetxt(save_path, results_null, delimiter=',')

if __name__ == "__main__":
    main()