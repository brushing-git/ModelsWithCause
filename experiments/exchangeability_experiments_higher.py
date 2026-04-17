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
import pandas as pd
import os
import multiprocessing as mp
from typing import Any
from tqdm import tqdm
from argparse import ArgumentParser
from scipy.stats import wasserstein_distance, kendalltau
from src.data.datasets import load_data
from src.data.generate_perms import build_permutations_order2
from src.nets.models import NADE, Transformer, DecoderTransformer, MoEDecoderTransformer

# Set seeds
torch.manual_seed(0)
np.random.seed(123)
rng = np.random.default_rng(123)

# Data parameters
FNS = {
    'NADE': 'nade_higher_markov_chain-dice-100-normal-training.txt',
    'Transformer': 'transformer_higher_markov_chain-dice-100-normal-training.txt',
    'DecoderTransformer': 'decodert_higher_markov_chain-dice-100-normal-training.txt',
    'MOE': 'moe_higher_markov_chain-dice-100-normal-training.txt'
}
PS_FN = 'transformer_higher_markov_chain-dice-100-normal-probabilities.csv'
MAT_FN = 'transformer_higher_markov_chain-dice-100-normal-markov_mats.csv'
MARKOV = True # Set this parameter to generate permutations of Markov Exchangeable sequences
TEXTLENGTH = 100
CAT = 6
SOS_TOKEN = 6 # This needs to be set depending on the type of dataset
EOS_TOKEN = 7 # This needs to be set depending on the type of dataset

# Model parameters
MODELS = {
    'NADE': 'NADE_160.0001_SEED_',
    'Transformer': 'Transformer_5-2-8-2048_SEED_',
    'DecoderTransformer': 'DecoderTransformer_10-4-4096_SEED_',
    'MOE': 'MoEDecoderTransformer_10-4-5-4096_SEED_'
}

# Experimental Data Parameters
N_SAMPLES = 1598 # This is for the permutation data, we aim for 80% power at effect size 0.1
N_PERMUTATIONS = 3 # Keep this low otherwise it will take forever to build the dataset
DATASETS_NAMES = ['SE-coin', 'SE-dice', 'ME-coin', 'ME-dice']

# Seeds
SEEDS = [51, 92, 14, 71, 60]

# Functions
def _backtrack_worker(
        variables: int,
        sequence: np.ndarray,
        n_perms: int,
        return_dict: dict[str, Any],
        idx: int
) -> dict[str, Any]:
    try:
        permutations = build_permutations_order2(
            variables=variables,
            sequence=sequence,
            n_perms=n_perms
        )

        return_dict['permutations'] = permutations
        return_dict['success'] = True
    except Exception as e:
        return_dict['success'] = False
        return_dict['error'] = e

def load_probabilities(ps_fn: str) -> np.ndarray:
    df = pd.read_csv(ps_fn)
    data = df['0'].to_numpy()
    return data

def backtrack_search(
        variables: int, 
        sequence: np.ndarray, 
        n_perms: int, 
        experiment_data: np.ndarray, 
        idx: int
) -> None:
    # Do the backtracking search
    permutations = build_permutations_order2(variables=variables, 
                                      sequence=sequence, 
                                      n_perms=n_perms)
    # Store the original sequence as the first element in dim=1
    experiment_data[idx,0,:] = sequence[:]
    # Store the remainder as the following elements in in dim=1
    experiment_data[idx,1:,:] = permutations[:,:]

def exchangeable_permutations(
        sequence: np.ndarray,
        n_perms: int,
        experiment_data: np.ndarray,
        idx: int
) -> None:
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

def build_experiment_data(
        dataset: np.ndarray, 
        ps: np.ndarray,
        n_samples: int, 
        n_permutations: int, 
        variables: int,
        markov: bool = True,
        timeout: int = 5
) -> np.ndarray:
    # Sample randomly from dataset
    data_indices = np.arange(dataset.shape[0])
    rng.shuffle(data_indices)

    # Build the permutations, we will store them in a (n_samples, n_permutations+1, sequence_length) array
    length = dataset.shape[1]
    experiment_data = np.zeros((n_samples, n_permutations+1, length))

    # Storage for the probabilities
    experiment_ps = np.zeros(n_samples)

    # Index counter for experiment_data
    i = 0

    # Track indices we've already tried to avoid infinite loops
    tried_indices = set()

    for idx in data_indices:
        if i >= n_samples:
            break

        # Skip if we've exhaused all possible indices
        if idx in tried_indices:
            continue
        tried_indices.add(idx)

        print(f'Trying sample {i}.')
        sequence = dataset[idx, :]

        if markov:
            # Use multiprocessing to handle timeouts
            manager = mp.Manager()
            return_dict = manager.dict()

            process = mp.Process(
                target=_backtrack_worker,
                args=(
                    variables,
                    sequence,
                    n_permutations,
                    return_dict,
                    i
                )
            )

            process.start()
            process.join(timeout=timeout) # Wait up to timeout seconds

            if process.is_alive():
                # Kill the process
                process.terminate()
                process.join(timeout=1)

                if process.is_alive():
                    # Force kill if not dead
                    process.kill()
                    process.join()
                
                print(f'Search timed out for index {idx}. Moving to the next sample.')
                continue

            # Check if the worker succeeded
            if return_dict.get('success', False):
                permutations = return_dict['permutations']

                # Verify we got a valid permtutations (not None)
                if permutations is not None and not np.any(np.isnan(permutations)):
                    experiment_data[i, 0, :] = sequence[:]
                    experiment_data[i, 1:, :] = permutations[:, :]
                    experiment_ps[i] = ps[idx]
                    i += 1
                else:
                    print(f'Invalid permutations for index {idx}. Moving to the next sample.')
            else:
                error = return_dict.get('error', 'Unknown error')
                print(f'Error for index {idx}: {error}. Moving to the next sample.')
        else:
            # Non-markov case - exchangeable permutations
            exchangeable_permutations(
                sequence=sequence,
                n_perms=n_permutations,
                experiment_data=experiment_data,
                idx=i
            )
            experiment_ps[i] = ps[idx]
            i += 1
    
    if i < n_samples:
        print(f'Warning: Only found {i} valid samples out of {n_samples} requested.')
        
        # Trim the array
        experiment_data = experiment_data[:i]
        experiment_ps = experiment_ps[:i]
    
    return experiment_data, experiment_ps

def calculate_wasserstein(
        ps: np.ndarray, 
        probs_model: np.ndarray, 
        probs_null: np.ndarray
) -> np.ndarray:
    # Create storage array
    results = np.zeros((3,3))

    # Convert to probabilities
    ps, probs_model, probs_null = np.exp(ps.copy()), np.exp(probs_model.copy()), np.exp(probs_null.copy())

    # Renormalize
    ps /= np.sum(ps)
    probs_model /= np.sum(probs_model)
    probs_null /= np.sum(probs_null)

    # Store in lists for iterating
    prob_list = [ps, probs_model, probs_null]

    # Iterate over list and store in array
    for i, p1 in enumerate(prob_list):
        for j, p2 in enumerate(prob_list):
            results[i,j] = wasserstein_distance(p1, p2)
    
    return results

def test_model(
        model: torch.nn.Module, 
        exp_dataset: np.ndarray, 
        ps_dataset: np.ndarray,
    ) -> tuple:
    """
    Loops through an experimental dataset and computes the difference between the first sequence and its 
    permutations assigned log probabilities.

    Args:
    model : torch.nn.Module : a pytorch model from one our built models
    exp_dataset : np.ndarray : a formatted dataset for experiments
    ps_dataset : np.ndarray : the probabilities of each sequence in the dataset
    ps_matrix : np.ndarray : the stochastic matrix for the dataset we are using

    Returns:
    results_perm : np.ndarray : the log difference between the permutations and the model sequence
    results_null : np.ndarray : the log difference between random sequences and the model sequence
    wass_distance : np.ndarray : the Wasserstein-1 distance between the model probabilities
    probs_model : np.ndarray : the actual probability the model assigns to the sequence
    probs_null : np.ndarray : the average probability the model assigns to random sequences
    comparison_indices : np.ndarray : the average agreement in ordering between the target sequence and the model's probs
    comparison_values : np.ndarray : the average difference between the target sequence and the model's conditional probs
    comparison_sim : np.ndarray : the average similarity between the target sequence and the model's probs
    """
    # Reset the random seed
    rng = np.random.default_rng(123)

    # Samples
    samples = exp_dataset.shape[0]

    # Results
    results_perm = np.zeros((samples, exp_dataset.shape[1]-1))
    results_null = np.zeros((samples, exp_dataset.shape[1]-1))

    # Transforms
    transform = torch.from_numpy
    indxs = [i for i in range(samples)]

    # Store the target probabilities for target and null
    probs_model = np.zeros(samples)
    probs_null = np.zeros(samples)

    for i in tqdm(range(samples)):
        # Set target sequence and permutations
        target_sequence = exp_dataset[i,0,:] # shape (1,L)
        permutations = exp_dataset[i,1:,:]

        # Convert target sequence
        target_sequence_x = transform(target_sequence).type(torch.float).unsqueeze(0)
        target_sequence_y = transform(target_sequence).type(torch.long).unsqueeze(0)

        # Register the original log probabilities
        target_log_prob = model.estimate_prob(target_sequence_x, target_sequence_y)
        target_log_prob = sum(target_log_prob[0])

        # Store the model probabilities
        probs_model[i] = target_log_prob

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
        
        for j, idx in enumerate(random_indxs):
            # Set the sequence and transform it
            alt_sequence = exp_dataset[idx,0,:]
            alt_sequence_x = transform(alt_sequence).type(torch.float).unsqueeze(0)
            alt_sequence_y = transform(alt_sequence).type(torch.long).unsqueeze(0)

            # Get the perm log probabilities
            alt_log_prob = model.estimate_prob(alt_sequence_x, alt_sequence_y)
            alt_log_prob = sum(alt_log_prob[0])

            # Store the alt probs
            probs_null[i] += alt_log_prob

            # Take the absolute value of the difference
            difference_log_prob = target_log_prob - alt_log_prob

            # Store the log probability
            results_null[i,j] = difference_log_prob

        # Average the probs null
        probs_null[i] = probs_null[i] / len(random_indxs)
    
    # Calculate the wasserstein distance
    wass_distance = calculate_wasserstein(ps=ps_dataset, probs_model=probs_model, probs_null=probs_null)
    
    return results_perm, results_null, wass_distance, probs_model, probs_null

def build_model(
        model_name: str, 
        file_path: str, 
        test_model=False
) -> torch.nn.Module:
    """
    Builds the appropriate model based on the model_name keyword.
    """

    if model_name == 'NADE':
        model = NADE(in_dim=TEXTLENGTH, hidden_dim=16, cat=CAT)
        if test_model:
            model.load_state_dict(torch.load(file_path, map_location=model.device))
        return model
    elif model_name == 'Transformer':
        model = Transformer(n_tokens=CAT+2, 
                            dim_model=100, 
                            n_heads=5, 
                            n_encoder_lyrs=2,
                            n_decoder_lyrs=8,
                            dropout_p=0.0,
                            ffn=2048,
                            SOS_token=SOS_TOKEN,
                            EOS_token=EOS_TOKEN)
        if test_model:
            model.load_state_dict(torch.load(file_path, map_location=model.device))
        return model
    elif model_name == 'DecoderTransformer':
        model = DecoderTransformer(n_tokens=CAT+2,
                                   dim_model=100,
                                   n_heads=10,
                                   n_decoder_lyrs=4,
                                   dropout_p=0.0,
                                   ffn=4096,
                                   SOS_token=SOS_TOKEN,
                                   EOS_token=EOS_TOKEN)
        if test_model:
            model.load_state_dict(torch.load(file_path, map_location=model.device))
        return model
    elif model_name == 'MOE':
        model = MoEDecoderTransformer(n_tokens=CAT+2,
                                      dim_model=100,
                                      n_heads=10,
                                      n_decoder_lyrs=4,
                                      dropout_p=0.0,
                                      n_experts=5,
                                      top_k=2,
                                      ffn=4096,
                                      SOS_token=SOS_TOKEN,
                                      EOS_token=EOS_TOKEN)
        if test_model:
            model.load_state_dict(torch.load(file_path, map_location=model.device))
        return model
    else:
        raise Exception('Incorrect model name.  Must be one of NADE, Transformer, DecoderTransformer, MOE.')
    
def main():
    parser = ArgumentParser(
        prog='exchangeability_experiments',
        description='Loads some models and data and runs some tests.',
        epilog='Only for use on experimental data.'
    )
    parser.add_argument('model', 
                        help='The model name to be tested.',
                        choices=list(MODELS.keys()),
                        type=str)
    parser.add_argument('data',
                        help='The type of data to be tested.',
                        choices=DATASETS_NAMES,
                        type=str)
    
    args = parser.parse_args()
    MODEL = args.model
    MODEL_PATH = MODELS[args.model]
    DATA_NAME = args.data
    FN = FNS[args.model]
    
    # Set the directory to save the experiment
    new_dir = "ExperimentNormal" + MODEL + DATA_NAME + str(TEXTLENGTH)
    path = os.getcwd()
    new_path = os.path.join(path, new_dir)
    os.makedirs(new_path, exist_ok=True)

    # Load and create the dataset
    print('Loading the data.')
    dataset = load_data(FN, TEXTLENGTH)
    ps = load_probabilities(PS_FN)
    print('Building the experiment dataset.')
    exp_data, exp_ps = build_experiment_data(
        dataset=dataset, 
        ps=ps,
        n_samples=N_SAMPLES, 
        n_permutations=N_PERMUTATIONS, 
        variables=CAT,
        markov=MARKOV
    )
    
    # Save the dataset
    print('Saving the experimental data.')
    fn = DATA_NAME + str(TEXTLENGTH) + 'samples-' + str(N_SAMPLES) + 'perms-' + str(N_PERMUTATIONS)
    flat_data = exp_data.reshape(-1, exp_data.shape[2]) # flatten out for saving
    save_path = os.path.join(new_path, fn + '.csv')
    np.savetxt(save_path, flat_data, delimiter=',')
    fn = DATA_NAME + str(TEXTLENGTH) + 'samples-' + str(N_SAMPLES) + '_ps'
    flat_data = exp_ps
    save_path = os.path.join(new_path, fn + '.csv')
    np.savetxt(save_path, flat_data, delimiter=',')

    # Save the parameters for recall (will need to reshape since it is flattened)
    save_path = os.path.join(new_path, fn + 'parameters.txt')
    with open(save_path, 'w') as param_file:
        param_file.write(f'{exp_data.shape[0]},{exp_data.shape[1]},{exp_data.shape[2]}')
        param_file.write('\n# The dimensions are in the order: m, n, p')        

    # Load the dataset stochastic matrix
    print('Loading the transition matrix.')
    ps_mats = pd.read_csv(MAT_FN).to_numpy()
    ps_mats = ps_mats[:,1:] # Drop the index rows and columns

    # Storage for seed experiments
    no_results = True

    # Loop through the seeds and test the models
    for i, seed in enumerate(SEEDS):
        print(f"running experiment {i+1}/{len(SEEDS)}")

        # Build the model
        print('Build the model.')
        model_path = MODEL_PATH + f"{seed}.pt"
        model = build_model(MODEL, model_path, test_model=True)

        # Test the model
        print('Testing the model.')
        seed_results = test_model(
            model=model, exp_dataset=exp_data, ps_dataset=exp_ps
        )

        if no_results:
            results = list(seed_results)
            no_results = False
        else:
            results = [result + seed_result for result, seed_result in zip(results, seed_results)]

    # Aggregate results
    results = [result / len(SEEDS) for result in results]
    results_perm, results_null, results_wasserstein, model_probs, null_probs = results
    
    # Save the perm results
    print('Saving the results.')
    fn = MODEL + '-' + DATA_NAME + str(TEXTLENGTH) + '-train-permutation_results'
    save_path = os.path.join(new_path, fn + '.csv')
    np.savetxt(save_path, results_perm, delimiter=',')

    # Save the null results
    fn = MODEL + '-' + DATA_NAME + str(TEXTLENGTH) + '-train-null_results'
    save_path = os.path.join(new_path, fn + '.csv')
    np.savetxt(save_path, results_null, delimiter=',')

    # Save the wasserstein
    fn = MODEL + '-' + DATA_NAME + str(TEXTLENGTH) + '-train_wasserstein_results'
    save_path = os.path.join(new_path, fn + '.csv')
    np.savetxt(save_path, results_wasserstein, delimiter=',')

    # Save the model probabilities and null probs
    fn = MODEL + '-' + DATA_NAME + str(TEXTLENGTH) + '-train_model_probs'
    save_path = os.path.join(new_path, fn + '.csv')
    np.savetxt(save_path, model_probs, delimiter=',')
    fn = MODEL + '-' + DATA_NAME + str(TEXTLENGTH) + '-train_null_probs'
    save_path = os.path.join(new_path, fn + '.csv')
    np.savetxt(save_path, null_probs, delimiter=',')

    # The No Train Results
    # Build the model
    print('Build the model.')
    model_path = MODEL_PATH
    model = build_model(MODEL, model_path, test_model=False)

    # Test the model
    print('Testing the model.')
    results_perm, results_null, results_wasserstein, model_probs, null_probs = test_model(
        model=model, exp_dataset=exp_data, ps_dataset=exp_ps
    )

    # Save the perm results
    print('Saving the results.')
    fn = MODEL + '-' + DATA_NAME + str(TEXTLENGTH) + '-notrain-permutation_results'
    save_path = os.path.join(new_path, fn + '.csv')
    np.savetxt(save_path, results_perm, delimiter=',')

    # Save the null results
    fn = MODEL + '-' + DATA_NAME + str(TEXTLENGTH) + '-notrain-null_results'
    save_path = os.path.join(new_path, fn + '.csv')
    np.savetxt(save_path, results_null, delimiter=',')

    # Save the wasserstein
    fn = MODEL + '-' + DATA_NAME + str(TEXTLENGTH) + '-notrain_wasserstein_results'
    save_path = os.path.join(new_path, fn + '.csv')
    np.savetxt(save_path, results_wasserstein, delimiter=',')

    # Save the model probabilities and null probs
    fn = MODEL + '-' + DATA_NAME + str(TEXTLENGTH) + '-notrain_model_probs'
    save_path = os.path.join(new_path, fn + '.csv')
    np.savetxt(save_path, model_probs, delimiter=',')
    fn = MODEL + '-' + DATA_NAME + str(TEXTLENGTH) + '-notrain_null_probs'
    save_path = os.path.join(new_path, fn + '.csv')
    np.savetxt(save_path, null_probs, delimiter=',')

if __name__ == "__main__":
    main()