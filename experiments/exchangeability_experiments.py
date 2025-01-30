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
import concurrent.futures
import os
import time
from tqdm import tqdm
from scipy.stats import wasserstein_distance, kendalltau
from csr.datasets import load_data
from csr.generate_perms import build_permutations
from csr.models import NADE, Transformer, DecoderTransformer, MoEDecoderTransformer

# Set seeds
torch.manual_seed(0)
np.random.seed(123)
rng = np.random.default_rng(123)

# Data parameters
FN = 'markov_chain-dice-100-normal-training.txt'
PS_FN = 'markov_chain-dice-100-normal-probabilities.csv'
MAT_FN = 'markov_chain-dice-100-normal-markov_mats.csv'
MARKOV = True # Set this parameter to generate permutations of Markov Exchangeable sequences
TEXTLENGTH = 100
CAT = 6
SOS_TOKEN = 6 # This needs to be set depending on the type of dataset
EOS_TOKEN = 7 # This needs to be set depending on the type of dataset

# Model parameters
MODELS = ['NADE', 'Transformer', 'DecoderTransformer', 'MOE']
MODEL = MODELS[3]
MODEL_PATH = 'MoEDecoderTransformer_10-4-5-4096.pt'

# Experimental Data Parameters
N_SAMPLES = 1598 # This is for the permutation data, we aim for 80% power at effect size 0.1
N_PERMUTATIONS = 3 # Keep this low otherwise it will take forever to build the dataset
DATASETS_NAMES = ['SE-coin', 'SE-dice', 'ME-coin', 'ME-dice']
DATA_NAME = DATASETS_NAMES[3] # Set the index to the right name

# Functions
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
    permutations = build_permutations(variables=variables, 
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

def calculate_order(
        model: torch.nn.Module,
        x: torch.tensor,
        y: torch.tensor
) -> tuple:
    # Get the model logits
    model.eval()
    model.to(model.device)

    with torch.no_grad():
        x, y = model._append_SOS_EOS(x), model._append_SOS_EOS(y)
        x, y = x.to(model.device), y.to(model.device)

        # Shift the tgt and mask
        y_input = y[:,:-1]
        y_expected = y[:,1:]
        sequence_length = y_input.size(1)
        tgt_mask = model._get_tgt_mask(sequence_length).to(model.device)

        y_hat = model.forward(x, y_input, tgt_mask)
        y_hat = y_hat.permute(1,2,0) # Permute to (batch, tokens, length)

        # Get the order ranking across the sequence
        ps = model.logprob(y_hat)
        ps = ps[:,:-2,2:-1] # We drop the first two items (sos token and the first item in the sequence) and the eos token
        values, indices = torch.sort(ps, dim=1, descending=True)

        # Convert to numpy
        values, indices, ps = values.squeeze().detach().cpu().numpy(), indices.squeeze().detach().cpu().numpy(), ps.squeeze().detach().cpu().numpy()
    
    return values, indices, ps

def compare_ordering(
        p1: np.ndarray,
        p2: np.ndarray
) -> float:
    """
    Takes two probability orderings for tokens of size (tokens, length) and compares them item by item in the length 
    and returns a similarity score of the ranking. The similarity score is:

    | order relations in p1 intersect order relations in p2 | / | order relations in p1 |

    We do this by taking the rankings in both and seeing what elements are less than it in both.

    Args:
    p1 : np.ndarray : array of ranked by tokens with lower index meaning highest and higher index meaning lower shape (tokens, length)
    p2 : np.ndarray : same as above

    Return:
    similarity_score : float : a float scoring the similarity
    """
    max = 0
    total = 0
    for i in range(p1.shape[1]):
        for j in range(p1.shape[0]):
            # Get the index of the value in j
            indx = np.argmax(p2[:,i] == p1[j,i])

            # Get the intersection of all common elements
            intersection = np.intersect1d(p1[j+1:,i], p2[indx+1:,i])

            # Get the quantity in common
            if j + 1 < p1.shape[0]:
                total += intersection.size / (p1.shape[0]-(j+1))
                max += 1
            else:
                total += intersection.size

    total /= max

    return total

def levenshtein_distance(
        p1: np.ndarray,
        p2: np.ndarray
) -> float:
    scores = np.zeros(p1.shape[1])

    for i in range(p1.shape[1]):
        seq1, seq2 = p1[:,i].flatten(), p2[:,i].flatten()

        m, n = len(seq1), len(seq2)

        matrix = np.zeros((m+1, n+1), dtype=int)
        matrix[:, 0] = np.arange(m+1)
        matrix[0, :] = np.arange(n+1)

        for j in range(1, m+1):
            for k in range(1, n+1):
                if seq1[j-1] == seq2[k-1]:
                    substitution_cost = 0
                else:
                    substitution_cost = 1
                
                matrix[j, k] = min(
                    matrix[j-1, k] + 1,
                    matrix[j, k-1] + 1,
                    matrix[j-1, k-1] + substitution_cost
                )
        
        scores[i] = 1 - matrix[m, n] / max(m, n)
    
    return scores.mean()

def build_experiment_data(
        dataset: np.ndarray, 
        ps: np.ndarray,
        n_samples: int, 
        n_permutations: int, 
        variables: int,
        markov: bool = True
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
                        result = future.result(timeout=5)
                        # Store the probabilities
                        experiment_ps[i] = ps[idx]
                        # Increment the counter
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
                    # Store the probabilities
                    experiment_ps[i] = ps[idx]
                    i += 1
            else:
                break
    
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
        ps_matrix: np.ndarray
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

    # Store the average comparison values
    comparison_indices = np.zeros(samples)
    comparison_values = np.zeros(samples)
    comparison_sim = np.zeros(samples)
    comparison_lev = np.zeros(samples)
    comparison_rand = np.zeros(samples)
    lev_rand = np.zeros(samples)

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

        # Get the difference between the probabilities after the first item and the transition matrix
        # Get the model's ordered probabilities
        values, indices, ps = calculate_order(model, target_sequence_x, target_sequence_y)

        # Create a comparison matrix
        matrix_seq_ps = np.zeros_like(ps)
        matrix_sort_ps = np.zeros_like(values)
        matrix_sort_order = np.zeros_like(indices)

        # Set the target portion of the ps_matrix
        start_indx = int(target_sequence[0])
        target_ps_matrix = ps_matrix[start_indx*CAT:(start_indx*CAT)+CAT]

        for j in range(1, matrix_sort_ps.shape[1] + 1):
            # Get the last item in the target sequence
            indx = int(target_sequence[j-1])

            # Set the corresponding row so that it is ordered
            log_probs = np.log(target_ps_matrix[indx])
            ps_sorted = np.sort(log_probs)[::-1]
            order_sorted = np.argsort(log_probs)[::-1]

            # Store the elements in the correct matrix
            matrix_seq_ps[:, j-1] = log_probs
            matrix_sort_ps[:, j-1] = ps_sorted
            matrix_sort_order[:, j-1] = order_sorted
        
        # Get average identity between the matrices
        comparison_order = (indices == matrix_sort_order).astype(np.float32)
        comparison_indices[i] = comparison_order.mean()

        # Get average difference between the probabilities
        comparison_differences = values - matrix_sort_ps
        comparison_values[i] = comparison_differences.mean()

        # Get the similarities
        comparison_sim[i] = compare_ordering(indices, matrix_sort_order)

        # Get the levenshtein distance
        comparison_lev[i] = levenshtein_distance(indices, matrix_sort_order)

        # Generate and check from a random array
        random_indices = np.concatenate([np.random.choice(np.arange(6), size=(indices.shape[0],1), replace=False) for _ in range(indices.shape[1])])
        comparison_rand[i] = compare_ordering(random_indices, matrix_sort_order)
        lev_rand[i] = levenshtein_distance(random_indices, matrix_sort_order)
    
    # Calculate the wasserstein distance
    wass_distance = calculate_wasserstein(ps=ps_dataset, probs_model=probs_model, probs_null=probs_null)

    # Print the average of the random indices
    print(f'The average for randomly generated indices is {comparison_rand.mean()}')

    # Print the test
    print(f'The random levenshtein value is {lev_rand.mean()}')
    
    return results_perm, results_null, wass_distance, probs_model, probs_null, comparison_indices, comparison_values, comparison_sim, comparison_lev

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
                            dim_model=TEXTLENGTH, 
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
                                   dim_model=TEXTLENGTH,
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
                                      dim_model=TEXTLENGTH,
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

    # Build the model
    print('Build the model.')
    model_path = MODEL_PATH
    model = build_model(MODEL, model_path, test_model=True)

    # Test the model
    print('Testing the model.')
    results_perm, results_null, results_wasserstein, model_probs, null_probs, avg_comp, avg_diff, avg_sim, avg_lev = test_model(
        model=model, exp_dataset=exp_data, ps_dataset=exp_ps, ps_matrix=ps_mats
    )

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

    # Save the model comparisons
    fn = MODEL + '-' + DATA_NAME + str(TEXTLENGTH) + '-train_avg_comparison'
    save_path = os.path.join(new_path, fn + '.csv')
    np.savetxt(save_path, avg_comp, delimiter=',')
    print(f'The average comparison was {avg_comp.mean()}')
    fn = fn = MODEL + '-' + DATA_NAME + str(TEXTLENGTH) + '-train_avg_difference'
    save_path = os.path.join(new_path, fn + '.csv')
    np.savetxt(save_path, avg_diff, delimiter=',')
    print(f'The average difference was {avg_diff.mean()}')
    fn = MODEL + '-' + DATA_NAME + str(TEXTLENGTH) + '-train_avg_sim'
    save_path = os.path.join(new_path, fn + '.csv')
    np.savetxt(save_path, avg_sim, delimiter=',')
    print(f'The average similarity was {avg_sim.mean()}')
    fn = MODEL + '-' + DATA_NAME + str(TEXTLENGTH) + '-train_avg_lev'
    save_path = os.path.join(new_path, fn + '.csv')
    np.savetxt(save_path, avg_lev, delimiter=',')
    print(f'The average levenshtein was {avg_lev.mean()}')

    # The No Train Results
    # Build the model
    print('Build the model.')
    model_path = MODEL_PATH
    model = build_model(MODEL, model_path, test_model=False)

    # Test the model
    print('Testing the model.')
    results_perm, results_null, results_wasserstein, model_probs, null_probs, avg_comp, avg_diff, avg_sim, avg_lev = test_model(
        model=model, exp_dataset=exp_data, ps_dataset=exp_ps, ps_matrix=ps_mats
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

    # Save the model comparisons
    fn = MODEL + '-' + DATA_NAME + str(TEXTLENGTH) + '-notrain_avg_comparison'
    save_path = os.path.join(new_path, fn + '.csv')
    np.savetxt(save_path, avg_comp, delimiter=',')
    print(f'The average comparison for no train was {avg_comp.mean()}')
    fn = fn = MODEL + '-' + DATA_NAME + str(TEXTLENGTH) + '-notrain_avg_difference'
    save_path = os.path.join(new_path, fn + '.csv')
    np.savetxt(save_path, avg_diff, delimiter=',')
    print(f'The average difference for no train was {avg_diff.mean()}')
    fn = MODEL + '-' + DATA_NAME + str(TEXTLENGTH) + '-notrain_avg_sim'
    save_path = os.path.join(new_path, fn + '.csv')
    np.savetxt(save_path, avg_sim, delimiter=',')
    print(f'The average similarity was {avg_sim.mean()}')
    fn = MODEL + '-' + DATA_NAME + str(TEXTLENGTH) + '-notrain_avg_lev'
    save_path = os.path.join(new_path, fn + '.csv')
    np.savetxt(save_path, avg_lev, delimiter=',')
    print(f'The average levenshtein was {avg_lev.mean()}')

if __name__ == "__main__":
    main()