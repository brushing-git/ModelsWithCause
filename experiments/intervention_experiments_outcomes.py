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
from tqdm import tqdm
from csr.datasets import load_data
from csr.models import NADE, Transformer, DecoderTransformer, MoEDecoderTransformer

# Set seeds
torch.manual_seed(0)
np.random.seed(123)
rng = np.random.default_rng(123)

# Data parameters
FN = 'markov_chain-dice-100-intervention-training.txt'
PS_FN = 'markov_chain-dice-100-intervention-probabilities.csv'
TEXTLENGTH = 100
CAT = 7
SOS_TOKEN = 7 # This needs to be set depending on the type of dataset
EOS_TOKEN = 8 # This needs to be set depending on the type of dataset

# Model parameters
MODELS = ['NADE', 'Transformer', 'DecoderTransformer', 'MOE']
MODEL = MODELS[2]
MODEL_PATH = 'DecoderTransformer_10-4-4096.pt'

# Experimental Data Parameters
N_SAMPLES = 1598 # This is for the permutation data, we aim for 80% power at effect size 0.1
DATASETS_NAMES = ['SE-coin', 'SE-dice', 'ME-coin', 'ME-dice']
DATA_NAME = DATASETS_NAMES[3] # Set the index to the right name

# Functions
def load_probabilities(ps_fn: str) -> np.ndarray:
    df = pd.read_csv(ps_fn)
    data = df['0'].to_numpy()
    return data

def build_experiment_data(dataset: np.ndarray, 
                          length: int,
                          rvs: int, 
                          n_samples: int) -> tuple:
    # Sample randomly from dataset
    subset_dataset = dataset[dataset[:,-1] != 6]
    data_indices = np.arange(subset_dataset.shape[0])
    samples = rng.choice(data_indices, n_samples, replace=False)

    # Loop through and construct a experimental dataset of n_samples randomly selected
    experiment_data = np.zeros((n_samples, rvs, length))
    for i, idx in enumerate(samples):
        # Save the sample
        experiment_data[i,0,:] = dataset[idx]
        # Generate different intervention values
        location = np.where(experiment_data[i,0,:] == 6)[0][0] + 1 # We add one to specify the intervention value
        value = experiment_data[i,0,location]
        # Generate the intervention values and remove
        intervention_values = [i for i in range(rvs)]
        intervention_values.remove(value)

        # Add the other values in to simulate a similar intervention
        for j, val in enumerate(intervention_values):
            new_sequence = experiment_data[i,0,:].copy()
            new_sequence[location] = val
            experiment_data[i,j+1,:] = new_sequence
    
    return experiment_data

def test_model(model, exp_dataset: np.ndarray) -> tuple:
    """
    Loops through and averages the 
    """

    samples = exp_dataset.shape[0]
    results_model = np.zeros(samples)
    results_null = np.zeros(samples)

    transform = torch.from_numpy
    indxs = [i for i in range(samples)]
    for i in tqdm(range(samples)):
        # Set target sequence and permutations
        target_sequence = exp_dataset[i,:]

        # Convert target sequence
        target_sequence_x = transform(target_sequence).type(torch.float).unsqueeze(0)
        target_sequence_y = transform(target_sequence).type(torch.long).unsqueeze(0)

        # Register the original log probabilities
        target_log_prob = model.estimate_prob(target_sequence_x, target_sequence_y)
        target_log_prob = sum(target_log_prob[0])

        # Compute and store the difference in log probabilities
        original_log_prob = ps_dataset[i]
        difference_log_prob = target_log_prob - original_log_prob

        results_model[i] = difference_log_prob
        
        # Loop through and build some null data to test
        random_indxs = indxs.copy()
        random_indxs.remove(i)
        random_indxs = rng.choice(random_indxs, rng_samples, replace=False)
        
        for idx in random_indxs:
            # Set the sequence and transform it
            alt_sequence = exp_dataset[idx,:]
            alt_sequence_x = transform(alt_sequence).type(torch.float).unsqueeze(0)
            alt_sequence_y = transform(alt_sequence).type(torch.long).unsqueeze(0)

            # Get the log probabilities
            alt_log_prob = model.estimate_prob(alt_sequence_x, alt_sequence_y)
            alt_log_prob = sum(alt_log_prob[0])

            # Take the value of the difference
            difference_log_prob = alt_log_prob - original_log_prob

            # Store the log probability
            results_null[i] += difference_log_prob
        
        # Average the null results
        results_null[i] = results_null[i] / rng_samples
    
    return results_model, results_null

def build_model(model_name: str, file_path: str, test_model=False):
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
    new_dir = "ExperimentInterventionOutcomes" + MODEL + DATA_NAME + str(TEXTLENGTH)
    path = os.getcwd()
    new_path = os.path.join(path, new_dir)
    os.makedirs(new_path, exist_ok=True)

    # Load and create the dataset
    print('Loading the data.')
    dataset = load_data(FN, TEXTLENGTH)
    ps = load_probabilities(PS_FN)
    exp_data = build_experiment_data(dataset=dataset,
                                     length=TEXTLENGTH,
                                     rvs=6,
                                     n_samples=N_SAMPLES)

    """
    # Save the dataset
    print('Saving the experimental data.')
    fn = DATA_NAME + str(TEXTLENGTH) + 'samples-' + str(N_SAMPLES) + 'interventions_seq'
    flat_data = exp_data.reshape(-1, exp_data.shape[1]) # flatten out for saving
    save_path = os.path.join(new_path, fn + '.csv')
    np.savetxt(save_path, flat_data, fmt='%d', delimiter=',')
    fn = DATA_NAME + str(TEXTLENGTH) + 'samples-' + str(N_SAMPLES) + 'interventions_ps'
    flat_data = ps_data
    save_path = os.path.join(new_path, fn + '.csv')
    np.savetxt(save_path, flat_data, delimiter=',')

    # Build the test model
    print('Building the test model.')
    model_path = MODEL_PATH
    model = build_model(MODEL, model_path, test_model=True)

    # Test the model
    print('Testing the model.')
    results_model, results_null = test_model(model=model, exp_dataset=exp_data, ps_dataset=ps_data)

    # Save the perm results
    print('Saving the results.')
    fn = MODEL + '-' + DATA_NAME + str(TEXTLENGTH) + '-intervention_results'
    save_path = os.path.join(new_path, fn + '.csv')
    np.savetxt(save_path, results_model, delimiter=',')

    # Save the null results
    fn = MODEL + '-' + DATA_NAME + str(TEXTLENGTH) + '-intervention_null_results'
    save_path = os.path.join(new_path, fn + '.csv')
    np.savetxt(save_path, results_null, delimiter=',')

    # Build the no train model
    print('Performing No Train experiments.\n Building the no train model.')
    model_path = MODEL_PATH
    model = build_model(MODEL, model_path, test_model=False)

    # Test the model
    print('Testing the model.')
    results_model, results_null = test_model(model=model, exp_dataset=exp_data, ps_dataset=ps_data)

    # Save the perm results
    print('Saving the results.')
    fn = MODEL + '-' + DATA_NAME + str(TEXTLENGTH) + '-no_train_intervention_results'
    save_path = os.path.join(new_path, fn + '.csv')
    np.savetxt(save_path, results_model, delimiter=',')

    # Save the null results
    fn = MODEL + '-' + DATA_NAME + str(TEXTLENGTH) + '-no_train_intervention_null_results'
    save_path = os.path.join(new_path, fn + '.csv')
    np.savetxt(save_path, results_null, delimiter=',')
    """
if __name__ == "__main__":
    main()