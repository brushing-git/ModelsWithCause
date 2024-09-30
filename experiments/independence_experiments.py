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
from tqdm import tqdm
from csr.datasets import load_data
from csr.independence_tests import *
from csr.models import NADE, Transformer, DecoderTransformer, MoEDecoderTransformer

# Set seeds
torch.manual_seed(0)
np.random.seed(123)
rng = np.random.default_rng(123)

# Data parameters
FN = 'markov_chain-dice-100-normal-training.txt'
PS_FN = 'markov_chain-dice-100-normal-probabilities.csv'
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
N_SAMPLES = 6388 # This is for the number of samples to test independence against; we aim for 80% power at effect size 0.05
DATASETS_NAMES = ['SE-coin', 'SE-dice', 'ME-coin', 'ME-dice']
DATA_NAME = DATASETS_NAMES[3] # Set the index to the right name

# Functions
def load_probabilities(ps_fn: str) -> np.ndarray:
    df = pd.read_csv(ps_fn)
    data = df['0'].to_numpy()
    return data

def build_experiment_data(
        dataset: np.ndarray, 
        ps: np.ndarray,
        n_samples: int 
    ) -> np.ndarray:
    # Sample randomly from dataset
    data_indices = np.arange(dataset.shape[0])
    rndm_indices = np.random.choice(data_indices, n_samples)

    # Get the experimental data
    experiment_data = dataset[rndm_indices, :]

    # Get the probabilities
    experiment_ps = ps[rndm_indices]
    
    return experiment_data, experiment_ps

def test_model(
        model, 
        exp_dataset: np.ndarray,
        seq_len: int
    ) -> tuple:
    """
    Loops through an experimental dataset and computes the difference between the first sequence and its 
    permutations assigned log probabilities.
    """

    results = test_markov_property(model, sequences=exp_dataset)

    return results

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
    new_dir = "ExperimentIndependence" + MODEL + DATA_NAME + str(TEXTLENGTH)
    path = os.getcwd()
    new_path = os.path.join(path, new_dir)
    os.makedirs(new_path, exist_ok=True)

    # Load and create the dataset
    print('Loading the data.')
    dataset = load_data(FN, TEXTLENGTH)
    ps = load_probabilities(PS_FN)
    print('Building the experiment dataset.')
    exp_data, exp_ps = build_experiment_data(
        dataset=dataset, ps=ps, n_samples=N_SAMPLES
    )
    
    # Save the dataset
    print('Saving the experimental data.')
    fn = DATA_NAME + str(TEXTLENGTH) + 'samples-' + str(N_SAMPLES) + '_independence_tests_data'
    flat_data = exp_data.flatten() # flatten out for saving
    save_path = os.path.join(new_path, fn + '.csv')
    np.savetxt(save_path, flat_data, delimiter=',')
    fn = DATA_NAME + str(TEXTLENGTH) + 'samples-' + str(N_SAMPLES) + '_independence_tests_ps'
    flat_data = exp_ps
    save_path = os.path.join(new_path, fn + '.csv')
    np.savetxt(save_path, flat_data, delimiter=',')    

    # Build the model
    print('Build the model.')
    model_path = MODEL_PATH
    model = build_model(MODEL, model_path, test_model=True)

    # Test the model
    print('Testing the model.')
    results = test_model(
        model=model, exp_dataset=exp_data, seq_len=TEXTLENGTH
    )

    # Save the independence results
    print('Saving the results.')
    fn = MODEL + '-' + DATA_NAME + str(TEXTLENGTH) + '-train-independence_results'
    save_path = os.path.join(new_path, fn + '.csv')
    np.savetxt(save_path, results, delimiter=',')

    # The No Train Results
    # Build the model
    print('Build the model.')
    model_path = MODEL_PATH
    model = build_model(MODEL, model_path, test_model=False)

    # Test the model
    print('Testing the model.')
    results = test_model(
        model=model, exp_dataset=exp_data, seq_len=TEXTLENGTH
    )

    # Save the no train results
    print('Saving the results.')
    fn = MODEL + '-' + DATA_NAME + str(TEXTLENGTH) + '-notrain-independence_results'
    save_path = os.path.join(new_path, fn + '.csv')
    np.savetxt(save_path, results, delimiter=',')

if __name__ == "__main__":
    main()