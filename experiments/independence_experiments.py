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
from typing import Any
from tqdm import tqdm
from argparse import ArgumentParser
from scipy import stats
from src.data.datasets import load_data
from src.tests.independence_tests import test_markov_property
from src.nets.models import NADE, Transformer, DecoderTransformer, MoEDecoderTransformer

# Set seeds
torch.manual_seed(0)
np.random.seed(123)
rng = np.random.default_rng(123)

# Data parameters
FNS = {
    'NADE': 'nade_markov_chain-dice-100-normal-training.txt',
    'Transformer': 'transformer_markov_chain-dice-100-normal-training.txt',
    'DecoderTransformer': 'decodert_markov_chain-dice-100-normal-training.txt',
    'MOE': 'moe_markov_chain-dice-100-normal-training.txt'
}
PS_FN = 'markov_chain-dice-100-normal-probabilities.csv'
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
N_SAMPLES = 6388 # This is for the number of samples to test independence against; we aim for 80% power at effect size 0.05
DATASETS_NAMES = ['SE-coin', 'SE-dice', 'ME-coin', 'ME-dice']

# Seeds
SEEDS = [51, 92, 14, 71, 60]

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
    ) -> np.ndarray:
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

def calculate_statistics(data: np.ndarray, confidence: float = 0.95) -> dict:
    """
    Calculate key statistics including confidence intervals.
    
    Args:
        data: numpy array of values
        confidence: confidence level for interval (default 0.95)
    
    Returns:
        dict with mean, variance, std, and confidence interval
    """
    mean = np.mean(data)
    variance = np.var(data, ddof=1)  # Sample variance
    std = np.std(data, ddof=1)  # Sample standard deviation
    n = len(data)
    
    # Calculate confidence interval using t-distribution
    confidence_level = confidence
    degrees_freedom = n - 1
    t_value = stats.t.ppf((1 + confidence_level) / 2, degrees_freedom)
    margin_error = t_value * (std / np.sqrt(n))
    
    ci_lower = mean - margin_error
    ci_upper = mean + margin_error
    
    return {
        'mean': mean,
        'variance': variance,
        'std': std,
        'ci_lower': ci_lower,
        'ci_upper': ci_upper,
        'confidence': confidence,
        'n': n
    }
    
def main():
    parser = ArgumentParser(
        prog='independence_experiments',
        description='Loads models and data and runs independence tests.',
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
    FN = FNS[args.model]
    DATA_NAME = args.data
    
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

    # Storage for seed experiments
    no_results = True
    seed_results_list = []

    # Loop through the seeds and test the models
    for i, seed in enumerate(SEEDS):
        print(f"Running experiment {i+1}/{len(SEEDS)} with seed {seed}")

        # Build the model
        print('Build the model.')
        model_path = MODEL_PATH + f"{seed}.pt"
        model = build_model(MODEL, model_path, test_model=True)

        # Test the model
        print('Testing the model.')
        seed_results = test_model(
            model=model, exp_dataset=exp_data, seq_len=TEXTLENGTH
        )

        # Store results for this seed
        seed_results_list.append(seed_results)

        if no_results:
            results = seed_results.copy()
            no_results = False
        else:
            results += seed_results

    # Aggregate results (average across seeds)
    results = results / len(SEEDS)
    
    # Convert seed_results_list to array for statistics calculation
    seed_results_array = np.array(seed_results_list)

    # Save the independence results
    print('Saving the results.')
    fn = MODEL + '-' + DATA_NAME + str(TEXTLENGTH) + '-train-independence_results'
    save_path = os.path.join(new_path, fn + '.csv')
    np.savetxt(save_path, results, delimiter=',')

    # Calculate and save statistics
    print('Calculating statistics.')
    stats_dict = calculate_statistics(results)
    
    # Also calculate statistics for each individual result if results is multidimensional
    fn = MODEL + '-' + DATA_NAME + str(TEXTLENGTH) + '-train-independence_statistics'
    save_path = os.path.join(new_path, fn + '.txt')
    with open(save_path, 'w') as f:
        f.write("=" * 80 + "\n")
        f.write("Independence Test Results - Trained Models\n")
        f.write("=" * 80 + "\n\n")
        f.write(f"Model: {MODEL}\n")
        f.write(f"Data: {DATA_NAME}\n")
        f.write(f"Text Length: {TEXTLENGTH}\n")
        f.write(f"Number of Samples: {N_SAMPLES}\n")
        f.write(f"Number of Seeds: {len(SEEDS)}\n")
        f.write(f"Seeds: {SEEDS}\n\n")
        
        f.write("-" * 80 + "\n")
        f.write("JSD Statistics (Aggregated)\n")
        f.write("-" * 80 + "\n")
        f.write(f"Mean: {stats_dict['mean']:.6f}\n")
        f.write(f"Variance: {stats_dict['variance']:.6f}\n")
        f.write(f"Standard Deviation: {stats_dict['std']:.6f}\n")
        f.write(f"{stats_dict['confidence']*100:.0f}% Confidence Interval: [{stats_dict['ci_lower']:.6f}, {stats_dict['ci_upper']:.6f}]\n")
        f.write(f"Sample Size (n): {stats_dict['n']}\n\n")
        
        # Add per-seed statistics
        f.write("-" * 80 + "\n")
        f.write("Per-Seed Results\n")
        f.write("-" * 80 + "\n")
        for i, seed in enumerate(SEEDS):
            seed_mean = np.mean(seed_results_list[i])
            f.write(f"Seed {seed}: Mean JSD = {seed_mean:.6f}\n")

    # The No Train Results

    # Reset all random states for reproducible non-trained comparison
    torch.manual_seed(42)
    np.random.seed(42)

    # Build the model
    print('Build the non-trained model.')
    model_path = MODEL_PATH
    model = build_model(MODEL, model_path, test_model=False)

    # Test the model
    print('Testing the non-trained model.')
    results_notrain = test_model(
        model=model, exp_dataset=exp_data, seq_len=TEXTLENGTH
    )

    # Save the no train results
    print('Saving the non-trained results.')
    fn = MODEL + '-' + DATA_NAME + str(TEXTLENGTH) + '-notrain-independence_results'
    save_path = os.path.join(new_path, fn + '.csv')
    np.savetxt(save_path, results_notrain, delimiter=',')

    # Calculate and save statistics for non-trained model
    print('Calculating statistics for non-trained model.')
    stats_dict_notrain = calculate_statistics(results_notrain)
    
    fn = MODEL + '-' + DATA_NAME + str(TEXTLENGTH) + '-notrain-independence_statistics'
    save_path = os.path.join(new_path, fn + '.txt')
    with open(save_path, 'w') as f:
        f.write("=" * 80 + "\n")
        f.write("Independence Test Results - Non-Trained Model\n")
        f.write("=" * 80 + "\n\n")
        f.write(f"Model: {MODEL}\n")
        f.write(f"Data: {DATA_NAME}\n")
        f.write(f"Text Length: {TEXTLENGTH}\n")
        f.write(f"Number of Samples: {N_SAMPLES}\n\n")
        
        f.write("-" * 80 + "\n")
        f.write("JSD Statistics\n")
        f.write("-" * 80 + "\n")
        f.write(f"Mean: {stats_dict_notrain['mean']:.6f}\n")
        f.write(f"Variance: {stats_dict_notrain['variance']:.6f}\n")
        f.write(f"Standard Deviation: {stats_dict_notrain['std']:.6f}\n")
        f.write(f"{stats_dict_notrain['confidence']*100:.0f}% Confidence Interval: [{stats_dict_notrain['ci_lower']:.6f}, {stats_dict_notrain['ci_upper']:.6f}]\n")
        f.write(f"Sample Size (n): {stats_dict_notrain['n']}\n")

    print("\n" + "=" * 80)
    print("Experiment Complete!")
    print("=" * 80)
    print(f"\nTrained Model - Mean JSD: {stats_dict['mean']:.6f}")
    print(f"Trained Model - 95% CI: [{stats_dict['ci_lower']:.6f}, {stats_dict['ci_upper']:.6f}]")
    print(f"\nNon-Trained Model - Mean JSD: {stats_dict_notrain['mean']:.6f}")
    print(f"Non-Trained Model - 95% CI: [{stats_dict_notrain['ci_lower']:.6f}, {stats_dict_notrain['ci_upper']:.6f}]")
    print("=" * 80 + "\n")

if __name__ == "__main__":
    main()
