import os
import numpy as np
import pandas as pd
import time
from itertools import product
from scipy.stats import chi2_contingency, ncx2
from tqdm import tqdm
from src.data.datasets import load_data

FN = "markov_chain-dice-100-normal-training.txt"
TEXTLENGTH = 100
SEED = 42
N = 100000
N_SAMPLES = 30
LAGS = [ i for i in range(2, 30) ]

def test_conditional_independence(
        sequences: np.ndarray,
        t: int,
        lag: int = 2,
        v: float = 0.15
) -> list[np.ndarray]:
    """
    Test X_t \bot X_{t-lag} | X_{t-1}, X_{0} across i.i.d. sequences using chi2 test.
    """
    # Check for each value of the conditioning variable X_{t-1}
    p_values = []
    cvs = []

    for x1 in range(6):
        # Store the p values for each
        x1_p_values = []
        x1_cvs = []

        for x_cond in range(6):
            mask = (sequences[:, t-1] == x_cond) & (sequences[:, 0] == x1)

            if mask.sum() > 30:
                # Construct contigency diagram
                contigency = np.histogram2d(
                    sequences[mask, t],
                    sequences[mask, t-lag],
                    bins=6
                )[0]

                # Smooth the data
                contigency = contigency + 1

                # Perform chi2 test for indepedence
                try:
                    chi2_stat, _, dof, _ = chi2_contingency(contigency)
                except:
                    raise ValueError(f"failed contigency test on start {x1} t {t} index {t-lag} with table\n{contigency}")

                # Calculate delta with Cramer's V
                n = contigency.sum()
                min_dim = contigency.shape[0]
                cramers_v_obs = np.sqrt(chi2_stat / (n * (min_dim - 1)))
                delta = n * (v ** 2)

                # Get the p value for dependence
                p_value = ncx2.cdf(chi2_stat, dof, delta)

                x1_p_values.append(p_value)
                x1_cvs.append(cramers_v_obs)

        # Store the average
        p_values.append(np.mean(x1_p_values))
        cvs.append(np.mean(x1_cvs))

    return p_values, cvs

def main():
    # Set the random seed
    np.random.seed(SEED)

    # Set the directory to save the experiment
    new_dir = "ExperimentBaseline"
    path = os.getcwd()
    new_path = os.path.join(path, new_dir)
    os.makedirs(new_path, exist_ok=True)

    # Load and create the dataset
    print("Loading the data.")
    data = load_data(FN, TEXTLENGTH)

    # Get randomly sampled sequences
    indxs = np.random.choice(data.shape[0], size=N, replace=False).tolist()
    sequences = data[indxs, :]
    print(f"Sequences shape {sequences.shape}")

    # Randomly sample the items up to maximum lag
    times = np.random.choice([i for i in range(max(LAGS)+2, TEXTLENGTH)], size=N_SAMPLES, replace=False).tolist()

    # Create results frame
    p_value_columns = [f"p_value_{i+1}" for i in range(6)]
    c_value_columns = [f"cramers_v_obs_{i+1}" for i in range(6)]
    columns = ["t", "lag"] + p_value_columns + c_value_columns
    df = pd.DataFrame(columns=columns)

    # Get the experiments
    experiments = list(product(*[times, LAGS]))

    print("Conducting the experiments.")

    for t, lag in tqdm(experiments):
        # Perform the experiment
        p, c = test_conditional_independence(
            sequences,
            t,
            lag
        )

        # Store the average p value
        results = [t, lag] + p + c
        df.loc[len(df)] = results
    
    print("Experiments completed succesfully.")

    for i, column in enumerate(p_value_columns):
        print(f"p value on index {i+1}: {df[column].mean():.4f} +/- {df[column].std():.4f}")
    
    for i, column in enumerate(c_value_columns):
        print(f"cramers v value on index {i+1}: {df[column].mean():.4f} +/- {df[column].std():.4f}")

    print("Saving the data.")
    fn = f"baseline_results_length_{TEXTLENGTH}_samples_{N_SAMPLES}.csv"
    f_path = os.path.join(new_path, fn)
    df.to_csv(f_path)

if __name__ == "__main__":
    main()