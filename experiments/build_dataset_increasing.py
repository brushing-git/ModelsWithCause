import numpy as np
import pandas as pd
from csr.intervention import build_intervention_big_dataset
from csr.markov_chain import build_markov_big_dataset

# Set random seed
np.random.seed(123)

# Build function to call for producing and saving datasets of increasing size
def build_dataset(name: str, 
                  size: int, 
                  rvs: int = 6, 
                  seq_len: int = 100, 
                  intervention_p: float = 0.0) -> None:

    # Build the dataset
    if intervention_p:
        data, ps, params = build_intervention_big_dataset(rvs=rvs, 
                                                         tr_size=size, 
                                                         seq_len=seq_len, 
                                                         intervention_p=intervention_p, 
                                                         num_processes=8)
    else:
        data, ps, params = build_markov_big_dataset(rvs=rvs, 
                                                    tr_size=size, 
                                                    seq_len=seq_len,
                                                    num_processes=8)

    # Save the dataset
    title = 'intervention' if intervention_p else 'normal'
    fn = name + f'{size}-{seq_len}-{title}-training.txt'
    np.savetxt(fn, data, fmt='%d', delimiter='', newline='')

    fn = name + f'{size}-{seq_len}-{title}-probabilities.csv'
    df = pd.DataFrame(ps)
    df.rename(columns={0: 'Log Probabilities'})
    df.to_csv(fn)

    # Save the parameters as csv
    fn = name + f'{size}-{seq_len}-{title}-priors.csv'
    df = pd.DataFrame(params[0])
    df.rename(columns={0: 'Priors'})
    df.to_csv(fn)

    fn = name + f'{size}-{seq_len}-{title}-markov_mats.csv'
    markov_mats = np.concatenate(params[1], axis=0)
    df = pd.DataFrame(markov_mats)
    df.to_csv(fn)

def main():
    # Build the transformer datasets
    sizes = [100000, 1000000, 10000000]
    name = 'markov-dice-transformer-'
    for s in sizes:
        # Build the normal dataset
        build_dataset(name, size=s)
    
        # Build the intervention dataset
        build_dataset(name, size=s, intervention_p=0.5)
    
    # Build the Decoder transformer datasets
    sizes = [187500, 1875000, 18750000]
    name = 'markov-dice-decoder-'
    for s in sizes:
        # Build the normal dataset
        build_dataset(name, size=s)

        # Build the intervention dataset
        build_dataset(name, size=s, intervention_p=0.5)
    
    # Build the MOE transformer datasets
    sizes = [625000, 6250000, 62500000]
    name = 'markov-dice-moe-'
    for s in sizes:
        # Build the normal dataset
        build_dataset(name, size=s)
    
        # Build the intervention dataset
        build_dataset(name, size=s, intervention_p=0.5)

if __name__ == "__main__":
    main()