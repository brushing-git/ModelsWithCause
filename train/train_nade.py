import torch
import numpy as np
import os
from src.nets.models import NADE
from src.data.datasets import load_data, build_datasets, build_trte_dataloader

torch.manual_seed(0)
np.random.seed(0)

# Constants to be changed based on dataset and model params
FN = 'nade_markov_chain-dice-100-normal-training.txt'
DATA_NAME = 'ME-Normal'
TEXTLENGTH = 100
CAT = 6
HIDDEN_DIM = 16
LR = 0.0001
EPOCHS = 250
BATCH_SIZE = 250
SEEDS = [51, 92, 14, 71, 60]

def main():
    # Create directory and 
    new_dir = "NADE" + DATA_NAME + str(TEXTLENGTH)
    path = os.getcwd()
    new_path = os.path.join(path, new_dir)
    os.makedirs(new_path, exist_ok=True)

    print("initializing and training nade")

    # Loop through and train the models with new seeds
    for i, seed in enumerate(SEEDS):
        # Reset the seed
        torch.manual_seed(seed)
        np.random.seed(seed)

        print(f"training on seed {i+1}/{len(SEEDS)}")

        # Load the data
        print('Loading the data.')
        dataset = load_data(FN, TEXTLENGTH)
        tr_load, te_load = build_trte_dataloader(dataset, te_size=0.2, batch_size=BATCH_SIZE)

        # Build the model
        net = NADE(in_dim=TEXTLENGTH, hidden_dim=HIDDEN_DIM, cat=CAT)
        net.fit(tr_loader=tr_load, te_loader=te_load, epochs=EPOCHS, lr=LR)

        # Save the model
        model_save_path = os.path.join(new_path, f'NADE_{HIDDEN_DIM}{LR}_SEED_{seed}.pt')
        torch.save(net.state_dict(), model_save_path)

main()
