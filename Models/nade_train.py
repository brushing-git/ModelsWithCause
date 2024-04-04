import torch
import numpy as np
import os
from torch.utils.data import DataLoader
from torch.nn import CrossEntropyLoss
from CSR.models import NADE
from CSR.datasets import load_data, build_datasets, build_trte_dataloader

torch.manual_seed(0)
np.random.seed(0)

# Constants to be changed based on dataset and model params
FN = 'markov_chain-dice-medium-training.txt'
DATA_NAME = 'ME'
TEXTLENGTH = 100
CAT = 6
HIDDEN_DIM = 16
LR = 0.0001
EPOCHS = 250
BATCH_SIZE = 50

def main():
    # Create directory and 
    new_dir = "NADE" + DATA_NAME + str(TEXTLENGTH)
    path = os.getcwd()
    new_path = os.path.join(path, new_dir)
    os.makedirs(new_path, exist_ok=True)

    # Load the data
    dataset = load_data(FN, TEXTLENGTH)
    tr_load, te_load = build_trte_dataloader(dataset, te_size=0.2, batch_size=BATCH_SIZE)
    final_test_load = te_load

    # Build the model
    net = NADE(in_dim=TEXTLENGTH, hidden_dim=HIDDEN_DIM, cat=CAT)
    net.fit(tr_loader=tr_load, te_loader=te_load, epochs=EPOCHS, lr=LR)

    # Save the model
    model_save_path = os.path.join(new_path, f'NADE_{HIDDEN_DIM}{LR}.pt')
    torch.save(net.state_dict(), model_save_path)

    # Final evaluation
    criterion = CrossEntropyLoss()
    loss = net._eval(final_test_load, criterion)
    print(f'The final loss is {loss}.')

main()