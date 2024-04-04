import torch
import numpy as np
import os
from torch.utils.data import DataLoader
from torch.nn import CrossEntropyLoss
from CSR.models import Transformer
from CSR.datasets import load_data, build_datasets, build_trte_dataloader

torch.manual_seed(0)
np.random.seed(0)

# Constants to be changed based on dataset and model params
FN = 'markov_chain-dice-medium-training.txt'
DATA_NAME = 'ME'
TEXTLENGTH = 100
CAT = 6
BATCH_SIZE = 250
N_HEADS = 5
ENCODER_LYRS = 2
DECODER_LYRS = 8
DROPOUT = 0.0
FFN = 2048
LR = 0.0001
EPOCHS = 4

def main():
    # Create directory and save
    new_dir = "Transformer" + DATA_NAME + str(TEXTLENGTH)
    path = os.getcwd()
    new_path = os.path.join(path, new_dir)
    os.makedirs(new_path, exist_ok=True)

    # Load the data
    dataset = load_data(FN, TEXTLENGTH)
    tr_load, te_load = build_trte_dataloader(dataset, te_size=0.2, batch_size=BATCH_SIZE)
    final_test_load = te_load

    # Build the model
    net = Transformer(n_tokens=CAT, dim_model=TEXTLENGTH, n_heads=N_HEADS, n_encoder_lyrs=ENCODER_LYRS, n_decoder_lyrs=DECODER_LYRS, dropout_p=DROPOUT, ffn=FFN)
    net.fit(tr_loader=tr_load, te_loader=te_load, epochs=EPOCHS, lr=LR)

    # Save the model
    model_save_path = os.path.join(new_path, f'Transformer_{N_HEADS}-{ENCODER_LYRS}-{DECODER_LYRS}-{FFN}.pt')
    torch.save(net.state_dict(), model_save_path)

    # Final evaluation
    criterion = CrossEntropyLoss()
    loss = net._eval(final_test_load, criterion)
    print(f'The final loss is {loss}.')

main()