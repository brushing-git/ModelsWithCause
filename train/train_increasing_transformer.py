import torch
import numpy as np
import os
from torch.utils.data import DataLoader
from torch.nn import CrossEntropyLoss
from src.nets.models import Transformer
from src.data.datasets import load_data, build_datasets, build_trte_dataloader

torch.manual_seed(0)
np.random.seed(0)

# Constants to be changed based on dataset and model params
DATA_NAME = 'ME-Normal'
TEXTLENGTH = 100
CAT = 6
SOS_TOKEN = 6 # This needs to be set depending on the type of dataset, 6 by default; 7 for intervention
EOS_TOKEN = 7 # This needs to be set depending on the type of dataset, 7 by default; 8 for intervention
BATCH_SIZE = 250
DIM_MODEL = 128
N_HEADS = 8
ENCODER_LYRS = 2
DECODER_LYRS = 8
DROPOUT = 0.1
FFN = 512
LR = 0.0001
EPOCHS = 4

def train_model(data_fn, data_name) -> None:
    # Create directory and save
    new_dir = "Transformer" + data_name + str(TEXTLENGTH)
    path = os.getcwd()
    new_path = os.path.join(path, new_dir)
    os.makedirs(new_path, exist_ok=True)

    # Load the data
    print('Loading the data.')
    dataset = load_data(data_fn, TEXTLENGTH)
    tr_load, te_load = build_trte_dataloader(dataset, te_size=0.2, batch_size=BATCH_SIZE)
    final_test_load = te_load

    # Build the model
    print('Building the model.')
    net = Transformer(n_tokens=CAT+2, 
                      dim_model=DIM_MODEL, 
                      n_heads=N_HEADS, 
                      n_encoder_lyrs=ENCODER_LYRS, 
                      n_decoder_lyrs=DECODER_LYRS, 
                      dropout_p=DROPOUT, 
                      ffn=FFN,
                      SOS_token=SOS_TOKEN,
                      EOS_token=EOS_TOKEN)
    print('Training the model.')
    net.fit(tr_loader=tr_load, te_loader=te_load, epochs=EPOCHS, lr=LR)

    # Save the model
    print('Saving the model.')
    model_save_path = os.path.join(new_path, f'Transformer_{N_HEADS}-{ENCODER_LYRS}-{DECODER_LYRS}-{FFN}.pt')
    torch.save(net.state_dict(), model_save_path)

def main():
    sizes = [10**(5+i) for i in range(3)]

    fns = [f"markov-dice-transformer-{s}-100-normal-training.txt" for s in sizes]

    dns = [DATA_NAME + f'-{s}' for s in sizes]

    for fn, dn in zip(fns, dns):
        train_model(fn, dn)

main()
