import torch
import numpy as np
import os
from src.nets.models import DecoderTransformer
from src.data.datasets import load_data, build_datasets, build_trte_dataloader

# Constants to be changed based on dataset and model params
FN = 'decodert_markov_chain-dice-100-normal-training.txt'
DATA_NAME = 'ME-Normal'
TEXTLENGTH = 100
CAT = 6 #6 for default, 7 for intervention
SOS_TOKEN = 6 # This needs to be set depending on the type of dataset, 6 by default; 7 for intervention
EOS_TOKEN = 7 # This needs to be set depending on the type of dataset, 7 by default; 8 for intervention
BATCH_SIZE = 250
DIM_MODEL = 100
N_HEADS = 10
DECODER_LYRS = 4
DROPOUT = 0.0
FFN = 4096
LR = 0.0001
EPOCHS = 4
SEEDS = [51, 92, 14, 71, 60]

def main():
    # Create directory and save
    new_dir = "DecoderTransformer" + DATA_NAME + str(TEXTLENGTH)
    path = os.getcwd()
    new_path = os.path.join(path, new_dir)
    os.makedirs(new_path, exist_ok=True)

    print("initializing and training decoder-only transformer")

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
        print('Building the model.')
        net = DecoderTransformer(n_tokens=CAT+2, 
                                dim_model=DIM_MODEL, 
                                n_heads=N_HEADS, 
                                n_decoder_lyrs=DECODER_LYRS, 
                                dropout_p=DROPOUT, 
                                ffn=FFN,
                                SOS_token=SOS_TOKEN,
                                EOS_token=EOS_TOKEN)
        print('Training the model.')
        net.fit(tr_loader=tr_load, te_loader=te_load, epochs=EPOCHS, lr=LR)

        # Save the model
        print('Saving the model.')
        model_save_path = os.path.join(new_path, f'DecoderTransformer_{N_HEADS}-{DECODER_LYRS}-{FFN}_SEED_{seed}.pt')
        torch.save(net.state_dict(), model_save_path)

main()
