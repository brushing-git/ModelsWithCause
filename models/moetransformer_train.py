import torch
import numpy as np
import os
from torch.utils.data import DataLoader
from torch.nn import CrossEntropyLoss
from csr.models import MoEDecoderTransformer
from csr.datasets import load_data, build_datasets, build_trte_dataloader

torch.manual_seed(0)
np.random.seed(0)

# Constants to be changed based on dataset and model params
FN = 'markov_chain-dice-100-normal-training.txt'
DATA_NAME = 'ME-Normal'
TEXTLENGTH = 6
CAT = 6 #6 for default, 7 for intervention
SOS_TOKEN = 6 # This needs to be set depending on the type of dataset, 6 by default; 7 for intervention
EOS_TOKEN = 7 # This needs to be set depending on the type of dataset, 7 by default; 8 for intervention
BATCH_SIZE = 100
N_HEADS = 10
DECODER_LYRS = 4
DROPOUT = 0.0
N_EXPERTS = 5
FFN = 4096
LR = 0.0001
EPOCHS = 4

def main():
    # Create directory and save
    new_dir = "MoEDecoderTransformer" + DATA_NAME + str(TEXTLENGTH)
    path = os.getcwd()
    new_path = os.path.join(path, new_dir)
    os.makedirs(new_path, exist_ok=True)

    # Load the data
    print('Loading the data.')
    dataset = load_data(FN, TEXTLENGTH)
    tr_load, te_load = build_trte_dataloader(dataset, te_size=0.2, batch_size=BATCH_SIZE)
    final_test_load = te_load

    # Build the model
    print('Building the model.')
    net = MoEDecoderTransformer(n_tokens=CAT+2, 
                                dim_model=TEXTLENGTH, 
                                n_heads=N_HEADS, 
                                n_decoder_lyrs=DECODER_LYRS, 
                                dropout_p=DROPOUT, 
                                n_experts=N_EXPERTS,
                                top_k=2, 
                                ffn=FFN,
                                SOS_token=SOS_TOKEN,
                                EOS_token=EOS_TOKEN)
    print('Training the model.')
    net.fit(tr_loader=tr_load, te_loader=te_load, epochs=EPOCHS, lr=LR)

    # Save the model
    print('Saving the model.')
    model_save_path = os.path.join(new_path, f'MoEDecoderTransformer_{N_HEADS}-{DECODER_LYRS}-{N_EXPERTS}-{FFN}.pt')
    torch.save(net.state_dict(), model_save_path)

    # Final evaluation
    criterion = CrossEntropyLoss()
    loss = net._eval(final_test_load, criterion)
    print(f'The final loss is {loss}.')

main()
