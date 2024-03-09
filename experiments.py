import argparse
import pandas as pd
import torch
import numpy as np
from CSR.gridsearch import grid_search, plot_grid
from CSR.models import NADE, Transformer, DecoderTransformer, MoEDecoderTransformer
from CSR.datasets import load_data, build_datasets

torch.manual_seed(0)
np.random.seed(0)

# Constants to be changed based on dataset
FN = 'markov_chain-dice-medium-training.txt'
DATA_NAME = 'ME'
TEXTLENGTH = 100
CAT = 6

# Experimental parameters
LEARNING_RATES = [10**(-(i+1)) for i in range(5)]
HIDDEN_DIMS = [2**(i+3) for i in range(10)]
N_HEADS = [2, 5, 10]
N_ENCODER_LYRS = [i for i in range(2, 22, 2)]
N_DENCODER_LYRS = [i for i in range(2, 22, 2)]
DROPOUT_P = [i*0.1 for i in range(6)]
FFN = [2**(i+3) for i in range(10)]
N_EXPERTS = [i for i in range(2, 10, 1)]
TOP_K = [i for i in range(2, 5, 1)]

EXP_VALUES = {
    'hidden_dim': HIDDEN_DIMS,
    'n_heads': N_HEADS, 
    'n_encoder_lyrs': N_ENCODER_LYRS, 
    'n_decoder_lyrs': N_DENCODER_LYRS,
    'dropout_p': DROPOUT_P, 
    'ffn': FFN, 
    'n_experts': N_EXPERTS, 
    'top_k': TOP_K,
    'lr': LEARNING_RATES
}

def main():
    parser = argparse.ArgumentParser(
        prog='Causal Reasoning Experiments',
        description='Runs a series of causal reasoning experiments'
    )

    parser.add_argument(
        'model',
        default='NADE',
        choices=['NADE', 'Transformer', 'DecoderTransformer', 'MOE'],
        metavar='M',
        type=str,
        help='Choose a model to perform grid search on:  NADE, Transformer, DecoderTransformer, MOE'
    )
    parser.add_argument(
        'epochs',
        default=25,
        metavar='E',
        type=int,
        help='Number of training epochs'
    )
    parser.add_argument(
        'kfolds',
        default=5,
        metavar='K',
        type=int,
        help='Specify the number of k-folds for cross-validation'
    )
    parser.add_argument(
        'lr',
        default=0.001,
        metavar='L',
        type=float,
        help='The learning rate'
    )
    parser.add_argument(
        'batch_size',
        default=64,
        metavar='B',
        type=int,
        help='The batch size.'
    )
    parser.add_argument(
        'param1',
        choices=['hidden_dim', 'n_heads', 'n_encoder_lyrs', 'n_decoder_lyrs',
                 'dropout_p', 'ffn', 'n_experts', 'top_k'],
        metavar='P1',
        type=str,
        help='The first parameter to be tested.'
    )
    parser.add_argument(
        'param2',
        choices=['n_heads', 'n_encoder_lyrs', 'n_decoder_lyrs',
                 'dropout_p', 'ffn', 'n_experts', 'top_k', 'lr'],
        default='lr',
        metavar='P2',
        type=str,
        help='The second parameter to be tested.  If none passed, learning rate is used.'
    )

    args = parser.parse_args()
    model = args.model
    epochs = args.epochs
    kfolds = args.kfolds
    lr = args.lr
    batch_size = args.batch_size
    param1_name = args.param1
    param2_name = args.param2

    # Make sure the parameters are different
    if param1_name == param2_name:
        raise Exception('Provide different parameter values.')

    # Load the data
    dataset = load_data(FN, TEXTLENGTH)
    tr_data, te_data = build_datasets(dataset)

    # Set up the models and parameters
    if model == 'NADE':
        net = NADE
        params = {'in_dim': TEXTLENGTH, 'hidden_dim': 512, 'cat': CAT}

        if param1_name not in params.keys():
            raise Exception('Parameter 1 incorrect value for this model.')
        
        param2_name = 'lr'
        params1 = HIDDEN_DIMS
        params2 = LEARNING_RATES
        lr = 0.0
    else:
        if model == 'Transformer':
            net = Transformer
            params = {'n_tokens': CAT+2, 
                  'dim_model': TEXTLENGTH,
                  'n_heads': 5,
                  'n_encoder_lyrs': 6,
                  'n_decoder_lyrs': 6,
                  'dropout_p': 0.1,
                  'ffn': 2048}
        elif model == 'DecoderTransformer':
            net = DecoderTransformer
            params = {'n_tokens': CAT+2, 
                  'dim_model': TEXTLENGTH,
                  'n_heads': 5,
                  'n_decoder_lyrs': 6,
                  'dropout_p': 0.1,
                  'ffn': 2048}
        else:
            net = MoEDecoderTransformer
            params = {'n_tokens': CAT+2, 
                  'dim_model': TEXTLENGTH,
                  'n_heads': 5,
                  'n_decoder_lyrs': 6,
                  'dropout_p': 0.1,
                  'n_experts': 2,
                  'top_k': 2,
                  'ffn': 2048}

        if param1_name not in params.keys():
            raise Exception('Parameter 1 incorrect for this model.')
        
        if param2_name not in params.keys() and param2_name != 'lr':
            raise Exception('Parameter 2 incorrect for this model.')
        
        params1 = EXP_VALUES[param1_name]
        params2 = EXP_VALUES[param2_name]
        if param2_name == 'lr':
            lr = 0.0


    # Perform grid search on the model
    print('Conducting grid search.')
    df = grid_search(model=net,
                     train_data=tr_data,
                     model_params=params,
                     param1_name=param1_name,
                     params1=params1,
                     param2_name=param2_name,
                     params2=params2,
                     epochs=epochs,
                     kfolds=kfolds,
                     lr=lr,
                     batch_size=batch_size)
    
    # Save the data
    print('Saving the search.')
    fn = DATA_NAME + 'grid_search' + model + '_' + param1_name + param2_name + \
        '_epochs=' + str(epochs) + '_kfolds=' + str(kfolds) + '_batch=' \
        + str(batch_size)
    df.to_csv(fn + '.csv')

    # Build the figure
    plot_grid(df=df, 
              param1_name=param1_name, 
              param2_name=param2_name,
              fn=fn)

if __name__ == "__main__":
    main()