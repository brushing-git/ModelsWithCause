import torch
import scipy
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, SubsetRandomSampler
from sklearn.model_selection import KFold
from CSR.datasets import load_data, build_datasets

def kfold_cross_val(model, 
                    train_data,
                    params: dict, 
                    epochs: int, 
                    lr: float, 
                    kfolds: int, 
                    batch_size: int = 64) -> tuple:
    kf = KFold(n_splits=kfolds, shuffle=True)
    metrics = []

    for fold, (train_idx, test_idx) in enumerate(kf.split(train_data)):
        print(f"Fold {fold + 1}")
        print("-------")

        # Define the data loaders
        tr_loader = DataLoader(
            dataset=train_data,
            batch_size=batch_size,
            sampler=SubsetRandomSampler(train_idx)
        )

        te_loader = DataLoader(
            dataset=train_data,
            batch_size=batch_size,
            sampler=SubsetRandomSampler(test_idx)
        )

        k_model = model(**params)
        hist = k_model.fit(tr_loader, te_loader, epochs, lr)
        tr_loss = hist['tr_loss'][-1]
        te_loss = hist['te_loss'][-1]

        print(f"Final metrics:\n  tr_loss: {tr_loss}, te_loss: {te_loss}")
        metrics.append(te_loss)
    
    avg_te_loss = np.mean(metrics)
    print(f"The average loss across {kfolds} k folds is {avg_te_loss}")
    return metrics, avg_te_loss

def grid_search(model, 
                train_data, 
                model_params: dict, 
                param1_name: str, 
                params1: list, 
                param2_name: str, 
                params2: list, 
                epochs: int,
                kfolds: int, 
                lr: float = 0.0, 
                batch_size: int = 64) -> pd.DataFrame:
    
    # Check if we have a positive learning rate and the params2 list has floats
    if not lr and not all(isinstance(p, float) for p in params2):
        raise ValueError('Non-positive learning rate and params2 is not all floats.')

    rows = [f'{param1_name}={p}' for p in params1]
    cols = [f'{param2_name}={p}' for p in params2]
    df = pd.DataFrame(index=rows, columns=cols)

    for i, r in enumerate(params1):
        for j, c in enumerate(params2):

            # Check if we are doing grid search on lr
            if lr:
                # Do kfold cross-validation
                print(f'Testing parameters {r} and {c} in grid item {(i*len(params2))+(j+1)} of {len(params1)*len(params2)}')
                print('------')
                model_params[param1_name] = r
                model_params[param2_name] = c
                _, te_loss = kfold_cross_val(model, train_data, model_params, 
                                                epochs, lr, kfolds, batch_size)
            else:
                # Do kfold cross-validation
                print(f'Testing parameters {r} and {c} in grid item {(i*len(params2))+(j+1)} of {len(params1)*len(params2)}')
                print('------')
                model_params[param1_name] = r
                _, te_loss = kfold_cross_val(model, train_data, model_params, 
                                             epochs, lr=c, kfolds=kfolds, batch_size=batch_size)
            df.iloc[i,j] = te_loss
    
    return df

def plot_grid(df: pd.DataFrame, param1_name: str, param2_name: str, fn: str) -> None:
    fig, ax = plt.subplots(1, 1, figsize=(10,8), dpi=900)

    x = df.to_numpy().astype(float)
    im = ax.imshow(x, cmap='viridis')
    rows = [''.join(c for c in r if (c.isdigit() or c == '.')) for r in df.index.to_list()]
    cols = [''.join(c for c in r if (c.isdigit() or c == '.')) for r in df.columns.to_list()]

    ax.set_xticks(np.arange(len(cols)), labels=cols)
    ax.set_yticks(np.arange(len(rows)), labels=rows)
    ax.set_xlabel(param2_name)
    ax.set_ylabel(param1_name)

    cbar = ax.figure.colorbar(im, ax=ax)
    cbar.ax.set_ylabel('Test Loss', rotation=-90, va="bottom")
    ax.set_title(f'Test Loss on Grid Search for {param1_name} and {param2_name}')

    fig.tight_layout()
    plt.show()

    print('Saving the figure.')
    fig.savefig(fn + '.pdf', dpi=900, format='pdf')