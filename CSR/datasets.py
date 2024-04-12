import torch
import numpy as np
import os
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split

class ExchangeData(Dataset):
    def __init__(self, X, transform=torch.from_numpy) -> None:
        self.X = X
        self.transform = transform
    
    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, idx):
        x = self.X[idx]
        y = self.X[idx]

        if self.transform:
            x = self.transform(x)
            x = x.type(torch.float)
            y = self.transform(y)
            y = y.type(torch.long)
        
        return x, y

def load_data(fn: str, item_len: int) -> np.ndarray:
    # Open the file and read all lines
    with open(fn, 'r') as file:
        data = file.read().replace(',', '')  # Read the whole file, remove commas

    # Split into chunks of specified length and convert to float
    X = np.array([[float(c) for c in data[i:i+item_len]] for i in range(0, len(data), item_len)])
    print(X)
    return X

def build_datasets(X: np.ndarray, te_size=0.2) -> tuple:
    X_train, X_test = train_test_split(X,test_size=te_size)

    tr_data = ExchangeData(X_train)
    te_data = ExchangeData(X_test)

    return tr_data, te_data

def build_trte_dataloader(X: np.ndarray, te_size=0.2, batch_size: int=64) -> tuple:
    tr_data, te_data = build_datasets(X, te_size)

    tr_loader = DataLoader(tr_data, batch_size=batch_size, shuffle=True)
    te_loader = DataLoader(te_data, batch_size=batch_size, shuffle=True)

    return tr_loader, te_loader