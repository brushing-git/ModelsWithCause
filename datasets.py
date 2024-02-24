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
    data = np.loadtxt(fn, dtype='str')
    data = data.item()

    X = np.array([[float(c) for c in data[i:i+10]] for i in range(0, len(data), item_len)])
    return X

def build_trte_dataloader(X: np.ndarray, te_size=0.2) -> tuple:
    X_train, X_test = train_test_split(X,test_size=te_size)

    tr_data = ExchangeData(X_train)
    te_data = ExchangeData(X_test)

    tr_loader = DataLoader(tr_data, batch_size=64, shuffle=True)
    te_loader = DataLoader(te_data, batch_size=64, shuffle=True)

    return tr_loader, te_loader