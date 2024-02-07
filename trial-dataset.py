# do not use

# Bruce Rushing 
# edited by Colin Roberson

import os
path = os.getcwd()
os.chdir(path + '/pytorch-generative') # may need to change this depending on folder/env

import numpy as np

np.random.seed(69)

# Build the data set
TRSIZE = 100000
TESIZE = 20000
RVL = 10

p1 = 0.7 # coin 1
p2 = 0.35 # coin 2
p3 = 0.5 # coin 3
prior = 0.33 # 1/3 : 2/3 mixture USE ARRAY HERE (SUM TO 1)
# choose different priors based on previous

training_d = np.zeros((TRSIZE, RVL))
for i in range(TRSIZE):
    # attempting to randomly cycle 3 coins
    switch: float
    switch = np.random.binomial(n=2, p=prior) / 2
    if switch == 1:
        training_d[i,:] = np.random.binomial(n=1, p=p1, size=RVL) # https://numpy.org/doc/stable/reference/random/generated/numpy.random.binomial.html 
    elif switch > 0:
        training_d[i,:] = np.random.binomial(n=1, p=p2, size=RVL)
    else:
        training_d[i,:] = np.random.binomial(n=1, p=p3, size=RVL)

training_d = training_d.astype(np.single)
print(training_d.dtype)
        
testing_d = np.zeros((TESIZE, RVL))
for i in range(TESIZE):
    # attempting to randomly cycle 3 coins
    switch: float
    switch = np.random.binomial(n=2, p=prior) / 2
    if switch == 1:
        testing_d[i,:] = np.random.binomial(n=1, p=p1, size=RVL)
    elif switch > 0:
        testing_d[i,:] = np.random.binomial(n=1, p=p2, size=RVL)
    else:
        testing_d[i,:] = np.random.binomial(n=1, p=p3, size=RVL)

testing_d = testing_d.astype(np.single)
print(testing_d.dtype)

print(training_d.shape)
print(testing_d.shape)

print(training_d)
print(testing_d)

# below taken from Bruce Rushing file

import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils

class bernDataset(Dataset):
    
    def __init__(self, variables, transform=None):
        self.variables = variables
        self.transform = transform
    
    def __len__(self):
        return len(self.variables)
    
    def __getitem__(self, idx):
        sample = self.variables[idx]
        
        if self.transform:
            sample = self.transform(sample)
        
        return sample
    

from pytorch_generative.models.autoregressive.nade import NADE
from torch import optim
from torch.nn import functional as F
from pytorch_generative import trainer

log_dir = os.makedirs(os.path.join(os.getcwd(), 'checkpoint'))
epochs = 50

transformers = lambda x: torch.from_numpy(x).float()

tr_set = bernDataset(training_d, transform=transformers)
te_set = bernDataset(testing_d, transform=transformers)
tr_loader = torch.utils.data.DataLoader(tr_set, batch_size=100, shuffle=True)
te_loader = torch.utils.data.DataLoader(te_set, batch_size=100, shuffle=True)

model = NADE(input_dim=RVL, hidden_dim=100)
optimizer = optim.Adam(model.parameters())

def loss_fn(x, _, preds):
    batch_size = x.shape[0]
    x, preds = x.view((batch_size, -1)), preds.view((batch_size, -1))
    loss = F.binary_cross_entropy_with_logits(preds, x, reduction="none")
    return loss.sum(dim=1).mean()

model_trainer = trainer.Trainer(
                model=model,
                loss_fn=loss_fn,
                optimizer=optimizer,
                train_loader=tr_loader,
                eval_loader=te_loader,
                log_dir=log_dir,
                n_gpus=0,
                device_id=0)

model_trainer.interleaved_train_and_eval(epochs)
