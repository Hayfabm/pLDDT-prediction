import pandas as pd
import numpy as np
import torch
import torch.nn.functional as F
import torch.nn as nn
from fastprogress.fastprogress import progress_bar, master_bar

from ast import literal_eval

DATA_FILE = 'data/all_data_pLDDT.csv'
BATCH_SIZE = 32
NUM_EPOCHS = 10
BIOTF_BATCH_SIZE = 32
MLP_LAYER_SIZE = 256
LEARNING_RATE = 1e-3
USE_CUDA = True
MODEL_SAVE_PATH = '/models/pLDDT_MLP.pt'

from biotransformers import BioTransformers

# Protbert transformer
bio_trans = BioTransformers(
    backend="protbert", 
    num_gpus=1 if USE_CUDA else 0
)

train_df = pd.read_csv('data/all_data_pLDDT.csv', converters={'labels': literal_eval})
# train_df.head(3)


# Protein Dataset
class Protien(torch.utils.data.Dataset):

    def __init__(self, df, is_test=False, is_valid=False):
        super().__init__()
        self.df = df
        self.type = type
        self.is_test = is_test
        self.is_valid = is_valid
    
    def __len__(self): return len(self.df)
    
    def __getitem__(self, idx):
        row = self.df.iloc[idx]

        inp = row.sequences
        
        if self.is_test: 
            targ = torch.zeros(len(inp), dtype=torch.float32)
        else: 
            targ = torch.tensor(row.labels, dtype=torch.float32)
        
        # Return: (sequence, sequence length, target)
        return inp, len(inp), targ

# Train-Validation split
valid_pct = 0.2
np.random.seed(5)
idx = np.random.permutation(np.arange(len(train_df)))
train_idx = idx[:int(len(train_df)*(1-valid_pct))]
valid_idx = idx[int(len(train_df)*(1-valid_pct)):]
_train_df = train_df.iloc[train_idx]
_valid_df = train_df.iloc[valid_idx]
print(f"Train ds is of len: {len(_train_df)} and validation ds is of len: {len(_valid_df)}")

train_ds = Protien(_train_df)
valid_ds = Protien(_valid_df, is_valid=True)

# Data Loader with custom collate function

from torch.utils.data import DataLoader
from torch.nn.utils.rnn import pad_sequence

# Collator calls protbert

def pad_collate(batch):
    xx = [s[0] for s in batch]
    ll = [s[1] for s in batch] # Length of each sequence in batch
    yy = [s[2] for s in batch]

    ll = torch.tensor(ll, dtype=torch.float)  
    
    xx = bio_trans.compute_embeddings(xx, pool_mode=('full'), batch_size=BIOTF_BATCH_SIZE, silent=True)['full']
    xx = [torch.tensor(x) for x in xx]
    xx_pad = pad_sequence(xx, batch_first=True, padding_value=0)
    yy_pad = pad_sequence(yy, batch_first=True, padding_value=0)
    
    return xx_pad, yy_pad, ll

train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, collate_fn=pad_collate)
valid_loader = DataLoader(valid_ds, batch_size=BATCH_SIZE, collate_fn=pad_collate)

# Simple MLP model
class MLP(nn.Module):

    def __init__(self, input_size, hidden_size):
        super().__init__()
            
        self.input_size = input_size
        self.hidden_size = hidden_size
        
        self.linear = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(0.1),
        )
        
        self.fc = nn.Linear(hidden_size, 1)

    def forward(self, x, l):
        # Embed sequence
        x = self.linear(x)
    
        # Apply head
        out = self.fc(x)
        
        return out.squeeze(-1)

# 1024 is protbert embedding size
model = MLP(1024, MLP_LAYER_SIZE).cuda()
print(model)

# Custom masked loss
def MSELossMasked(pred, target, lengths):
    loss = ((pred-target)**2)
    
    loss_ = torch.tensor(0.).cuda()
    for j in range(x.shape[0]):
        loss_ += loss[j,:lengths[j].long()].mean()/x.shape[0]
    
    return loss_

import torch.optim as optim

optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
criterion = MSELossMasked

mb = master_bar(range(NUM_EPOCHS))

print('\nTraining...')

for epoch in mb:
    
    # Train loop
    model.train()
    running_loss = 0.0
    for i, data in progress_bar(enumerate(train_loader), total=len(_train_df)//16+1, parent=mb):
        # get the inputs; data is a list of [inputs, labels]
        x, y_true, l = data
        y_true = y_true.cuda()
        
        # zero the parameter gradients
        optimizer.zero_grad()

        # forward
        y_pred = model(x.cuda(), l)
        
        # backward + optimize
        loss = criterion(y_pred, y_true, l)
        
        loss.backward()
        optimizer.step()
        running_loss += 0.1 * (loss - running_loss)
        mb.child.comment = f'loss: {running_loss.item()}'
    
    # Eval loop
    model.eval()
    val_loss = []
    for i, data in progress_bar(enumerate(valid_loader), total=len(_valid_df)//16+1, parent=mb):
        # get the inputs; data is a list of [inputs, labels]
        x, y_true, l = data
        y_true = y_true.cuda()

        # forward
        with torch.no_grad():
            y_pred = model(x.cuda(), l)
        
        # mask zero for padded values
        val_loss.append(criterion(y_pred, y_true, l).item())
        
    print(f'Epoch {epoch:2d}: train={running_loss:4.1f}, valid={np.mean(val_loss):4.1f}')

torch.save(model.state_dict(), MODEL_SAVE_PATH)