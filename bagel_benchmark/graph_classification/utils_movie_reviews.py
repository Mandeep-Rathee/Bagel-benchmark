########### train gnn for movie_reviews dataset

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from torch_geometric.loader import DataLoader

from torch_geometric.nn import global_mean_pool

sys.path.insert(0, '../')

from models import *

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

import torch_geometric

from dataset.create_movie_reviews import *

from torch_geometric.nn import GCNConv
from torch_geometric.nn import GATConv
from torch_geometric.nn import APPNP
from torch_geometric.nn import GINConv


import torch_geometric.utils as pyg_utils
from torch.utils.data.dataloader import default_collate
from torch_geometric.datasets import TUDataset
import os.path as osp
import sys
import tqdm
import json



dataset_type = 'linear'


assert dataset_type in ['linear', 'complex'], "dataset type needs to be 'linear' or 'complex'"





dataset_dim = [300,2]

model = GCN(dataset_dim)
model.to(device)

optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
criterion = torch.nn.CrossEntropyLoss()

#path_base = osp.join('..', '..')

path_base = '/home/rathee/bagel/Dataset'

if dataset_type == 'linear':
    dataset_class = AnnotatedMoviesLinear
else:
    dataset_class = AnnotatedMoviesComplex

def load_dataset():
    train_dataset = dataset_class(path_base, preload_to=device)
    test_dataset = dataset_class(path_base, dataset_type='test', preload_to=device)
    print(len(test_dataset), 'test_dataset')
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)
    print(test_dataset[1])

    return train_loader, test_loader

train_loader, test_loader = load_dataset()

def train():
    model.train()
    for data in train_loader:  # Iterate in batches over the training dataset.
         out = model(data.x, data.edge_index, data.batch)  # Perform a single forward pass.
         loss = criterion(out, data.y)  # Compute the loss.
         loss.backward()  # Derive gradients.
         optimizer.step()  # Update parameters based on gradients.
         optimizer.zero_grad()  # Clear gradients.

def test(loader):
     model.eval()
     correct = 0
     for data in loader:  # Iterate in batches over the training/test dataset.
         out = model(data.x, data.edge_index, data.batch)
         pred = out.argmax(dim=1)  # Use the class with highest probability.
         correct += int((pred == data.y).sum())  # Check against ground-truth labels.
     return correct / len(loader.dataset)  # Derive ratio of correct predictions.

def load_model(path, model):
     if not torch.cuda.is_available():
         model.load_state_dict(torch.load(path, map_location="cpu"))
     else:
         model.load_state_dict(torch.load(path))
     model.eval()

    
def train_gnn():
    for epoch in range(1, 201):
        train()
        train_acc = test(train_loader)
        test_acc = test(test_loader)
        print(f'Epoch: {epoch:03d}, Train Acc: {train_acc:.4f}, Test Acc: {test_acc:.4f}')

