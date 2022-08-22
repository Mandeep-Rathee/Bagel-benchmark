######## graph classification models

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from torch_geometric.nn import global_mean_pool


import torch_geometric

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

sys.path.insert(0, '../')

def load_dataset(data_set):
    if data_set == 'PROTEINS':
        dataset = TUDataset(root='data/TUDataset', name=data_set)
    elif data_set=="MUTAG":
        dataset = TUDataset(root='data/TUDataset', name=data_set)
    elif data_set=='ENZYMES':
        dataset = TUDataset(root='data/ENYYMES',name=data_set)
    return dataset



hidden_channels=64

class GCN(torch.nn.Module):
    def __init__(self, dataset):
        super(GCN, self).__init__()
        torch.manual_seed(12345)
        self.conv1 = GCNConv(dataset[0], hidden_channels)
        self.conv2 = GCNConv(hidden_channels, hidden_channels)
        self.lin = nn.Linear(hidden_channels, dataset[1])

    def forward(self, x, edge_index, batch):
        # 1. Obtain node embeddings
        x = self.conv1(x, edge_index)
        x = x.relu()
        x = self.conv2(x, edge_index)
        # 2. Readout layer
        x = global_mean_pool(x, batch)  # [batch_size, hidden_channels]
        # 3. Apply a final classifier
        x = F.dropout(x, p=0.5, training=self.training)
        x = self.lin(x)
        return x

class GATNet(torch.nn.Module):
    # based on https://github.com/rusty1s/pytorch_geometric/blob/master/examples/gat.py
    def __init__(self, dataset):
        super(GATNet, self).__init__()
        self.conv1 = GATConv(dataset[0], 8, heads=8, dropout=0.6)
        # On the Pubmed dataset, use heads=8 in conv2.
        self.conv2 = GATConv(8 * 8, hidden_channels, heads=1, concat=False,
                             dropout=0.6)
        self.lin = nn.Linear(hidden_channels, dataset[1])

    def forward(self, x, edge_index, batch):
        x = F.dropout(x, p=0.6, training=self.training)
        x = F.elu(self.conv1(x, edge_index))
        x = F.dropout(x, p=0.6, training=self.training)
        x = self.conv2(x, edge_index)
        x = global_mean_pool(x, batch)  # [batch_size, hidden_channels]
        x = F.dropout(x, p=0.5, training=self.training)
        x = self.lin(x)

        return x



class GINConvNet(torch.nn.Module):
    def __init__(self, dataset):
        super(GINConvNet, self).__init__()

        num_features = dataset[0]
        dim = 32
        nn1 = nn.Sequential(nn.Linear(num_features, dim), nn.ReLU(), nn.Linear(dim, dim))
        self.conv1 = GINConv(nn1)
        self.bn1 = torch.nn.BatchNorm1d(dim)

        nn2 = nn.Sequential(nn.Linear(dim, dim), nn.ReLU(), nn.Linear(dim, dim))
        self.conv2 = GINConv(nn2)
        self.bn2 = torch.nn.BatchNorm1d(dim)

        self.fc1 = nn.Linear(dim, dim)
        self.fc2 = nn.Linear(dim, dim)
        self.lin = nn.Linear(dim, dataset[1])


    def forward(self, x, edge_index,batch):
        x = F.relu(self.conv1(x, edge_index))
        x = self.bn1(x)
        x = F.relu(self.conv2(x, edge_index))
        x = self.bn2(x)

        x = F.relu(self.fc1(x))
        x = F.dropout(x, p=0.5, training=self.training)
        x = self.fc2(x)

        x = global_mean_pool(x, batch)  # [batch_size, hidden_channels]
        x = F.dropout(x, p=0.5, training=self.training)
        x = self.lin(x)

        return x


class APPNP2Net(torch.nn.Module):
    def __init__(self, dataset):
        super(APPNP2Net, self).__init__()
        # default values from https://github.com/rusty1s/pytorch_geometric/blob/master/benchmark/citation/appnp.py
        self.dropout = 0.5
        self.hidden = 64
        self.K = 2  # adjusted to two layers
        self.alpha = 0.1
        self.lin1 = nn.Linear(dataset[0], self.hidden)
        self.lin2 = nn.Linear(self.hidden, self.hidden)
        self.prop1 = APPNP(self.K, self.alpha)
        self.lin = nn.Linear(self.hidden, dataset[1])


    def reset_parameters(self):
        self.lin1.reset_parameters()
        self.lin2.reset_parameters()

    def forward(self, x, edge_index,batch):
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = F.relu(self.lin1(x))
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.lin2(x)
        x = self.prop1(x, edge_index)

        x = global_mean_pool(x, batch)  # [batch_size, hidden_channels]
        x = F.dropout(x, p=0.5, training=self.training)
        x = self.lin(x)

        return x


