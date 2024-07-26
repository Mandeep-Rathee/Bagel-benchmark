############# load datasets and GNN models 
import torch
import torch.nn.functional as F
from torch.nn import Sequential, Linear, ReLU
import torch_geometric
from torch_geometric.nn import GCNConv
from torch_geometric.nn import GATConv
from torch_geometric.nn import APPNP
from torch_geometric.nn import GINConv, global_add_pool
from torch_geometric.datasets import Planetoid, Amazon
from torch_geometric.utils import from_networkx
from pathlib import Path
import numpy as np


def load_dataset(data_set, working_directory=None):
    if working_directory is None:
        working_directory = Path(".").resolve()
    if data_set == "Cora":
        dataset = Planetoid(root=working_directory.joinpath('Datasets/tmp/Cora'), name='Cora')
        data = dataset[0]
        results_path = "cora"
    elif data_set == "CiteSeer":
        dataset = Planetoid(root=working_directory.joinpath('Datasets/tmp/CiteSeer'), name='CiteSeer')
        data = dataset[0]
        results_path = "citeseer"
    elif data_set == "PubMed":
        dataset = Planetoid(root=working_directory.joinpath('Datasets/tmp/PubMed'), name='PubMed')
        data = dataset[0]
        results_path = "pubmed"
    else:
        raise ValueError("Dataset " + data_set + "not implemented")

    return dataset, data, results_path


class GCNNet(torch.nn.Module):
    def __init__(self, dataset):
        super(GCNNet, self).__init__()
        self.conv1 = GCNConv(dataset.num_node_features, 16)
        self.conv2 = GCNConv(16, dataset.num_classes)

    def forward(self, x, edge_index, edge_mask):
        x = self.conv1(x, edge_index, edge_mask)
        x = F.relu(x)
        x = F.dropout(x, training=self.training)
        x = self.conv2(x, edge_index, edge_mask)

        return F.log_softmax(x, dim=1)




class GATNet(torch.nn.Module):
    # based on https://github.com/rusty1s/pytorch_geometric/blob/master/examples/gat.py
    def __init__(self, dataset):
        super(GATNet, self).__init__()
        self.conv1 = GATConv(dataset.num_features, 8, heads=8, dropout=0.6)
        # On the Pubmed dataset, use heads=8 in conv2.
        self.conv2 = GATConv(8 * 8, dataset.num_classes, heads=1, concat=False,
                             dropout=0.6)

    def forward(self, x, edge_index):
        x = F.dropout(x, p=0.6, training=self.training)
        x = F.elu(self.conv1(x, edge_index))
        x = F.dropout(x, p=0.6, training=self.training)
        x = self.conv2(x, edge_index)
        return F.log_softmax(x, dim=1)



class APPNP2Net(torch.nn.Module):
    def __init__(self, dataset):
        super(APPNP2Net, self).__init__()
        # default values from https://github.com/rusty1s/pytorch_geometric/blob/master/benchmark/citation/appnp.py
        self.dropout = 0.5
        self.hidden = 64
        self.K = 2  # adjusted to two layers
        self.alpha = 0.1
        self.lin1 = Linear(dataset.num_features, self.hidden)
        self.lin2 = Linear(self.hidden, dataset.num_classes)
        self.prop1 = APPNP(self.K, self.alpha)

    def reset_parameters(self):
        self.lin1.reset_parameters()
        self.lin2.reset_parameters()

    def forward(self, x, edge_index):
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = F.relu(self.lin1(x))
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.lin2(x)
        x = self.prop1(x, edge_index)
        return F.log_softmax(x, dim=1)


class GINConvNet(torch.nn.Module):
    def __init__(self, dataset):
        super(GINConvNet, self).__init__()

        num_features = dataset.num_features
        dim = 32

        nn1 = Sequential(Linear(num_features, dim), ReLU(), Linear(dim, dim))
        self.conv1 = GINConv(nn1)
        self.bn1 = torch.nn.BatchNorm1d(dim)

        nn2 = Sequential(Linear(dim, dim), ReLU(), Linear(dim, dim))
        self.conv2 = GINConv(nn2)
        self.bn2 = torch.nn.BatchNorm1d(dim)

        self.fc1 = Linear(dim, dim)
        self.fc2 = Linear(dim, dataset.num_classes)

    def forward(self, x, edge_index):
        x = F.relu(self.conv1(x, edge_index))
        x = self.bn1(x)
        x = F.relu(self.conv2(x, edge_index))
        x = self.bn2(x)
        # x = F.relu(self.conv3(x, edge_index))
        # x = self.bn3(x)
        # x = F.relu(self.conv4(x, edge_index))
        # x = self.bn4(x)
        # x = F.relu(self.conv5(x, edge_index))
        # x = self.bn5(x)
        # x = global_add_pool(x, batch)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, p=0.5, training=self.training)
        x = self.fc2(x)
        return F.log_softmax(x, dim=-1)


def load_model(path, model):
    if not torch.cuda.is_available():
        model.load_state_dict(torch.load(path, map_location="cpu"))
    else:
        model.load_state_dict(torch.load(path))
    model.eval()


def train_model(model, data, epochs=200, lr=0.01, weight_decay=5e-4, clip=None, loss_function="nll_loss",
                epoch_save_path=None, no_output=False):
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

    accuracies = []

    model.train()
    for epoch in range(epochs):
        optimizer.zero_grad()
        out = model(data.x, data.edge_index, None)
        if loss_function == "nll_loss":
            loss = F.nll_loss(out[data.train_mask], data.y[data.train_mask])
        elif loss_function == "cross_entropy":
            loss = F.cross_entropy(out[data.train_mask], data.y[data.train_mask], size_average=True)
        else:
            raise Exception()
        if clip is not None:
            torch.nn.utils.clip_grad_norm(model.parameters(), clip)
        loss.backward()
        optimizer.step()

        if epoch_save_path is not None:
            # circumvent .pt ending
            save_model(model, epoch_save_path[:-3] + "_epoch_" + str(epoch) + epoch_save_path[-3:])
            accuracies.append(retrieve_accuracy(model, data, value=True))
            print('Accuracy: {:.4f}'.format(accuracies[-1]), "Epoch", epoch)
        else:
            if epoch % 25 == 0 and not no_output:
                print(retrieve_accuracy(model, data))

    model.eval()

    return accuracies


def save_model(model, path):
    torch.save(model.state_dict(), path)


def retrieve_accuracy(model, data, test_mask=None, value=False):
    _, pred = model(data.x, data.edge_index, None).max(dim=1)
    if test_mask is None:
        test_mask = data.test_mask
    correct = float(pred[test_mask].eq(data.y[test_mask]).sum().item())
    acc = correct / test_mask.sum().item()
    if value:
        return acc
    else:
        return 'Accuracy: {:.4f}'.format(acc)

