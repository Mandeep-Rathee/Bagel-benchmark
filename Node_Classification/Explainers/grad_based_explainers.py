######### gnn explaining methods
import torch
import torch.nn as nn

from torch.autograd import Variable

import torch_geometric.nn as pyg_nn
from torch_geometric.utils.convert import to_networkx
from torch_geometric.utils import subgraph

import networkx as nx
from torch_geometric.data import DataLoader, Dataset, Data

import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.cm as cm

from copy import deepcopy
from tqdm import tqdm

from math import log10
import re


EPS = 1e-30


def plot_graph(data, weights=None, mask=None):
    if mask is not None and "influence" in mask:
        lower = re.search(r'\d+%', mask)
        lower = 0. if lower is None else int(lower.group()[:-1])/100
        assert weights is not None, "attribution weights are needed to mask by influence"
        mask = influence_mask(weights, min_attr=lower)
    if mask is not None:
        masked_x = data.x[mask]
        masked_y = data.y[mask]
        if weights is not None:
            weights = weights[mask]
        mask_edges, _ = subgraph(mask, data.edge_index, relabel_nodes=True)
        data = Data(x=masked_x, y=masked_y, edge_index=mask_edges)

    graph = to_networkx(data)
    color_map = None

    plt.figure(1)

    if weights is not None:
        max_weight = weights.max().item()
        normalize = mcolors.Normalize(vmin=min([0, weights.min().item()]), vmax=max_weight)
        colormap = cm.jet

        color_map = [colormap(normalize(weight)) for weight in weights]

        scalar_mappable = cm.ScalarMappable(norm=normalize, cmap=colormap)
        plt.colorbar(scalar_mappable)

    nx.draw(graph, cmap=plt.get_cmap('Set1'), node_size=75, node_color=color_map, linewidths=6,
            pos=nx.drawing.kamada_kawai_layout(graph))

    plt.show()


def influence_mask(weights, min_attr=0.):
    return weights.gt(min_attr * torch.ones_like(weights))


def attr_mask(data, node_index):
    attr_mask = torch.zeros_like(data.train_mask, dtype=bool)
    attr_mask[node_index] = True
    return attr_mask


class PointDataset(Dataset):
    def __init__(self, graph):
        super().__init__()
        if hasattr(graph, "batch"):
            assert graph.batch.max().item() == 0
            self.data = Data(x=graph.x.float(), edge_index=graph.edge_index, y=graph.y)
        else:
            self.data = graph

    @staticmethod
    def len():
        return 1

    def get(self, idx):
        return self.data


def prediction_val(out, idx=None):
    if idx is None:
        return out.max()
    else:
        if len(out.shape) == 2:
            out = out.view(-1)
        return out[idx]


def normalize(tensor):
    if tensor.abs().max().item() > 0.:
        return tensor/tensor.sum(dim=0).item()
    return tensor


def autograd_data(model, data, attribution_mask=None, is_cam=False):
    if isinstance(attribution_mask, int):
        attribution_mask = attr_mask(data, attribution_mask)
    point_loader = DataLoader(PointDataset(data), batch_size=1)
    for batch in point_loader:
        batch.x.type(torch.FloatTensor)
        batch.x.requires_grad = True
        if is_cam:
            prediction = model.forward(batch, is_cam)
        else:
            prediction = model.forward(batch)
        if attribution_mask is not None:
            prediction = prediction[attribution_mask]
        loss = prediction_val(prediction)
        loss.backward()
    return batch


def get_factor(x):
    base10 = log10(abs(x+EPS))
    return float(10**min([-int(base10), 0]))


def grad_weights(model, data, p=2, attribution_mask=None):
    batch = autograd_data(model, data, attribution_mask)
    # get into computational range
    fac = get_factor(batch.x.grad.abs().max().item())
    grads = fac * batch.x.grad
    grads = grads.norm(p=p, dim=1)
    return normalize(grads)


def grads_times_input_weights(model, data, p=2, attribution_mask=None):
    batch = autograd_data(model, data, attribution_mask)
    grads = batch.x.grad.detach()
    # get into computational range
    fac = get_factor(grads.abs().min().item())
    grads = fac * grads
    vals = batch.x.detach()
    weights = grads * vals
    weights = weights.norm(p=p, dim=1)
    return normalize(weights)


class SmoothingDataset(Dataset):
    def __init__(self, graph, sigma=0.15, n=100):
        super().__init__()
        if hasattr(graph, "batch"):
            assert graph.batch.max().item() == 0
            self.data = Data(x=graph.x.float(), edge_index=graph.edge_index, y=graph.y)
        else:
            self.data = graph
        self.sigma = sigma * torch.ones_like(graph.x)
        self.n = n

    def len(self):
        return self.n

    def get(self, idx):
        sample = torch.normal(mean=0.0, std=self.sigma)
        data = deepcopy(self.data)
        data.x = data.x + sample
        return data


def smooth_grad_weights(model, data, p=2, n=100, sigma=0.15, attribution_mask=None, show_progress=False):
    if isinstance(attribution_mask, int):
        attribution_mask = attr_mask(data, attribution_mask)
    smoothing_loader = DataLoader(SmoothingDataset(data, n=n, sigma=sigma), batch_size=1, shuffle=False)
    smooth_grad = 0
    loader = tqdm(smoothing_loader) if show_progress else smoothing_loader
    for batch in loader:
        batch.x.requires_grad = True
        prediction = model(batch)
        if attribution_mask is not None:
            prediction = prediction[attribution_mask]
        loss = prediction_val(prediction)
        loss.backward()

        grads = batch.x.grad
        vals = batch.x.detach()
        smooth_grad += grads * vals
    grads = smooth_grad.norm(p=p, dim=1)
    return normalize(grads)


class IntegrationPointDataset(Dataset):
    def __init__(self, graph, n=100):
        super().__init__()
        if hasattr(graph, "batch"):
            assert graph.batch.max().item() == 0
            self.data = Data(x=graph.x.float(), edge_index=graph.edge_index, y=graph.y)
        else:
            self.data = graph
        self.n = n

    def len(self):
        return self.n

    def get(self, idx):
        data = deepcopy(self.data)
        data.x = idx / self.n * data.x
        return data


def integrated_gradients_weights(model, data, n=100, p=2, attribution_mask=None, show_progress=False):
    if isinstance(attribution_mask, int):
        attribution_mask = attr_mask(data, attribution_mask)
    ig_loader = DataLoader(IntegrationPointDataset(data, n=n), batch_size=1, shuffle=True)
    for batch in DataLoader(PointDataset(data), batch_size=1):
        base_prediction = model(batch)
    if attribution_mask is not None:
        base_prediction = base_prediction[attribution_mask]
    prediction_idx = base_prediction.argmax().item()
    # approximate the integral by a riemann sum
    ig_sum = 0
    loader = tqdm(ig_loader) if show_progress else ig_loader
    for batch in loader:
        batch.x.requires_grad = True
        prediction = model(batch)
        if attribution_mask is not None:
            prediction = prediction[attribution_mask]
        loss = prediction_val(prediction, idx=prediction_idx)
        loss.backward()

        ig_sum += batch.x.grad
    weights = 1/n * data.x * ig_sum
    weights = weights.norm(p=p, dim=1)
    return normalize(weights)


def cam_weights(model, data, attribution_mask=None):
    assert attribution_mask is None, "CAM only works for graph classification"

    # run graph through model to set pre_pool_val and pool_val
    point_loader = DataLoader(PointDataset(data), batch_size=1)
    for batch in point_loader:
        model.forward(batch, is_cam=True)

    pre = model.pre_pool_val
    post = model.pool_val
    weights = post @ pre.T
    weights = weights.detach().flatten().abs()
    return normalize(weights)


def grad_cam_weights(model, data, attribution_mask=None):
    assert attribution_mask is None, "GradCAM only works for graph classification"
    autograd_data(model, data, is_cam=True)
    weights = model.pool_val @ model.pre_pool_val.grad.T
    weights = weights.detach().flatten().abs()
    return normalize(weights)
