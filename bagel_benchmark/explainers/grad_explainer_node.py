import numpy as np
import torch
from torch_geometric.nn import MessagePassing
import os
import torch_geometric.utils as ut
from pathlib import Path
from torch_geometric.utils import to_dense_adj,k_hop_subgraph


def save_model(model, path):
    torch.save(model.state_dict(), path)
from torch_geometric.data import Data
from torch_geometric.datasets import Planetoid


def edge_mask_to_node_mask(data, edge_mask, aggregation="mean"):
    node_weights = torch.zeros(data.x.shape[0])
    if aggregation == "sum":
        for weight, nodes in zip(edge_mask, data.edge_index.T):
            node_weights[nodes[0].item()] += weight.item() / 2
            node_weights[nodes[1].item()] += weight.item() / 2
    elif aggregation == "mean":
        node_degrees = torch.zeros(data.x.shape[0])
        for weight, nodes in zip(edge_mask, data.edge_index.T):
            node_weights[nodes[0].item()] += weight.item()
            node_weights[nodes[1].item()] += weight.item()
            node_degrees[nodes[0].item()] += 1
            node_degrees[nodes[1].item()] += 1
        node_weights = node_weights / node_degrees.clamp(min=1.)
    elif aggregation == "max":
        for weight, nodes in zip(edge_mask, data.edge_index.T):
            node_weights[nodes[0].item()] = max(weight.item(), node_weights[nodes[0].item()])
            node_weights[nodes[1].item()] = max(weight.item(), node_weights[nodes[1].item()])
    else:
        raise NotImplementedError(f"No such aggregation method: {aggregation}")
    return node_weights




def execute_model_with_gradient(model, node, x, edge_index):
    ypred = model(x, edge_index)

    predicted_labels = ypred.argmax(dim=-1)
    predicted_label = predicted_labels[node]
    logit = torch.nn.functional.softmax((ypred[node, :]).squeeze(), dim=0)

    logit = logit[predicted_label]
    loss = -torch.log(logit)
    loss.backward()


def grad_edge_explanation(model, node, x, edge_index):
    model.zero_grad()

    E = edge_index.size(1)
    edge_mask = torch.nn.Parameter(torch.ones(E))

    for module in model.modules():
        if isinstance(module, MessagePassing):
            module.__explain__ = True
            module.__edge_mask__ = edge_mask

    edge_mask.requires_grad = True
    x.requires_grad = True

    if edge_mask.grad is not None:
        edge_mask.grad.zero_()
    if x.grad is not None:
        x.grad.zero_()

    execute_model_with_gradient(model, node, x, edge_index)

    adj_grad = edge_mask.grad
    adj_grad = torch.abs(adj_grad)
    masked_adj = adj_grad + adj_grad.t()
    masked_adj = torch.sigmoid(masked_adj)
    masked_adj = masked_adj.cpu().detach().numpy()

    feature_mask = torch.abs(x.grad).cpu().detach().numpy()

    return np.max(feature_mask, axis=0), masked_adj


def grad_node_explanation(model, node, x, edge_index):
    model.zero_grad()

    num_nodes, num_features = x.size()

    node_grad = torch.nn.Parameter(torch.ones(num_nodes))
    feature_grad = torch.nn.Parameter(torch.ones(num_features))

    node_grad.requires_grad = True
    feature_grad.requires_grad = True

    mask = node_grad.unsqueeze(0).T.matmul(feature_grad.unsqueeze(0))

    execute_model_with_gradient(model, node, mask*x, edge_index)

    node_mask = torch.abs(node_grad.grad).cpu().detach().numpy()
    feature_mask = torch.abs(feature_grad.grad).cpu().detach().numpy()

    return feature_mask, node_mask


def gradinput_node_explanation(model, node, x, edge_index):
    model.zero_grad()

    x.requires_grad = True
    if x.grad is not None:
        x.grad.zero_()

    execute_model_with_gradient(model, node, x, edge_index)

    feature_mask = torch.abs(x.grad * x).cpu().detach().numpy()

    return np.mean(feature_mask, axis=0), np.mean(feature_mask, axis=1)

def integrated_gradients_weight(model, data, n=100, p=2, attribution_mask=None, show_progress=False):
    if isinstance(attribution_mask, int):
        attribution_mask = attr_mask(data, attribution_mask)
    ig_loader = DataLoader(IntegrationPointDataset(data, n=n), batch_size=1, shuffle=True)
    for batch in DataLoader(PointDataset(data), batch_size=1):
        base_prediction = model(batch.x,batch.edge_index)
    if attribution_mask is not None:
        base_prediction = base_prediction[attribution_mask]
    prediction_idx = base_prediction.argmax().item()
    # approximate the integral by a riemann sum
    ig_sum = 0
    loader = tqdm(ig_loader) if show_progress else ig_loader
    for batch in loader:
        batch.x.requires_grad = True
        prediction = model(batch.x,batch.edge_index)
        if attribution_mask is not None:
            prediction = prediction[attribution_mask]
        loss = prediction_val(prediction, idx=prediction_idx)
        loss.backward()

        ig_sum += batch.x.grad
    weights = 1/n * data.x * ig_sum
    weights = weights.norm(p=p, dim=1)
    return normalize(weights)


def smooth_grad_weight(model, data, p=2, n=100, sigma=0.15, attribution_mask=None, show_progress=False):
    if isinstance(attribution_mask, int):
        attribution_mask = attr_mask(data, attribution_mask)
    smoothing_loader = DataLoader(SmoothingDataset(data, n=n, sigma=sigma), batch_size=1, shuffle=False)
    smooth_grad = 0
    loader = tqdm(smoothing_loader) if show_progress else smoothing_loader
    for batch in loader:
        batch.x.requires_grad = True
        prediction = model(batch.x,batch.edge_index)
        if attribution_mask is not None:
            prediction = prediction[attribution_mask]
        loss = prediction_val(prediction)
        loss.backward()

        grads = batch.x.grad
        vals = batch.x.detach()
        smooth_grad += grads * vals
    grads = smooth_grad.norm(p=p, dim=1)
    return normalize(grads)


