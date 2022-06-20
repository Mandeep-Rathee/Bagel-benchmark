from bagel_benchmark import metrics
from bagel_benchmark.node_classification import utils
from bagel_benchmark.explainers.grad_explainer_node import grad_node_explanation

import torch
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
data_set="Cora"
dataset, data, results_path = utils.load_dataset(data_set)
data.to(device)


model = utils.GCNNet(dataset)
model.to(device)

#### train the GNN model 
accuracy = utils.train_model(model,data)
print(accuracy)


node = 10
feature_mask, node_mask = grad_node_explanation(model,node,data.x, data.edge_index)
print(feature_mask)
print(node_mask)

feature_sparsity = False
Node_sparsity = True


sparsity = metrics.sparsity(feature_sparsity, Node_sparsity, feature_mask, node_mask)

print(sparsity)
feature_mask = torch.from_numpy(feature_mask).reshape(1,-1)

fidelity = metrics.fidelity(model, node, data.x,data.edge_index, feature_mask=feature_mask)

print(fidelity)
