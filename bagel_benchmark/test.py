from bagel_benchmark import metrics
from bagel_benchmark.node_classification import utils
from bagel_benchmark.explainers.grad_based_explainers import grad_weights

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



explanation = grad_weights(model,data)
