from bagel_benchmark import metrics
from bagel_benchmark.node_classification import utils
from bagel_benchmark.explainers.grad_explainer_node import grad_node_explanation

import torch
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
device ='cpu'


from bagel_benchmark.graph_classification.utils_movie_reviews import load_dataset
from bagel_benchmark.graph_classification.models import GCN


dataset_dim = [300,2] ### features size is 300 and there are 2 labels. 
model = GCN(dataset_dim)

train_loader, test_loader = load_dataset()
print(train_loader)
print(test_loader)



utils_movie_reviews.train_gnn(model, train_loader, test_loader)


