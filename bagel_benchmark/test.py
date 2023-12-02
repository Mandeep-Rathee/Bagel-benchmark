import sys

sys.path.append('/home/rathee/Bagel-benchmark')


from bagel_benchmark.metrics import suff_and_comp
from bagel_benchmark.node_classification import utils
from bagel_benchmark.explainers.grad_explainer_node import grad_node_explanation
from bagel_benchmark.explainers.grad_explainer_graph import grad_weights


import torch
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


from bagel_benchmark.graph_classification.utils_movie_reviews import load_dataset, train_gnn
from bagel_benchmark.graph_classification.models import GCN


dataset_dim = [300,2] ### features size is 300 and there are 2 labels. 
model = GCN(dataset_dim)
model.to(device)

train_loader, test_loader, test_dataset = load_dataset()


# for data in train_loader:
#     print(data.x.device)
#     exit()


train_gnn(model, train_loader, test_loader)


data = test_dataset[10]

print(data)
data.batch = torch.zeros(data.x.shape[0], device=device).long()

print(data)

explanation = grad_weights(model, data)


suff, comp = suff_and_comp(10, model,explanation,test_dataset)

print("suf", suff, "comp: ", comp)
exit()



print(explanation)


