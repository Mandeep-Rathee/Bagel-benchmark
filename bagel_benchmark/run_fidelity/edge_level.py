
import sys
import numpy as np
import torch


from torch_geometric.utils import k_hop_subgraph
from torch_geometric.nn import MessagePassing, APPNP



# sys.path.append('/home/rathee/Bagel-benchmark/bagel_benchmark')
sys.path.append('/home/rathee/Bagel-benchmark')



from bagel_benchmark.node_classification.utils import GCNNet, GATNet, APPNP2Net, GINConvNet
from bagel_benchmark.node_classification.utils import load_dataset, train_model
from bagel_benchmark.explainers.grad_explainer_edge import grad_edge_explanation
from bagel_benchmark.explainers.gnnexplainer import GNNExplainer
from bagel_benchmark.metrics import fidelity

data_set = "PubMed"
explainer = "random"

device = torch.device("cuda" if torch.cuda.is_available else "cpu")

dataset, data, results_path = load_dataset(data_set)
model = GCNNet(dataset)
model = model.to(device)
data = data.to(device)
acc = train_model(model, data, epochs=200, lr=0.01, weight_decay=5e-4, clip=None, loss_function="nll_loss",
                epoch_save_path=None, no_output=False)


filename3 = "/home/rathee/Bagel-benchmark/bagel_benchmark/dataset/Cora_selected_nodes.npy"
selected_nodes = np.load(filename3)

gnn_explainer = GNNExplainer(model, log=False)


def subgraph(model, node_idx, x, edge_index, **kwargs):
    num_nodes, num_edges = x.size(0), edge_index.size(1)

    flow = 'source_to_target'
    for module in model.modules():
        if isinstance(module, MessagePassing):
            flow = module.flow
            break

    num_hops = 0
    for module in model.modules():
        if isinstance(module, MessagePassing):
            if isinstance(module, APPNP):
                num_hops += module.K
            else:
                num_hops += 1

    subset, edge_index, mapping, edge_mask = k_hop_subgraph(
        node_idx, num_hops, edge_index, relabel_nodes=True,
        num_nodes=num_nodes, flow=flow)

    x = x[subset]
    for key, item in kwargs:
        if torch.is_tensor(item) and item.size(0) == num_nodes:
            item = item[subset]
        elif torch.is_tensor(item) and item.size(0) == num_edges:
            item = item[edge_mask]

        kwargs[key] = item

    return x, edge_index, mapping, edge_mask, kwargs


node_level_fid =[]

for node in selected_nodes:
    node  = torch.tensor(node)
    node = node.item()

    computation_graph_feature_matrix, computation_graph_edge_index, mapping, hard_edge_mask, kwargs = \
        subgraph(model, node, data.x, data.edge_index)

    ## for gnn explainer
    if explainer=="gnnexplainer":
        feature_mask,edge_mask = gnn_explainer.explain_node(node_idx=mapping,x=computation_graph_feature_matrix,edge_index=computation_graph_edge_index)

    ## for grad edge
    if explainer =="grad":
        feature_mask, edge_mask = grad_edge_explanation(model, mapping, computation_graph_feature_matrix, computation_graph_edge_index)
        edge_mask  = torch.from_numpy(edge_mask).to(device)

    ## for random
    if explainer =="random":
        edge_mask = torch.rand(computation_graph_edge_index.shape[1]).to(device)

    bin_list = [0.01, 0.05, 0.1, 0.2, 0.5]

    bin_level_fid = []
    for topk in bin_list:
        fid = fidelity(model=model, node_idx=node,full_feature_matrix=data.x, edge_index=data.edge_index, edge_mask= edge_mask, topk=topk, device=device)
        bin_level_fid.append(fid)

    node_level_fid.append(np.mean(bin_level_fid))


print(f"Over all fidelity for {explainer} and {data_set} with GCN is {np.mean(node_level_fid)}")







