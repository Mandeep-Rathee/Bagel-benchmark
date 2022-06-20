### metrics used in bagel becnhmark

import torch
from torch_geometric.nn import MessagePassing
from torch_geometric.utils import k_hop_subgraph
from torch_geometric.nn import APPNP
from scipy.stats import entropy

from dataset.create_movie_reviews import to_eraser_dict

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


def fidelity(model,  # is a must
             node_idx,  # is a must
             full_feature_matrix,  # must
             edge_index=None,  # the whole, so data.edge_index
             node_mask=None,  # at least one of these three node, feature, edge
             feature_mask=None,
             edge_mask=None,
             samples=100,
             random_seed=12345,
             device="cpu"
             ):
    """
    Distortion/Fidelity (for Node Classification)
    :param model: GNN model which is explained
    :param node_idx: The node which is explained
    :param full_feature_matrix: The feature matrix from the Graph (X)
    :param edge_index: All edges
    :param node_mask: Is a (binary) tensor with 1/0 for each node in the computational graph
    => 1 means the features of this node will be fixed
    => 0 means the features of this node will be pertubed/randomized
    if not available torch.ones((1, num_computation_graph_nodes))
    :param feature_mask: Is a (binary) tensor with 1/0 for each feature
    => 1 means this features is fixed for all nodes with 1
    => 0 means this feature is randomized for all nodes
    if not available torch.ones((1, number_of_features))
    :param edge_mask:
    :param samples:
    :param random_seed:
    :param device:
    :param validity:
    :return:
    """
    if edge_mask is None and feature_mask is None and node_mask is None:
        raise ValueError("At least supply one mask")

    computation_graph_feature_matrix, computation_graph_edge_index, mapping, hard_edge_mask, kwargs = \
        subgraph(model, node_idx, full_feature_matrix, edge_index)

    # get predicted label
    log_logits = model(x=computation_graph_feature_matrix,
                       edge_index=computation_graph_edge_index)
    predicted_labels = log_logits.argmax(dim=-1)

    predicted_label = predicted_labels[mapping]

    # fill missing masks
    if feature_mask is None:
        (num_nodes, num_features) = full_feature_matrix.size()
        feature_mask= torch.ones((1, num_features), device=device)

    num_computation_graph_nodes = computation_graph_feature_matrix.size(0)
    if node_mask is None:
        # all nodes selected
        node_mask = torch.ones((1, num_computation_graph_nodes), device=device)


    # set edge mask
    if edge_mask is not None:
        for module in model.modules():
            if isinstance(module, MessagePassing):
                module.__explain__ = False
                module.__edge_mask__ = edge_mask

    (num_nodes, num_features) = full_feature_matrix.size()

    num_nodes_computation_graph = computation_graph_feature_matrix.size(0)

    # retrieve complete mask as matrix
    mask = node_mask.T.matmul(feature_mask)

    correct = 0.0

    rng = torch.Generator(device=device)
    rng.manual_seed(random_seed)
    random_indices = torch.randint(num_nodes, (samples, num_nodes_computation_graph, num_features),
                                   generator=rng,
                                   device=device,
                                   )
    random_indices = random_indices.type(torch.int64)

    for i in range(samples):
        random_features = torch.gather(full_feature_matrix,
                                       dim=0,
                                       index=random_indices[i, :, :])

        randomized_features = mask * computation_graph_feature_matrix + (1 - mask) * random_features

        log_logits = model(x=randomized_features, edge_index=computation_graph_edge_index)
        distorted_labels = log_logits.argmax(dim=-1)

        if distorted_labels[mapping] == predicted_label:
            correct += 1

    # reset mask
    if edge_mask is not None:
        for module in model.modules():
            if isinstance(module, MessagePassing):
                module.__explain__ = False
                module.__edge_mask__ = None

    return correct / samples
  
#### entropy based sparsity,for further deatils read the documentation https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.entropy.html

def sparsity(feature_sparsity, node_sparsity, feature_mask=None, node_mask=None):
    if feature_sparsity:
        return entropy(feature_mask)
    elif:
        return entropy(node_mask)


def f1(pre, rec):
    if pre == 0 or rec == 0:
        return 0
    return 2 * pre * rec / (pre + rec)

######## defining the correctness of the explanation ################

def correctness(node_mask, ground_truth, mask_type,topk):
    gt_positive = 0
    true_positive = 0
    pred_positive = 0
    
    if mask_type=="soft":
        values,explanation_nodes = node_mask.topk(topk) ## make sure node_mask is a tensor, we recomnd to use topk=20
    else: 
        explanation_nodes = node_mask.nonzero()     ## if masks are hard then simply retrieve the indices where the mask=1
        
    gt_positive = gt_positive + len(ground_truth)
    pred_positive = pred_positive + len(explanation_nodes)  
    
    for ex_node in explanation_nodes:
        if ex_node in ground_truth:
            true_positive = true_positive + 1
    recall = true_positive / gt_positive
    precision = true_positive / pred_positive 
    f1_score = f1(precision, recall)
    
    return precision, recall, f1_score


def aopc_overall(aopc, thresholds, key):
    instance_scores =[]
    for t in thresholds:
        klass = aopc[t]['classification']
        beta_0_list = aopc[t]['classification_scores']
        beta_0 = beta_0_list[0][klass]
        beta_k_list = aopc[t][key]
        beta_k = beta_k_list[0][klass]
        delta = beta_0 - beta_k
        instance_scores.append(delta.item())
                
    return np.mean(instance_scores)


############# the sufficiency and comprehensiveness and only for soft masks explanations 

def suff_and_comp(idx, model, explanation, test_dataset):
    
    ''' We follow the definition of suff and comp from eraser benchmark https://arxiv.org/abs/1911.03429
        
        idx: this is the id of the graph from test dataset
        model: trianed GNN model
        explanation: the soft mask explanations generated by exlainers
        test_datset: this is the test dataset for which we are computing sufficieny and comprehensiveness 
    '''    
    
    
    thresholds = [0.01, 0.05, 0.1, 0.2, 0.5]
    
    aopc_dict_all = {}
    for t in thresholds:
        dict_obj, aopc_dict_ = to_eraser_dict(test_dataset, idx, weights, model=model, k=t)
        aopc_dict_all[t] = aopc_dict_
        comp = aopc_overall(aopc_dict_all, thresholds, key='comprehensiveness_classification_scores')
        suff = aopc_overall(aopc_dict_all, thresholds, key='sufficiency_classification_scores')   
        
    
    return suff, comp











    
    
    
    
        
        

        
        
    
    
  
