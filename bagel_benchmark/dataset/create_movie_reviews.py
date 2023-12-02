############### create graph of text, we use movie reviews dataset with human annotations form eraser bechmark https://arxiv.org/abs/1911.03429

import torch
from torchtext.vocab import GloVe
from torch_geometric.data import Data, Dataset
import json
import os.path as osp


_removed_tokens = ['', ',', '.', '\'s', '"', '-', '?', '!', '/', '(', ')', '_', 'the', 'be', 'to', 'of', 'and', 'a',
                       'an', 'in', 'that', 'it', 'you', 'me', 'i', 'is', 'at']


class AnnotatedMoviesLinear(Dataset):
    def __init__(self, base_path, dataset_name='movie_reviews', dataset_type='train', preload_to=None):
        super().__init__()
        assert dataset_type in ['val', 'test', 'train'], "val, test or train dataset possible"
        self.movies_list = []
        self.data_path = osp.join(base_path, dataset_name, 'docs')
        self.remove = _removed_tokens
        with open(osp.join(base_path, dataset_name, dataset_type + '.jsonl'), "r") as f:
            for line in f:
                movie_entry = json.loads(line)
                self.movies_list.append(movie_entry)
        self.glove = GloVe()
        self.data_preloaded = False;
        if preload_to is not None:
            print(f"preloading dataset to {preload_to}")
            self.data_list = [self.get(idx).to(preload_to) for idx in range(self.len())]
            self.data_preloaded = True

    def len(self):
        return len(self.movies_list)

    def get(self, idx):
        if self.data_preloaded:
            return self.data_list[idx]
        movie_obj = self.movies_list[idx]
        movie_path = osp.join(self.data_path, movie_obj["annotation_id"])
        with open(movie_path, 'r') as f:
            review = f.read().replace('\n', ' ')
        label = 0 if movie_obj["classification"] == "NEG" else 1
        label = torch.LongTensor([label])
        review = [token for token in review.split(' ') if token not in self.remove]
        embedded = torch.zeros(len(review), 300)
        for i, txt in enumerate(review):
            embedded[i] = self.glove[txt]
        edges = torch.zeros(4 * len(review) - 6, 2, dtype=torch.long)
        for i in range(len(review) - 1):
            edges[2 * i] = torch.Tensor([i, i + 1])
            edges[2 * i + 1] = torch.Tensor([i + 1, i])
        offset = 2 * len(review) - 2
        for i in range(len(review) - 2):
            edges[offset + 2 * i] = torch.Tensor([i, i + 2])
            edges[offset + 2 * i + 1] = torch.Tensor([i + 2, i])
        data_obj = Data(x=embedded, y=label, edge_index=edges.T)
        data_obj.dataset_idx = torch.LongTensor([idx])
        return data_obj

    def get_text(self, idx):
        movie_obj = self.movies_list[idx]
        movie_path = osp.join(self.data_path, movie_obj["annotation_id"])
        with open(movie_path, 'r') as f:
            review = f.read().replace('\n', ' ')
        indices = [idx for idx, token in enumerate(review.split(' ')) if token not in self.remove]
        return review, indices

    def get_ground_truth(self, idx):
        movie_obj = self.movies_list[idx]
        _, indices = self.get_text(idx)
        mask = torch.zeros(len(indices))
        i = 0
        for evidence in movie_obj["evidences"][0]:
            start_token = evidence["start_token"]
            end_token = evidence["end_token"]
            while indices[i] < start_token:
                i += 1
            while i < len(indices) and indices[i] <= end_token:
                mask[i] = 1
                i += 1
        return mask


class AnnotatedMoviesComplex(Dataset):
    def __init__(self, base_path, dataset_name='movie_reviews', dataset_type='train', preload_to=None, max_connection_distance=2):
        super().__init__()
        assert dataset_type in ['val', 'test', 'train'], "val, test or train dataset possible"
        self.movies_list = []
        self.max_dist = max_connection_distance
        self.data_path = osp.join(base_path, dataset_name, 'docs')
        self.remove = _removed_tokens
        with open(osp.join(base_path, dataset_name, dataset_type + '.jsonl'), "r") as f:
            for line in f:
                movie_entry = json.loads(line)
                self.movies_list.append(movie_entry)
        self.text_indices = [[-1]] * len(self)
        self.glove = GloVe()
        self.data_preloaded = False
        if preload_to is not None:
            print(f"preloading dataset to {preload_to}")
            self.data_list = [self.get(idx).to(preload_to) for idx in range(self.len())]
            self.data_preloaded = True

    def len(self):
        return len(self.movies_list)

    def get(self, idx):
        if self.data_preloaded:
            return self.data_list[idx]
        movie_obj = self.movies_list[idx]
        movie_path = osp.join(self.data_path, movie_obj["annotation_id"])
        with open(movie_path, 'r') as f:
            full_review = f.read().replace('\n', ' ')
        label = 0 if movie_obj["classification"] == "NEG" else 1
        label = torch.LongTensor([label])
        review = [token for token in full_review.split(' ') if token not in self.remove]
        tokens = list(set(review))
        edges = []

        for d in range(1, self.max_dist+1):
            for txt_1, txt_2 in zip(review[:-d], review[d:]):
                idx_1 = tokens.index(txt_1)
                idx_2 = tokens.index(txt_2)
                if (idx_1, idx_2) not in edges:
                    edges.append((idx_1, idx_2))
                    edges.append((idx_2, idx_1))

        embedded = torch.zeros(len(tokens), 300)
        for i, txt in enumerate(tokens):
            embedded[i] = self.glove[txt]

        self.text_indices[idx] = [tokens.index(token) if token not in self.remove else -1 for token in full_review.split(' ')]

        edges = torch.LongTensor(edges)
        data_obj = Data(x=embedded, y=label, edge_index=edges.T)
        data_obj.dataset_idx = torch.LongTensor([idx])
        return data_obj

    def get_text(self, idx):
        movie_obj = self.movies_list[idx]
        movie_path = osp.join(self.data_path, movie_obj["annotation_id"])
        with open(movie_path, 'r') as f:
            review = f.read().replace('\n', ' ')
        return review, self.text_indices[idx]

    def get_ground_truth(self, idx):
        movie_obj = self.movies_list[idx]
        _, indices = self.get_text(idx)
        mask = torch.zeros(len(indices))
        i = 0
        for evidence in movie_obj["evidences"][0]:
            start_token = evidence["start_token"]
            end_token = evidence["end_token"]
            while indices[i] < start_token:
                i += 1
            while i < len(indices) and indices[i] <= end_token:
                mask[i] = 1
                i += 1
        return mask


def expand_weights(weights, indices, text_len):
    expanded_weights = [0.] * text_len
    if len(weights) == len(indices):
        for idx, weight in zip(indices, weights):
            expanded_weights[idx] = weight
    else:
        assert len(indices) == text_len, f"{len(indices)} vs. {text_len}"
        for t_idx, w_idx in enumerate(indices):
            if w_idx == -1:
                continue
            expanded_weights[t_idx] = weights[w_idx]
    return expanded_weights


def _tensor_to_str_dict(t):
    t = t.view(-1)
    assert len(t) == 2, f"tensor needs to be of length 2, but has shape {t.shape} and length {len(t)}"
    return {"NEG": float(t[0]), "POS": float(t[1])}

device = 'cuda'


def to_eraser_dict(dataset, idx, weights, model=None, odd=False, device=device, k=None):
    txt, indices = dataset.get_text(idx)
    txt_len = len(txt.split(' '))
    annotation_id = dataset.movies_list[idx]["annotation_id"]
    docid = annotation_id

    if not len(weights) == txt_len:
        expanded_weights = expand_weights(weights, indices, txt_len)
    else:
        expanded_weights = weights

    rational_weights = [float(w) for w in expanded_weights]
    rationale_obj = {"annotation_id": annotation_id, "rationales": [{"docid": docid, "soft_rationale_predictions": rational_weights}]}
    aopc_dic = {}
    if model is not None:
        # do comprehensiveness & sufficiency
        data = dataset[idx].to(device)
        data.batch = torch.zeros(data.x.shape[0], device=device).long()
        assert len(weights) == data.x.shape[0]
        # get top k_d nodes
        top_k_node_mask = torch.zeros(data.x.shape[0], device=device)
        k_d = int((data.x.shape[0]*k))
        if k_d <=1:
            k_d=3
        top_k_edge_index = []
        non_top_k_edge_index = []
        for _ in range(k_d):
            max_weight = 0.
            max_idx = -1
            for i, w in enumerate(weights):
                if (max_idx == -1 or w > max_weight) and top_k_node_mask[i] == 0:
                    max_weight = w
                    max_idx = i
            top_k_node_mask[max_idx] = 1

        # construct subgraphs with only top k_d nodes and without top k_d nodes

        top_k_node_map = list(range(data.x.shape[0]))
        non_top_k_node_map = list(range(data.x.shape[0]))
        for i, b in enumerate(top_k_node_mask):
            if b == 1:
                non_top_k_node_map.remove(i)
            else:
                top_k_node_map.remove(i)
        top_k_node_map = {j: i for i, j in enumerate(top_k_node_map)}
        non_top_k_node_map = {j: i for i, j in enumerate(non_top_k_node_map)}
        for i, edge in enumerate(data.edge_index.T):
            if top_k_node_mask[edge[0].item()] == top_k_node_mask[edge[1].item()]:
                if top_k_node_mask[edge[0].item()] == 0:
                    non_top_k_edge_index.append([non_top_k_node_map[edge[0].item()], non_top_k_node_map[edge[1].item()]])
                else:
                    top_k_edge_index.append([top_k_node_map[edge[0].item()], top_k_node_map[edge[1].item()]])
        if len(top_k_edge_index) == 0:
            top_k_edge_index = [[0, 0]]
        top_k_data = Data(x=data.x[top_k_node_mask.bool()], edge_index=torch.tensor(top_k_edge_index, device=device).long().T)
        non_top_k_data = Data(x=data.x[~top_k_node_mask.bool()], edge_index=torch.tensor(non_top_k_edge_index, device=device).long().T)
        top_k_data.batch = torch.zeros(top_k_data.x.shape[0], device=device).long()
        non_top_k_data.batch = torch.zeros(non_top_k_data.x.shape[0], device=device).long()

        # get model predictions of all 3 graphs
        data.to(device)
        top_k_data.to(device)
        top_k_data.to(device)
        with torch.no_grad():
            pred = model(data.x, data.edge_index, data.batch)
            top_k_pred = model(top_k_data.x, top_k_data.edge_index, top_k_data.batch)
            non_top_k_pred = model(non_top_k_data.x, non_top_k_data.edge_index, non_top_k_data.batch)
        rationale_obj["classification"] = "NEG" if pred.argmax() == 0 else "POS"
        rationale_obj["classification_scores"] = _tensor_to_str_dict(pred)
        rationale_obj["comprehensiveness_classification_scores"] = _tensor_to_str_dict(top_k_pred)
        rationale_obj["sufficiency_classification_scores"] = _tensor_to_str_dict(non_top_k_pred)
        aopc_dic["classification"] = 0 if pred.argmax() == 0 else 1
        aopc_dic["classification_scores"] = pred
        aopc_dic["comprehensiveness_classification_scores"] = top_k_pred
        aopc_dic["sufficiency_classification_scores"] = non_top_k_pred
        aopc_dic["threshold"] = k


    return rationale_obj, aopc_dic
