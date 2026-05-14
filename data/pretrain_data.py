"""
Pretraining data pipeline: VirtualNodeAugmentor (Task-Tree construction) and
multi-domain graph merging.

This is the core of GIT's task-tree approach — different graph tasks
(node / edge / graph) are unified into a common format via virtual task nodes.
"""
import torch
from torch_geometric.data import Batch
from torch_geometric.utils import to_undirected

from data.finetune_data import (
    citation_datasets, ecommerce_datasets, kg_datasets,
    molecule_datasets, temporal_datasets, datasets, get_data,
)

# ── Dataset registries ──────────────────────────────────────────────

pretrain_datasets = {
    'default': ['arxiv', 'products', 'WN18RR', 'FB15K237', 'chemblpre', 'chempcba'],
    'citation': citation_datasets,
    'ecommerce': ecommerce_datasets,
    'kg': kg_datasets,
    'molecule': molecule_datasets,
    'cora': ['cora'],
    'citeseer': ['citeseer'],
    'pubmed': ['pubmed'],
    'dblp': ['dblp'],
    'arxiv23': ['arxiv23'],
    'arxiv': ['arxiv'],
    'bookhis': ['bookhis'],
    'bookchild': ['bookchild'],
    'elecomp': ['elecomp'],
    'elephoto': ['elephoto'],
    'sportsfit': ['sportsfit'],
    'amazonratings': ['amazonratings'],
    'products': ['products'],
    'chemblpre': ['chemblpre'],
    'chempcba': ['chempcba'],
    'chemhiv': ['chemhiv'],
    'bbbp': ['bbbp'],
    'bace': ['bace'],
    'toxcast': ['toxcast'],
    'cyp450': ['cyp450'],
    'tox21': ['tox21'],
    'muv': ['muv'],
    'WN18RR': ['WN18RR'],
    'FB15K237': ['FB15K237'],
    'codex_s': ['codex_s'],
    'codex_m': ['codex_m'],
    'codex_l': ['codex_l'],
    'NELL995': ['NELL995'],
    'GDELT': ['GDELT'],
    'ICEWS1819': ['ICEWS1819'],
    'Enron': ['Enron'],
    'Googlemap_CT': ['Googlemap_CT'],
    'scaling_law_1': ['arxiv', 'chempcba', 'FB15K237'],
    'scaling_law_2': ['arxiv', 'chempcba', 'FB15K237', 'products', 'WN18RR'],
    'scaling_law_3': ['arxiv', 'chempcba', 'FB15K237', 'products', 'WN18RR',
                      'chemblpre', 'arxiv23', 'amazonratings', 'NELL995', 'Enron'],
    'scaling_law_4': ['arxiv', 'cora', 'citeseer', 'pubmed', 'arxiv23', 'dblp',
                      'bookhis', 'bookchild', 'elecomp', 'elephoto', 'sportsfit',
                      'amazonratings', 'products', 'chemblpre', 'chempcba', 'chemhiv',
                      'bbbp', 'bace', 'toxcast', 'cyp450', 'tox21', 'muv',
                      'WN18RR', 'FB15K237', 'codex_s', 'codex_m', 'codex_l',
                      'NELL995', 'GDELT', 'ICEWS1819', 'Enron', 'Googlemap_CT'],
}

domain2task = {
    'citation': 'node',
    'ecommerce': 'node',
    'kg': 'edge',
    'temporal': 'edge',
    'molecule': 'graph',
}

dataset2domain = (
    {d: 'citation' for d in citation_datasets}
    | {d: 'ecommerce' for d in ecommerce_datasets}
    | {d: 'kg' for d in kg_datasets}
    | {d: 'molecule' for d in molecule_datasets}
    | {d: 'temporal' for d in temporal_datasets}
)


# ── VirtualNodeAugmentor ────────────────────────────────────────────

class VirtualNodeAugmentor:
    """
    Builds the GIT Task-Tree by injecting virtual task nodes into a graph.

    Each original node/edge/graph gets a dedicated virtual node that serves
    as the target for task-specific representation learning during pretraining.
    """

    def augment(self, data, task):
        assert data.x.ndim == 1, "Node features must be 1D index tensor"
        if task == 'node':
            return self.add_virtual_nodes_node_classification(data)
        elif task == 'edge':
            return self.add_virtual_nodes_edge_classification(data)
        elif task == 'graph':
            return self.add_virtual_nodes_graph_classification(data)
        else:
            raise ValueError(f"Unknown task: {task}")

    def add_virtual_nodes_node_classification(self, data):
        num_nodes = data.num_nodes
        node_dim = data.node_text_feat.size(1)

        data.x = torch.cat([data.x, torch.ones(num_nodes, dtype=torch.long) * num_nodes])
        data.node_text_feat = torch.cat([data.node_text_feat, torch.zeros(1, node_dim)])
        task_node_idx = torch.arange(num_nodes, num_nodes * 2, dtype=torch.long)

        new_edge = torch.tensor(
            [[i, num_nodes + i] for i in range(num_nodes)], dtype=torch.long
        ).t()
        new_edge = to_undirected(new_edge)
        data.edge_index = torch.cat([data.edge_index, new_edge], dim=1)

        return data, task_node_idx

    def add_virtual_nodes_edge_classification(self, data):
        num_edges = data.edge_index.size(1)
        num_nodes = data.num_nodes
        node_dim = data.node_text_feat.size(1)

        data.x = torch.cat([data.x, torch.ones(num_edges, dtype=torch.long) * num_nodes])
        data.node_text_feat = torch.cat([data.node_text_feat, torch.zeros(1, node_dim)])
        task_node_idx = torch.arange(num_nodes, num_nodes + num_edges, dtype=torch.long)

        new_edge = []
        for i in range(num_edges):
            src, dst = data.edge_index[:, i]
            new_edge.append([src, num_nodes + i])
            new_edge.append([num_nodes + i, dst])
        new_edge = torch.tensor(new_edge, dtype=torch.long).t()
        new_edge = to_undirected(new_edge)

        data.edge_index = torch.cat([data.edge_index, new_edge], dim=1)

        return data, task_node_idx

    def add_virtual_nodes_graph_classification(self, data):
        num_nodes = data.x.shape[0]
        num_node_texts = data.node_text_feat.shape[0]
        node_dim = data.node_text_feat.shape[1]

        groups = data.groups
        num_groups = groups.max().item() + 1

        data.x = torch.cat([data.x, torch.ones(num_groups, dtype=torch.long) * num_node_texts])
        data.node_text_feat = torch.cat([data.node_text_feat, torch.zeros(1, node_dim)])
        task_node_idx = torch.arange(num_nodes, num_nodes + num_groups, dtype=torch.long)

        i_indices = torch.arange(num_nodes, dtype=torch.long)
        new_edge = torch.stack([i_indices, num_nodes + groups], dim=1).t()
        new_edge = to_undirected(new_edge)

        data.edge_index = torch.cat([data.edge_index, new_edge], dim=1)

        return data, task_node_idx


# ── Data preparation helpers ────────────────────────────────────────

def preprocess(data):
    """Convert dataset-specific formats to the unified index-feature representation."""
    dataset_name = data.name
    if dataset_name in citation_datasets + ecommerce_datasets + kg_datasets + temporal_datasets:
        data.x = torch.arange(data.num_nodes, dtype=torch.long)
    elif dataset_name in molecule_datasets:
        data = data.data
        data.edge_index = data.pre_edge_index
        data.node_text_feat = data.node_embs
    return data


def postprocess(data):
    """Keep only essential keys before merging into a unified batch."""
    keys = ['x', 'edge_index', 'node_text_feat']
    for k, v in data.to_dict().items():
        if k not in keys:
            data[k] = None
    return data


def preprocess_data_dict(data_dict, task_node_idx_dict):
    """
    Re-index nodes and task nodes across datasets so indices are globally unique.
    """
    x_start = 0
    cnt = 0
    for dataset_name, data in data_dict.items():
        task_node_idx = task_node_idx_dict[dataset_name]

        num_nodes = data.x.shape[0]
        num_unique_nodes = data.node_text_feat.shape[0]

        print(f"Preprocessing {dataset_name} with {num_nodes} nodes "
              f"and {num_unique_nodes} unique node-text features")

        data.x = data.x + x_start
        x_start += num_unique_nodes

        task_node_idx = task_node_idx + cnt
        cnt += num_nodes

        data_dict[dataset_name] = data
        task_node_idx_dict[dataset_name] = task_node_idx

    return data_dict, task_node_idx_dict


# ── Main entry point ────────────────────────────────────────────────

def unified_data(params):
    """
    Build the unified multi-domain pretraining graph.

    For each dataset in the pretrain set, loads the raw graph, preprocesses it,
    augments with task-specific virtual nodes, and merges everything into a
    single large graph batch.

    Returns
    -------
    unified_dataset : torch_geometric.data.Batch
    task_node_idx_dict : dict[str, torch.Tensor]
        Maps dataset name → task-node index tensor (global indices).
    """
    data_path = params['data_path']
    pre_datasets = pretrain_datasets[params['pretrain_dataset']]

    vn = VirtualNodeAugmentor()

    data_dict = {}
    task_node_idx_dict = {}
    for dataset in pre_datasets:
        data = get_data({
            'data_path': data_path,
            'dataset': dataset,
            'task': domain2task[dataset2domain[dataset]],
        })
        data = preprocess(data)
        data, task_node_idx = vn.augment(data, task=domain2task[dataset2domain[dataset]])
        data = postprocess(data)
        data_dict[dataset] = data
        task_node_idx_dict[dataset] = task_node_idx

    data_dict, task_node_idx_dict = preprocess_data_dict(data_dict, task_node_idx_dict)
    unified_dataset = Batch.from_data_list(list(data_dict.values()))

    return unified_dataset, task_node_idx_dict
