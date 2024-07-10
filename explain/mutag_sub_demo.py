import math
from argparse import ArgumentParser, Namespace
from collections import Counter
from dataclasses import dataclass
from functools import partial
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple, Union
from typing_extensions import Literal

import ipywidgets as widgets
import matplotlib.pyplot as plt
import networkx as nx
import networkx.algorithms.isomorphism as iso
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from easydict import EasyDict as edict
from sklearn.metrics import accuracy_score, roc_auc_score
from torch import FloatTensor, LongTensor, Tensor
from torch_geometric.data import Batch, Data, InMemoryDataset
from torch_geometric.datasets import TUDataset
from torch_geometric.loader import DataLoader
from torch_geometric.nn import GCNConv, GINConv, GNNExplainer, global_max_pool
from torch_geometric.nn.conv import MessagePassing
from torch_geometric.nn.models import GIN
from torch_geometric.utils import k_hop_subgraph, remove_self_loops, to_networkx
from tqdm.notebook import trange, tqdm
import torch_geometric
#from singleomic_GAT import *
from subgraph_demo import *
from scipy.sparse import coo_matrix  #
from gnn import *
# Env
from utils import *
# Load MUTAG dataset (graph classification)
mutag_dataset = edict()
mutag_dataset.ds = TUDataset(
    './dataset/mutag/',
    name='MUTAG',
    use_node_attr=True,
    use_edge_attr=True
)
mutag_dataset.ds.shuffle()
mutag_size = len(mutag_dataset.ds)
mutag_dataset.train_ds = mutag_dataset.ds[:int(0.8 * mutag_size)]
mutag_dataset.valid_ds = mutag_dataset.ds[int(0.8 * mutag_size) : int(0.9 * mutag_size)]
mutag_dataset.test_ds = mutag_dataset.ds[int(0.9 * mutag_size):]
c = mutag_dataset.ds[:5]
net = GIN(
    in_channels=mutag_dataset.ds.num_features,
    hidden_channels=24,
    num_layers=3,
    out_channels=1,
    dropout=0.2
)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
net.to(device)
#
def visualize_subgraph_mutag(graph: nx.Graph,
                             node_set: Optional[Set[int]] = None,
                             edge_set: Optional[Set[int]] = None,
                             title: Optional[str] = None) -> None:
    if node_set is None:
        node_set = set(graph.nodes())

    if edge_set is None:
        edge_set = {(n_from, n_to) for (n_from, n_to) in graph.edges() if n_from in node_set and n_to in node_set}

    node_dict = {0: 'C', 1: 'N', 2: 'O', 3: 'F', 4: 'I', 5: 'Cl', 6: 'Br'}
    node_idxs = {node: node_x.index(1.0) for node, node_x in graph.nodes(data='x')}
    node_labels = {k: node_dict[v] for k, v in node_idxs.items()}
    node_color = ['#E49D1C', '#4970C6', '#FF5357', '#29A329', 'brown', 'darkslategray', '#F0EA00']
    colors = [node_color[v % len(node_color)] for k, v in node_idxs.items()]

    pos = nx.kamada_kawai_layout(graph)

    nx.draw_networkx_nodes(G=graph, pos=pos, nodelist=list(graph.nodes()), node_color=colors, node_size=300)
    nx.draw_networkx_edges(G=graph, pos=pos, width=3, edge_color='gray', arrows=False)
    nx.draw_networkx_edges(G=graph, pos=pos, edgelist=list(edge_set), width=6, edge_color='black', arrows=False)
    nx.draw_networkx_labels(G=graph, pos=pos, labels=node_labels)

    if title is not None:
        plt.title(title)

    plt.axis('off')
    plt.show()
    plt.close()

# Run and visualize
num_nodes = 4
model_explain = Explain(model=net, min_nodes=num_nodes)

for i, mutag_data in enumerate(mutag_dataset.ds[:1]):
    graph = to_networkx(mutag_data, node_attrs=['x'], edge_attrs=['edge_attr'], to_undirected=True)
    subgraph = model_explain.explain(x=mutag_data.x.to(device), edge_index=mutag_data.edge_index.to(device), max_nodes=num_nodes)
    visualize_subgraph_mutag(
        graph=graph,
        node_set=set(subgraph.coalition),
        title=f'Identified subgraph in bold'
    )

