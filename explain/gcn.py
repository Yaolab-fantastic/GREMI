import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, GINConv, GNNExplainer, global_max_pool
from torch_geometric.nn.conv import MessagePassing
from torch_geometric.nn.models import GIN
from torch import FloatTensor, LongTensor, Tensor
class GIN_SubgraphX(torch.nn.Module):

    def __init__(self,
                in_channels: int,
                hidden_channels: int,
                num_layers: int,
                out_channels: int,
                dropout: float = 0.5):
        super(GIN_SubgraphX, self).__init__()

        self.dropout = dropout

        convs_list = [
            GINConv(nn.Sequential(
                nn.Linear(in_channels, hidden_channels),
                nn.ReLU(),
                nn.Linear(hidden_channels, hidden_channels),
                nn.ReLU())
            )
        ]
        convs_list += [
            GINConv(nn.Sequential(
                nn.Linear(hidden_channels, hidden_channels),
                nn.ReLU(),
                nn.Linear(hidden_channels, hidden_channels),
                nn.ReLU())
            )
            for _ in range(num_layers - 1)
        ]

        self.convs = nn.ModuleList(convs_list)

        self.classifier = nn.Sequential(
            nn.Linear(hidden_channels, hidden_channels),
            nn.ReLU(),
            nn.Dropout(self.dropout),
            nn.Linear(hidden_channels, out_channels)
        )

    def forward(self,
                x: LongTensor,
                edge_index: LongTensor,
                batch: LongTensor) -> FloatTensor:
        for i in range(len(self.convs)):
            x = self.convs[i](x, edge_index)

        x = global_max_pool(x, batch)

        x = self.classifier(x)

        return x
