from __future__ import annotations

import torch
import torch.nn as nn

from torch_geometric.data import Data
from torch_geometric.nn import MetaLayer
from torch_geometric.utils import scatter


class MLP(nn.Module):
    def __init__(self, in_dim: int, hidden: int, out_dim: int, p_drop: float = 0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.ReLU(),
            nn.Dropout(p_drop),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Dropout(p_drop),
            nn.Linear(hidden, out_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class EdgeModel(nn.Module):
    def __init__(self, node_dim: int, edge_dim: int, hidden: int):
        super().__init__()
        self.mlp = MLP(node_dim * 2 + edge_dim, hidden, edge_dim)

    def forward(self, src, dst, edge_attr, u=None, batch=None):
        h = torch.cat([src, dst, edge_attr], dim=1)
        return self.mlp(h)


class NodeModel(nn.Module):
    def __init__(self, node_dim: int, edge_dim: int, hidden: int):
        super().__init__()
        self.mlp = MLP(node_dim + edge_dim, hidden, node_dim)

    def forward(self, x, edge_index, edge_attr, u=None, batch=None):
        _, col = edge_index  # messages to node col
        # agg = scatter_add(edge_attr, col, dim=0, dim_size=x.size(0))
        agg = scatter(edge_attr, col, dim=0, dim_size=x.size(0), reduce="sum")
        h = torch.cat([x, agg], dim=1)
        return self.mlp(h)


class FareGNN(nn.Module):
    def __init__(self, node_dim: int, edge_dim: int, hidden: int = 64, n_layers: int = 2):
        super().__init__()
        self.layers = nn.ModuleList([
            MetaLayer(EdgeModel(node_dim, edge_dim, hidden), NodeModel(node_dim, edge_dim, hidden), None)
            for _ in range(n_layers)
        ])
        self.readout = MLP(edge_dim, hidden, 1)

    def forward(self, g: Data) -> torch.Tensor:
        x, edge_index, edge_attr = g.x, g.edge_index, g.edge_attr
        for layer in self.layers:
            x, edge_attr, _ = layer(x, edge_index, edge_attr, None, None)
        return self.readout(edge_attr)  # (E, 1)