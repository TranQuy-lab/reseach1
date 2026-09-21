"""Edge-aware GraphSAGE matching the upstream message/mean/update structure."""

from __future__ import annotations

import torch
from torch import nn
from torch_geometric.nn import MessagePassing


class EdgeSAGELayer(MessagePassing):
    def __init__(self, node_dim: int, edge_dim: int, out_dim: int):
        super().__init__(aggr="mean", flow="source_to_target")
        self.w_msg = nn.Linear(node_dim + edge_dim, out_dim)
        self.w_apply = nn.Linear(node_dim + out_dim, out_dim)

    def forward(self, x, edge_index, edge_attr):
        aggregated = self.propagate(edge_index, x=x, edge_attr=edge_attr)
        return torch.relu(self.w_apply(torch.cat([x, aggregated], dim=1)))

    def message(self, x_j, edge_attr):
        return self.w_msg(torch.cat([x_j, edge_attr], dim=1))


class EGraphSAGE(nn.Module):
    def __init__(self, edge_dim: int, n_classes: int, hidden: int = 128,
                 dropout: float = 0.2, direct_edge: bool = False):
        super().__init__()
        self.edge_dim = edge_dim
        self.direct_edge = direct_edge
        self.layer1 = EdgeSAGELayer(edge_dim, edge_dim, hidden)
        self.layer2 = EdgeSAGELayer(hidden, edge_dim, hidden)
        self.dropout = nn.Dropout(dropout)
        self.head = nn.Linear(2 * hidden + (edge_dim if direct_edge else 0), n_classes)

    def forward(self, edge_index, edge_attr, label_edge_index, label_edge_attr, num_nodes):
        x = torch.ones((num_nodes, self.edge_dim), dtype=edge_attr.dtype, device=edge_attr.device)
        h = self.layer1(x, edge_index, edge_attr)
        h = self.layer2(self.dropout(h), edge_index, edge_attr)
        parts = [h[label_edge_index[0]], h[label_edge_index[1]]]
        if self.direct_edge:
            parts.append(label_edge_attr)
        return self.head(torch.cat(parts, dim=1))


class EdgeMLP(nn.Module):
    def __init__(self, edge_dim: int, n_classes: int, hidden: int = 128, dropout: float = 0.2):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(edge_dim, hidden), nn.ReLU(), nn.Dropout(dropout), nn.Linear(hidden, n_classes)
        )

    def forward(self, edge_attr):
        return self.net(edge_attr)


def build_model(name: str, edge_dim: int, n_classes: int, hidden: int = 128, dropout: float = 0.2):
    if name == "edge_mlp":
        return EdgeMLP(edge_dim, n_classes, hidden, dropout)
    if name in {"sage", "sage_edge"}:
        return EGraphSAGE(edge_dim, n_classes, hidden, dropout, direct_edge=name == "sage_edge")
    raise ValueError(f"Unknown model: {name}")
