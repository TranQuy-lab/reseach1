"""FP32 native PyTorch port of the upstream message/mean/update equations.

Reference: waimorris/E-GraphSAGE e05eb74, NF-ToN-IoT multiclass notebook.
This is a protocol-corrected v2 adaptation, NOT an exact paper reproduction.
"""
from dataclasses import dataclass

import numpy as np
import pandas as pd
import torch
from torch import nn


@dataclass
class FlowGraph:
    edge_features: torch.Tensor
    src: torch.Tensor
    dst: torch.Tensor
    message_src: torch.Tensor
    message_dst: torch.Tensor
    message_flow: torch.Tensor
    num_nodes: int

    @property
    def node_features(self):
        return torch.ones((self.num_nodes, self.edge_features.shape[1]),
                          dtype=torch.float32, device=self.edge_features.device)


def make_graph(frame, features, bidirectional=True):
    if len(frame) == 0:
        raise ValueError('Empty graph')
    if features.shape[0] != len(frame) or not np.isfinite(features).all():
        raise ValueError('Invalid feature array')
    if frame[['src_node', 'dst_node']].isna().any().any():
        raise ValueError('Missing endpoint')
    codes, nodes = pd.factorize(pd.concat([frame.src_node, frame.dst_node], ignore_index=True), sort=True)
    src = torch.from_numpy(codes[:len(frame)].copy()).long()
    dst = torch.from_numpy(codes[len(frame):].copy()).long()
    flow = torch.arange(len(frame))
    if bidirectional:
        reverse = src != dst
        ms, md = torch.cat([src, dst[reverse]]), torch.cat([dst, src[reverse]])
        mf = torch.cat([flow, flow[reverse]])
    else:
        ms, md, mf = src, dst, flow
    return FlowGraph(torch.as_tensor(features, dtype=torch.float32), src, dst, ms, md, mf, len(nodes))


class SAGELayer(nn.Module):
    def __init__(self, node_dim, edge_dim, out_dim):
        super().__init__()
        self.w_msg = nn.Linear(node_dim + edge_dim, out_dim)
        self.w_apply = nn.Linear(node_dim + out_dim, out_dim)

    def forward(self, x, g):
        msg = self.w_msg(torch.cat([x[g.message_src], g.edge_features[g.message_flow]], dim=1))
        # FP32 accumulation, including degrees, even if a caller uses autocast.
        aggregated = torch.zeros((g.num_nodes, msg.shape[1]), device=x.device, dtype=torch.float32)
        aggregated.index_add_(0, g.message_dst, msg.float())
        degree = torch.bincount(g.message_dst, minlength=g.num_nodes).float().clamp_min(1).unsqueeze(1)
        return torch.relu(self.w_apply(torch.cat([x.float(), aggregated / degree], dim=1)))


class EGraphSAGE(nn.Module):
    def __init__(self, n_features, n_classes, hidden=128, dropout=.2, direct_edge=False):
        super().__init__()
        self.direct_edge = direct_edge
        self.layer1 = SAGELayer(n_features, n_features, hidden)
        self.layer2 = SAGELayer(hidden, n_features, hidden)
        self.dropout = nn.Dropout(dropout)
        self.head = nn.Linear(2*hidden + (n_features if direct_edge else 0), n_classes)

    def forward(self, g):
        h = self.layer1(g.node_features, g)
        h = self.layer2(self.dropout(h), g)
        parts = [h[g.src], h[g.dst]]
        if self.direct_edge:
            parts.append(g.edge_features)
        return self.head(torch.cat(parts, dim=1))


class EdgeMLP(nn.Module):
    def __init__(self, n_features, n_classes, hidden=128, dropout=.2):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(n_features, hidden), nn.ReLU(),
                                 nn.Dropout(dropout), nn.Linear(hidden, n_classes))

    def forward(self, g):
        return self.net(g.edge_features)


def build_model(name, n_features, n_classes, hidden=128, dropout=.2):
    if name == 'edge_mlp':
        return EdgeMLP(n_features, n_classes, hidden, dropout)
    if name in ('sage', 'sage_edge'):
        return EGraphSAGE(n_features, n_classes, hidden, dropout, direct_edge=(name == 'sage_edge'))
    raise ValueError(f'Unknown neural model: {name}')
