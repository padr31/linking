from __future__ import annotations
from typing import Tuple
from torch_geometric.nn import GCNConv, GATConv, GatedGraphConv
import torch


class GCNEncoder(torch.nn.Module):
    def __init__(self, in_channels: int, out_channels: int):
        super(GCNEncoder, self).__init__()
        '''self.conv1 = GATConv(
            in_channels, 2 * out_channels, add_self_loops=False
        )
        self.conv2 = GATConv(
            2 * out_channels, out_channels, add_self_loops=False
        )'''
        self.conv1 = GatedGraphConv(
            out_channels, 2
        )
        self.conv2 = GatedGraphConv(
            out_channels, 2
        )

    def forward(self, x, edge_index) -> torch.Tensor:
        x = self.conv1(x, edge_index).relu()
        for i in range(7):
            x = self.conv2(x, edge_index).relu()
        return self.conv2(x, edge_index)


class VariationalGCNEncoder(torch.nn.Module):
    def __init__(self, in_channels: int, out_channels: int):
        super(VariationalGCNEncoder, self).__init__()
        self.conv1 = GATConv(
            in_channels, 2 * out_channels, cached=True, add_self_loops=False
        )
        self.conv_mu = GATConv(
            2 * out_channels, out_channels, cached=True, add_self_loops=False
        )
        self.conv_logstd = GATConv(
            2 * out_channels, out_channels, cached=True, add_self_loops=False
        )

    def forward(self, x, edge_index) -> Tuple[torch.Tensor.torch.Tensor]:
        x = self.conv1(x, edge_index).relu()
        return self.conv_mu(x, edge_index), self.conv_logstd(x, edge_index)
