from __future__ import annotations
from typing import Tuple
from torch_geometric.nn import GCNConv, GATConv, GatedGraphConv, CGConv
import torch

class CGCEncoder(torch.nn.Module):
    def __init__(self, in_channels: int, out_channels: int, edge_dim: int, hidden_layers: int):
        super(CGCEncoder, self).__init__()
        self.hidden_layers = hidden_layers
        self.conv1 = CGConv(
            in_channels, dim=edge_dim
        )
        self.conv2 = CGConv(
            in_channels, dim=edge_dim
        )
        self.conv3 = CGConv(
            in_channels, dim=edge_dim
        )

    def forward(self, x, edge_index, edge_attr) -> torch.Tensor:
        x = self.conv1(x, edge_index, edge_attr).relu()
        for i in range(self.hidden_layers):
           x = self.conv2(x, edge_index, edge_attr).relu()
        return self.conv3(x, edge_index, edge_attr).softmax(dim=1)

class GATEncoder(torch.nn.Module):
    def __init__(self, in_channels: int, out_channels: int, hidden_layers: int):
        super(GATEncoder, self).__init__()
        self.hidden_layers = hidden_layers
        self.conv1 = GATConv(
            in_channels, 2 * out_channels, add_self_loops=False
        )
        self.conv2 = GATConv(
            2 * out_channels, 2 * out_channels, add_self_loops=False
        )
        self.conv3 = GATConv(
            2 * out_channels, out_channels, add_self_loops=False
        )

    def forward(self, x, edge_index) -> torch.Tensor:
        x = self.conv1(x, edge_index).relu()
        for i in range(self.hidden_layers):
            x = self.conv2(x, edge_index).relu()
        return self.conv3(x, edge_index).softmax(dim=1)

class VariationalGATEncoder(torch.nn.Module):
    def __init__(self, in_channels: int, out_channels: int, hidden_layers: int):
        super(VariationalGATEncoder, self).__init__()
        self.hidden_layers = hidden_layers
        self.conv1 = GATConv(
            in_channels, 2 * out_channels, add_self_loops=False
        )
        self.conv2 = GATConv(
            2 * out_channels, 2 * out_channels, add_self_loops=False
        )
        self.conv_mu = GATConv(
            2 * out_channels, out_channels, add_self_loops=False
        )
        self.conv_log_var = GATConv(
            2 * out_channels, out_channels, add_self_loops=False
        )

    def forward(self, x, edge_index) -> torch.Tensor:
        x = self.conv1(x, edge_index).relu()
        for i in range(self.hidden_layers):
            x = self.conv2(x, edge_index).relu()
        return torch.sum(self.conv_mu(x, edge_index), dim=0), torch.sum(self.conv_log_var(x, edge_index), dim=0)

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

    def forward(self, x, edge_index, edge_attr) -> torch.Tensor:
        x = self.conv1(x, edge_index, edge_attr).relu()
        for i in range(7):
            x = self.conv2(x, edge_index, edge_attr).relu()
        return self.conv2(x, edge_index, edge_attr)