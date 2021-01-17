from __future__ import annotations
from typing import Tuple
from torch_geometric.nn import GCNConv
import torch
import torch.nn.functional as F



class LinearEncoder(torch.nn.Module):
    def __init__(self, in_channels: int, out_channels: int) -> None:
        super(LinearEncoder, self).__init__()
        self.conv = GCNConv(in_channels, out_channels, cached=True)

    def forward(self, x, edge_index) -> Tuple[torch.Tensor, 0]:
        return self.conv(x, edge_index), 0


class VariationalLinearEncoder(torch.nn.Module):
    def __init__(self, in_channels: int, out_channels: int) -> None:
        super(VariationalLinearEncoder, self).__init__()
        self.conv_mu = GCNConv(in_channels, out_channels, cached=True)
        self.conv_logstd = GCNConv(in_channels, out_channels, cached=True)

    def forward(self, x, edge_index) -> Tuple[torch.Tensor, 0]:
        return self.conv_mu(x, edge_index), self.conv_logstd(x, edge_index)

class LinearAtomLabelClassifier(torch.nn.Module):
    def __init__(self, in_channels: int, out_channels: int) -> None:
        super(LinearAtomLabelClassifier, self).__init__()
        self.linear = torch.nn.Linear(in_channels, out_channels)
        self.out_channels = out_channels

    def forward(self, x):
        x = self.linear(x)
        return F.one_hot(torch.argmax(F.softmax(x, dim=1), dim=1), num_classes=self.out_channels)