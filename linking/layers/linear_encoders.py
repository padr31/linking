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

class LinearAtomClassifier(torch.nn.Module):
    def __init__(self, in_channels: int, out_channels: int) -> None:
        super(LinearAtomClassifier, self).__init__()
        self.linear = torch.nn.Linear(in_channels, out_channels)
        self.out_channels = out_channels

    def forward(self, x, mask=None, gumbel=False):
        x = self.linear(x)
        x = F.relu(x)
        if not mask is None:
            x = x + mask
        if gumbel:
            x = F.gumbel_softmax(x, hard=True, dim=1)
        else:
            x = F.softmax(x, dim=1)
        return x

class LinearEdgeSelector(torch.nn.Module):
    '''
        Edges are passed in as features of type phi(u,v) = [t, z_pocket, z_ligand, z_u, l_u, z_v, l_v, z_g]
        We pick one and return its index. One of these needs to be the termination node, probably best if it's the 0th.
    '''
    def __init__(self, in_channels: int) -> None:
        super(LinearEdgeSelector, self).__init__()
        self.linear = torch.nn.Linear(in_channels, 1)

    def forward(self, x, mask=None, gumbel=False):
        x = self.linear(x)
        x = F.relu(x)
        if not mask is None:
            x = x + mask
        if gumbel:
            x = F.gumbel_softmax(x, hard=True, dim=0)
            x = x.squeeze(1)
            x = x.dot(torch.tensor(range(0, x.size()[0])).to(device=x.device).float())
            x = x.long()
        else:
            x = F.softmax(x, dim=0)
        return x

class LinearEdgeRowClassifier(torch.nn.Module):
    '''
           Edges are passed in as features of type phi(u,v) = [t, z_pocket, z_ligand, z_u, l_u, z_v, l_v, z_g]
           We reduce the vector into 3 numbers, and return the argmax, as
       '''

    def __init__(self, in_channels: int, out_channels: int) -> None:
        super(LinearEdgeRowClassifier, self).__init__()
        self.linear = torch.nn.Linear(in_channels, out_channels)

    def forward(self, x, mask=None, gumbel=False):
        x = self.linear(x)
        x = F.relu(x)
        if not mask is None:
            x = x + mask

        if gumbel:
            x = F.gumbel_softmax(x, hard=True, dim=1)
        else:
            x = F.softmax(x, dim=1)
        return x[0]

class LinearEdgeClassifier(torch.nn.Module):
    '''
        Edges are passed in as features of type phi(u,v) = [t, z_pocket, z_ligand, z_u, l_u, z_v, l_v, z_g]
        We reduce the vector into 3 numbers, and return the argmax, as
    '''
    def __init__(self, in_channels: int, out_channels: int) -> None:
        super(LinearEdgeClassifier, self).__init__()
        self.linear = torch.nn.Linear(in_channels, out_channels)

    def forward(self, x, mask=None, gumbel=False):
        x = self.linear(x)
        x = F.relu(x)
        if not mask is None:
            x = x + mask

        if gumbel:
            x = F.gumbel_softmax(x, hard=True, dim=1)
        else:
            x = F.softmax(x, dim=1)
        return x

class MLP(torch.nn.Module):
    def __init__(self, in_channels: int, out_channels: int, hidden_layers: int) -> None:
        super(MLP, self).__init__()
        self.l1 = torch.nn.Linear(in_channels, in_channels*2)
        self.l2 = torch.nn.Linear(in_channels*2, in_channels*2)
        self.l3 = torch.nn.Linear(in_channels*2, out_channels)
        self.hidden_layers = hidden_layers

    def forward(self, x):
        x = self.l1(x).relu()
        for i in range(self.hidden_layers):
            x = self.l2(x).relu()
        return self.l3(x)
