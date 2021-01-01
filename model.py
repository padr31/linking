
import os.path as osp
import argparse
import torch
import torch_geometric.transforms as T
from torch_geometric.nn import GCNConv, GAE, VGAE
from torch_geometric.utils import train_test_split_edges
from pdb_loader import LigandDataset
from torch_geometric.data import DataLoader


path = osp.join(osp.dirname(osp.realpath(__file__)), '..', 'data', 'Planetoid')
dataset = LigandDataset(root='./datasets')
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

d = []
for data in dataset:
    data.train_mask = data.val_mask = data.test_mask = data.y = None
    data = train_test_split_edges(data)
    data.x = data.x.to(device)
    data.train_pos_edge_index = data.train_pos_edge_index.to(device)
    d.append(data)

class GCNEncoder(torch.nn.Module):
    def __init__(self, in_channels, out_channels):
        super(GCNEncoder, self).__init__()
        self.conv1 = GCNConv(in_channels, 2 * out_channels, cached=True, add_self_loops=False)
        self.conv2 = GCNConv(2 * out_channels, out_channels, cached=True, add_self_loops=False)

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index).relu()
        return self.conv2(x, edge_index), 0


class VariationalGCNEncoder(torch.nn.Module):
    def __init__(self, in_channels, out_channels):
        super(VariationalGCNEncoder, self).__init__()
        self.conv1 = GCNConv(in_channels, 2 * out_channels, cached=True, add_self_loops=False)
        self.conv_mu = GCNConv(2 * out_channels, out_channels, cached=True, add_self_loops=False)
        self.conv_logstd = GCNConv(2 * out_channels, out_channels, cached=True, add_self_loops=False)

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index).relu()
        return self.conv_mu(x, edge_index), self.conv_logstd(x, edge_index)


class LinearEncoder(torch.nn.Module):
    def __init__(self, in_channels, out_channels):
        super(LinearEncoder, self).__init__()
        self.conv = GCNConv(in_channels, out_channels, cached=True)

    def forward(self, x, edge_index):
        return self.conv(x, edge_index), 0


class VariationalLinearEncoder(torch.nn.Module):
    def __init__(self, in_channels, out_channels):
        super(VariationalLinearEncoder, self).__init__()
        self.conv_mu = GCNConv(in_channels, out_channels, cached=True)
        self.conv_logstd = GCNConv(in_channels, out_channels, cached=True)

    def forward(self, x, edge_index):
        return self.conv_mu(x, edge_index), self.conv_logstd(x, edge_index)


out_channels = 16
num_features = dataset.num_features

model = VGAE(VariationalGCNEncoder(num_features, out_channels))
model = model.to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

def train():
    model.train()
    total_loss = 0
    for data in d[0:3500]:
        optimizer.zero_grad()
        print(data.x)
        z = model.encode(data.x, data.train_pos_edge_index)
        loss = model.recon_loss(z, data.train_pos_edge_index)
        loss = loss + (1 / data.num_nodes) * model.kl_loss()
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return float(total_loss)


def test(pos_edge_index, neg_edge_index):
    auc_total = 0
    ap_total = 0
    for data in d[3500:]:
        model.eval()
        with torch.no_grad():
            z = model.encode(data.x, data.train_pos_edge_index)
        auc_, ap_ = model.test(z, pos_edge_index, neg_edge_index)
        auc_total += auc_
        ap_total += ap_

    return auc_total / len(d[3500:]), ap_total / len(d[3500:])


for epoch in range(1, 400 + 1):
    loss = train()
    auc, ap = test(data.test_pos_edge_index, data.test_neg_edge_index)
    print('Epoch: {:03d}, AUC: {:.4f}, AP: {:.4f}'.format(epoch, auc, ap))
