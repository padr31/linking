import argparse
import os.path as osp

import torch
import torch_geometric.transforms as T
from pdb_loader import LigandDataset
from torch_geometric.data import DataLoader
from torch_geometric.nn import GAE, VGAE, GCNConv
from torch_geometric.utils import train_test_split_edges

path = osp.join(osp.dirname(osp.realpath(__file__)), "..", "data", "Planetoid")
dataset = LigandDataset(root="./datasets")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

d = []
for data in dataset:
    data.train_mask = data.val_mask = data.test_mask = data.y = None
    data = train_test_split_edges(data)
    data.x = data.x.to(device)
    data.train_pos_edge_index = data.train_pos_edge_index.to(device)
    d.append(data)
