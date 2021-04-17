from __future__ import annotations
from linking.data.torchgeom_dude_loader import DudeLigandDataset, DudePocketDataset
from linking.data.torchgeom_pdb_loader import PDBLigandDataset, PDBPocketDataset
from linking.config.config import Config
import os
import torch

def train_test_split(dataset, num_train, train_test_ratio, device):
    num_train = min(len(dataset), num_train)
    num_test = num_train/train_test_ratio
    X_train = []
    X_test = []
    i = 0
    for data in dataset:
        data.train_mask = data.val_mask = data.test_mask = None
        data.x = data.x.to(device)
        if data.y != None:
            data.y = data.y.to(device)
        data.edge_index = data.edge_index.to(device)
        if data.edge_attr != None:
            data.edge_attr = data.edge_attr.to(device)
        if hasattr(data, 'bfs_index') and data.bfs_index != None:
            data.bfs_index = torch.stack(data.bfs_index).to(device)
        if hasattr(data, 'bfs_attr') and data.bfs_attr != None:
            data.bfs_attr = torch.stack(data.bfs_attr).to(device)
        if i < num_train:
            X_train.append(data)
        elif i < num_train + num_test:
            X_test.append(data)
        else:
            break
        i += 1
    return X_train, X_test

def create_data(config: Config, device: torch.device):
    if config.dataset == 'dude':
        ligand_data = DudeLigandDataset(root=os.path.join(config.dataset_root, config.dataset))
        pocket_data = DudePocketDataset(root=os.path.join(config.dataset_root, config.dataset))
    elif config.dataset == 'pdb':
        ligand_data = PDBLigandDataset(root=os.path.join(config.dataset_root, config.dataset))
        pocket_data = PDBPocketDataset(root=os.path.join(config.dataset_root, config.dataset))
    else:
        raise Exception('Non-existing dataset identifier provided.')

    X_ligand_train, X_ligand_test = train_test_split(ligand_data, config.num_train, config.train_test_ratio, device)
    X_pocket_train, X_pocket_test = train_test_split(pocket_data, config.num_train, config.train_test_ratio, device)

    return X_ligand_train, X_ligand_test, X_pocket_train, X_pocket_test