from __future__ import annotations
import torch
from data.torchgeom_pdb_loader import LigandDataset, PocketDataset
from linking.config.config import Config

def train_test_split(dataset, num_train, train_test_ratio, device):
    num_train = min(len(dataset), num_train)
    num_test = num_train/train_test_ratio
    X_train = []
    X_test = []
    i = 0
    for data in dataset:
        data.train_mask = data.val_mask = data.test_mask = data.y = None
        data.x = data.x.to(device)
        data.edge_index = data.edge_index.to(device)
        if i < num_train:
            X_train.append(data)
        elif i < num_train + num_test:
            X_test.append(data)
        else:
            break
        i += 1
    return X_train, X_test

def create_data(config: Config, device: torch.device):
    ligand_data = LigandDataset(root=config.dataset_root)
    pocket_data = PocketDataset(root="/Users/padr/repos/linking/datasets")

    X_ligand_train, X_ligand_test = train_test_split(ligand_data, config.num_train, config.train_test_ratio, device)
    X_pocket_train, X_pocket_test = train_test_split(pocket_data, config.num_train, config.train_test_ratio, device)

    return X_ligand_train, X_ligand_test, X_pocket_train, X_pocket_test