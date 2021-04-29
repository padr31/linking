from __future__ import annotations

import copy
import random

from linking.data.torchgeom_dude_loader import DudeLigandDataset, DudePocketDataset
from linking.data.torchgeom_pdb_loader import PDBLigandDataset, PDBPocketDataset
from linking.config.config import Config
import os
import torch

def train_test_split(dataset, num_train, num_test, device):
    num_train = min(len(dataset), num_train)
    X_train = []
    X_test = []
    i = 0
    for data in dataset:
        if i < num_train:
            X_train.append(data)
        elif i < num_train + num_test:
            X_test.append(data)
        else:
            break
        i += 1
    return X_train, X_test

def train_test_split_dude(ligand_data, pocket_data, num_train, num_test, device, specific_pockets, ligands_per_pocket):
    pockets_by_protein_name = {}
    if specific_pockets is None:
        for pocket in pocket_data:
            if pocket.protein_name in pockets_by_protein_name:
                pockets_by_protein_name[pocket.protein_name].append(pocket)
            else:
                pockets_by_protein_name[pocket.protein_name] = [pocket]
    else:
        for pocket in pocket_data:
            if pocket.protein_name in specific_pockets:
                pockets_by_protein_name[pocket.protein_name] = pocket

    # num_train ligands for each specific pocket
    if (not specific_pockets is None) and (not ligands_per_pocket is None):
        ligands_by_protein_name = { pocket: [] for pocket in specific_pockets }
        for ligand in ligand_data:
            if ligand.protein_name in specific_pockets:
                ligands_by_protein_name[ligand.protein_name].append(ligand)

        ligands = []
        ligands_test = []
        pockets = []
        pockets_test = []
        for protein_name in specific_pockets:
            ligands.extend(ligands_by_protein_name[protein_name][:num_train])
            ligands_test.extend(ligands_by_protein_name[protein_name][num_train:num_train+num_test])
            pockets.extend([copy.deepcopy(pockets_by_protein_name[protein_name]) for _ in range(num_train)])
            pockets_test.extend([copy.deepcopy(pockets_by_protein_name[protein_name]) for _ in range(num_test)])
        return ligands, ligands_test, pockets, pockets_test

    # random selection of num_train ligands with corresponding pockets
    else:
        ligand_data = random.sample(list(ligand_data), num_train + num_test)
        pocket_data = []
        for li in ligand_data:
            pocket_data.extend(copy.deepcopy(pockets_by_protein_name[li.protein_name]))
        return ligand_data[0:num_train], ligand_data[num_train:], pocket_data[0:num_train], pocket_data[num_train:]


def to_dev(X_ligand_train, X_ligand_test, X_pocket_train, X_pocket_test, device):
    for d in [X_ligand_train, X_ligand_test, X_pocket_train, X_pocket_test]:
        for data in d:
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

def create_data(config: Config, device: torch.device):
    if config.dataset == 'dude':
        ligand_data = DudeLigandDataset(root=os.path.join(config.dataset_root, config.dataset))
        pocket_data = DudePocketDataset(root=os.path.join(config.dataset_root, config.dataset))
    elif config.dataset == 'pdb':
        ligand_data = PDBLigandDataset(root=os.path.join(config.dataset_root, config.dataset))
        pocket_data = PDBPocketDataset(root=os.path.join(config.dataset_root, config.dataset))
    else:
        raise Exception('Non-existing dataset identifier provided.')

    if config.dataset == 'dude':
        X_ligand_train, X_ligand_test, X_pocket_train, X_pocket_test = train_test_split_dude(ligand_data, pocket_data, config.num_train, config.num_test, device, config.specific_pockets, config.ligands_per_pocket)

    elif config.dataset == 'pdb':
        X_ligand_train, X_ligand_test = train_test_split(ligand_data, config.num_train, config.train_test_ratio, device)
        X_pocket_train, X_pocket_test = train_test_split(pocket_data, config.num_train, config.train_test_ratio, device)

    to_dev(X_ligand_train, X_ligand_test, X_pocket_train, X_pocket_test, device)
    return X_ligand_train, X_ligand_test, X_pocket_train, X_pocket_test
