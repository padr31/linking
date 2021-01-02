from __future__ import annotations

import os


def create_data(config: Config, device: torch.device):
    path = osp.join(osp.dirname(osp.realpath(__file__)), "..", "data", "Planetoid")
    dataset = LigandDataset(root="./datasets")

    d = []
    for data in dataset:
        data.train_mask = data.val_mask = data.test_mask = data.y = None
        data = train_test_split_edges(data)
        data.x = data.x.to(device)
        data.train_pos_edge_index = data.train_pos_edge_index.to(device)
        d.append(data)

    return d
