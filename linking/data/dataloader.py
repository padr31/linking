from __future__ import annotations


from data.torchgeom_pdb_loader import LigandDataset


def create_data(config: Config, device: torch.device):
    dataset = LigandDataset(root="/Users/padr/repos/linking/datasets")
'''
    d = []
    for data in dataset:
        data.train_mask = data.val_mask = data.test_mask = data.y = None
        data = train_test_split_edges(data)
        data.x = data.x.to(device)
        data.train_pos_edge_index = data.train_pos_edge_index.to(device)
        d.append(data)

    return d
'''
create_data(None, None)