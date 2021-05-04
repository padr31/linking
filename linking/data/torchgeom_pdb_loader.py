import os
import pandas as pd
import torch
from torch_geometric.data import InMemoryDataset
from linking.data.data_util import pdb_file_to_torch_geometric, mol2_file_to_torch_geometric
from linking.util.encoding import allowable_atoms, ligand_bond_to_one_hot, pocket_bond_to_one_hot

bad_data = ["1g7v", "1r1h", "2a5b", "2zjw", "1cps", "4abd"]
pd.set_option("display.max_columns", None)

def process_dir(dir, file_ending, bad_data):
    # Read data into huge `Data` list.
    files_to_process = []
    for path, dirs, files in os.walk(dir):
        for file in files:
            if file.endswith(file_ending) and not file.split("_")[0] in bad_data:
                full_path = path + os.sep + file
                files_to_process.append(full_path)

    return files_to_process

class PDBLigandDataset(InMemoryDataset):
    def __init__(self, root, transform=None, pre_transform=None, config=None):
        self.config = config
        super(PDBLigandDataset, self).__init__(root, transform, pre_transform)
        self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def processed_file_names(self):
        return ["ligands.pt"]

    def process(self):
        # Read data into huge `Data` list. and save
        files_to_process = process_dir(
            self.raw_dir, "ligand.mol2", bad_data
        )

        graphs = []
        total = len(files_to_process)
        print("Starting to process " + str(total) + " files...")
        i = 0
        for path in sorted(files_to_process):
            i += 1
            print(
                "(" + str(int(100 * i / total)) + "%) Processing " + os.path.basename(path)
            )
            protein_name = path.split('/')[-2]
            g = mol2_file_to_torch_geometric(path, allowable_atoms, ligand_bond_to_one_hot, protein_name, self.config.remove_hydrogens)
            # torchgeom_plot_3D(g, 90)
            graphs.append(g)

        if self.pre_filter is not None:
            graphs = [g for g in graphs if self.pre_filter(g)]

        if self.pre_transform is not None:
            graphs = [self.pre_transform(g) for g in graphs]

        data, slices = self.collate(graphs)
        torch.save((data, slices), self.processed_paths[0])


class PDBPocketDataset(InMemoryDataset):
    def __init__(self, root, transform=None, pre_transform=None, config=None):
        self.config = config
        super(PDBPocketDataset, self).__init__(root, transform, pre_transform)
        self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def processed_file_names(self):
        return ["pockets.pt"]

    def process(self):
        # Read data into huge `Data` list. and save
        files_to_process = process_dir(
            self.raw_dir, "pocket.pdb", bad_data
        )

        graphs = []
        total = len(files_to_process)
        print("Starting to process " + str(total) + " files...")
        i = 0
        for path in sorted(files_to_process):
            i += 1
            print(
                "(" + str(int(100 * i / total)) + "%) Processing " + os.path.basename(path)
            )
            protein_name = path.split('/')[-2]
            g = pdb_file_to_torch_geometric(path, allowable_atoms, pocket_bond_to_one_hot, protein_name)
            # torchgeom_plot_3D(g, 90)
            graphs.append(g)

        if self.pre_filter is not None:
            graphs = [g for g in graphs if self.pre_filter(g)]

        if self.pre_transform is not None:
            graphs = [self.pre_transform(g) for g in graphs]

        data, slices = self.collate(graphs)
        torch.save((data, slices), self.processed_paths[0])

# d = PDBLigandDataset(root="/Users/padr/repos/linking/datasets/pdb/")
# d = PDBPocketDataset(root="/Users/padr/repos/linking/datasets/pdb/")
# g = pdb_file_to_torch_geometric('/Users/padr/repos/linking/datasets/raw/refined-set/1a1e/1a1e_pocket.pdb')
# g = mol2_file_to_torch_geometric('/Users/padr/repos/linking/datasets/raw/refined-set/4rdn/4rdn_ligand.mol2')

# torchgeom_plot_3D(g, 0)
