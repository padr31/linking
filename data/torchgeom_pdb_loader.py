import os
import re
import dgl
import networkx as nx
import numpy as np
import pandas as pd
import torch
from biopandas.mol2 import PandasMol2
from biopandas.pdb import PandasPdb
from rdkit import Chem
from torch_geometric import data
from torch_geometric.data import InMemoryDataset
from data.data_plotting import torchgeom_plot_3D

pdb_dir = "./datasets/raw/refined-set"
bad_data = ["1g7v", "1r1h", "2a5b", "2zjw", "1cps", "4abd"]
pd.set_option("display.max_columns", None)

def parse_bonds(filename):
    with open(filename, "r") as f:
        f_text = f.read()
        bonds = np.array(
            re.sub(
                r"\s+", " ", re.search(r"@<TRIPOS>BOND([a-z0-9\s]*)@", f_text).group(1)
            ).split()
        ).reshape((-1, 4))
        df_bonds = pd.DataFrame(
            bonds, columns=["bond_id", "atom1", "atom2", "bond_type"]
        )
        df_bonds.set_index(["bond_id"], inplace=True)
        return df_bonds


allowable_atoms = [
    "C",
    "N",
    "O",
    "S",
    "F",
    "Si",
    "P",
    "Cl",
    "Br",
    "Mg",
    "Na",
    "Ca",
    "Fe",
    "As",
    "Al",
    "I",
    "B",
    "V",
    "K",
    "Tl",
    "Yb",
    "Sb",
    "Sn",
    "Ag",
    "Pd",
    "Co",
    "Se",
    "Ti",
    "Zn",
    "H",
    "Li",
    "Ge",
    "Cu",
    "Au",
    "Ni",
    "Cd",
    "In",
    "Mn",
    "Zr",
    "Cr",
    "Pt",
    "Hg",
    "Pb",
    "H",
    "Du",
    "LP",
]
allowable_bonds = ["ar", "1", "2", "3", "am", "du", "un", "nc"]

allowable_rdkit_bonds = [
    Chem.rdchem.BondType.AROMATIC,
    Chem.rdchem.BondType.SINGLE,
    Chem.rdchem.BondType.DOUBLE,
    Chem.rdchem.BondType.TRIPLE,
]


def to_one_hot(x, allowable_set):
    """
    Function for one hot encoding
    :param x: value to one-hot
    :param allowable_set: set of options to encode
    :return: one-hot encoding as torch tensor
    """
    return [1 if x == s else 0 for s in allowable_set]

def featurise_ligand_atoms(atoms_df):
    atoms_df["atom_id"] = atoms_df["atom_id"] - 1
    atoms_df.loc[:, "atom_type"] = atoms_df["atom_type"].apply(
        lambda a: to_one_hot(a.split(".")[0], allowable_atoms)
    )


def featurise_ligand_bonds(bonds_df):
    bonds_df.loc[:, "atom1"] = bonds_df["atom1"].apply(lambda s: int(s) - 1)
    bonds_df.loc[:, "atom2"] = bonds_df["atom2"].apply(lambda s: int(s) - 1)
    bonds_df.loc[:, "bond_type"] = bonds_df["bond_type"].apply(
        lambda b: to_one_hot(b, allowable_bonds)
    )


def mol2_file_to_networkx(path):
    bonds = parse_bonds(path)
    atoms = PandasMol2().read_mol2(path).df

    g = nx.Graph()

    for index, row in atoms.iterrows():
        g.add_node(
            int(row["atom_id"]),
            x=row["x"],
            y=row["y"],
            z=row["z"],
            atom_type=row["atom_type"],
        )

    for index, row in bonds.iterrows():
        g.add_edge(int(row["atom1"]), int(row["atom2"]), bond_type=row["bond_type"])

    return g


def mol2_file_to_torch_geometric(path):
    bonds = parse_bonds(path)
    atoms = PandasMol2().read_mol2(path).df
    featurise_ligand_atoms(atoms)
    featurise_ligand_bonds(bonds)

    bonds_other_direction = bonds.copy(deep=True)
    bonds_other_direction = bonds_other_direction.rename(
        columns={"atom1": "atom2", "atom2": "atom1"}
    )
    bonds = pd.concat([bonds, bonds_other_direction])

    features = [
        torch.tensor([f for f in atoms[feat].tolist()], dtype=torch.float)
        for feat in ["atom_id", "x", "y", "z", "atom_type"]
    ]
    features = [f.unsqueeze(dim=1) if len(f.shape) == 1 else f for f in features]
    node_features = torch.cat(features, dim=1)

    # Get edge features from DGL graph and concatenate them
    edge_features = [
        torch.tensor([edge for edge in bonds[feat].tolist()], dtype=torch.float)
        for feat in ["bond_type"]
    ]
    edge_features = [
        e.unsqueeze(dim=1) if len(e.shape) == 1 else e for e in edge_features
    ]
    edge_features = torch.cat(edge_features, dim=1)

    # Create the Torch Geometric graph
    geom_graph = data.Data(
        x=node_features,
        edge_index=torch.tensor(
            [bonds["atom1"].tolist(), bonds["atom2"].tolist()], dtype=torch.long
        ).contiguous(),
        edge_attr=edge_features,
    )
    return geom_graph


def mol2_file_to_dgl(path):
    nxg = mol2_file_to_networkx(path)
    g = dgl.DGLGraph()
    g.from_networkx(
        nxg, node_attrs=["x", "y", "z", "atom_type"], edge_attrs=["bond_type"]
    )
    return g


def pdb_file_to_torch_geometric(path):
    mol = Chem.MolFromPDBFile(path)

    # TODO: add hydrogens?
    atoms = PandasPdb().read_pdb(path).df

    node_features = []

    num_atoms = len(atoms["ATOM"]) - 1
    for atom in mol.GetAtoms():
        pdb_loc = (
            atom.GetPDBResidueInfo().GetSerialNumber() - 1
        )  # minus 1 because iloc starts at 0 and serial numbers start at 1
        if pdb_loc > num_atoms:
            continue
        pandas_atom = atoms["ATOM"].iloc[[pdb_loc]]
        feature_vector = [
            atom.GetIdx(),
            float(pandas_atom["x_coord"]),
            float(pandas_atom["y_coord"]),
            float(pandas_atom["z_coord"]),
        ]
        feature_vector.extend(to_one_hot(atom.GetSymbol(), allowable_atoms))
        node_features.append(feature_vector)

    # Add edges
    edge_src = []
    edge_dst = []
    edge_types = []
    num_bonds = mol.GetNumBonds()
    for i in range(num_bonds):
        bond = mol.GetBondWithIdx(i)
        u = bond.GetBeginAtomIdx()
        v = bond.GetEndAtomIdx()
        if (
            mol.GetAtomWithIdx(u).GetPDBResidueInfo().GetSerialNumber() - 1 > num_atoms
            or mol.GetAtomWithIdx(u).GetPDBResidueInfo().GetSerialNumber() - 1
            > num_atoms
        ):
            continue
        edge_src.extend([u, v])
        edge_dst.extend([v, u])
        type = to_one_hot(bond.GetBondType(), allowable_rdkit_bonds)
        edge_types.extend([type, type])

    # Create the Torch Geometric graph
    geom_graph = data.Data(
        x=torch.tensor(node_features, dtype=torch.float),
        edge_index=torch.tensor([edge_src, edge_dst], dtype=torch.long).contiguous(),
        edge_attr=torch.tensor(edge_types, dtype=torch.float).contiguous(),
    )
    return geom_graph


def process_dir(dir, file_ending, graph_constructor, bad_data):
    # Read data into huge `Data` list.
    files_to_process = []
    for path, dirs, files in os.walk(dir):
        for file in files:
            if file.endswith(file_ending) and not file.split("_")[0] in bad_data:
                full_path = path + os.sep + file
                files_to_process.append(full_path)

    graphs = []
    total = len(files_to_process)
    print("Starting to process " + str(total) + " files...")
    i = 0
    for path in sorted(files_to_process):
        i += 1
        print(
            "(" + str(int(100 * i / total)) + "%) Processing " + os.path.basename(path)
        )
        g = graph_constructor(path)
        #torchgeom_plot_3D(g, 90)
        graphs.append(g)

    return graphs


class LigandDataset(InMemoryDataset):
    def __init__(self, root, transform=None, pre_transform=None):
        super(LigandDataset, self).__init__(root, transform, pre_transform)
        self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def processed_file_names(self):
        return ["ligands.pt"]

    def process(self):
        # Read data into huge `Data` list. and save
        graphs = process_dir(
            self.raw_dir, "ligand.mol2", mol2_file_to_torch_geometric, bad_data
        )

        if self.pre_filter is not None:
            graphs = [g for g in graphs if self.pre_filter(g)]

        if self.pre_transform is not None:
            graphs = [self.pre_transform(g) for g in graphs]

        data, slices = self.collate(graphs)
        torch.save((data, slices), self.processed_paths[0])


class PocketDataset(InMemoryDataset):
    def __init__(self, root, transform=None, pre_transform=None):
        super(PocketDataset, self).__init__(root, transform, pre_transform)
        self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def processed_file_names(self):
        return ["pockets.pt"]

    def process(self):
        # Read data into huge `Data` list. and save
        graphs = process_dir(
            self.raw_dir, "pocket.pdb", pdb_file_to_torch_geometric, bad_data
        )

        if self.pre_filter is not None:
            graphs = [g for g in graphs if self.pre_filter(g)]

        if self.pre_transform is not None:
            graphs = [self.pre_transform(g) for g in graphs]

        data, slices = self.collate(graphs)
        torch.save((data, slices), self.processed_paths[0])


# d = PocketDataset(root="./datasets")
# g = pdb_file_to_torch_geometric('/Users/padr/repos/linking/datasets/raw/refined-set/1a1e/1a1e_pocket.pdb')
#g = mol2_file_to_torch_geometric('/Users/padr/repos/linking/datasets/raw/refined-set/4rdn/4rdn_ligand.mol2')

# torchgeom_plot_3D(g, 0)
