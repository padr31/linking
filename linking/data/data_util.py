from biopandas.pdb import PandasPdb
from rdkit import Chem
from pathlib import Path
from torch_geometric.utils.convert import to_networkx
from biopandas.mol2 import PandasMol2
from torch_geometric import data
from linking.util.encoding import empty_bond, to_one_hot
import shutil
import torch
import os
import re
import numpy as np
import dgl
import networkx as nx
import pandas as pd

def bfs(geom_graph):
    G = to_networkx(geom_graph, to_undirected=True)
    '''
    plotting bfs step-by-step for debugging
    l = nx.spring_layout(G)
    if geom_graph.name.endswith('10gs_ligand.mol2'):
        n_list = []
        print([(u, v) if not (u,v) in G.edges else 'ok' for (u, v) in nx.bfs_edges(G, 0)])
        for (u,v) in nx.bfs_edges(G, 0):
            n_list.append(u)
            n_list.append(v)
            nx.draw_networkx(G, nodelist=n_list, pos=l)
            plt.show()
    '''
    bfs_edges = list(nx.edge_bfs(G, 0))

    # build attribute map
    attrib_map = {}
    for i in range(geom_graph.edge_index.size(1)):
        attrib_map[str((geom_graph.edge_index[0][i].item(), geom_graph.edge_index[1][i].item()))] = geom_graph.edge_attr[i]

    # add edge attributes
    bfs_attributes = []
    for e in bfs_edges:
        bfs_attributes.append(attrib_map[str(e)])

    edges_copy = []
    attrib_copy = []
    for i in range(len(bfs_edges)):
        edges_copy.append(bfs_edges[i])
        attrib_copy.append(bfs_attributes[i])
        if i == len(bfs_edges)-1 or bfs_edges[i+1][0] != bfs_edges[i][0]:
            edges_copy.append((bfs_edges[i][0], -1))
            attrib_copy.append(torch.tensor(empty_bond, dtype=torch.float))

    return [torch.tensor([e[0], e[1]], dtype=torch.long) for e in edges_copy], attrib_copy

def bfs_distance(start, adj):
    '''
        Perform BFS on the adj and return distances to start node.
    '''
    dist = torch.ones_like(adj[0], device=adj.device) * 100
    visited = torch.zeros_like(adj[0], dtype=torch.bool, device=adj.device)
    q = torch.tensor([start], device=adj.device)

    visited[start] = True
    dist[start] = 0

    while q.size()[0] > 0:
        vis = q[0]
        q = q[1:]
        for i in range(adj.size()[0]):
            if adj[vis][i] and (not visited[i]):
                q = torch.cat([q, torch.unsqueeze(torch.tensor(i, device=adj.device), 0)])
                dist[i] = dist[vis] + 1
            visited[i] = True
    return dist.unsqueeze(1)

def parse_mol2_bonds(filename):
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

def mol2_file_to_networkx(path):
    bonds = parse_mol2_bonds(path)
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

def mol2_file_to_torch_geometric(path, allowable_atoms, bond_to_one_hot, protein_name=None, remove_hydrogens=True):
    def featurise_ligand_atoms(atoms_df):
        atoms_df.loc[:, "atom_type"] = atoms_df["atom_type"].apply(
            lambda a: a.split(".")[0]
        )
        atoms_df.loc[:, "atom_type_feature"] = atoms_df["atom_type"].apply(
            lambda a: to_one_hot(a, allowable_atoms)
        )

    def featurise_ligand_bonds(bonds_df):
        bonds_df.loc[:, "atom1"] = bonds_df["atom1"].apply(lambda s: int(s))
        bonds_df.loc[:, "atom2"] = bonds_df["atom2"].apply(lambda s: int(s))
        bonds_df.loc[:, "bond_type_feature"] = bonds_df["bond_type"].apply(
            lambda b: to_one_hot(b, mapping=bond_to_one_hot)
        )

    bonds = parse_mol2_bonds(path)
    atoms = PandasMol2().read_mol2(path).df
    featurise_ligand_atoms(atoms)
    featurise_ligand_bonds(bonds)

    if remove_hydrogens:
        hydrogen_idx = atoms.index[atoms['atom_type'] == 'H'].tolist()
        hydrogen_atom_idx = atoms.loc[hydrogen_idx]['atom_id'].to_list()
        hydrogen_bond_idx = []
        for bond_index, bond_row in bonds.iterrows():
            if int(bond_row['atom1']) in hydrogen_atom_idx or int(bond_row['atom2']) in hydrogen_atom_idx:
                hydrogen_bond_idx.append(bond_index)
        atoms = atoms.drop(hydrogen_idx)
        bonds = bonds.drop(hydrogen_bond_idx)

    current_atom_index = 0
    atom_id_to_index = {}
    for atoms_index, atoms_row in atoms.iterrows():
        atom_id_to_index[atoms_row['atom_id']] = current_atom_index
        current_atom_index += 1

    bonds_other_direction = bonds.copy(deep=True)
    bonds_other_direction = bonds_other_direction.rename(
        columns={"atom1": "atom2", "atom2": "atom1"}
    )
    bonds = pd.concat([bonds, bonds_other_direction])

    features = [
        torch.tensor([f for f in atoms[feat].tolist()], dtype=torch.float)
        for feat in ["atom_id", "x", "y", "z", "atom_type_feature"]
    ]
    features = [f.unsqueeze(dim=1) if len(f.shape) == 1 else f for f in features]
    node_features = torch.cat(features, dim=1)

    edge_indices = [
        list(map(lambda atom_id: atom_id_to_index[atom_id], bonds["atom1"].tolist())),
        list(map(lambda atom_id: atom_id_to_index[atom_id], bonds["atom2"].tolist()))]
    edge_indices = torch.tensor(
        edge_indices, dtype=torch.long
    ).contiguous()

    # Get edge features and concatenate them
    edge_features = [
        torch.tensor([edge for edge in bonds[feat].tolist()], dtype=torch.float)
        for feat in ["bond_type_feature"]
    ]
    edge_features = [
        e.unsqueeze(dim=1) if len(e.shape) == 1 else e for e in edge_features
    ]
    edge_features = torch.cat(edge_features, dim=1)

    # Create the Torch Geometric graph
    geom_graph = data.Data(
        x=node_features,
        edge_index=edge_indices,
        edge_attr=edge_features,
        name=path,
        protein_name=protein_name,
        bfs_index=None,
        bfs_attr=None,
    )

    geom_graph.bfs_index, geom_graph.bfs_attr = bfs(geom_graph)
    return geom_graph

def mol2_file_to_dgl(path):
    nxg = mol2_file_to_networkx(path)
    g = dgl.DGLGraph()
    g.from_networkx(
        nxg, node_attrs=["x", "y", "z", "atom_type"], edge_attrs=["bond_type"]
    )
    return g

def pdb_file_to_torch_geometric(path, allowable_atoms, bond_to_one_hot, protein_name=None):
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
        type = to_one_hot(bond.GetBondType(), mapping=bond_to_one_hot)
        edge_types.extend([type, type])

    # Create the Torch Geometric graph
    geom_graph = data.Data(
        x=torch.tensor(node_features, dtype=torch.float),
        edge_index=torch.tensor([edge_src, edge_dst], dtype=torch.long).contiguous(),
        edge_attr=torch.tensor(edge_types, dtype=torch.float).contiguous(),
        name=path,
        protein_name=protein_name
    )

    return geom_graph

def split_multi_mol2_file(path, dir_name):
    delimiter = '@<TRIPOS>MOLECULE'
    write_dir = Path(path).parent / dir_name
    if not os.path.exists(str(write_dir)):
        os.makedirs(str(write_dir))

    def write(m):
        wf = open(str(write_dir / (m[0] + '_actives_ligand.mol2')), "w")
        for line in m[1]:
            wf.write(line)
        wf.close()

    with open(path, 'r') as f:
        mol2 = ['', []]
        while True:
            try:
                line = next(f)
                if line.startswith(delimiter):
                    if mol2[0]:
                        mol2[1].append('@')
                        write(mol2)
                    mol2 = ['', []]
                    mol2_id = next(f)
                    mol2[0] = mol2_id.rstrip()
                    mol2[1].append(line)
                    mol2[1].append(mol2_id)
                else:
                    mol2[1].append(line)
            except StopIteration:
                mol2[1].append('@')
                write(mol2)
                return

def split_dude_dataset():
    actives_or_decoys = 'actives'
    # removing existing dir (optional)
    for path, dirs, files in os.walk('/Users/padr/repos/linking/datasets/dude/raw/'):
        for dir in dirs:
            if dir == actives_or_decoys:
                shutil.rmtree(path + os.sep + dir)

    files_to_process = []
    for path, dirs, files in os.walk('/Users/padr/repos/linking/datasets/dude/raw/'):
        for file in files:
            if file.endswith(actives_or_decoys + '_final.mol2'):
                full_path = path + os.sep + file
                files_to_process.append(full_path)
    print(files_to_process)
    print(len(files_to_process))
    for file in files_to_process:
        split_multi_mol2_file(file, actives_or_decoys)