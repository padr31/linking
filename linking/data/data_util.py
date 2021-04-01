import shutil

from biopandas.pdb import PandasPdb
from rdkit import Chem
import torch
from torch_geometric.utils.convert import to_networkx
from pathlib import Path
import os
import re
import numpy as np
from biopandas.mol2 import PandasMol2
from torch_geometric import data
import dgl
import networkx as nx
import pandas as pd

allowable_atoms = ['C', 'F', 'N', 'Cl', 'O', 'I', 'P', 'Br', 'S', 'H', 'Stop']
allowable_ligand_bonds = ["1", "2", "3", "ar", "am"]

allowable_rdkit_bonds = [
    Chem.rdchem.BondType.AROMATIC,
    Chem.rdchem.BondType.SINGLE,
    Chem.rdchem.BondType.DOUBLE,
    Chem.rdchem.BondType.TRIPLE,
]

ligand_bond_to_one_hot = {
    "1": [1., 0., 0., 0.],
    "2": [0., 1., 0., 0.],
    "3": [0., 0., 1., 0.],
    "ar": [0., 0., 0., 1.],
    "am": [1., 0., 0., 0.],
}

pocket_bond_to_one_hot = {
    Chem.rdchem.BondType.AROMATIC: [0., 0., 0., 1.],
    Chem.rdchem.BondType.SINGLE: [1., 0., 0., 0.],
    Chem.rdchem.BondType.DOUBLE: [0., 1., 0., 0.],
    Chem.rdchem.BondType.TRIPLE: [0., 0., 1., 0.],
}

empty_bond = [0., 0., 0., 0.]

allowable_angles = [120, 109, 60]
allowable_dyhedrals = [180, 120, 60, 0]

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
    #for path, dirs, files in os.walk('/Users/padr/Desktop/aa2ar'):
        for file in files:
            if file.endswith(actives_or_decoys + '_final.mol2'):
                full_path = path + os.sep + file
                files_to_process.append(full_path)
    print(files_to_process)
    print(len(files_to_process))
    for file in files_to_process:
        split_multi_mol2_file(file, actives_or_decoys)

# split_dude_dataset()

def to_one_hot(x, allowable_set=None, mapping=None):
    """
    Function for one hot encoding
    :param x: value to one-hot
    :param allowable_set: set of options to encode
    :param mapping: exact mapping from x to one hot, preferred over allowable set
    :return: one-hot encoding as torch tensor
    """
    if not mapping is None:
        return mapping[x].copy()
    else:
        return [1 if x == s else 0 for s in allowable_set]

def bfs(geom_graph):
    G = to_networkx(geom_graph, to_undirected=True)

    '''
    plotting bfs step-by-step
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
    bfs_edges = list(nx.bfs_edges(G, 0))

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

def mol2_file_to_torch_geometric(path, allowable_atoms, bond_to_one_hot):
    def featurise_ligand_atoms(atoms_df):
        atoms_df["atom_id"] = atoms_df["atom_id"] - 1
        atoms_df.loc[:, "atom_type"] = atoms_df["atom_type"].apply(
            lambda a: to_one_hot(a.split(".")[0], allowable_atoms)
        )

    def featurise_ligand_bonds(bonds_df):
        bonds_df.loc[:, "atom1"] = bonds_df["atom1"].apply(lambda s: int(s) - 1)
        bonds_df.loc[:, "atom2"] = bonds_df["atom2"].apply(lambda s: int(s) - 1)
        bonds_df.loc[:, "bond_type"] = bonds_df["bond_type"].apply(
            lambda b: to_one_hot(b, mapping=bond_to_one_hot)
        )

    bonds = parse_mol2_bonds(path)
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

    # Get edge features and concatenate them
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
        name=path,
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

def pdb_file_to_torch_geometric(path, allowable_atoms, bond_to_one_hot):
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
        name=path
    )

    return geom_graph

def to_atom(t, device=None):
    return allowable_atoms[int(torch.dot(t, torch.tensor(range(t.size()[0]), dtype=torch.float, device=device)).item())]

def to_bond_valency(t, device=None):
    t_s = t.squeeze()
    return [1, 2, 3, 2][int(torch.dot(t_s, torch.tensor(range(t_s.size()[0]), dtype=torch.float, device=device)).item())]

def to_bond_index(t, device=None):
    t_s = t.squeeze()
    return [1, 2, 3, 4][int(torch.dot(t_s, torch.tensor(range(t_s.size()[0]), dtype=torch.float, device=device)).item())]

def to_bond_symbol(t, device=None):
    return ['-', '=', ':=:', '..'][to_bond_index(t, device) - 1]

def calc_angle(v2, v3):
    uvec1 = v2 / np.linalg.norm(v2)
    uvec2 = v3 / np.linalg.norm(v3)
    return np.arccos(np.dot(uvec1, uvec2))

def calc_dihedral(v1, v2, v3):
    """
       Calculate dihedral angle between 4 atoms
       For more information, see:
           http://math.stackexchange.com/a/47084
    """
    # Normal vector of plane containing v1,v2
    n1 = np.cross(v1, v2)
    n1 = n1 / np.linalg.norm(n1)

    # Normal vector of plane containing v2,v3
    n2 = np.cross(v2, v3)
    n2 = n2 / np.linalg.norm(n2)

    # un1, ub2, and um1 form orthonormal frame
    uv2 = v2 / np.linalg.norm(v2)
    m1 = np.cross(n1, uv2)
    m1 = m1 / np.linalg.norm(m1)

    # dot(ub2, n2) is always zero
    x = np.dot(n1, n2)
    y = np.dot(m1, n2)

    dihedral = np.arctan2(y, x)
    return dihedral

def calc_dihedral_2(v1, v2, v3):
    return np.arctan2(np.dot(v2, np.cross(np.cross(v1, v2), np.cross(v2, v3))), np.linalg.norm(v2)*np.dot(np.cross(v1, v2), np.cross(v2, v3)))

dst = np.linalg.norm(np.array([0., 2., -2.]))
print(dst)
ang = calc_angle(np.array([1., 0., 0.]), np.array([0., 2., -2.]))
print(ang)
dih = calc_dihedral(np.array([0., 0., -1.]), np.array([1., 0., 0.]), np.array([0., 2., -2.]))
print(dih)

def calc_position(v1, v2, p3, dst, ang, dih):
    """Calculate position x of another atom based on
       internal coordinates between v1, v2, (p3,x)
       using distance, angle, and dihedral angle.
    """
    # Normal vector of plane containing v1,v2
    n1 = np.cross(v1, v2)
    n1 = n1 / np.linalg.norm(n1)

    # un1, ub2, and um1 form orthonormal frame
    uv2 = v2 / np.linalg.norm(v2)
    m1 = np.cross(n1, uv2)
    m1 = m1 / np.linalg.norm(m1)

    n2 = np.cos(dih)*n1 + np.sin(dih)*m1
    n2 = n2 / np.linalg.norm(n2)

    nn2 = np.cross(n2, uv2)
    nn2 = nn2 / np.linalg.norm(nn2)
    v3 = np.cos(ang)*uv2 + np.sin(ang)*nn2
    v3 = v3 / np.linalg.norm(v3)

    position = p3 + dst * v3

    return position

print(calc_position(np.array([0., 0., -1.]), np.array([1., 0., 0.]), np.array([1., 0., 0.]), dst, ang, dih))

def get_angle(v2, v3) -> float:
    """
    Compute angle between points i, j, and k
    :param p_i: point i
    :param p_j: point j
    :param p_k: point k
    :return: angle in radians
    """
    rij = v3
    rkj = -v2

    sin_theta = np.linalg.norm(np.cross(rij, rkj))
    cos_theta = np.dot(rij, rkj)
    return np.arctan2(sin_theta, cos_theta)

def get_dihedral(v1, v2, v3) -> float:
    """
    Return dihedral between points i, j, k, and l.
    :param p_i: point i
    :param p_j: point j
    :param p_k: point k
    :param p_l: point l
    :return: dihedral angle in radians
    """
    r_ji = -v3
    r_kj = -v2
    r_lk = -v1

    v1 = np.cross(r_ji, r_kj)
    v1 = v1 / np.linalg.norm(v1)

    v2 = np.cross(r_lk, r_kj)
    v2 = v2 / np.linalg.norm(v2)

    m1 = np.cross(v1, r_kj) / np.linalg.norm(r_kj)

    x = np.dot(v1, v2)
    y = np.dot(m1, v2)

    psi = np.arctan2(y, x)
    if psi < 0:
        return -psi - np.pi
    else:
        return np.pi - psi


def position_point(v1: np.ndarray, v2: np.ndarray, p3: np.ndarray, distance: float, angle: float,
                   dihedral: float) -> np.ndarray:
    """
    Determine point p in space that is:
        - <distance> far from p2
        - <angle> between p2 and p1
        - <dihedral> between p2, p1, and p0
    :param p0: position for dihedral
    :param p1: position for angle
    :param p2: position for distance
    :param distance: distance between p and v2
    :param angle: angle between p, p2 and p1
    :param dihedral: dihedral angle between p, p2, p1, and p0
    :return: coordinates of p
    """
    x = distance * np.cos(angle)
    y = distance * np.cos(dihedral) * np.sin(angle)
    z = distance * np.sin(dihedral) * np.sin(angle)

    v_a = v1

    v_b = v2
    v_b = v_b / np.linalg.norm(v_b)

    c_ab = np.cross(v_a, v_b)
    c_ab = c_ab / np.linalg.norm(c_ab)

    c_ab_b = np.cross(c_ab, v_b)

    return p3 - v_b * x + c_ab_b * y + c_ab * z

bond_lengths = {
"C-N":1.4107755,
"C-C":1.5150901,
"C..O":1.255712,
"C=O":1.2334914,
"N-C":1.4107754,
"C-S":1.7459718,
"S-C":1.7459718,
"C..C":1.4013792,
"N-H":1.0122557,
"C-H":1.095748,
"O..C":1.255712,
"O=C":1.2334915,
"H-N":1.0122557,
"H-C":1.095748,
"C=C":1.3827366,
"C-O":1.4193139,
"O-P":1.603689,
"P..O":1.5154448,
"O-C":1.4193139,
"P-O":1.603689,
"O..P":1.5154448,
"P-N":1.6737605,
"C=N":1.3358338,
"O-H":0.9500009,
"N-P":1.6737605,
"N=C":1.3358338,
"H-O":0.9500009,
"S=O":1.4604245,
"N..C":1.3490263,
"C..N":1.3490263,
"N-S":1.6302401,
"O=S":1.4604245,
"S-N":1.6302402,
"N-N":1.3555375,
"N=O":1.2924565,
"O=N":1.2924569,
"C-F":1.3404396,
"C-Cl":1.7358768,
"F-C":1.3404396,
"Cl-C":1.7358768,
"P-C":1.8017883,
"C-P":1.8017882,
"S-H":1.008008,
"H-S":1.008008,
"N-O":1.374657,
"O-N":1.374657,
"C-I":2.1217341,
"I-C":2.1217341,
"C-Br":1.897452,
"Br-C":1.897452,
"S..O":1.4803548,
"O..S":1.4803548,
"P=O":1.5383024,
"O=P":1.5383024,
"N=N":1.3202432,
"C:=:N":1.1414758,
"N:=:C":1.1414758,
"C:=:C":1.1998755,
"S=N":1.5981185,
"N=S":1.5981185,
"S-S":2.046078,
"O-S":1.5391866,
"S-O":1.5391866,
"P=S":1.9899822,
"S=P":1.9899822,
"N..N":1.3403707,
"C=S":1.6924675,
"S=C":1.6924675,
"P-H":1.0079916,
"H-P":1.0079916,
"P-S":2.0013974,
"S-P":2.0013974,
"N:=:N":1.1924888,
}