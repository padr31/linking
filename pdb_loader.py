import os
import numpy as np
import pandas as pd
import re
from biopandas.mol2 import PandasMol2
import networkx as nx
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from torch_geometric.utils import from_networkx
from torch_geometric import data
import dgl
import torch

pdb_dir = './datasets/refined-set'
pd.set_option('display.max_columns', None)


def network_plot_3D(G, angle, save=False):
    # Get node positions
    pos = {node[0]: (node[1]['x'], node[1]['y'], node[1]['z']) for node in G.nodes.data()}

    # 3D network plot
    with plt.style.context(('ggplot')):

        fig = plt.figure(figsize=(10, 7))
        ax = Axes3D(fig)

        # Loop on the pos dictionary to extract the x,y,z coordinates of each node
        for key, value in pos.items():
            xi = value[0]
            yi = value[1]
            zi = value[2]

            # Scatter plot
            ax.scatter(xi, yi, zi, c='r', s=20 + 20 * G.degree(key), edgecolors='k', alpha=0.7)

        # Loop on the list of edges to get the x,y,z, coordinates of the connected nodes
        # Those two points are the extrema of the line to be plotted
        for i, j in enumerate(G.edges()):
            x = np.array((pos[j[0]][0], pos[j[1]][0]))
            y = np.array((pos[j[0]][1], pos[j[1]][1]))
            z = np.array((pos[j[0]][2], pos[j[1]][2]))

            # Plot the connecting lines
            ax.plot(x, y, z, c='black', alpha=0.5)

    # Set the initial view
    ax.view_init(30, angle)

    # Hide the axes
    ax.set_axis_off()

    plt.show()
    return

def torchgeom_plot_3D(graph, angle, save=False):
    # Get node positions
    pos = {int(node[0].item()): (node[1].item(), node[2].item(), node[3].item()) for node in graph.x}

    # 3D network plot
    with plt.style.context(('ggplot')):

        fig = plt.figure(figsize=(10, 7))
        ax = Axes3D(fig)

        # Loop on the pos dictionary to extract the x,y,z coordinates of each node
        for key, value in pos.items():
            xi = value[0]
            yi = value[1]
            zi = value[2]

            # Scatter plot
            ax.scatter(xi, yi, zi, c='r', s=20 + 20*1, edgecolors='k', alpha=0.7)

        # Loop on the list of edges to get the x,y,z, coordinates of the connected nodes
        # Those two points are the extrema of the line to be plotted
        for e in range(len(graph.edge_index[0])):
            i = int(graph.edge_index[0][e].item())
            j = int(graph.edge_index[1][e].item())
            x = np.array((pos[i][0], pos[j][0]))
            y = np.array((pos[i][1], pos[j][1]))
            z = np.array((pos[i][2], pos[j][2]))

            # Plot the connecting lines
            ax.plot(x, y, z, c='black', alpha=0.5)

    # Set the initial view
    ax.view_init(30, angle)

    # Hide the axes
    ax.set_axis_off()

    plt.show()
    return

def plot_nx_3D(g):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    n = 100

    xs = [node[1]['x'] for node in g.nodes.data()]
    ys = [node[1]['y'] for node in g.nodes.data()]
    zs = [node[1]['z'] for node in g.nodes.data()]

    ax.set_xlabel('X Label')
    ax.set_ylabel('Y Label')
    ax.set_zlabel('Z Label')
    ax.scatter(xs, ys, zs, marker='o')

    plt.show()

def parse_bonds(filename):
    with open(filename, 'r') as f:
        f_text = f.read()
        bonds = np.array(re.sub(r'\s+', ' ', re.search(r'@<TRIPOS>BOND([a-z0-9\s]*)@', f_text).group(1)).split()).reshape(
            (-1, 4))
        df_bonds = pd.DataFrame(bonds, columns=['bond_id', 'atom1', 'atom2', 'bond_type'])
        df_bonds.set_index(['bond_id'], inplace=True)
        return df_bonds

allowable_atoms = ['C', 'N', 'O', 'S', 'F', 'Si', 'P', 'Cl', 'Br', 'Mg', 'Na', 'Ca',
                   'Fe', 'As', 'Al', 'I', 'B', 'V', 'K', 'Tl', 'Yb', 'Sb', 'Sn',
                   'Ag', 'Pd', 'Co', 'Se', 'Ti', 'Zn', 'H', 'Li', 'Ge', 'Cu', 'Au',
                   'Ni', 'Cd', 'In', 'Mn', 'Zr', 'Cr', 'Pt', 'Hg', 'Pb', 'H', 'Du', 'LP']
allowable_bonds = ['ar', '1', '2', '3', 'am', 'du', 'un', 'nc']

def featurise_ligand_atoms(atoms_df):
    atoms_df['atom_id'] = atoms_df['atom_id'] - 1
    atoms_df.loc[:, 'atom_type'] = atoms_df["atom_type"].apply(lambda a: allowable_atoms.index(a.split('.')[0]))

def featurise_ligand_bonds(bonds_df):
    bonds_df.loc[:, 'atom1'] = bonds_df["atom1"].apply(lambda s: int(s)-1)
    bonds_df.loc[:, 'atom2'] = bonds_df["atom2"].apply(lambda s: int(s)-1)
    bonds_df.loc[:, 'bond_type'] = bonds_df["bond_type"].apply(lambda b: allowable_bonds.index(b))

def mol2_file_to_networkx(path):
    bonds = parse_bonds(path)
    atoms = PandasMol2().read_mol2(path).df

    g = nx.Graph()

    for index, row in atoms.iterrows():
        g.add_node(int(row['atom_id']), x=row['x'], y=row['y'], z=row['z'], atom_type=row['atom_type'])

    for index, row in bonds.iterrows():
        g.add_edge(int(row['atom1']), int(row['atom2']), bond_type=row['bond_type'])

    print(bonds.head(3))
    print(atoms.head(3))

    return g

def mol2_file_to_torch_geometric(path):
    bonds = parse_bonds(path)
    atoms = PandasMol2().read_mol2(path).df
    featurise_ligand_atoms(atoms)
    featurise_ligand_bonds(bonds)

    bonds_other_direction = bonds.copy(deep=True)
    bonds_other_direction = bonds_other_direction.rename(columns={'atom1': 'atom2', 'atom2': 'atom1'})
    bonds = pd.concat([bonds, bonds_other_direction])

    # Get node features from DGL graph and concatenate them
    features = [torch.tensor([float(i) for i in atoms[feat].tolist()]) for feat in ['atom_id', 'x', 'y', 'z', 'atom_type']]
    features = [
        f.unsqueeze(dim=1) if len(f.shape) == 1 else f for f in features
    ]
    node_features = torch.cat(features, dim=1)

    # Get edge features from DGL graph and concatenate them
    edge_feats = [torch.tensor([float(edge) for edge in bonds[feat].tolist()]) for feat in ['bond_type']]
    edge_feats = [
        e.unsqueeze(dim=1) if len(e.shape) == 1 else e for e in edge_feats
    ]
    edge_feats = torch.cat(edge_feats, dim=1)

    # Create the Torch Geometric graph
    geom_graph = data.Data(
        x=node_features,
        edge_index=torch.tensor([bonds['atom1'].tolist(), bonds['atom2'].tolist()]).contiguous(),
        edge_attr=edge_feats,
    )
    return geom_graph

def mol2_file_to_dgl(path):
    nxg = mol2_file_to_networkx(path)
    g = dgl.DGLGraph()
    g.from_networkx(nxg, node_attrs=['x', 'y', 'z', 'atom_type'], edge_attrs=['bond_type'])
    return g

def pdb_file_to_torch_geometric(path):
    return ''

def prepare_data(pdb_dir):
    ligands = []
    pockets = []
    for _, dirs, _ in os.walk(pdb_dir):
        i = 0
        total = len(dirs)
        for dir in dirs:
            i += 1
            if i == 2:
                break
            print('(' + str(int(100*i/total)) + '%) Processing ' + dir)
            for path, _, protein_files in os.walk(pdb_dir + os.sep + dir):
                for file in protein_files:
                    full_path = path + os.sep + file
                    if file.endswith('ligand.mol2'):
                        g = mol2_file_to_torch_geometric(full_path)
                        torchgeom_plot_3D(g, 90)
                        ligands.append(g)
                        print(g.edge_index)
                        print(g.edge_attr)
                        print(g.num_nodes)
                        print(g.num_edges)
                    elif file.endswith('pocket.pdb'):
                        pockets.append(pdb_file_to_torch_geometric(full_path))

    return ligands, pockets
prepare_data(pdb_dir)