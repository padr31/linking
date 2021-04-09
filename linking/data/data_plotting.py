import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from rdkit import Chem
from rdkit.Chem import rdDepictor
from rdkit.Chem.Draw import rdMolDraw2D
import numpy as np
import torch
import networkx as nx
from torch_geometric.utils.convert import to_networkx

from linking.data.data_util import to_atom


def torchgeom_plot(graph):
    def to_atom(t):
        return ['C', 'F', 'N', 'Cl', 'O', 'I', 'P', 'Br', 'S', 'H', 'Stop'][
            int(torch.dot(t, torch.tensor(range(t.size()[0]), dtype=torch.float)).item())
        ]

    G = to_networkx(graph)
    # Need to create a layout when doing
    # separate calls to draw nodes and edges
    pos = nx.spring_layout(G)
    G.remove_nodes_from(list(nx.isolates(G)))
    label_dict = {}
    for n in nx.nodes(G):
        label_dict[n] = to_atom(graph.x[n])
    nx.draw(G, pos, cmap=plt.get_cmap('jet'), node_size=300, with_labels = True, labels=label_dict)
    #nx.draw_networkx_labels(G, pos)
    nx.draw_networkx_edges(G, pos)
    plt.show()

def networkx_plot_3D(G, angle):
    # Get node positions
    pos = {
        node[0]: (node[1]["x"], node[1]["y"], node[1]["z"]) for node in G.nodes.data()
    }

    # 3D network plot
    with plt.style.context(("ggplot")):

        fig = plt.figure(figsize=(10, 7))
        ax = Axes3D(fig)

        # Loop on the pos dictionary to extract the x,y,z coordinates of each node
        for key, value in pos.items():
            xi = value[0]
            yi = value[1]
            zi = value[2]

            # Scatter plot
            ax.scatter(
                xi, yi, zi, c="r", s=20 + 20 * G.degree(key), edgecolors="k", alpha=0.7
            )

        # Loop on the list of edges to get the x,y,z, coordinates of the connected nodes
        # Those two points are the extrema of the line to be plotted
        for i, j in enumerate(G.edges()):
            x = np.array((pos[j[0]][0], pos[j[1]][0]))
            y = np.array((pos[j[0]][1], pos[j[1]][1]))
            z = np.array((pos[j[0]][2], pos[j[1]][2]))

            # Plot the connecting lines
            ax.plot(x, y, z, c="black", alpha=0.5)

    # Set the initial view
    ax.view_init(30, angle)

    # Hide the axes
    ax.set_axis_off()

    plt.show()
    return


def pos_plot_3D(pos, edge_index, atoms, angle, save_name=None):
    # Get node positions
    pos = {
        int(id): (node[0].item(), node[1].item(), node[2].item())
        for id, node in enumerate(pos)
    }

    # 3D network plot
    with plt.style.context(("ggplot")):

        fig = plt.figure(figsize=(10, 7))
        ax = Axes3D(fig)

        # Loop on the pos dictionary to extract the x,y,z coordinates of each node
        for key, value in pos.items():
            if to_atom(atoms[key]) == 'H':
                continue
            xi = value[0]
            yi = value[1]
            zi = value[2]

            # Scatter plot
            ax.scatter(xi, yi, zi, c="r", s=20 + 20 * 1, edgecolors="k", alpha=0.7)

        # Loop on the list of edges to get the x,y,z, coordinates of the connected nodes
        # Those two points are the extrema of the line to be plotted
        for e in range(len(edge_index[0])):
            i = int(edge_index[0][e].item())
            j = int(edge_index[1][e].item())
            if to_atom(atoms[i]) == 'H' or to_atom(atoms[j]) == 'H':
                continue
            x = np.array((pos[i][0], pos[j][0]))
            y = np.array((pos[i][1], pos[j][1]))
            z = np.array((pos[i][2], pos[j][2]))

            # Plot the connecting lines
            ax.plot(x, y, z, c="black", alpha=0.5)

    # Set the initial view
    ax.view_init(30, angle)

    # Hide the axes
    ax.set_axis_off()

    if not save_name is None:
        plt.savefig(save_name)
    else:
        plt.show()
    return

def torchgeom_plot_3D(graph, angle):
    # Get node positions
    pos = {
        int(node[0].item()): (node[1].item(), node[2].item(), node[3].item())
        for node in graph.x
    }

    # 3D network plot
    with plt.style.context(("ggplot")):

        fig = plt.figure(figsize=(10, 7))
        ax = Axes3D(fig)

        # Loop on the pos dictionary to extract the x,y,z coordinates of each node
        for key, value in pos.items():
            xi = value[0]
            yi = value[1]
            zi = value[2]

            # Scatter plot
            ax.scatter(xi, yi, zi, c="r", s=20 + 20 * 1, edgecolors="k", alpha=0.7)

        # Loop on the list of edges to get the x,y,z, coordinates of the connected nodes
        # Those two points are the extrema of the line to be plotted
        for e in range(len(graph.edge_index[0])):
            i = int(graph.edge_index[0][e].item())
            j = int(graph.edge_index[1][e].item())
            x = np.array((pos[i][0], pos[j][0]))
            y = np.array((pos[i][1], pos[j][1]))
            z = np.array((pos[i][2], pos[j][2]))

            # Plot the connecting lines
            ax.plot(x, y, z, c="black", alpha=0.5)

    # Set the initial view
    ax.view_init(30, angle)

    # Hide the axes
    ax.set_axis_off()

    plt.show()
    return

def plot_nx_3D(g):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")

    n = 100

    xs = [node[1]["x"] for node in g.nodes.data()]
    ys = [node[1]["y"] for node in g.nodes.data()]
    zs = [node[1]["z"] for node in g.nodes.data()]

    ax.set_xlabel("X Label")
    ax.set_ylabel("Y Label")
    ax.set_zlabel("Z Label")
    ax.scatter(xs, ys, zs, marker="o")

    plt.show()

'''
Usage
mol = Chem.MolFromPDBFile(
    "/Users/padr/repos/linking/datasets/raw/refined-set/1g7v/1g7v_pocket.pdb"
)
with open("out.svg", "w") as svg_file:
    svg_file.write(mol_to_svg(mol))
'''
def mol_to_svg(mol, molSize=(300, 300), kekulize=True):
    mc = Chem.Mol(mol.ToBinary())
    if kekulize:
        try:
            Chem.Kekulize(mc)
        except:
            mc = Chem.Mol(mol.ToBinary())
    if not mc.GetNumConformers():
        rdDepictor.Compute2DCoords(mc)
    drawer = rdMolDraw2D.MolDraw2DSVG(molSize[0], molSize[1])
    drawer.DrawMolecule(mc)
    drawer.FinishDrawing()
    svg = drawer.GetDrawingText()
    return svg.replace("svg:", "")
