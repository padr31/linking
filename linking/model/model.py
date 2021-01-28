import torch

from linking.config.config import Config
from linking.layers.gcn_encoders import GCNEncoder
from linking.layers.linear_encoders import LinearAtomClassifier, LinearEdgeSelector, LinearEdgeClassifier, LinearScorePredictor
from data.torchgeom_pdb_loader import to_one_hot, allowable_atoms
import numpy as np

class Node:
    def __init(self, latent, label):
        self.latent = latent
        self.label = label

class MoleculeGenerator(torch.nn.Module):
    def __init__(self, pocket_encoder: GCNEncoder, ligand_encoder: GCNEncoder, graph_encoder: GCNEncoder, f: LinearAtomClassifier, g: LinearEdgeSelector, h: LinearEdgeClassifier, config: Config):
        super(MoleculeGenerator, self).__init__()
        self.pocket_encoder = pocket_encoder
        self.ligand_encoder = ligand_encoder
        self.graph_encoder = graph_encoder
        self.f = f
        self.g = g
        self.h = h
        self.decoder = None
        self.config = config

    def forward(self, data_pocket, data_ligand):
        # these contain both coordinates and atom types, can be separated which does not need to be differentiable
        x_p, edge_index_p, edge_wight_p = data_pocket.x, data_pocket.edge_index, data_pocket.edge_attr
        x_l, edge_index_l, edge_weight_l = data_ligand.x, data_ligand.edge_index, data_ligand.edge_attr

        z_pocket = self.pocket_encoder(x_p, edge_index_p)
        z_pocket = torch.mean(z_pocket, dim=0)
        z_ligand = self.ligand_encoder(x_l, edge_index_l)
        z_ligand = torch.mean(z_ligand, dim=0)

        # Initialise a graph that will be built in the loop
        x_latent = []  # the latent node representations, recomputed at every node addition
        x_label = []  # labels are softmax (not exactly one-hot atom types)
        edge_index = torch.tensor([[], []], dtype=torch.long)  # edge indices
        edge_attr = []  # edge types

        # initialise graph with C atom
        x_label_init = torch.tensor(to_one_hot("C", allowable_atoms), dtype=torch.float32)
        x_label.append(x_label_init)
        # initial path to itself to be able to run GNN
        edge_index_init = torch.tensor([[0], [0]], dtype=torch.long)
        x_latent = self.graph_encoder(torch.stack(x_label), edge_index_init)
        z_g = torch.mean(x_latent, dim=0)

        # create a queue of nodes Q
        Q = []
        # append the index of the node
        Q.append(0)

        # sample expansion nodes
        nodes = [torch.normal(0, torch.ones(self.config.ligand_encoder_out_channels)) for _ in range(self.config.num_expansion_nodes)]
        # compute atom labels for expansion nodes
        nodes_labels = list(torch.unbind(self.f(torch.stack(nodes))))

        # append a stop node
        l_stop = torch.zeros(self.config.num_allowable_atoms)
        z_stop = torch.zeros(self.config.graph_encoder_out_channels)
        nodes.append(z_stop)
        nodes_labels.append(l_stop)

        t = torch.tensor(0)

        while len(Q) != 0:
            # pop the node on top of the queue and consider all chemically valid edges (u,v)

            # is it differentiable to select only chemically valid edges?
            u = Q[0]
            z_u = x_latent[u]  # use cat to get this T = torch.cat([T[0:i], T[i+1:]])
            l_u = x_label[u]

            # for top of queue u and each atom v in N compute

            # make edge feature phi = [t, z_pocket, z_ligand, z_u, l_u, z_v, l_v, G]
            num_nodes = len(nodes_labels)
            phi = torch.cat((
                torch.tensor([t]).unsqueeze(0).repeat(num_nodes, 1),
                z_pocket.unsqueeze(0).repeat(num_nodes, 1),
                z_ligand.unsqueeze(0).repeat(num_nodes, 1),
                z_u.unsqueeze(0).repeat(num_nodes, 1),
                l_u.unsqueeze(0).repeat(num_nodes, 1),
                torch.stack(nodes),
                torch.stack(nodes_labels),
                z_g.unsqueeze(0).repeat(num_nodes, 1)
            ), dim=1)

            # add v of the highest rated edge as rated by a linear layer g(phi) -> softmax
            selected_node_index = self.g(phi)

            already_selected = []
            # if terminator node was selected keep u out of Queue and don't add anything to the graph
            if selected_node_index.item() == num_nodes - 1:
                Q.pop(0)
                continue
            else:
                #l_v -> x_labels
                x_label.append(torch.index_select(torch.stack(nodes_labels), 0, selected_node_index).squeeze(0))
                # remove v from expansion nodes
                z_pop = nodes[int(selected_node_index.item())]
                l_pop = nodes_labels[int(selected_node_index.item())]
                # add index of v to Q
                Q.append(len(x_label) - 1)
                # l(u,v) -> edge_index
                edge_index = torch.cat([
                    edge_index,
                    torch.tensor([torch.tensor(u, dtype=torch.long), torch.tensor(len(x_label) - 1, dtype=torch.long)]).unsqueeze(1),
                    torch.tensor([torch.tensor(len(x_label) - 1, dtype=torch.long), torch.tensor(u, dtype=torch.long)]).unsqueeze(1)
                ], dim=1)

                # add edge type into edge_attr as classified by a layer h(phi) -> {single, double, tripple}
                phi = torch.cat((
                    torch.tensor([t]).unsqueeze(0),
                    z_pocket.unsqueeze(0),
                    z_ligand.unsqueeze(0),
                    z_u.unsqueeze(0),
                    l_u.unsqueeze(0),
                    z_pop.unsqueeze(0),
                    l_pop.unsqueeze(0),
                    z_g.unsqueeze(0)
                ), dim=1)
                edge_type = self.h(phi)

                edge_attr.append(edge_type)
                edge_attr.append(edge_type)

                # compute z_graph = gnn(x_label, edge_index) of the graph
                x_latent = self.graph_encoder(torch.stack(x_label), edge_index)
                z_g = torch.mean(x_latent, dim=0)

            t = t + torch.tensor(1)

        return torch.stack(x_label), edge_index_init if len(x_label) == 1 else edge_index, torch.tensor([]) if len(edge_attr) == 0 else torch.stack(edge_attr)

    def parameters(self):
       return [
            dict(params=self.pocket_encoder.conv1.parameters(), weight_decay=5e-4),
            dict(params=self.pocket_encoder.conv2.parameters(), weight_decay=0),
            dict(params=self.ligand_encoder.conv1.parameters(), weight_decay=5e-4),
            dict(params=self.ligand_encoder.conv2.parameters(), weight_decay=0),
            dict(params=self.graph_encoder.conv1.parameters(), weight_decay=5e-4),
            dict(params=self.graph_encoder.conv2.parameters(), weight_decay=5e-4),
            dict(params=self.f.linear.parameters()),
            dict(params=self.g.linear.parameters()),
            dict(params=self.h.linear.parameters())
        ]

class SimpleModel(torch.nn.Module):
    def __init__(self, pocket_encoder: GCNEncoder, ligand_encoder: GCNEncoder, score_predictor: LinearScorePredictor):
        super(SimpleModel, self).__init__()
        self.pocket_encoder = pocket_encoder
        self.ligand_encoder = ligand_encoder
        self.score_predictor = score_predictor
        self.decoder = None

    def forward(self, data_pocket, data_ligand):
        x_p, edge_index_p, edge_wight_p = data_pocket.x, data_pocket.edge_index, data_pocket.edge_attr
        x_l, edge_index_l, edge_weight_l = data_ligand.x, data_ligand.edge_index, data_ligand.edge_attr

        z_pocket = self.pocket_encoder(x_p, edge_index_p)
        z_pocket = torch.mean(z_pocket, dim=0)
        z_ligand = self.ligand_encoder(x_l, edge_index_l)
        z_ligand = torch.mean(z_ligand, dim=0)
        return self.score_predictor(torch.cat([z_pocket, z_ligand]))

    def parameters(self):
       return [
            dict(params=self.pocket_encoder.conv1.parameters(), weight_decay=5e-4),
            dict(params=self.pocket_encoder.conv2.parameters(), weight_decay=0),
            dict(params=self.ligand_encoder.conv1.parameters(), weight_decay=5e-4),
            dict(params=self.ligand_encoder.conv2.parameters(), weight_decay=0),
            dict(params=self.score_predictor.linear.parameters())
        ]