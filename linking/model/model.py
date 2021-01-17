import torch

from linking.config.config import Config
from linking.layers.gcn_encoders import GCNEncoder
from linking.layers.linear_encoders import LinearAtomLabelClassifier
from data.torchgeom_pdb_loader import to_one_hot, allowable_atoms
import numpy as np

class Node:
    def __init(self, latent, label):
        self.latent = latent
        self.label = label

class MoleculeGenerator(torch.nn.Module):
    def __init__(self, pocket_encoder: GCNEncoder, ligand_encoder: GCNEncoder, f: LinearAtomLabelClassifier, g, h, config: Config):
        super(MoleculeGenerator, self).__init__()
        self.pocket_encoder = pocket_encoder
        self.ligand_encoder = ligand_encoder
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
        # the latent node representations, recomputed at every node addition
        x_latent = []
        # labels are one-hot atom types
        x_label = []
        edge_index = [[],[]]
        edge_attr = []

        # initialise graph with a carbon node
        lab = to_one_hot("C", allowable_atoms)
        x_label.append(lab)

        # run gnn on graph
        x_label_train = torch.tensor(x_label, dtype=torch.float32)
        x_label_train = torch.nn.functional.pad(x_label_train, (self.config.ligand_encoder_in_channels-x_label_train.shape[1], 0), value=0)

        # initial path to itself
        edge_index_train = torch.tensor([[0],[0]], dtype=torch.long)
        x_latent = self.ligand_encoder(x_label_train, edge_index_train)
        graph_latent = torch.mean(x_latent, dim=0)

        # create a queue of nodes Q (does Q need to be a tensor? probably not because its only part of control flow)
        Q = []
        # append the index of the node
        Q.append(0)

        # N = sample num_max_generated_atoms expansion nodes from N(0, I) - num of dimensions is ligand_encoder_out_channels
        # this will be just N vectors
        N = np.random.multivariate_normal(np.zeros(self.config.ligand_encoder_out_channels),
                                             np.identity(self.config.ligand_encoder_out_channels),
                                             (self.config.num_max_generated_atoms))
        # for each of these compute the atom labels using a linear encoder f(z \in N, z_pocket)
        # pick atom type using linear classifier f(z_pocket, z_ligand?, z \in N) so we have structural information
        N_train = torch.tensor(N, dtype=torch.float32)
        N_lab = self.f(N_train)

        while len(Q) != 0:
            # pop the node on top of the queue and consider all chemically valid edges (u,v)
            # is it differentiable to select only chemically valid edges?
            u = Q.pop(0)
            u_z = x_latent[u]
            u_l = x_label[u]
            z_graph_vectors = self.pocket_encoder(torch.tensor(x_p), torch.tensor(edge_index_p))
            z_graph = torch.mean(z_graph_vectors, dim=0)

            # for top of queue u and each atom v in N compute
            # make edge feature phi = [t, z_pocket, z_ligand, z_u, l_u, z_v, l_v, G]
            # add v of the highest rated edge as rated by a linear layer g(phi) -> softmax
            # z_v -> x_latent, l_v -> x_labels, l(u,v) -> edge_index
            # add edge type into edge_attr as classified by a layer h(phi) -> {single, double, tripple, no edge}
            # remove v from N
            # add v to Q
            # compute z_graph = gnn(x_label, edge_index) of the graph

        return x_label, edge_index, edge_attr

    def parameters(self):
       return [
            dict(params=self.pocket_encoder.conv1.parameters(), weight_decay=5e-4),
            dict(params=self.pocket_encoder.conv2.parameters(), weight_decay=0),
            dict(params=self.ligand_encoder.conv1.parameters(), weight_decay=5e-4),
            dict(params=self.ligand_encoder.conv2.parameters(), weight_decay=0)
        ]

class SimpleModel(torch.nn.Module):
    def __init__(self, pocket_encoder: GCNEncoder, ligand_encoder: GCNEncoder):
        super(SimpleModel, self).__init__()
        self.pocket_encoder = pocket_encoder
        self.ligand_encoder = ligand_encoder
        self.decoder = None

    def forward(self, data_pocket, data_ligand):
        x_p, edge_index_p, edge_wight_p = data_pocket.x, data_pocket.edge_index, data_pocket.edge_attr
        x_l, edge_index_l, edge_weight_l = data_ligand.x, data_ligand.edge_index, data_ligand.edge_attr

        z_pocket = self.pocket_encoder(x_p, edge_index_p)
        z_pocket = torch.mean(z_pocket, dim=0)
        z_ligand = self.ligand_encoder(x_l, edge_index_l)
        z_ligand = torch.mean(z_ligand, dim=0)
        return z_pocket, z_ligand

    def parameters(self):
       return [
            dict(params=self.pocket_encoder.conv1.parameters(), weight_decay=5e-4),
            dict(params=self.pocket_encoder.conv2.parameters(), weight_decay=0),
            dict(params=self.ligand_encoder.conv1.parameters(), weight_decay=5e-4),
            dict(params=self.ligand_encoder.conv2.parameters(), weight_decay=0)
        ]