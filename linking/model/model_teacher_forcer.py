import torch

from linking.config.config import Config
from linking.layers.gcn_encoders import GCNEncoder
from linking.layers.linear_encoders import LinearAtomClassifier, LinearEdgeSelector, LinearEdgeClassifier, LinearScorePredictor
from data.torchgeom_pdb_loader import to_one_hot, allowable_atoms

class TeacherForcer(torch.nn.Module):
    def __init__(self, pocket_encoder: GCNEncoder, ligand_encoder: GCNEncoder, graph_encoder: GCNEncoder, f: LinearAtomClassifier, g: LinearEdgeSelector, h: LinearEdgeClassifier, config: Config):
        super(TeacherForcer, self).__init__()
        self.pocket_encoder = pocket_encoder
        self.ligand_encoder = ligand_encoder
        self.graph_encoder = graph_encoder
        self.f = f
        self.g = g
        self.h = h
        self.decoder = None
        self.config = config
        self.valency_map = {'C': 4, 'F': 1, 'N': 3, 'Cl': 1, 'O': 2, 'I': 1, 'P': 5, 'Br': 1, 'S': 6, 'H': 1, 'Stop': 100}

    def to_atom(self, t):
        return allowable_atoms[int(torch.dot(t, torch.tensor(range(t.size()[0]), dtype=torch.float)).item())]

    def to_bond(self, t):
        t_s = t.squeeze()
        return [1, 2, 3][int(torch.dot(t_s, torch.tensor(range(t_s.size()[0]), dtype=torch.float)).item())]

    def calculate_node_mask_list(self, valencies, u, closed, edges):
        mask_list = [(
                 (float('-inf') if i == u else 0.0) + # no self edges allowed
                 (float('-inf') if valencies[i] <= 0 else 0.0) + # valency of new node kept
                 (float('-inf') if valencies[u] <= 0 else 0.0) + # valency of old node kept
                 (float('-inf') if i in closed else 0.0) + # new node is not closed
                 (float('-inf') if (u, i) in edges else 0.0))
                 for i in range(len(valencies))]
        return mask_list

    def calculate_node_mask(self, valencies, u, closed, edges):
        mask_list = self.calculate_node_mask_list(valencies, u, closed, edges)
        t = torch.stack([torch.tensor(0., dtype=torch.float) if x == 0.0 else torch.tensor(float('-inf'), dtype=torch.float) for x in mask_list])
        return torch.unsqueeze(t, 1)

    def calculate_bond_mask(self, valencies, u, closed, edges):
        node_mask = self.calculate_node_mask_list(valencies, u, closed, edges)
        mask_list = [
            torch.stack([(
            node_mask[i] +
            torch.tensor(float('-inf'), dtype=torch.float) if valencies[i] < b else torch.tensor(0.0,  dtype=torch.float))
            for b in (1, 2, 3)])
            for i in range(len(valencies))]
        return torch.stack(mask_list)

    def forward(self, data_pocket, data_ligand):
        # these contain both coordinates and atom types, can be separated which does not need to be differentiable
        x_p, edge_index_p, edge_wight_p = data_pocket.x, data_pocket.edge_index, data_pocket.edge_attr
        x_l, edge_index_l, edge_weight_l, bfs_index, bfs_attr = data_ligand.x, data_ligand.edge_index, data_ligand.edge_attr, list(data_ligand.bfs_index), data_ligand.bfs_attr

        log_prob = torch.tensor(0.0,  dtype=torch.float)

        # time
        t = torch.tensor(0)

        z_pocket_atoms = self.pocket_encoder(x_p, edge_index_p)
        z_pocket = torch.mean(z_pocket_atoms, dim=0)

        z_ligand_atoms = self.ligand_encoder(x_l, edge_index_l)
        z_ligand = torch.mean(z_ligand_atoms, dim=0)
        x_latent = z_ligand_atoms

        # need to train f decoding
        x_label = self.f(x_latent)
        x_l = x_l[:, 4:]
        # calculate loss from label prediction
        log_prob += torch.sum(torch.log(torch.sum(x_label*x_l, dim=1)))

        edge_index = torch.tensor([[], []], dtype=torch.long)  # edge indices
        edge_attr = torch.tensor([], dtype=torch.float)  # edge types

        # calculate z_graph
        # initial path to itself to be able to run GNN
        edge_index_init = torch.tensor([[bfs_index[0][0]], [bfs_index[0][1]]], dtype=torch.long)
        x_latent = self.graph_encoder(x_label, edge_index_init)
        z_g = torch.mean(x_latent, dim=0)

        # append a stop node
        z_stop = torch.zeros(self.config.graph_encoder_out_channels)
        l_stop = torch.tensor(to_one_hot('Stop', allowable_atoms))
        x_latent = torch.cat([x_latent, torch.unsqueeze(z_stop, 0)], 0)
        x_label = torch.cat([x_label, torch.unsqueeze(l_stop, 0)], 0)
        i_stop = torch.tensor(x_label.size()[0] - 1)  # index of stop node

        last_node = bfs_index[0][0]
        while not len(bfs_index) == 0:
            # pop the node on top of the queue and consider all chemically valid edges (u,v)
            u, v = bfs_index[0]
            # need to select stop node?
            stopping = last_node.item() != u.item()
            if stopping:
                temp = u
                u = last_node
                last_node = temp
                v = i_stop
            else:
                bfs_index.pop(0)

            z_u = torch.index_select(x_latent, 0, u)
            l_u = torch.index_select(x_label, 0, u)

            # for top of queue u and each atom v in nodes compute
            # edge feature phi = [t, z_pocket, z_ligand, z_u, l_u, z_v, l_v, G]
            num_nodes = len(x_latent)
            phi = torch.cat((
                torch.tensor([t]).unsqueeze(0).repeat(num_nodes, 1),
                z_pocket.unsqueeze(0).repeat(num_nodes, 1),
                z_ligand.unsqueeze(0).repeat(num_nodes, 1),
                z_u.repeat(num_nodes, 1),
                l_u.repeat(num_nodes, 1),
                x_latent,
                x_label,
                z_g.unsqueeze(0).repeat(num_nodes, 1)
            ), dim=1)

            # prob of selecting edge from u to v
            prob_v = torch.log(torch.squeeze(self.g(phi, mask=None, gumbel=False)[v]))
            log_prob += prob_v

            # l(u,v) -> edge_index
            edge_index = torch.cat([
                edge_index,
                torch.tensor([u, v]).unsqueeze(1),
                torch.tensor([v, u]).unsqueeze(1)
            ], dim=1)

            ''' edge label probability
            # add edge type into edge_attr as classified by a layer h(phi) -> {single, double, tripple}
            edge_types = self.h(phi, self.calculate_bond_mask(valencies, int(u.item()), closed_nodes, edges))
            edge_type = torch.index_select(edge_types, 0, v)
            edge_attr = torch.cat([edge_attr, edge_type.unsqueeze(0), edge_type.unsqueeze(0)])

            # update valencies
            bond_valency = self.to_bond(edge_type.unsqueeze(0))
            

            # update edges
            edges.append((int(u.item()), int(v.item())))
            edges.append((int(v.item()), int(u.item())))
            '''

            # compute z_graph = gnn(x_label, edge_index) of the graph
            x_latent = self.graph_encoder(x_label, edge_index)
            z_g = torch.mean(x_latent, dim=0)

            t = t + torch.tensor(1)

        return log_prob

    def forward_generate(self, data_pocket, data_ligand):
        # these contain both coordinates and atom types, can be separated which does not need to be differentiable
        x_p, edge_index_p, edge_wight_p = data_pocket.x, data_pocket.edge_index, data_pocket.edge_attr
        x_l, edge_index_l, edge_weight_l, bfs = data_ligand.x, data_ligand.edge_index, data_ligand.edge_attr, data_ligand.bfs

        valencies = [self.valency_map[self.to_atom(x_l[i][4:])] for i in range(x_l.size()[0])]
        valencies.append(100)
        edges = []
        closed_nodes = []
        # time
        t = torch.tensor(0)

        z_pocket_atoms = self.pocket_encoder(x_p, edge_index_p)
        z_pocket = torch.mean(z_pocket_atoms, dim=0)

        z_ligand_atoms = self.ligand_encoder(x_l, edge_index_l)
        z_ligand = torch.mean(z_ligand_atoms, dim=0)
        x_latent = z_ligand_atoms

        # Initialise a graph that will be built in the loop
        x_label = self.f(x_latent)  # labels are softmax (not exactly one-hot atom types)
        edge_index = torch.tensor([[], []], dtype=torch.long)  # edge indices
        edge_attr = torch.tensor([], dtype=torch.float)  # edge types

        # create a queue of nodes Q
        Q = []
        # append the index of the node
        Q.append(torch.tensor(0))

        # calculate z_graph
        # initial path to itself to be able to run GNN
        edge_index_init = torch.tensor([[0], [0]], dtype=torch.long)
        x_latent = self.graph_encoder(x_label, edge_index_init)
        z_g = torch.mean(x_latent, dim=0)

        # append a stop node
        z_stop = torch.zeros(self.config.graph_encoder_out_channels)
        l_stop = torch.zeros(self.config.num_allowable_atoms)
        x_latent = torch.cat([x_latent, torch.unsqueeze(z_stop, 0)], 0)
        x_label = torch.cat([x_label, torch.unsqueeze(l_stop, 0)], 0)
        i_stop = x_label.size()[0]-1  # index of stop node

        while len(Q) != 0:
            # pop the node on top of the queue and consider all chemically valid edges (u,v)
            u = Q[0]
            z_u = torch.index_select(x_latent, 0, u)
            l_u = torch.index_select(x_label, 0, u)

            # for top of queue u and each atom v in nodes compute
            # edge feature phi = [t, z_pocket, z_ligand, z_u, l_u, z_v, l_v, G]
            num_nodes = len(x_latent)
            phi = torch.cat((
                torch.tensor([t]).unsqueeze(0).repeat(num_nodes, 1),
                z_pocket.unsqueeze(0).repeat(num_nodes, 1),
                z_ligand.unsqueeze(0).repeat(num_nodes, 1),
                z_u.repeat(num_nodes, 1),
                l_u.repeat(num_nodes, 1),
                x_latent,
                x_label,
                z_g.unsqueeze(0).repeat(num_nodes, 1)
            ), dim=1)

            # select node v to add
            # add v of the highest rated edge as rated by a linear layer g(phi) -> softmax
            v = self.g(phi, self.calculate_node_mask(valencies, int(u.item()), closed_nodes, edges))

            # if terminator node was selected or we can't add more edges to node, remove u from Q
            if v.item() == i_stop or valencies[int(u.item())] <= 0:
                closed_nodes.append(Q.pop(0))
                continue
            else:
                # add index of v to Q
                Q.append(v)
                # l(u,v) -> edge_index
                edge_index = torch.cat([
                    edge_index,
                    torch.tensor([u, v]).unsqueeze(1),
                    torch.tensor([v, u]).unsqueeze(1)
                ], dim=1)

                # add edge type into edge_attr as classified by a layer h(phi) -> {single, double, tripple}
                edge_types = self.h(phi, self.calculate_bond_mask(valencies, int(u.item()), closed_nodes, edges))
                edge_type = torch.index_select(edge_types, 0, v)
                edge_attr = torch.cat([edge_attr, edge_type.unsqueeze(0), edge_type.unsqueeze(0)])

                # update valencies
                bond_valency = self.to_bond(edge_type.unsqueeze(0))
                valencies[int(u.item())] -= bond_valency
                valencies[int(v.item())] -= bond_valency

                # update edges
                edges.append((int(u.item()), int(v.item())))
                edges.append((int(v.item()), int(u.item())))

                # compute z_graph = gnn(x_label, edge_index) of the graph
                x_latent = self.graph_encoder(x_label, edge_index)
                z_g = torch.mean(x_latent, dim=0)

            t = t + torch.tensor(1)

        return x_label, edge_index_init if len(edges) == 0 else edge_index, torch.tensor([]) if len(edges) == 0 else edge_attr

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