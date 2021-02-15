import torch

from linking.config.config import Config
from linking.layers.gcn_encoders import GCNEncoder, VariationalGCNEncoder
from linking.layers.linear_encoders import LinearAtomClassifier, LinearEdgeSelector, LinearEdgeClassifier
from data.torchgeom_pdb_loader import to_one_hot, allowable_atoms

class TeacherForcer(torch.nn.Module):
    def __init__(self, pocket_encoder: GCNEncoder, ligand_encoder: GCNEncoder, g_dec: GCNEncoder, f: LinearAtomClassifier, g: LinearEdgeSelector, h: LinearEdgeClassifier, config: Config):
        super(TeacherForcer, self).__init__()
        self.pocket_encoder = pocket_encoder
        self.ligand_encoder = ligand_encoder
        self.g_dec = g_dec
        self.f = f
        self.g = g
        self.h = h
        self.decoder = None
        self.config = config
        self.valency_map = {'C': 4, 'F': 1, 'N': 3, 'Cl': 1, 'O': 2, 'I': 1, 'P': 5, 'Br': 1, 'S': 6, 'H': 1, 'Stop': 1000}

    def to_atom(self, t):
        return allowable_atoms[int(torch.dot(t, torch.tensor(range(t.size()[0]), dtype=torch.float)).item())]

    def to_bond(self, t):
        t_s = t.squeeze()
        return [1, 2, 3][int(torch.dot(t_s, torch.tensor(range(t_s.size()[0]), dtype=torch.float)).item())]

    def calculate_node_mask_list(self, valencies, u, closed, edges, unmask=None):
        mask_list = [(
                 (float('-inf') if i == u else 0.0) + # no self edges allowed
                 (float('-inf') if valencies[i] <= 0 else 0.0) + # valency of new node kept
                 (float('-inf') if valencies[u] <= 0 else 0.0) + # valency of old node kept
                 (float('-inf') if i in closed else 0.0) + # new node is not closed
                 (float('-inf') if (u, i) in edges else 0.0))
                 for i in range(len(valencies))]
        if not unmask is None:
            for i in unmask:
                mask_list[i] = 0
        return mask_list

    def calculate_node_mask(self, valencies, u, closed, edges, unmask=None):
        mask_list = self.calculate_node_mask_list(valencies, u, closed, edges, unmask)
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

    def forward_train(self, data_pocket, data_ligand):
        # init data, prob, time, and generated molecule
        x_p, edge_index_p, edge_wight_p = data_pocket.x, data_pocket.edge_index, data_pocket.edge_attr
        x_l, edge_index_l, edge_weight_l, bfs_index, bfs_attr = data_ligand.x, data_ligand.edge_index, data_ligand.edge_attr, list(data_ligand.bfs_index), list(data_ligand.bfs_attr)

        log_prob = torch.tensor(0.0,  dtype=torch.float)

        t = torch.tensor(0)

        edge_index = torch.tensor([[], []], dtype=torch.long)  # edge indices
        edge_attr = torch.tensor([], dtype=torch.float)  # edge types

        # init valencies
        valencies = [self.valency_map[self.to_atom(x_l[i][4:])] for i in range(x_l.size()[0])]
        # for stop node
        valencies.append(100)
        edges = []
        closed_nodes = []

        # init latent pocket and ligand representations
        z_pocket_atoms = self.pocket_encoder(x_p, edge_index_p)
        z_pocket = torch.mean(z_pocket_atoms, dim=0)

        z_ligand_atoms = self.ligand_encoder(x_l, edge_index_l)
        # probs not needed
        z_ligand = torch.mean(z_ligand_atoms, dim=0)
        z_v = z_ligand_atoms

        # decode molecule from latent train sample and calculate decoding loss
        x_label = self.f(z_v)
        x_l = x_l[:, 4:]
        log_prob += torch.sum(torch.log(torch.sum(x_label*x_l, dim=1)))

        # init z_graph and x_latent using the first atom with a path to itself
        edge_index_init = torch.tensor([[bfs_index[0][0]], [bfs_index[0][1]]], dtype=torch.long)
        z_v = self.g_dec(x_label, edge_index_init)
        z_g = torch.mean(z_v, dim=0)

        # append a stop node to latent and label
        z_stop = torch.zeros(self.config.graph_encoder_out_channels)
        l_stop = torch.tensor(to_one_hot('Stop', allowable_atoms))
        z_v = torch.cat([z_v, torch.unsqueeze(z_stop, 0)], 0)
        x_label = torch.cat([x_label, torch.unsqueeze(l_stop, 0)], 0)
        i_stop = torch.tensor(x_label.size()[0] - 1)  # index of stop node

        # build graph according to bfs index and calculate its probability|model
        last_node = bfs_index[0][0]
        while not len(bfs_index) == 0:
            assert len(bfs_index) == len(bfs_attr)
            # pop the node on top of the queue and consider all chemically valid edges (u,v)
            u, v = bfs_index[0]
            # select stop node if we finished adding edges to the last_node, i.e. start adding to u != last_node
            stopping = last_node.item() != u.item()
            if stopping: # TODO handle this sooner and don't let it be selected
                temp = u
                u = last_node
                last_node = temp
                v = i_stop
                closed_nodes.append(u)
            else:
                bfs_index.pop(0)

            # select latent and label of u
            z_u = torch.index_select(z_v, 0, u)
            l_u = torch.index_select(x_label, 0, u)

            # for top of queue u and each atom v in nodes compute edge likelihood
            # edge feature phi = [t, z_pocket, z_ligand, z_u, l_u, z_v, l_v, G]
            num_nodes = len(z_v)
            phi = torch.cat((
                torch.tensor([t]).unsqueeze(0).repeat(num_nodes, 1),
                z_pocket.unsqueeze(0).repeat(num_nodes, 1),
                z_ligand.unsqueeze(0).repeat(num_nodes, 1),
                z_u.repeat(num_nodes, 1),
                l_u.repeat(num_nodes, 1),
                z_v,
                x_label,
                z_g.unsqueeze(0).repeat(num_nodes, 1)
            ), dim=1)

            # select prob of selecting edge from u to v
            prob_v = torch.log(torch.squeeze(self.g(phi, mask=self.calculate_node_mask(valencies, int(u.item()), closed_nodes, edges, unmask=[v.item(), -1]), gumbel=False)[v]))
            log_prob += prob_v

            # (u,v), (v,u) -> edge_index for purpose of getting z_graph
            if not stopping: # stop node does not have edge types
                edge_index = torch.cat([
                    edge_index,
                    torch.tensor([u, v]).unsqueeze(1),
                    torch.tensor([v, u]).unsqueeze(1)
                ], dim=1)

                # add edge type into edge_attr as classified by a layer h(phi) -> {single, double, tripple}
                edge_type = self.h(phi, mask=self.calculate_bond_mask(valencies, int(u.item()), closed_nodes, edges), gumbel=False)[v]
                prob_u_v = torch.log(torch.dot(edge_type, bfs_attr[0]))
                log_prob += prob_u_v
                bfs_attr.pop(0)
                edge_attr = torch.cat([edge_attr, edge_type.unsqueeze(0), edge_type.unsqueeze(0)])

                bond_valency = self.to_bond(edge_type.unsqueeze(0))
                valencies[int(u.item())] -= bond_valency
                valencies[int(v.item())] -= bond_valency
            
                edges.append((int(u.item()), int(v.item())))
                edges.append((int(v.item()), int(u.item())))

            # compute z_graph = gnn(x_label, edge_index) of the graph
            z_v = self.g_dec(x_label, edge_index)
            z_g = torch.mean(z_v, dim=0)

            t = t + torch.tensor(1)

# TODO optimisation results in NaN weights - remake code, rewrite better, I don't know, do sth
        return log_prob

    def forward(self, data_pocket, data_ligand, generate=False):
        # Initialise data variables -------------------------
        x_p, edge_index_p, edge_wight_p = data_pocket.x, data_pocket.edge_index, data_pocket.edge_attr
        x_l, edge_index_l, edge_weight_l, bfs_index, bfs_attr = data_ligand.x, data_ligand.edge_index, data_ligand.edge_attr, list(data_ligand.bfs_index), list(data_ligand.bfs_attr)
        log_prob = torch.tensor(0.0,  dtype=torch.float)

        # Encode -------------------------
        # pocket
        z_pocket_atoms = self.pocket_encoder(x_p, edge_index_p)
        z_pocket = torch.mean(z_pocket_atoms, dim=0)

        # ligand
        z_ligand_atoms = self.ligand_encoder(x_l, edge_index_l)
        z_ligand = torch.mean(z_ligand_atoms, dim=0)
        z_v = z_ligand_atoms

        # guess ligand labels
        lab_v = self.f(z_v, gumbel=generate)  # gumbel if generative returns one-hots

        if not generate:
            log_prob += torch.sum(torch.log(torch.sum(lab_v * x_l[:, 4:], dim=1)))
            lab_v = x_l[:, 4:]

        # Initialise decoding -------------------------
        # tensor variables
        H_init = torch.mean(torch.cat([z_v, lab_v], dim=1), dim=0)
        edge_index = torch.tensor([[], []], dtype=torch.long)  # edge indices
        edge_attr = torch.tensor([], dtype=torch.float)  # edge types
        # append a stop node
        l_stop = torch.zeros(self.config.num_allowable_atoms)
        lab_v = torch.cat([lab_v, torch.unsqueeze(l_stop, 0)], 0)
        i_stop = lab_v.size()[0] - 1  # index of stop node
        # init z_graph and x_latent using the first atom with a path to itself
        edge_index_init = torch.tensor([[0], [0]], dtype=torch.long) if generate else torch.tensor([[bfs_index[0][0]], [bfs_index[0][1]]], dtype=torch.long)
        z_v = self.g_dec(lab_v, edge_index_init)
        H_t = torch.mean(torch.cat([z_v, lab_v], dim=1), dim=0)

        # non-tensor helper variables
        valencies = [self.valency_map[self.to_atom(lab_v[i])] for i in range(lab_v.size()[0])]
        edges = []
        closed_nodes = []
        time = torch.tensor(0)
        Q = [torch.tensor(0) if generate else bfs_index[0][0]]  # first atom into queue

        while len(Q) != 0:
            u = Q[0]

            # for top of queue u and each atom v in nodes compute
            # edge feature phi = [t, z_pocket, z_ligand, z_u, l_u, z_v, l_v, G]
            num_nodes = len(z_v)
            phi = torch.cat((
                torch.tensor([time]).unsqueeze(0).repeat(num_nodes, 1),
                z_pocket.unsqueeze(0).repeat(num_nodes, 1),  # pocket agg latent
                z_v[u].repeat(num_nodes, 1),  # u latent feature
                lab_v[u].repeat(num_nodes, 1),  # u label
                z_v,  # latent s
                lab_v,  # labels
                H_t.unsqueeze(0).repeat(num_nodes, 1),  # graph agg latent
                H_init.unsqueeze(0).repeat(num_nodes, 1)  # graph initial agg latent
            ), dim=1)

            # select node v to add
            # add v of the highest rated edge as rated by a linear layer g(phi) -> softmax
            if generate:
                v = self.g(phi, self.calculate_node_mask(valencies, int(u.item()), closed_nodes, edges), gumbel=generate)
            else:
                v = bfs_index[0][1]
                v_attr = bfs_attr[0]
                # remove front from bfs index as v was selected
                bfs_index.pop(0)
                bfs_attr.pop(0)
                p_uv = self.g(phi, self.calculate_node_mask(valencies, int(u.item()), closed_nodes, edges, unmask=[v.item(), -1]), gumbel=generate)
                # select prob of selecting edge from u to v
                prob_v = torch.log(torch.squeeze(p_uv[v]))
                log_prob += prob_v

            # stop node selected
            if v.item() == i_stop:
                closed_nodes.append(Q.pop(0))
                continue

            # add index of v to Q
            Q.append(v)
            # l(u,v) -> edge_index
            edge_index = torch.cat([
                edge_index,
                torch.tensor([u, v]).unsqueeze(1),
                torch.tensor([v, u]).unsqueeze(1)
            ], dim=1)

            # add edge type into edge_attr as classified by a layer h(phi) -> {single, double, tripple}
            # need to allow some edge types
            p_att_uv = self.h(phi, self.calculate_bond_mask(valencies, int(u.item()), closed_nodes, edges), gumbel=generate)[v]
            edge_type = p_att_uv
            if not generate:
                log_prob += torch.log(torch.dot(p_att_uv, v_attr))
                edge_type = v_attr

            edge_attr = torch.cat([edge_attr, edge_type.unsqueeze(0), edge_type.unsqueeze(0)])

            # update valencies
            bond_valency = self.to_bond(edge_type.unsqueeze(0))
            valencies[int(u.item())] -= bond_valency
            valencies[int(v.item())] -= bond_valency

            # update edges
            edges.append((int(u.item()), int(v.item())))
            edges.append((int(v.item()), int(u.item())))

            # compute z_graph = gnn(x_label, edge_index) of the graph
            z_v = self.g_dec(lab_v, edge_index)
            H_t = torch.mean(z_v, dim=0)
            time = time + torch.tensor(1)

        return lab_v, edge_index_init if len(edges) == 0 else edge_index, torch.tensor([]) if len(edges) == 0 else edge_attr if generate else log_prob

    def parameters(self):
       return [
            dict(params=self.pocket_encoder.conv1.parameters(), weight_decay=5e-4),
            dict(params=self.pocket_encoder.conv2.parameters(), weight_decay=0),
            dict(params=self.ligand_encoder.conv1.parameters(), weight_decay=5e-4),
            dict(params=self.ligand_encoder.conv2.parameters(), weight_decay=0),
            dict(params=self.g_dec.conv1.parameters(), weight_decay=5e-4),
            dict(params=self.g_dec.conv2.parameters(), weight_decay=5e-4),
            dict(params=self.f.linear.parameters()),
            dict(params=self.g.linear.parameters()),
            dict(params=self.h.linear.parameters())
        ]