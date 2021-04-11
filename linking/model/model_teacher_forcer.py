import torch
from linking.config.config import Config
from linking.layers.gcn_encoders import GCNEncoder
from linking.layers.geom_encoders import Sch
from linking.layers.linear_encoders import LinearAtomClassifier, LinearEdgeSelector, LinearEdgeRowClassifier, LinearEdgeClassifier
from linking.data.data_util import to_one_hot, allowable_atoms, to_bond_valency, to_bond_index, to_atom, to_bond_length, \
    calc_position, calc_angle, calc_dihedral, allowable_angles, allowable_dihedrals, encode_angle, encode_dihedral, \
    to_angle, to_dihedral, calc_angle_p, calc_position_p, calc_dihedral_p
import numpy as np

class TeacherForcer(torch.nn.Module):
    def __init__(self,
                 pocket_encoder: GCNEncoder,
                 ligand_encoder: GCNEncoder,
                 g_dec: GCNEncoder,
                 f: LinearAtomClassifier,
                 g: LinearEdgeSelector,
                 h: LinearEdgeRowClassifier,
                 sch: Sch,
                 mlp_alpha: LinearEdgeRowClassifier,
                 mlp_theta: LinearEdgeRowClassifier,
                 config: Config,
                 device):
        super(TeacherForcer, self).__init__()

        # graph
        self.pocket_encoder = pocket_encoder
        self.ligand_encoder = ligand_encoder
        self.g_dec = g_dec  # partial mol encoder (used in decoding)
        self.f = f  # label predictor
        self.g = g  # next atom selector
        self.h = h  # edge classifier

        # coords
        self.sch = sch  # encoding
        self.mlp_alpha = mlp_alpha  # angle
        self.mlp_theta = mlp_theta  # dihedral

        self.config = config
        self.device = device
        self.valency_map = {'C': 4, 'F': 1, 'N': 3, 'Cl': 1, 'O': 2, 'I': 1, 'P': 5, 'Br': 1, 'S': 2, 'H': 1, 'Stop': 1000}

    def calculate_node_mask_list(self, valencies, u, closed_mask, adj, unmask=None):
        #  mask for node u --> v
        if valencies[u] < 1:
            m_list = torch.ones_like(valencies, device=self.device, dtype=torch.float)*float('-inf')  #  valency of u full
        else:
            m_list = torch.zeros_like(valencies, device=self.device, dtype=torch.float)
            m_list[valencies < 1] = float('-inf')  # valency of v is full
            m_list[u] = float('-inf')  # no self edges allowed
            m_list[closed_mask] = float('-inf')
            #  we don't deal with closed nodes, because if they can bear an edge we don't mind
            m_list[adj[u]] = float('-inf')  # edge already present
        if not unmask is None:
            m_list[unmask] = 0.0

        return m_list

    def calculate_node_mask(self, valencies, u, closed_mask, adj, unmask=None):
        m_list = self.calculate_node_mask_list(valencies, u, closed_mask, adj, unmask)
        return torch.unsqueeze(m_list, 1)

    def calculate_bond_mask(self, valencies, u, closed_mask, adj, unmask=None, unmask_bond=None):
        m_list = self.calculate_node_mask_list(valencies, u, closed_mask, adj, unmask)

        zero_row = torch.tensor([float('-inf'), float('-inf'), float('-inf'), float('-inf')], device=self.device, dtype=torch.float)
        one_row = torch.tensor([0., float('-inf'), float('-inf'), 0.0], device=self.device, dtype=torch.float)
        two_row = torch.tensor([0., 0., float('-inf'), 0.0], device=self.device, dtype=torch.float)
        e_mask = torch.zeros(valencies.size(0), self.config.num_allowable_bonds, device=self.device)
        e_mask[valencies <= 2] = two_row
        e_mask[valencies <= 1] = one_row
        e_mask[valencies <= 0] = zero_row
        e_mask[m_list == float('-inf')] = zero_row
        if not (unmask is None or unmask_bond is None):
            for i in unmask:
                for j in unmask_bond:
                    e_mask[i, j-1] = 0.

        return e_mask

    def calculate_bond_mask_row(self, valencies, u, v, closed_mask, adj, unmask=None, unmask_bond=None):
        m_list = self.calculate_node_mask_list(valencies, u, closed_mask, adj, unmask)

        zero_row = torch.tensor([float('-inf'), float('-inf'), float('-inf'), float('-inf')], device=self.device, dtype=torch.float)
        one_row = torch.tensor([0., float('-inf'), float('-inf'), 0.0], device=self.device, dtype=torch.float)
        two_row = torch.tensor([0., 0., float('-inf'), 0.0], device=self.device, dtype=torch.float)
        e_mask = torch.zeros(self.config.num_allowable_bonds, device=self.device)
        if min(valencies[u], valencies[v]) <= 2:
            e_mask = two_row
        if min(valencies[u], valencies[v]) <= 1:
            e_mask = one_row
        if min(valencies[u], valencies[v]) <= 0:
            e_mask = zero_row
        #  TODO should not be true (although we sometimes need to force wrong valencies) assert m_list[v] != float('-inf')
        if m_list[v] == float('-inf'):
            e_mask = zero_row

        if not (unmask_bond is None):
            for j in unmask_bond:
                e_mask[j - 1] = 0.
        return e_mask.unsqueeze(0)

    def calculate_label_mask(self, length):
        l_mask = torch.zeros(len(self.valency_map), dtype=torch.float, device=self.device)
        l_mask[-1] = float('-inf')
        return l_mask.repeat(length, 1)

    def forward(self, data_pocket, data_ligand, generate=False, coords=False):
        # Initialise data variables -------------------------
        x_p, edge_index_p, edge_attr_p = data_pocket.x, data_pocket.edge_index, data_pocket.edge_attr
        x_l, edge_index_l, edge_attr_l, bfs_index, bfs_attr = data_ligand.x, data_ligand.edge_index, data_ligand.edge_attr, data_ligand.bfs_index.clone(), data_ligand.bfs_attr.clone()
        if coords:
            x_pos_p = data_pocket.x[:, 1:4].detach()
            x_pos_l = data_ligand.x[:, 1:4].detach()
            log_prob_coords = torch.tensor(0.0,  dtype=torch.float, device=self.device)
            coord_v = torch.zeros_like(x_pos_l, dtype=torch.float, device=self.device)
            coord_v[0 if generate else bfs_index[0][0]] = x_pos_l[0 if generate else bfs_index[0][0]]
            coord_helper = CoordinateHelper()
            coord_helper.add_point(torch.tensor(0, device=self.device) if generate else bfs_index[0][0], x_pos_l[0 if generate else bfs_index[0][0]])

        log_prob = torch.tensor(0.0,  dtype=torch.float, device=self.device)

        # Encode -------------------------
        # pocket gnn encoding
        z_pocket_atoms = self.pocket_encoder(x_p, edge_index_p)
        z_pocket = torch.mean(z_pocket_atoms, dim=0)

        # ligand
        z_ligand_atoms = self.ligand_encoder(x_l, edge_index_l)
        z_ligand = torch.mean(z_ligand_atoms, dim=0)
        z_v = z_ligand_atoms

        # guess ligand labels
        lab_v = self.f(z_v, gumbel=generate, mask=self.calculate_label_mask(z_v.size()[0]))  # gumbel if generative returns one-hots

        if not generate:
            log_prob += torch.sum(torch.log(torch.sum(lab_v * x_l[:, 4:], dim=1)))
            lab_v = x_l[:, 4:]

        # just for testing
        if generate:
            lab_v = x_l[:, 4:]

        # Initialise decoding -------------------------
        # tensor variables
        H_init = torch.mean(torch.cat([z_v, lab_v], dim=1), dim=0)
        edge_index = torch.tensor([[], []], dtype=torch.long, device=self.device)  # edge indices
        edge_attr = torch.tensor([], dtype=torch.float, device=self.device)  # edge types
        # append a stop node
        l_stop = torch.tensor(to_one_hot('Stop', allowable_atoms), device=self.device)
        lab_v = torch.cat([lab_v, torch.unsqueeze(l_stop, 0)], 0)
        i_stop = torch.tensor(lab_v.size()[0] - 1, device=self.device)  # index of stop node
        # init z_graph and x_latent using the first atom with a path to itself
        edge_index_init = torch.tensor([[0], [0]], dtype=torch.long, device=self.device) if generate else torch.stack([bfs_index[0][0].unsqueeze(0), bfs_index[0][1].unsqueeze(0)])
        z_v = self.g_dec(lab_v, edge_index_init)
        H_t = torch.mean(torch.cat([z_v, lab_v], dim=1), dim=0)
        adj = torch.zeros(lab_v.shape[0], lab_v.shape[0], device=self.device, dtype=torch.bool)

        # helper variables
        valencies = torch.tensor([self.valency_map[to_atom(lab_v[i])] for i in range(lab_v.size()[0])], device=self.device)
        closed_mask = torch.zeros_like(valencies, device=self.device, dtype=torch.bool)
        time = torch.tensor(0, device=self.device)

        # non-tensor Queue, convert to queue
        Q = [torch.tensor(0, device=self.device) if generate else bfs_index[0][0]]  # first atom into queue

        while len(Q) != 0:
            if not generate:
                if bfs_index.shape[0] == 0:
                    break
            u = Q[0]

            # for top of queue u and each atom v in nodes compute
            # edge feature phi = [t, z_pocket, z_ligand, z_u, l_u, z_v, l_v, G]
            num_nodes = len(z_v)
            phi = torch.cat((
                torch.tensor([time], device=self.device).unsqueeze(0).repeat(num_nodes, 1),
                z_pocket.unsqueeze(0).repeat(num_nodes, 1),  # pocket agg latent
                z_v[u].repeat(num_nodes, 1),  # u latent feature
                lab_v[u].repeat(num_nodes, 1),  # u label
                z_v,  # latents
                lab_v,  # labels
                H_t.unsqueeze(0).repeat(num_nodes, 1),  # graph agg latent
                H_init.unsqueeze(0).repeat(num_nodes, 1)  # graph initial agg latent
            ), dim=1)

            # select node v to add
            # add v of the highest rated edge as rated by a linear layer g(phi) -> softmax
            if generate:
                v = self.g(phi, self.calculate_node_mask(valencies, u, closed_mask, adj, unmask=torch.tensor([i_stop])), gumbel=generate)
            else:
                v = bfs_index[0][1]
                v_attr = bfs_attr[0]
                # remove front from bfs index as v was selected
                bfs_index = bfs_index[1:]  # behaves like pop(0)
                bfs_attr = bfs_attr[1:]
                # TODO dont mask things during generation
                p_uv = self.g(phi, self.calculate_node_mask(valencies, u, closed_mask, adj, unmask=torch.tensor([v, i_stop])), gumbel=generate)
                # select prob of selecting edge from u to v
                prob_v = torch.log(torch.squeeze(p_uv[v]))
                log_prob += prob_v

            # stop node selected
            # TODO set stop node to last elem not -1
            if v == i_stop or v == -1:
                popped = Q.pop(0)
                closed_mask[popped] = True
                continue

            # add index of v to Q
            Q.append(v)
            # l(u,v) -> edge_index
            edge_index = torch.cat([
                edge_index,
                torch.tensor([u, v], device=self.device).unsqueeze(1),
                torch.tensor([v, u], device=self.device).unsqueeze(1)
            ], dim=1)

            # add edge type into edge_attr as classified by a layer h(phi) -> {single, double, tripple}
            # need to allow some edge types
            if generate:
                p_att_uv = self.h(phi[v].unsqueeze(0), mask=self.calculate_bond_mask_row(valencies, u, v, closed_mask, adj), gumbel=generate)
            else:
                # only mask
                p_att_uv = self.h(phi[v].unsqueeze(0), mask=self.calculate_bond_mask_row(valencies, u, v, closed_mask, adj, unmask_bond=torch.tensor([to_bond_index(v_attr.unsqueeze(0))]))
                                  , gumbel=generate)
            edge_type = p_att_uv
            if not generate:
                prob = torch.log(torch.sum(p_att_uv*v_attr))
                log_prob += prob
                edge_type = v_attr

            edge_attr = torch.cat([edge_attr, edge_type.unsqueeze(0), edge_type.unsqueeze(0)])

            # update valencies
            bond_valency = to_bond_valency(edge_type.unsqueeze(0))
            valencies[int(u.item())] -= bond_valency
            valencies[int(v.item())] -= bond_valency

            # update edges
            adj[u, v] = True
            adj[v, u] = True

            lab_v = lab_v.detach()
            edge_index = edge_index.detach()
            # compute z_graph = gnn(x_label, edge_index) of the graph
            z_v = self.g_dec(lab_v, edge_index)
            H_t = torch.mean(torch.cat([z_v, lab_v], dim=1), dim=0)

            # once new edge added, compute its coordinates
            if coords and not coord_helper.contains(v):
                # TODO only if v does not have a coord yet
                # embedding of pocket + graph
                pocket_graph_coords = torch.cat([coord_v, x_pos_p]).detach()
                pocket_graph_features = torch.cat([torch.zeros(coord_v.size(0), device=self.device, dtype=torch.long), torch.ones(x_pos_p.size(0), device=self.device, dtype=torch.long)]).detach()
                C = self.sch(pocket_graph_features, pocket_graph_coords)
                C_avg = torch.mean(C, dim=0)

                phi_c_uv = torch.cat([C_avg, C[u], C[v], z_v[u].detach(), lab_v[u].detach(), z_v[v].detach(), lab_v[v].detach(), edge_type.detach()]).unsqueeze(0)
                d = to_bond_length(lab_v[u], lab_v[v], edge_type)
                pred_alpha = self.mlp_alpha(phi_c_uv, gumbel=generate)
                pred_theta = self.mlp_theta(phi_c_uv, gumbel=generate)
                pred_alpha_dec = to_angle(pred_alpha)
                pred_theta_dec = to_dihedral(pred_theta)

                if not generate:
                    v_pos = x_pos_l[v].clone().detach()
                    true_alpha = encode_angle(coord_helper.ground_truth_angle(coord_v[u].clone().detach(), v_pos), self.device)
                    true_theta = encode_dihedral(coord_helper.ground_truth_dihedral(coord_v[u].clone().detach(), v_pos), self.device)
                    prob_alpha = torch.log(torch.sum(pred_alpha*true_alpha))
                    prob_theta = torch.log(torch.sum(pred_theta*true_theta))
                    log_prob_coords += prob_alpha + prob_theta

                coord_v[v] = torch.tensor(coord_helper.next_coord(coord_v[u], d, (pred_alpha_dec / 180) * np.pi,
                                                                  (pred_theta_dec / 180) * np.pi),
                                          device=self.device) if generate else x_pos_l[v].clone().detach()
                coord_helper.add_point(v, coord_v[v].clone().detach())

            time = time + torch.tensor(1, device=self.device)
        return (lab_v, edge_index, edge_attr, coord_v if coords else lab_v, edge_index, edge_attr) if generate else (log_prob + log_prob_coords if coords else log_prob)

    def parameters(self):
       return [
            dict(params=self.pocket_encoder.conv1.parameters(), weight_decay=5e-4),
            dict(params=self.pocket_encoder.conv2.parameters(), weight_decay=0),
            dict(params=self.ligand_encoder.conv1.parameters(), weight_decay=5e-4),
            dict(params=self.ligand_encoder.conv2.parameters(), weight_decay=0),
            dict(params=self.g_dec.conv1.parameters(), weight_decay=5e-4),
            dict(params=self.g_dec.conv2.parameters(), weight_decay=0),
            dict(params=self.f.linear.parameters()),
            dict(params=self.g.linear.parameters()),
            dict(params=self.h.linear.parameters()),
            dict(params=self.mlp_alpha.linear.parameters()),
            dict(params=self.mlp_theta.linear.parameters()),
            dict(params=self.sch.sch_layer.parameters()),
       ]

class VecHelper:
    def __init__(self):
        self.v1 = np.array([1., 0., 0.])
        self.v2 = np.array([0., 1., 0.])

    def add_vec(self, v3: torch.Tensor):
        v3 = v3.cpu().numpy()
        self.v1 = self.v2
        self.v2 = v3 / np.linalg.norm(v3)

    def next_coord(self, p3, d, ang, dih):
        return calc_position(self.v1, self.v2, p3.cpu().numpy(), d.cpu().numpy(), ang, dih)

    def ground_truth_angle(self, v3: torch.Tensor):
        return calc_angle(self.v2, v3.cpu().numpy())

    def ground_truth_dihedral(self, v3: torch.Tensor):
        return calc_dihedral(self.v1, self.v2, v3.cpu().numpy())

class CoordinateHelper:
    def __init__(self):
        self.p1_aux = np.array([1., 0., 0.])
        self.p2_aux = np.array([0., 1., 0.])
        self.coords = {}

    def add_point(self, i: torch.Tensor, p: torch.Tensor):
        self.coords[str(i.cpu().numpy())] = p.cpu().numpy()

    def contains(self, i: torch.Tensor):
        return str(i.cpu().numpy()) in self.coords

    def min_dist_points(self, p: torch.Tensor):
        p = p.cpu().numpy()
        sorted_coords = [v for k, v in sorted(self.coords.items(), key=lambda item: np.linalg.norm(item[1] - p))]
        num_coords = len(sorted_coords)
        if num_coords == 0 or num_coords == 1:  # if there is only one coord, then it is the origin
            return p+self.p1_aux, p+self.p2_aux
        elif len(sorted_coords) == 2:
            return self.p1_aux + self.p2_aux + sorted_coords[1] + p, sorted_coords[1]
        else:
            return sorted_coords[2], sorted_coords[1]

    def next_coord(self, p3: torch.Tensor, d: torch.Tensor, ang: int, dih: int):
        p1, p2 = self.min_dist_points(p3)
        return calc_position_p(p1, p2, p3.cpu().numpy(), d.cpu().numpy(), ang, dih)

    def ground_truth_angle(self, p3: torch.Tensor, p4: torch.Tensor):
        p1, p2 = self.min_dist_points(p3)
        return calc_angle_p(p2, p3.cpu().numpy(), p4.cpu().numpy())

    def ground_truth_dihedral(self, p3: torch.Tensor, p4: torch.Tensor):
        p1, p2 = self.min_dist_points(p3)
        return calc_dihedral_p(p1, p2, p3.cpu().numpy(), p4.cpu().numpy())
