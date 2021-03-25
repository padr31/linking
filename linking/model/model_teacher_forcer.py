import torch
from rdkit import Chem
from linking.config.config import Config
from linking.layers.gcn_encoders import GCNEncoder
from linking.layers.linear_encoders import LinearAtomClassifier, LinearEdgeSelector, LinearEdgeRowClassifier, LinearEdgeClassifier
from linking.data.data_util import to_one_hot, allowable_atoms
from rdkit.Chem import rdDepictor, rdmolops, Draw
from rdkit.Chem.Draw import rdMolDraw2D

class TeacherForcer(torch.nn.Module):
    def __init__(self, pocket_encoder: GCNEncoder, ligand_encoder: GCNEncoder, g_dec: GCNEncoder, f: LinearAtomClassifier, g: LinearEdgeSelector, h: LinearEdgeClassifier, h_row: LinearEdgeRowClassifier, config: Config, device):
        super(TeacherForcer, self).__init__()
        self.pocket_encoder = pocket_encoder
        self.ligand_encoder = ligand_encoder
        self.g_dec = g_dec
        self.f = f
        self.g = g
        self.h = h
        self.h_row = h_row
        self.decoder = None
        self.config = config
        self.device = device
        self.valency_map = {'C': 4, 'F': 1, 'N': 3, 'Cl': 1, 'O': 2, 'I': 1, 'P': 5, 'Br': 1, 'S': 6, 'H': 1, 'Stop': 1000}

    def to_atom(self, t):
        return allowable_atoms[int(torch.dot(t, torch.tensor(range(t.size()[0]), dtype=torch.float, device=self.device)).item())]

    def to_bond(self, t):
        t_s = t.squeeze()
        return [1, 2, 3][int(torch.dot(t_s, torch.tensor(range(t_s.size()[0]), dtype=torch.float, device=self.device)).item())]

    def calculate_node_mask_list(self, valencies, u, closed, closed_mask, edges, adj, unmask=None):
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

        mask_list = [(
                 (float('-inf') if i == u else 0.0) +  # no self edges allowed
                 (float('-inf') if valencies[i] <= 0 else 0.0) +  # valency of new node kept
                 (float('-inf') if valencies[u] <= 0 else 0.0) +  # valency of old node kept
                 (float('-inf') if i in closed else 0.0) +  # new node is not closed
                 (float('-inf') if (u, i) in edges else 0.0))  # edge already present
                 for i in range(valencies.size()[0])]
        if not unmask is None:
            for i in unmask:
                mask_list[i] = 0

        for i in range(len(mask_list)):
            assert mask_list[i] == m_list[i]
        return m_list

    def calculate_node_mask(self, valencies, u, closed, closed_mask, edges, adj, unmask=None):
        m_list = self.calculate_node_mask_list(valencies, u, closed, closed_mask, edges, adj, unmask)
        # t = torch.stack([torch.tensor(0., dtype=torch.float, device=self.device) if x == 0.0 else torch.tensor(float('-inf'), dtype=torch.float, device=self.device) for x in mask_list])
        return torch.unsqueeze(m_list, 1)

    def calculate_bond_mask(self, valencies, u, closed, closed_mask, edges, adj, unmask=None, unmask_bond=None):
        m_list = self.calculate_node_mask_list(valencies, u, closed, closed_mask, edges, adj, unmask)

        zero_row = torch.tensor([float('-inf'), float('-inf'), float('-inf')], device=self.device, dtype=torch.float)
        one_row = torch.tensor([0., float('-inf'), float('-inf')], device=self.device, dtype=torch.float)
        two_row = torch.tensor([0., 0., float('-inf')], device=self.device, dtype=torch.float)
        three_row = torch.tensor([0., 0., 0.], device=self.device, dtype=torch.float)
        e_mask = torch.zeros(valencies.size(0), 3)
        e_mask[valencies <= 2] = two_row
        e_mask[valencies <= 1] = one_row
        e_mask[valencies <= 0] = zero_row
        e_mask[m_list == float('-inf')] = zero_row
        if not (unmask is None or unmask_bond is None):
            for i in unmask:
                for j in unmask_bond:
                    e_mask[i, j-1] = 0.

        mask_list = [
            torch.stack([(
            m_list[i] +
            (torch.tensor(float('-inf'), dtype=torch.float, device=self.device) if valencies[i] < b else torch.tensor(0.0, dtype=torch.float, device=self.device)))
            for b in (1, 2, 3)])
            for i in range(len(valencies))]
        if not (unmask is None or unmask_bond is None):
            for i in unmask:
                for j in unmask_bond:
                    mask_list[i][j-1] = 0

        for i in range(len(mask_list)):
            for j in range(3):
                assert mask_list[i][j] == e_mask[i, j]

        return e_mask

    def calculate_bond_mask_row(self, valencies, u, v, closed, closed_mask, edges, adj, unmask=None, unmask_bond=None):
        m_list = self.calculate_node_mask_list(valencies, u, closed, closed_mask, edges, adj, unmask)

        zero_row = torch.tensor([float('-inf'), float('-inf'), float('-inf')], device=self.device, dtype=torch.float)
        one_row = torch.tensor([0., float('-inf'), float('-inf')], device=self.device, dtype=torch.float)
        two_row = torch.tensor([0., 0., float('-inf')], device=self.device, dtype=torch.float)
        e_mask = torch.zeros(3)
        if min(valencies[u], valencies[v]) <= 2:
            e_mask = two_row
        if min(valencies[u], valencies[v]) <= 1:
            e_mask = one_row
        if min(valencies[u], valencies[v]) <= 0:
            e_mask = zero_row
        # TODO should not be true (although we sometimes need to force wrong valencies) assert m_list[v] != float('-inf')
        if m_list[v] == float('-inf'):
            e_mask = zero_row

        if not (unmask_bond is None):
            for j in unmask_bond:
                e_mask[j - 1] = 0.
        return e_mask.unsqueeze(0)

    def calculate_label_mask(self, length):
        l_mask = torch.zeros(len(self.valency_map), dtype=torch.float, device=self.device)
        l_mask[-1] = float('-inf')
        l = [torch.tensor(0.0, dtype=torch.float, device=self.device) for i in range(len(self.valency_map) - 1)]
        l.append(torch.tensor(float('-inf'), dtype=torch.float, device=self.device))

        for i in range(len(l)):
            assert l[i] == l_mask[i]

        return l_mask.repeat(length, 1)


    def to_rdkit(self, data):
        node_list = []
        for i in range(data.x.size()[0]):
            node_list.append(self.to_atom(data.x[i]))

        # create empty editable mol object
        mol = Chem.RWMol()
        # add atoms to mol and keep track of index
        node_to_idx = {}
        invalid_idx = set([])
        for i in range(len(node_list)):
            if node_list[i] == 'Stop' or node_list[i] == 'H':
                invalid_idx.add(i)
                continue
            a = Chem.Atom(node_list[i])
            molIdx = mol.AddAtom(a)
            node_to_idx[i] = molIdx

        added_bonds = set([])
        for i in range(0, data.edge_index.size()[1]):
            ix = data.edge_index[0][i].item()
            iy = data.edge_index[1][i].item()
            bond = self.to_bond(data.edge_attr[i])
            # add bonds between adjacent atoms
            if (str((ix, iy)) in added_bonds) or (str((iy, ix)) in added_bonds) or (iy in invalid_idx or ix in invalid_idx):
                continue
            # add relevant bond type (there are many more of these)
            if bond == 0:
                continue
            elif bond == 1:
                bond_type = Chem.rdchem.BondType.SINGLE
                mol.AddBond(node_to_idx[ix], node_to_idx[iy], bond_type)
            elif bond == 2:
                bond_type = Chem.rdchem.BondType.DOUBLE
                mol.AddBond(node_to_idx[ix], node_to_idx[iy], bond_type)
            elif bond == 3:
                bond_type = Chem.rdchem.BondType.TRIPLE
                mol.AddBond(node_to_idx[ix], node_to_idx[iy], bond_type)

            added_bonds.add(str((ix, iy)))

        # Convert RWMol to Mol object
        mol = mol.GetMol()
        mol_frags = rdmolops.GetMolFrags(mol, asMols=True, sanitizeFrags=False)
        largest_mol = max(mol_frags, default=mol, key=lambda m: m.GetNumAtoms())

        return largest_mol

    def mol_to_svg(self, mol, molSize=(300, 300), kekulize=True, sanitize=True):
        mc = Chem.Mol(mol.ToBinary())
        if kekulize:
            try:
                Chem.Kekulize(mc)
            except:
                mc = Chem.Mol(mol.ToBinary())
        if sanitize:
            try:
                Chem.SanitizeMol(mc)
            except:
                mc = Chem.Mol(mol.ToBinary())
        if not mc.GetNumConformers():
            rdDepictor.Compute2DCoords(mc)
        drawer = rdMolDraw2D.MolDraw2DSVG(molSize[0], molSize[1])
        drawer.DrawMolecule(mc)
        drawer.FinishDrawing()
        svg = drawer.GetDrawingText()
        return svg.replace('svg:', '')

    def forward(self, data_pocket, data_ligand, generate=False):
        # Initialise data variables -------------------------
        x_p, edge_index_p, edge_wight_p = data_pocket.x, data_pocket.edge_index, data_pocket.edge_attr
        x_l, edge_index_l, edge_weight_l, bfs_index, bfs_attr = data_ligand.x, data_ligand.edge_index, data_ligand.edge_attr, data_ligand.bfs_index.clone(), data_ligand.bfs_attr.clone()

        log_prob = torch.tensor(0.0,  dtype=torch.float, device=self.device)
        # Encode -------------------------
        # pocket
        z_pocket_atoms = self.pocket_encoder(x_p, edge_index_p)
        z_pocket = torch.mean(z_pocket_atoms, dim=0)

        # ligand
        z_ligand_atoms = self.ligand_encoder(x_l, edge_index_l)
        z_ligand = torch.mean(z_ligand_atoms, dim=0)
        z_v = z_ligand_atoms

        # guess ligand labels
        # TODO labels cant be stop node labels, need to mask it out
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
        valencies = torch.tensor([self.valency_map[self.to_atom(lab_v[i])] for i in range(lab_v.size()[0])], device=self.device)
        edges = []
        closed_nodes = []
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
                v = self.g(phi, self.calculate_node_mask(valencies, u, closed_nodes, closed_mask, edges, adj, unmask=torch.tensor([i_stop])), gumbel=generate)
            else:
                v = bfs_index[0][1]
                v_attr = bfs_attr[0]
                # remove front from bfs index as v was selected
                bfs_index = bfs_index[1:]  # behaves like pop(0)
                bfs_attr = bfs_attr[1:]
                # TODO dont mask things during generation
                p_uv = self.g(phi, self.calculate_node_mask(valencies, u, closed_nodes, closed_mask, edges, adj, unmask=torch.tensor([v, i_stop])), gumbel=generate)
                # select prob of selecting edge from u to v
                prob_v = torch.log(torch.squeeze(p_uv[v]))
                log_prob += prob_v

            # stop node selected
            # TODO set stop node to last elem not -1
            if v == i_stop or v == -1:
                popped = Q.pop(0)
                closed_nodes.append(popped)
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
                p_att_uv = self.h_row(phi[v].unsqueeze(0), mask=self.calculate_bond_mask_row(valencies, u, v, closed_nodes, closed_mask, edges, adj), gumbel=generate)
            else:
                # only mask
                p_att_uv = self.h_row(phi[v].unsqueeze(0), mask=self.calculate_bond_mask_row(valencies, u, v, closed_nodes, closed_mask, edges, adj, unmask_bond=torch.tensor([self.to_bond(v_attr.unsqueeze(0))]))
                                  , gumbel=generate)
            edge_type = p_att_uv
            if not generate:
                prob = torch.log(torch.sum(p_att_uv*v_attr))
                log_prob += prob
                edge_type = v_attr

            edge_attr = torch.cat([edge_attr, edge_type.unsqueeze(0), edge_type.unsqueeze(0)])

            # update valencies
            bond_valency = self.to_bond(edge_type.unsqueeze(0))
            valencies[int(u.item())] -= bond_valency
            valencies[int(v.item())] -= bond_valency

            # update edges
            adj[u, v] = True
            adj[v, u] = True
            edges.append((int(u.item()), int(v.item())))
            edges.append((int(v.item()), int(u.item())))

            # compute z_graph = gnn(x_label, edge_index) of the graph
            z_v = self.g_dec(lab_v, edge_index)
            H_t = torch.mean(torch.cat([z_v, lab_v], dim=1), dim=0)
            time = time + torch.tensor(1, device=self.device)
            for (e_i, e_j) in edges:
                assert adj[e_i, e_j]

        assert sum(adj.type(torch.FloatTensor).matmul(torch.ones(adj.shape[0]))) == edge_index.shape[1]
        return (lab_v, torch.tensor([[], []], dtype=torch.long, device=self.device) if len(edges) == 0 else edge_index, torch.tensor([[]], device=self.device) if len(edges) == 0 else edge_attr) if generate else log_prob

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
            dict(params=self.h_row.linear.parameters()),
       ]