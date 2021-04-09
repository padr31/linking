import torch
from linking.config.config import Config
from linking.layers.gcn_encoders import VariationalGATEncoder
from linking.layers.geom_encoders import Sch
from linking.layers.linear_encoders import MLP

class QED(torch.nn.Module):
    def __init__(self,
                 ligand_encoder: VariationalGATEncoder,
                 sch: Sch,
                 mlp: MLP,
                 config: Config,
                 device):
        super(QED, self).__init__()

        # graph
        self.ligand_encoder = ligand_encoder

        # coords
        self.sch = sch  # encoding

        # prediction
        self.mlp = mlp

        self.config = config
        self.device = device

    def forward(self, data_pocket, data_ligand):
        # Initialise data variables -------------------------
        # x_p, edge_index_p, edge_wight_p = data_pocket.x, data_pocket.edge_index, data_pocket.edge_attr
        x_l, edge_index_l, edge_attr_l, bfs_index, bfs_attr = data_ligand.x, data_ligand.edge_index, data_ligand.edge_attr, data_ligand.bfs_index.clone(), data_ligand.bfs_attr.clone()
        # x_pos_p = data_pocket.x[:, 1:4].copy()
        x_pos_l = data_ligand.x[:, 1:4]

        # Encode -------------------------
        # pocket gnn encoding
        # z_pocket_atoms = self.pocket_encoder(x_p, edge_index_p)
        # z_pocket = torch.mean(z_pocket_atoms, dim=0)

        # ligand gnn encoding
        z_mu, z_log_var = self.ligand_encoder(x_l, edge_index_l)

        std = torch.exp(0.5 * z_log_var)
        eps = torch.randn_like(std)
        z_ligand = z_mu + std*eps

        # ligand schnet encoding
        # C = self.sch(torch.zeros(x_pos_l.size(0), device=self.device, dtype=torch.long), x_pos_l)
        # C_avg = torch.mean(C, dim=0)

        y = self.mlp(z_ligand)

        return y.squeeze(0), z_mu, z_log_var

    def parameters(self):
       return [
            dict(params=self.ligand_encoder.conv1.parameters(), weight_decay=5e-4),
            dict(params=self.ligand_encoder.conv2.parameters(), weight_decay=0),
            dict(params=self.ligand_encoder.conv_mu.parameters(), weight_decay=0),
            dict(params=self.ligand_encoder.conv_log_var.parameters(), weight_decay=0),
            dict(params=self.mlp.l1.parameters()),
            dict(params=self.mlp.l2.parameters()),
            dict(params=self.mlp.l3.parameters()),
            dict(params=self.sch.sch_layer.parameters()),
       ]