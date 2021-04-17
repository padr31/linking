from linking.config.config import Config
from linking.layers.gcn_encoders import VariationalGATEncoder
from linking.layers.linear_encoders import MLP
from torch.distributions import Normal
import torch
import torch.nn.functional as F

class QED(torch.nn.Module):
    def __init__(self,
                 ligand_encoder: VariationalGATEncoder,
                 mlp: MLP,
                 config: Config,
                 device):
        super(QED, self).__init__()

        # graph
        self.ligand_encoder = ligand_encoder

        # coords

        # prediction
        self.mlp = mlp

        self.config = config
        self.device = device

    def forward(self, data_pocket, data_ligand):
        # Initialise data variables -------------------------
        x_l, edge_index_l, edge_attr_l, bfs_index, bfs_attr = data_ligand.x, data_ligand.edge_index, data_ligand.edge_attr, data_ligand.bfs_index.clone(), data_ligand.bfs_attr.clone()

        # ligand gnn encoding
        z_mu, z_log_var = self.ligand_encoder(x_l, edge_index_l)
        q = Normal(z_mu, 0.1 + F.softplus(z_log_var))

        preds = []
        for _ in range(20):
            z_sample = q.rsample()
            y = self.mlp(z_sample)
            preds.append(y)

        pred = torch.mean(torch.stack(preds, 1), 1)

        return pred, z_mu, z_log_var, q

    def parameters(self):
       return [
            dict(params=self.ligand_encoder.conv1.parameters(), weight_decay=5e-4),
            dict(params=self.ligand_encoder.conv2.parameters(), weight_decay=0),
            dict(params=self.ligand_encoder.conv_mu.parameters(), weight_decay=0),
            dict(params=self.ligand_encoder.conv_log_var.parameters(), weight_decay=0),
            dict(params=self.mlp.l1.parameters()),
            dict(params=self.mlp.l2.parameters()),
            dict(params=self.mlp.l3.parameters()),
       ]