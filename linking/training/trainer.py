from __future__ import annotations

from typing import Dict, Tuple

import torch
from linking.layers.gcn_encoders import GCNEncoder
from torch_geometric.data import Data
from linking.config.config import Config
from linking.data.pdb_loader import torchgeom_plot


class Trainer:
    def __init__(self, model: torch.nn.Module, data, optimizer, config: Config):
        self.model = model
        self.data = data
        self.optimizer = optimizer
        self.config = config
        self.X_ligand_train, self.X_ligand_test, self.X_pocket_train, self.X_pocket_test = data
        self.loss_history: Dict[str, float] = {}
        self.auc_history: Dict[str, float] = {}
        self.ap_history: Dict[str, float] = {}

    def training_epoch(self) -> float:
        self.model.train()
        total_loss = 0
        for i in range(len(self.X_ligand_train)):
            x_ligand = self.X_ligand_train[i]
            x_pocket = self.X_pocket_train[i]
            self.optimizer.zero_grad()
            loss_enc = GCNEncoder(in_channels=self.config.num_allowable_atoms, out_channels=self.config.ligand_encoder_out_channels)
            prediction = self.model(x_pocket, x_ligand)
            if i == 0:
                torchgeom_plot(Data(x=prediction[0], edge_index=prediction[1]))
            loss_f = torch.nn.MSELoss()
            indices = torch.tensor(list(range(4, x_ligand.x.size()[1])), dtype=torch.long)
            x_ligand_x = torch.index_select(x_ligand.x, 1, indices)

            truth = loss_enc(x_ligand_x, x_ligand.edge_index)
            truth = torch.sum(truth, dim=0)

            pred = loss_enc(prediction[0], prediction[1])
            pred = torch.sum(pred, dim=0)

            loss = loss_f(pred, truth)
            loss.backward()

            self.optimizer.step()
            total_loss += loss.item()
        return float(total_loss/len(self.X_ligand_train))

    def train(self) -> None:
        for epoch in range(1, self.config.num_epochs):
            loss = self.training_epoch()
            self.loss_history[epoch] = loss

            '''
            auc, ap = self.test(
                self.data.test_pos_edge_index, self.data.test_neg_edge_index
            )
            self.auc_history[epoch] = auc
            self.ap_history[epoch] = ap

            print("Epoch: {:03d}, AUC: {:.4f}, AP: {:.4f}, LOSS: {:.4f}".format(epoch, auc, ap, loss))
            '''
            print("Epoch: {:03d}, LOSS: {:.4f}".format(epoch, loss))

    def test(self) -> None:
            total_loss = 0
            for i in range(len(self.X_ligand_test)-1):
                x_ligand = self.X_ligand_test[i+1]
                x_pocket = self.X_pocket_test[i+1]
                self.model.eval()
                with torch.no_grad():
                    loss_enc = GCNEncoder(in_channels=self.config.num_allowable_atoms,
                                          out_channels=self.config.ligand_encoder_out_channels)
                    prediction = self.model(x_pocket, x_ligand)
                    #torchgeom_plot(Data(x=prediction[0], edge_index=prediction[1]))
                    loss_f = torch.nn.MSELoss()
                    indices = torch.tensor(list(range(4, x_ligand.x.size()[1])), dtype=torch.long)
                    x_ligand_x = torch.index_select(x_ligand.x, 1, indices)

                    truth = loss_enc(x_ligand_x, x_ligand.edge_index)
                    truth = torch.sum(truth, dim=0)

                    pred = loss_enc(prediction[0], prediction[1])
                    pred = torch.sum(pred, dim=0)

                    loss = loss_f(pred, truth)
                total_loss += loss

            print("Loss on test set: {:.4f}".format(total_loss / len(self.X_ligand_test)))
