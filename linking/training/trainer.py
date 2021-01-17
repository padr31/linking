from __future__ import annotations

from typing import Dict, Tuple

import torch

from linking.config.config import Config


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
            '''print(str(i) + " ----- ")
            print("Ligand")
            print(x_ligand)
            print("Pocket")
            print(x_pocket)'''
            z_pocket, z_ligand = self.model(x_pocket, x_ligand)
            loss_f = torch.nn.L1Loss()
            loss = loss_f(z_pocket, z_ligand)
            loss.backward()
            self.optimizer.step()
            total_loss += loss.item()
        return float(total_loss)

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
                x_ligand = self.X_ligand_test[i]
                x_pocket = self.X_pocket_test[i+1]
                self.model.eval()
                with torch.no_grad():
                    z_pocket, z_ligand = self.model(x_pocket, x_ligand)
                    loss_f = torch.nn.L1Loss()
                    loss = loss_f(z_pocket, z_ligand)
                total_loss += loss

            print("Loss on test set: {:.4f}".format(total_loss / len(self.X_ligand_test)))
