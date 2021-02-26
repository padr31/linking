from __future__ import annotations

from typing import Dict, Tuple

import torch
from linking.layers.gcn_encoders import GCNEncoder
from torch_geometric.data import Data
from linking.config.config import Config
from data.data_plotting import torchgeom_plot


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

    def training_epoch(self, epoch) -> float:
        self.model.train()
        total_loss = 0
        for i in range(len(self.X_ligand_train)):
            x_ligand = self.X_ligand_train[i]
            x_pocket = self.X_pocket_train[i]
            assert x_ligand.name.split('/')[-1].split('_')[0] == x_pocket.name.split('/')[-1].split('_')[0]
            self.optimizer.zero_grad()

            prediction = self.model(x_pocket, x_ligand, generate=False)
            pred_generate = self.model(x_pocket, x_ligand, generate=True)

            if i != 100:
                ligand = self.model.mol_to_svg(self.model.to_rdkit(
                    Data(x=x_ligand.x[:, 4:], edge_index=x_ligand.edge_index, edge_attr=x_ligand.edge_attr)))
                generated_ligand = self.model.mol_to_svg(self.model.to_rdkit(
                    Data(x=pred_generate[0], edge_index=pred_generate[1], edge_attr=pred_generate[2])))

                #if epoch == 1:
                #    with open("out_svg/ligand_" + str(i) + "_" + str(x_ligand.name.split('/')[-1].split('_')[0]) + ".svg", "w") as svg_file:
                #        svg_file.write(ligand)
                with open("out_svg/generated_ligand_" + str(epoch) + ".svg", "w") as svg_file:
                   svg_file.write(generated_ligand)

                #torchgeom_plot(Data(x=x_ligand.x[:, 4:], edge_index=x_ligand.edge_index))
                #torchgeom_plot(Data(x=pred_generate[0], edge_index=pred_generate[1]))

            loss_f = torch.nn.L1Loss()

            loss = loss_f(-prediction, torch.tensor(0.0))
            ''' loss for gumbel
            loss_enc = GCNEncoder(in_channels=self.config.num_allowable_atoms, out_channels=self.config.ligand_encoder_out_channels)
            prediction = self.model(x_pocket, x_ligand)
            if i == 0 or i == 1 or i == 2:
                torchgeom_plot(Data(x=prediction[0], edge_index=prediction[1]))
            loss_f = torch.nn.MSELoss()
            indices = torch.tensor(list(range(4, x_ligand.x.size()[1])), dtype=torch.long)
            x_ligand_x = torch.index_select(x_ligand.x, 1, indices)

            truth = loss_enc(x_ligand_x, x_ligand.edge_index)
            truth = torch.sum(truth, dim=0)

            pred = loss_enc(prediction[0], prediction[1])
            pred = torch.sum(pred, dim=0)

            loss = loss_f(pred, truth)
            '''

            loss.backward()

            self.optimizer.step()
            total_loss += loss.item()
        return float(total_loss/len(self.X_ligand_train))

    def train(self) -> None:
        for epoch in range(1, self.config.num_epochs):
            loss = self.training_epoch(epoch=epoch)
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
                    pred_generate = self.model(x_pocket, x_ligand, generate=True)

                    #torchgeom_plot(Data(x=pred_generate[0], edge_index=pred_generate[1]))
                    loss_f = torch.nn.L1Loss()

                    '''
                    gumbel loss
                    indices = torch.tensor(list(range(4, x_ligand.x.size()[1])), dtype=torch.long)
                    x_ligand_x = torch.index_select(x_ligand.x, 1, indices)

                    truth = loss_enc(x_ligand_x, x_ligand.edge_index)
                    truth = torch.sum(truth, dim=0)

                    pred = loss_enc(prediction[0], prediction[1])
                    pred = torch.sum(pred, dim=0)
                    '''

                    loss = loss_f(prediction, torch.tensor(0.0))
                total_loss += loss

            print("Loss on test set: {:.4f}".format(total_loss / len(self.X_ligand_test)))
