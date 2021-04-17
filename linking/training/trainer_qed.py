from __future__ import annotations
from typing import Dict
from torch.distributions import Normal
from tqdm import tqdm
from torch_geometric.data import Data
from linking.config.config import Config
from linking.util.eval import to_rdkit, qed_score
from torch.utils.tensorboard import SummaryWriter
from datetime import datetime
import torch
import numpy as np
import matplotlib.pyplot as plt

class QEDTrainer:
    def __init__(self, model: torch.nn.Module, data, optimizer, config: Config):
        self.model = model
        self.data = data
        self.optimizer = optimizer
        self.config = config
        self.X_ligand_train, self.X_ligand_test, self.X_pocket_train, self.X_pocket_test = data
        self.loss_history: Dict[str, float] = {}
        logdir = config.logdir_qed + datetime.now().strftime("%Y%m%d-%H%M%S")
        self.general_writer = SummaryWriter(logdir + "/general")
        self.Y_qed = {}
        self.Y_qed_list=[]
        self.Y_qed_test = {}

        print('Preprocessing QED scores')
        for i in tqdm(range(len(self.X_ligand_train))):
            lig_i = self.X_ligand_train[i]
            ligand_mol = to_rdkit(Data(x=lig_i.x[:, 4:], edge_index=lig_i.edge_index, edge_attr=lig_i.edge_attr), device=self.model.device)
            qed_s = qed_score(ligand_mol)
            if qed_s is None:
                qed_s = 0.0
            self.Y_qed[lig_i.name] = torch.tensor(qed_s, dtype=torch.float, device=model.device)
            self.Y_qed_list.append(qed_s)

        l = np.array(self.Y_qed_list)
        plt.hist(l, bins=30)
        plt.show()
        print("mean, std: " + str(l.mean()) + ", " + str(l.std()))

        print('Preprocessing test QED scores')
        for i in tqdm(range(len(self.X_ligand_test) - 1)):
            x_ligand = self.X_ligand_test[i + 1]
            ligand_mol = to_rdkit(Data(x=x_ligand.x[:, 4:], edge_index=x_ligand.edge_index, edge_attr=x_ligand.edge_attr), device=self.model.device)
            qed_s = qed_score(ligand_mol)
            if qed_s is None:
                qed_s = 0.0
            self.Y_qed_test[x_ligand.name] = torch.tensor(qed_s, dtype=torch.float, device=model.device)

    def training_epoch(self, epoch) -> float:
        print('Epoch ' + str(epoch))
        self.model.train()
        total_loss = 0
        total_loss_kl = 0
        for i in tqdm(range(len(self.X_ligand_train))):
            x_ligand = self.X_ligand_train[i]
            x_pocket = self.X_pocket_train[i]

            prediction, mu, log_var, q = self.model(x_pocket, x_ligand)
            KL = torch.distributions.kl_divergence(Normal(torch.zeros_like(mu), torch.ones_like(log_var)), q).mean()
            #KL = -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp(), 0)
            loss_f = torch.nn.MSELoss()
            loss_normal = loss_f(prediction, self.Y_qed[x_ligand.name])
            loss = loss_normal + KL
            loss.backward()
            if (i + 1) % self.config.batch_size == 0:
                # every config.batch iterations
                self.optimizer.step()
                self.optimizer.zero_grad()

            total_loss += loss_normal.item()
            total_loss_kl += KL.item()
        return float(total_loss/len(self.X_ligand_train)), float(total_loss_kl/len(self.X_ligand_train))

    def train(self) -> None:
        for epoch in range(1, self.config.num_epochs):
            loss = self.training_epoch(epoch=epoch)
            self.loss_history[epoch] = loss

            print("Epoch: {:03d}, LOSS: {:.4f}, kl: {:.4f}".format(epoch, loss[0], loss[1]))
            self.general_writer.add_scalar('Training', loss[0], epoch)

    def test(self) -> None:
            total_loss = 0
            total_loss_kl = 0
            self.model.eval()
            for i in range(len(self.X_ligand_test)-1):
                x_ligand = self.X_ligand_test[i+1]
                x_pocket = self.X_pocket_test[i+1]
                with torch.no_grad():
                    prediction, mu, log_var = self.model(x_pocket, x_ligand)
                    negative_KL = 0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp(), 0)
                    loss_f = torch.nn.L1Loss()
                    loss_normal = loss_f(prediction, self.Y_qed_test[x_ligand.name])
                    loss = loss_normal - negative_KL
                    loss.backward()
                total_loss += loss_normal.item()
                total_loss_kl += negative_KL.item()

            print("Loss on test set: {:.4f}, kl: {:.4f}".format(total_loss / len(self.X_ligand_test), total_loss_kl / len(self.X_ligand_test)))
