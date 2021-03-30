from __future__ import annotations

from typing import Dict

import torch
from rdkit import Chem
from tqdm import tqdm

from linking.layers.gcn_encoders import GCNEncoder
from torch_geometric.data import Data
from linking.config.config import Config
from linking.data.data_eval import rdkit_tanimoto, rdkit_fingerprint, lipinski_nhoh_count, lipinski_ring_count, \
    to_rdkit, rdkit_sanitize, mol_to_svg, qed_score
from torch.utils.tensorboard import SummaryWriter
from datetime import datetime

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
        logdir = config.logdir + datetime.now().strftime("%Y%m%d-%H%M%S")
        self.general_writer = SummaryWriter(logdir + "/general")
        self.writers = {}
        for a in range(len(config.eval_data)):
            self.writers[str(a)] = SummaryWriter(logdir + "/" + str(a))

    def training_epoch(self, epoch) -> float:
        print('Epoch ' + str(epoch))
        self.model.train()
        total_loss = 0
        for i in tqdm(range(len(self.X_ligand_train))):
            x_ligand = self.X_ligand_train[i]
            x_pocket = self.X_pocket_train[i]
            assert x_ligand.name.split('/')[-1].split('_')[0] == x_pocket.name.split('/')[-1].split('_')[0]

            prediction = self.model(x_pocket, x_ligand, generate=False)
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
            if (i + 1) % self.config.batch_size == 0:
                # every config.batch iterations
                self.optimizer.step()
                self.optimizer.zero_grad()

            total_loss += loss.item()
        return float(total_loss/len(self.X_ligand_train))

    def train(self) -> None:
        for epoch in range(1, self.config.num_epochs):
            loss = self.training_epoch(epoch=epoch)
            self.loss_history[epoch] = loss

            print("Epoch: {:03d}, LOSS: {:.4f}".format(epoch, loss))
            self.general_writer.add_scalar('Training', loss, epoch)
            self.eval(epoch=epoch)

    def eval(self, epoch):
        self.model.eval()
        eval_data = self.config.eval_data

        for i in eval_data:
            x_ligand = self.X_ligand_train[i]
            x_pocket = self.X_pocket_train[i]
            protein_name = str(x_ligand.name.split('/')[-1].split('_')[0])

            tanimoto = 0

            nhoh_count = 0
            nhoh_count_items = 1

            ring_count = 0
            ring_count_items = 1

            qed_score_count = 0
            qed_score_count_items = 1

            ligand_mol = to_rdkit(Data(x=x_ligand.x[:, 4:], edge_index=x_ligand.edge_index, edge_attr=x_ligand.edge_attr))
            ligand_fingerprint = rdkit_fingerprint(ligand_mol)
            ligand_svg = mol_to_svg(ligand_mol)
            with open("out_svg/ligand_" + "_" + protein_name + ".svg", "w") as svg_file:
                svg_file.write(ligand_svg)

            for j in range(self.config.num_eval_generate):
                pred_generate = self.model(x_pocket, x_ligand, generate=True)

                generated_ligand_mol = to_rdkit(
                    Data(x=pred_generate[0], edge_index=pred_generate[1], edge_attr=pred_generate[2]))

                generated_ligand_mol = rdkit_sanitize(generated_ligand_mol)
                generated_ligand_fingerprint = rdkit_fingerprint(generated_ligand_mol)
                generated_ligand_tanimoto = rdkit_tanimoto(ligand_fingerprint, generated_ligand_fingerprint)
                tanimoto += generated_ligand_tanimoto / self.config.num_eval_generate

                nhoh_c = lipinski_nhoh_count(generated_ligand_mol)
                if not nhoh_c is None:
                    nhoh_count += nhoh_c
                    nhoh_count_items += 1

                ring_c = lipinski_ring_count(generated_ligand_mol)
                if not ring_c is None:
                    ring_count += ring_c
                    ring_count_items += 1

                qed_s = qed_score(generated_ligand_mol)
                if not qed_s is None:
                    qed_score_count += qed_s
                    qed_score_count_items += 1

                generated_ligand_svg = mol_to_svg(generated_ligand_mol)
                with open("out_svg/generated_ligand_" + str(epoch) + "_" + str(j) + "_" + protein_name + ".svg", "w") as svg_file:
                    svg_file.write(generated_ligand_svg)

                # torchgeom_plot(Data(x=x_ligand.x[:, 4:], edge_index=x_ligand.edge_index))
                # torchgeom_plot(Data(x=pred_generate[0], edge_index=pred_generate[1]))

            self.writers[str(i)].add_scalar('Tanimoto', tanimoto, epoch)
            self.writers[str(i)].add_scalar('NHOH Count', nhoh_count/nhoh_count_items, epoch)
            self.writers[str(i)].add_scalar('Ring Count', ring_count/nhoh_count_items, epoch)
            self.writers[str(i)].add_scalar('QED Score', qed_score_count/qed_score_count_items, epoch)

            print("Valid QED scores: " + str(qed_score_count_items/self.config.num_eval_generate))
            print("Valid ring counts: " + str(ring_count_items/self.config.num_eval_generate))
            print("Valid nhoh counts: " + str(nhoh_count_items/self.config.num_eval_generate))
            print("Tanimoto coefficient of " + protein_name + ": " + str(tanimoto))

    def test(self) -> None:
            total_loss = 0
            self.model.eval()
            for i in range(len(self.X_ligand_test)-1):
                x_ligand = self.X_ligand_test[i+1]
                x_pocket = self.X_pocket_test[i+1]
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
