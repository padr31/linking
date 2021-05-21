from __future__ import annotations

import random
from typing import Dict

from ase.formula import Formula
from tqdm import tqdm
from linking.layers.gcn_encoders import GCNEncoder
from torch_geometric.data import Data, data
from linking.config.config import Config
from linking.util.docking import score
from linking.util.encoding import molgym_formula_to_ligand
from linking.util.eval import rdkit_tanimoto, rdkit_fingerprint, lipinski_nhoh_count, lipinski_ring_count, \
    to_rdkit, rdkit_sanitize, qed_score, tanimoto_score, rdkit_sascore, rdkit_logp
from linking.util.plotting import mol_to_svg, mol_to_3d_svg, pos_plot_3D
from torch.utils.tensorboard import SummaryWriter
from datetime import datetime
from rdkit.Chem import PyMol
import torch

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
        logdir = config.logdir + ("_gpu/" if torch.cuda.is_available() else "/") + datetime.now().strftime("%Y%m%d-%H%M%S")
        self.general_writer = SummaryWriter(logdir + "/general")
        self.writers = {}
        for a in config.eval_data:
            self.writers[str(a)] = SummaryWriter(logdir + "/" + str(a))
        self.writers['filtered'] = SummaryWriter(logdir + "/" + 'filtered')

        self.metric_descriptions = {
            'tanimoto': 'Tanimoto',
            'nhoh': 'NHOH Count',
            'ring': 'Ring Count',
            'qed': 'QED Score',
            'sascore': 'SAS Score',
            'logp': 'LogP',
            'reward': 'Reward',
            'docking': 'Docking',
        }

        if self.config.molgym_eval:
            for f in config.molgym_eval_formulas:
                self.writers[f] = SummaryWriter(logdir + "/" + f)

        if self.config.coords:
            try:
                self.viewer = PyMol.MolViewer()
            except:
                self.viewer = None

    def training_epoch(self, epoch) -> float:
        print('Epoch ' + str(epoch))
        self.model.train()
        total_loss = 0
        for i in tqdm(range(len(self.X_ligand_train))):
            x_ligand = self.X_ligand_train[i]
            x_pocket = self.X_pocket_train[i]
            assert (not x_ligand.protein_name is None) and (x_ligand.protein_name == x_pocket.protein_name)

            prediction = self.model(x_pocket, x_ligand, generate=False, coords=self.config.coords)
            loss_f = torch.nn.L1Loss()
            loss = loss_f(-prediction, torch.tensor(0.0))
            loss.backward()
            if (i + 1) % self.config.batch_size == 0:
                # every config.batch iterations
                self.optimizer.step()
                self.optimizer.zero_grad()

            total_loss += loss.item()
        return float(total_loss/len(self.X_ligand_train))

    def train(self) -> None:
        # self.eval(epoch=0)
        for epoch in range(1, self.config.num_epochs):
            loss = self.training_epoch(epoch=epoch)
            self.loss_history[epoch] = loss

            print("Epoch: {:03d}, LOSS: {:.4f}".format(epoch, loss))
            self.general_writer.add_scalar('Training', loss, epoch)
            self.eval(epoch=epoch)

    def eval(self, epoch):
        self.model.eval()

        # evaluate overall filtered molecules
        scores = {'tanimoto': 0, 'nhoh': 0, 'ring': 0, 'qed': 0, 'sascore': 0, 'logp': 0, 'reward': 0, 'docking': 0}
        filtered_count = 0
        random_molecules = random.choices(range(len(self.X_ligand_train)), k=self.config.num_eval_filtered)
        for i in tqdm(random_molecules):
            x_ligand = self.X_ligand_train[i]
            x_pocket = self.X_pocket_train[i]
            protein_name = str(x_ligand.name.split('/')[-1].split('_')[0])
            prot_path = '/'.join(x_ligand.name.split('/')[:-1]) + '/' + x_ligand.protein_name + '_pocket.pdb'
            ligand_write_id = str(i) + "_random_ligand_" + protein_name

            # generate ligand
            try:
                pred_generate = self.model(x_pocket, x_ligand, generate=True, coords=self.config.coords,
                                           molgym_eval=self.config.molgym_eval)
            except:
                raise Exception("Exception in generating: " + x_ligand.name)

            generated_ligand_mol = to_rdkit(
                Data(x=pred_generate[0], edge_index=pred_generate[1], edge_attr=pred_generate[2],
                     pos=(pred_generate[3] if self.config.coords else None)), device=self.model.device)
            generated_ligand_mol = rdkit_sanitize(generated_ligand_mol)

            ligand_mol = to_rdkit(
                Data(x=x_ligand.x[:, 4:], edge_index=x_ligand.edge_index, edge_attr=x_ligand.edge_attr,
                     pos=x_ligand.x[:, 1:4]), device=self.model.device)
            ligand_mol = rdkit_sanitize(ligand_mol)

            metrics = {'tanimoto': 0, 'nhoh': 0, 'ring': 0, 'qed': 0, 'reward': 0, 'sascore':0, 'logp': 0, 'docking': 0}
            # metrics to pass filter
            metrics['tanimoto'] = tanimoto_score(generated_ligand_mol, ligand_mol)
            if metrics['tanimoto'] is None:
                continue

            metrics['nhoh'] = lipinski_nhoh_count(generated_ligand_mol)
            if metrics['nhoh'] is None:
                continue

            metrics['ring'] = lipinski_ring_count(generated_ligand_mol)
            if metrics['ring'] is None:
                continue

            metrics['qed'] = qed_score(generated_ligand_mol)
            if metrics['qed'] is None:
                continue

            metrics['sascore'] = rdkit_sascore(generated_ligand_mol)
            if metrics['sascore'] is None:
                continue

            metrics['logp'] = rdkit_logp(generated_ligand_mol)
            if metrics['logp'] is None:
                continue

            metrics['reward'] = pred_generate[4]

            # docking - only when filter was passed
            if self.config.coords and self.config.dock_eval:
                # print(score(ligand_mol, prot_path, str(epoch) + "ep_" + ligand_write_id + '_lig', dock=False))
                d_score = score(generated_ligand_mol, prot_path, str(epoch) + "ep_" + ligand_write_id + '_gen', dock=True, embed=True,
                          bounding_box=x_pocket.bounding_box)
                metrics['docking'] = d_score
                if metrics['docking'] is None:
                    continue

            filtered_count += 1
            for metric_name in scores.keys():
                scores[metric_name] += metrics[metric_name]

            # performing writeouts and visualisations
            # ligand svg writeout
            ligand_svg = mol_to_svg(ligand_mol)
            with open(self.config.svg_dir + str(epoch) + "ep_" + ligand_write_id + "_lig.svg", "w") as svg_file:
                svg_file.write(ligand_svg)

            generated_ligand_svg = mol_to_svg(generated_ligand_mol)
            # generated ligand svg writeout
            with open(self.config.svg_dir + str(epoch) + "ep_" + ligand_write_id + "_gen.svg", "w") as svg_file:
                svg_file.write(generated_ligand_svg)

            # show in 3D in PyMol if Viewer running
            if self.config.coords and (not self.viewer is None):
                generated_ligand_png = mol_to_3d_svg(generated_ligand_mol, viewer=self.viewer,
                                                     pocket_file=x_pocket.name)
                try:
                    generated_ligand_png.save(self.config.svg_dir + str(epoch) + "ep_" + ligand_write_id + "_3D.png")
                except:
                    print('error saving png from pymol')

        # log aggregated scores of overall generated ligands
        if self.config.num_eval_filtered > 0 and filtered_count > 0:
            for metric_name in scores.keys():
                self.writers['filtered'].add_scalar(self.metric_descriptions[metric_name], scores[metric_name]/filtered_count, epoch)

            print("Overall metrics (" + str(filtered_count/self.config.num_eval_filtered) + " passed filter) --- tanimoto: " + str(scores['tanimoto']/filtered_count) + " qed: " + str(scores['qed']/filtered_count) + " sascore: " + str(scores['sascore']/filtered_count))

        # evaluate specific molecules
        for i in self.config.eval_data:
            x_ligand = self.X_ligand_train[i]
            x_pocket = self.X_pocket_train[i]
            protein_name = str(x_ligand.name.split('/')[-1].split('_')[0])

            tanimoto = 0

            nhoh_count = 0
            nhoh_count_items = 0

            ring_count = 0
            ring_count_items = 0

            qed_score_count = 0
            qed_score_count_items = 0

            sascore_count = 0
            sascore_count_items = 0

            logp_count = 0
            logp_count_items = 0

            reward = 0

            docking = 0
            docking_items = 0

            ligand_mol = to_rdkit(Data(x=x_ligand.x[:, 4:], edge_index=x_ligand.edge_index, edge_attr=x_ligand.edge_attr, pos=x_ligand.x[:, 1:4]), device=self.model.device)
            ligand_mol = rdkit_sanitize(ligand_mol)
            ligand_fingerprint = rdkit_fingerprint(ligand_mol)
            prot_path = '/'.join(x_ligand.name.split('/')[:-1]) + '/' + x_ligand.protein_name + '_pocket.pdb'

            # docking ligand - we don't need it necessarily
            # if self.config.coords and self.config.dock_eval:
            #    print(score(ligand_mol, prot_path, "ligand_" + "_" + protein_name, dock=False))

            # ligand svg writeout
            ligand_svg = mol_to_svg(ligand_mol)
            with open(self.config.svg_dir + "ligand_" + "_" + protein_name + ".svg", "w") as svg_file:
                svg_file.write(ligand_svg)

            # generate ligands
            for j in range(self.config.num_eval_generate):
                try:
                    pred_generate = self.model(x_pocket, x_ligand, generate=True, coords=self.config.coords, molgym_eval=self.config.molgym_eval)
                except:
                    raise Exception("Exception in generating: " + x_ligand.name)

                generated_ligand_mol = to_rdkit(
                    Data(x=pred_generate[0], edge_index=pred_generate[1], edge_attr=pred_generate[2], pos=(pred_generate[3] if self.config.coords else None)), device=self.model.device)
                generated_ligand_mol = rdkit_sanitize(generated_ligand_mol)

                # docking
                if self.config.coords and self.config.dock_eval:
                    docking_s = score(generated_ligand_mol, prot_path, "generated_ligand_" + str(epoch) + "_" + str(j) + "_" + protein_name, dock=True, embed=True, bounding_box=x_pocket.bounding_box)
                    print(docking_s)
                    if not docking_s is None:
                        docking += docking_s
                        docking_items += 1

                generated_ligand_fingerprint = rdkit_fingerprint(generated_ligand_mol)
                generated_ligand_tanimoto = rdkit_tanimoto(ligand_fingerprint, generated_ligand_fingerprint)
                tanimoto += generated_ligand_tanimoto / self.config.num_eval_generate
                reward += pred_generate[4]

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

                sascore_s = rdkit_sascore(generated_ligand_mol)
                if not sascore_s is None:
                    sascore_count += sascore_s
                    sascore_count_items += 1

                logp_s = rdkit_logp(generated_ligand_mol)
                if not logp_s is None:
                    logp_count += logp_s
                    logp_count_items += 1

                generated_ligand_svg = mol_to_svg(generated_ligand_mol)

                # show in 3D in PyMol if Viewer running
                if self.config.coords and (not self.viewer is None):
                     generated_ligand_png = mol_to_3d_svg(generated_ligand_mol, viewer=self.viewer, pocket_file=x_pocket.name)
                     try:
                        generated_ligand_png.save(self.config.svg_dir + "generated_ligand_" + str(epoch) + "_" + str(j) + "_" + protein_name + "_3D.png")
                     except:
                         print('error saving png from pymol')

                # svg writeout of generated ligand
                with open(self.config.svg_dir + "generated_ligand_" + str(epoch) + "_" + str(j) + "_" + protein_name + ".svg", "w") as svg_file:
                    svg_file.write(generated_ligand_svg)

                # save 3D .png (not very useful)
                if self.config.coords:
                    file_save_name = self.config.svg_dir + "generated_ligand_" + str(epoch) + "_" + str(j) + "_" + protein_name
                    pos_plot_3D(pred_generate[3], pred_generate[1], pred_generate[0], 90, save_name=file_save_name)
                # torchgeom_plot(Data(x=pred_generate[0], edge_index=pred_generate[1]))

            # log aggregated scores of ligand i
            if self.config.num_eval_generate > 0:
                self.writers[str(i)].add_scalar('Tanimoto', tanimoto, epoch)
                self.writers[str(i)].add_scalar('NHOH Count', nhoh_count/nhoh_count_items, epoch)
                self.writers[str(i)].add_scalar('Ring Count', ring_count/ring_count_items, epoch)
                self.writers[str(i)].add_scalar('QED Score', qed_score_count/qed_score_count_items, epoch)
                self.writers[str(i)].add_scalar('LogP', logp_count/logp_count_items, epoch)
                self.writers[str(i)].add_scalar('SAS Score', sascore_count/sascore_count_items, epoch)
                self.writers[str(i)].add_scalar('Docking', docking/docking_items, epoch)

                # print("Valid QED scores: " + str(qed_score_count_items/qed_score_count_items))
                # print("Valid ring counts: " + str(ring_count_items/ring_count_items))
                # print("Valid nhoh counts: " + str(nhoh_count_items/nhoh_count_items))
                print("Tanimoto coefficient of " + protein_name + ": " + str(tanimoto))

                if self.config.molgym_eval:
                    self.writers[str(i)].add_scalar('Reward', reward/self.config.num_eval_generate, epoch)


        if self.config.molgym_eval:
            rewards = 0
            for formula_string in self.config.molgym_eval_formulas:
                formula = Formula(formula_string)
                x_pocket = self.X_pocket_train[0]
                x_ligand = data.Data(
                    x=molgym_formula_to_ligand(formula, device=x_pocket.x.device),
                    edge_index=torch.tensor([[0], [0]], device=x_pocket.x.device),
                    edge_attr=torch.tensor([], device=x_pocket.x.device),
                    name=x_pocket.protein_name,
                    protein_name=x_pocket.protein_name,
                    bfs_index=torch.tensor([], device=x_pocket.x.device),
                    bfs_attr=torch.tensor([], device=x_pocket.x.device),
                )
                protein_name = x_pocket.protein_name

                reward = 0

                for j in range(self.config.num_molgym_eval):
                    pred_generate = self.model(x_pocket, x_ligand, generate=True, coords=self.config.coords, molgym_eval=self.config.molgym_eval)

                    generated_ligand_mol = to_rdkit(
                        Data(x=pred_generate[0], edge_index=pred_generate[1], edge_attr=pred_generate[2], pos=(pred_generate[3] if self.config.coords else None)), device=self.model.device)
                    generated_ligand_mol = rdkit_sanitize(generated_ligand_mol)
                    generated_ligand_svg = mol_to_svg(generated_ligand_mol)

                    reward += pred_generate[4]

                    with open(self.config.svg_dir + "generated_ligand_" + str(epoch) + "_" + str(j) + "_" + protein_name + ".svg", "w") as svg_file:
                        svg_file.write(generated_ligand_svg)
                    if self.config.coords:
                        file_save_name = self.config.svg_dir + "generated_ligand_" + str(epoch) + "_" + str(j) + "_" + protein_name
                        pos_plot_3D(pred_generate[3], pred_generate[1], pred_generate[0], 90, save_name=file_save_name)

                self.writers[formula_string].add_scalar('Reward', reward/self.config.num_molgym_eval, epoch)
                rewards += reward / self.config.num_molgym_eval
                print("Reward for " + formula_string + ": " + str(reward/self.config.num_molgym_eval))
            self.writers['filtered'].add_scalar('Reward', rewards / len(self.config.molgym_eval_formulas), epoch)
            print("Reward for " + formula_string + ": " + str(rewards / len(self.config.molgym_eval_formulas)))

    def test(self) -> None:
            total_loss = 0
            self.model.eval()
            for i in range(len(self.X_ligand_test)-1):
                x_ligand = self.X_ligand_test[i+1]
                x_pocket = self.X_pocket_test[i+1]
                with torch.no_grad():
                    loss_enc = GCNEncoder(in_channels=self.config.num_allowable_atoms,
                                          out_channels=self.config.ligand_encoder_out_channels)
                    prediction = self.model(x_pocket, x_ligand, generate=False, coords=self.config.coords)
                    pred_generate = self.model(x_pocket, x_ligand, generate=True, coords=self.config.coords)

                    #torchgeom_plot(Data(x=pred_generate[0], edge_index=pred_generate[1]))
                    loss_f = torch.nn.L1Loss()

                    loss = loss_f(prediction, torch.tensor(0.0))
                total_loss += loss

            print("Loss on test set: {:.4f}".format(total_loss / len(self.X_ligand_test)))
