from __future__ import annotations

from typing import Dict, Tuple

import torch


class Trainer:
    def __init__(self, model: nn.Module, data, optimizer, config: Config):
        self.model = model
        self.data = data
        self.optimizer = optimizer
        self.config = config

        self.loss_history: Dict[str, float] = {}
        self.auc_history: Dict[str, float] = {}
        self.ap_history: Dict[str, float] = {}

    def train(self) -> None:
        for epoch in range(1, config.num_epochs):
            loss = self.trainining_iteration()
            self.loss_dict[epoch] = loss

            auc, ap = self.test(
                self.data.test_pos_edge_index, self.data.test_neg_edge_index
            )
            self.auc_history[epoch] = auc
            self.ap_history[epoch] = ap

            print("Epoch: {:03d}, AUC: {:.4f}, AP: {:.4f}".format(epoch, auc, ap))

    def training_iteration(self) -> float:
        self.model.train()
        total_loss = 0
        for data in d[0:3500]:
            self.optimizer.zero_grad()
            print(data.x)
            z = self.model.encode(data.x, data.train_pos_edge_index)
            loss = self.model.recon_loss(z, data.train_pos_edge_index)
            loss = loss + (1 / data.num_nodes) * model.kl_loss()
            loss.backward()
            self.optimizer.step()
            total_loss += loss.item()
        return float(total_loss)

    def test(self, pos_edge_index, neg_edge_index) -> Tuple[float, float]:
        auc_total = 0
        ap_total = 0
        for data in d[3500:]:
            model.eval()
            with torch.no_grad():
                z = model.encode(data.x, data.train_pos_edge_index)
            auc_, ap_ = model.test(z, pos_edge_index, neg_edge_index)
            auc_total += auc_
            ap_total += ap_

        return auc_total / len(d[3500:]), ap_total / len(d[3500:])
