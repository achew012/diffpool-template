import os
from torch import nn
import torch
from typing import List, Any, Dict

#from common.utils import *
import pytorch_lightning as pl

import torch
import torch.nn.functional as F

from clearml import Dataset as ClearML_Dataset
from common.utils import *
from model.DiffPool import DiffPool
import ipdb


class GraphEmbedding(pl.LightningModule):
    """Pytorch Lightning module. It wraps up the model, data loading and training code"""

    def __init__(self, cfg, task):
        """Loads the model, the tokenizer and the metric."""
        super().__init__()
        self.cfg = cfg
        self.task = task
        self.clearml_logger = self.task.get_logger()

        self.model = DiffPool(self.cfg)
        # clearml_data_object = ClearML_Dataset.get(
        #     dataset_name=self.cfg.clearml_dataset_name,
        #     dataset_project=self.cfg.clearml_dataset_project_name,
        #     dataset_tags=list(self.cfg.clearml_dataset_tags),
        #     only_published=False,
        # )
        # self.dataset_path = clearml_data_object.get_local_copy()

    def forward(self, batch):
        output, l_norm, entropy_reg = self.model(
            batch.x, batch.adj, batch.mask)
        sim_mat = cos_sim(output, output)
        loss = torch.nn.MSELoss()(sim_mat, generate_indicator_matrix(batch).type_as(output))
        return loss, output

    def training_step(self, batch, batch_nb):
        ipdb.set_trace()
        """Call the forward pass then return loss"""
        loss, output = self.forward(batch)
        return {"loss": loss}

    def training_epoch_end(self, outputs):
        total_loss = []
        for batch in outputs:
            total_loss.append(batch["loss"])
        self.log("train_loss", sum(total_loss) / len(total_loss))

    def eval_step(self, batch):
        loss, output = self.model(batch.x, batch.adj, batch.mask)
        pred = output[0]
        pred = cos_sim(pred, pred)  # N * N
        top_k = torch.topk(pred, self.cfg.topk, dim=-1).indices[:, 1:]
        #groundtruth_matrix = batch.y.squeeze(dim=-1).repeat(batch.y.shape[0], 1)
        #print("gt: ", groundtruth_matrix.shape)
        groundtruth_matrix = generate_indicator_matrix(batch)
        selected_graph = torch.gather(groundtruth_matrix, dim=-1, index=top_k)
        hit_k = selected_graph.sum()
        return loss, (hit_k / (ground_truth_count(batch) - batch.y.shape[0]))

    def val_step(self, batch, batch_nb):
        """Call the forward pass then return loss"""
        if batch.x.shape[0] < 30:
            loss = None
            hit_k_normalized = None
        else:
            loss, hit_k_normalized = self.eval_step(**batch)
        return {"loss": loss, "hit_k_normalized": hit_k_normalized}

    def validation_epoch_end(self, outputs):
        hit_rate = []
        for batch in outputs:
            batch_hit_k_normalized = batch["hit_k_normalized"]
            if batch_hit_k_normalized:
                hit_rate.append(batch_hit_k_normalized)
        self.log("hits_K", torch.tensor(hit_rate).mean())

    def test_step(self, batch, batch_nb):
        """Call the forward pass then return loss"""
        loss, output = self(batch.x, batch.adj, batch.mask)
        pred = output[0].max(dim=1)[1]
        return {"loss": loss, "batch": batch, "pred": pred}

    def test_epoch_end(self, outputs):
        correct = 0
        num_batches = len(outputs)
        for batch in outputs:
            pred = batch["pred"]
            correct += pred.eq(batch['batch'].y.view(-1)).sum().item()
        self.log("accuracy", correct / len(num_batches)*self.cfg.batch_size)

    def configure_optimizers(self):
        """Configure the optimizer and the learning rate scheduler"""
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.cfg.lr)

        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": torch.optim.lr_scheduler.ReduceLROnPlateau(
                    optimizer, "min", verbose=True
                ),
                "monitor": "val_loss",
                "frequency": 1,
            },
        }
