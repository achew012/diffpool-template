import os
from torch import nn
import torch
from typing import List, Any, Dict

# from data import NERDataset
from common.utils import *
from metric.eval import eval_ceaf
from transformers import (
    AutoTokenizer,
    AutoModelForQuestionAnswering,
    AutoConfig,
    set_seed,
    get_linear_schedule_with_warmup,
)
import torch
import pytorch_lightning as pl
from clearml import Dataset as ClearML_Dataset
import ipdb


class DiffPool(pl.LightningModule):
    """Pytorch Lightning module. It wraps up the model, data loading and training code"""

    def __init__(self, cfg, task):
        """Loads the model, the tokenizer and the metric."""
        super().__init__()
        self.cfg = cfg
        self.task = task
        self.clearml_logger = self.task.get_logger()

        clearml_data_object = ClearML_Dataset.get(
            dataset_name=self.cfg.clearml_dataset_name,
            dataset_project=self.cfg.clearml_dataset_project_name,
            dataset_tags=list(self.cfg.clearml_dataset_tags),
            only_published=False,
        )
        self.dataset_path = clearml_data_object.get_local_copy()



    def forward(self, **batch):
        return outputs

    def training_step(self, batch, batch_nb):
        """Call the forward pass then return loss"""
        docids = batch.pop("docid", None)
        gold_mentions = batch.pop("gold_mentions", None)

        outputs = self.forward(**batch)
        return {"loss": outputs.loss}

    def training_epoch_end(self, outputs):
        total_loss = []
        for batch in outputs:
            total_loss.append(batch["loss"])
        self.log("train_loss", sum(total_loss) / len(total_loss))

    # def validation_step(self, batch, batch_nb):
    #     out = self._evaluation_step("val", batch, batch_nb)
    #     return {"results": out["preds"], "loss": out["loss"]}

    # def validation_epoch_end(self, outputs):

    # def test_step(self, batch, batch_nb):

    # #################################################################################
    # def test_epoch_end(self, outputs):
    #     return {"results": results}

    # def configure_optimizers(self):
    #     """Configure the optimizer and the learning rate scheduler"""

    #     optimizer = torch.optim.AdamW(self.parameters(), lr=self.cfg.lr)
    #     return {
    #         "optimizer": optimizer,
    #         "lr_scheduler": {
    #             "scheduler": torch.optim.lr_scheduler.ReduceLROnPlateau(
    #                 optimizer, "min", verbose=True
    #             ),
    #             "monitor": "val_loss",
    #             "frequency": 1,
    #         },
    #     }
