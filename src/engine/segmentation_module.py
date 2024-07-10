import logging

import numpy as np
import pytorch_lightning as pl
import segmentation_models_pytorch as smp
import torch
from omegaconf import DictConfig


class SegmentationModule(pl.LightningModule):
    """
    Pytorch Lighting module to perform segmentation

    Attributes
    ----------
    cfg: DictConfig
        The configuration
    loss: smp.losses.DiceLoss
        The Dice loss
    accuracy: list
        Pixel accuracies
    training_step_outputs: list
        Batch losses

    """

    def __init__(self, cfg: DictConfig):
        """
        Init function

        Parameters
        ----------
        cfg: DictConfig
            the configuration

        """
        super().__init__()
        self.cfg = cfg
        if cfg.model.name == "DeepLabV3Plus":
            self.model = smp.DeepLabV3Plus(
                encoder_name=cfg.model.encoder_name,
                encoder_weights=cfg.model.encoder_weights,
                classes=1,
            )
        else:
            logging.error("{cfg.model_name} is not supported")

        self.loss = smp.losses.DiceLoss(
            smp.losses.BINARY_MODE, from_logits=True
        )
        self.accuracy = []
        self.training_step_outputs = []
        self.save_hyperparameters()

    def forward(self, batch):
        imgs, _ = batch
        return self.model(imgs)

    def training_step(self, batch):
        imgs, labels = batch
        logits_mask = self.model(imgs)
        assert labels.max() <= 1.0 and labels.min() >= 0
        batch_loss = self.loss(logits_mask, labels)
        self.training_step_outputs.append(batch_loss)
        return {"loss": batch_loss}

    def on_train_epoch_end(self):
        loss = np.array([])
        for batch_loss in self.training_step_outputs:
            loss = np.append(loss, batch_loss.detach().cpu())
        self.training_step_outputs.clear()
        self.log("metrics/epoch/train_loss", loss.mean(), prog_bar=True)

    def validation_step(self, batch, batch_idx):
        imgs, labels = batch
        logits_mask = self.model(imgs)
        tp, fp, fn, tn = smp.metrics.get_stats(
            logits_mask, labels, mode="binary", threshold=0.5
        )
        self.accuracy.append(
            smp.metrics.accuracy(tp, fp, fn, tn, reduction="macro").item()
        )
        return tp, fp, fn, tn

    def on_validation_epoch_end(self):
        """
        Log pixel accuracy
        """
        accuracy = np.mean(self.accuracy)
        self.log("metrics/epoch/acc", accuracy)
        self.accuracy = []

    def configure_optimizers(self):
        return torch.optim.Adam(
            self.parameters(), lr=self.cfg.train.learning_rate
        )

    def predict(self, batch):
        imgs, labels = batch
        return self.model(imgs)
