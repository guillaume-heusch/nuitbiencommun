import hydra
import pytorch_lightning as pl
import torch
from omegaconf import DictConfig
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.callbacks.progress import TQDMProgressBar

from src.data.segmentation_dataloader import SegmentationDataLoader
from src.engine.segmentation_module import SegmentationModule


@hydra.main(
    version_base=None,
    config_path="../../configs",
    config_name="segmentation_trainer",
)
def run_training(cfg: DictConfig):
    """
    Launch the training of a model performing
    semantic segmentation of panels with numbers 

    """
    segmenter = SegmentationModule(cfg)

    train_dataset = SegmentationDataLoader(cfg)

    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=cfg.train.batch_size,
        shuffle=True,
        pin_memory=False,
        num_workers=cfg.data.num_workers,
    )
    val_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=1,
        shuffle=False,
        pin_memory=False,
        num_workers=1,
    )

    model_checkpoint = ModelCheckpoint(
        dirpath=cfg.save_model_dir,
        filename="model-{epoch:02d}",
        every_n_epochs=1,
        save_top_k=3,
        save_on_train_epoch_end=True,
        monitor="metrics/epoch/train_loss",
    )

    trainer = pl.Trainer(
        accelerator="cpu",
        devices=1,
        max_epochs=cfg.train.max_epochs,
        callbacks=[
            TQDMProgressBar(refresh_rate=cfg.train.epoch_refresh_rate),
            model_checkpoint,
        ],
        log_every_n_steps=1,
    )

    trainer.fit(segmenter, train_loader, val_loader)


if __name__ == "__main__":
    run_training()
