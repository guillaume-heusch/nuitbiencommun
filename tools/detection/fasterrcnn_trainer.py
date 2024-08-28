import hydra
import pytorch_lightning as pl
import torch
from omegaconf import DictConfig
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.callbacks.progress import TQDMProgressBar

from src.data.detection_dataloader import DetectionDataLoader
from src.engine.fasterrcnn_module import FasterRCNNModule

from torch.utils.data import random_split


@hydra.main(
    version_base=None,
    config_path="../../configs",
    config_name="fasterrcnn_trainer",
)
def run_training(cfg: DictConfig):
    """
    Launch the training of a Faster RCNN performing
    detection of panels with numbers

    """
    torch.manual_seed(cfg.seed)

    # data
    dataset = DetectionDataLoader(cfg)
    dataset_size = len(dataset)
    train_size = int(cfg.data.train_ratio * dataset_size)
    val_size = dataset_size - train_size

    print(f"train dataset size = {train_size}")
    print(f"validation dataset size = {val_size}")

    train_dataset, valid_dataset = random_split(dataset, [train_size, val_size])

    detector = FasterRCNNModule(cfg)

    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=cfg.train.batch_size,
        shuffle=True,
        pin_memory=False,
        num_workers=cfg.data.num_workers,
        collate_fn=collate_fn,
    )

    val_loader = torch.utils.data.DataLoader(
        valid_dataset,
        batch_size=1,
        shuffle=False,
        pin_memory=False,
        num_workers=cfg.data.num_workers,
        collate_fn=collate_fn,
    )

    # TODO: monitor should be on validation F-score
    model_checkpoint = ModelCheckpoint(
        dirpath=cfg.save_model_dir,
        filename="model-{epoch:02d}",
        every_n_epochs=1,
        save_top_k=3,
        save_on_train_epoch_end=True,
        monitor="train_loss",
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

    trainer.fit(detector, train_loader, val_loader)


def collate_fn(batch):
    return tuple(zip(*batch))


if __name__ == "__main__":
    run_training()
