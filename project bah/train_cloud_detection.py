"""
src/training/train_cloud_detection.py

PyTorch Lightning trainer for the cloud segmentation model (Layer 2).

Trains U-Net++ to classify each pixel into:
  0=clear, 1=thin cloud, 2=thick cloud, 3=cloud shadow

Loss: Dice + Cross-Entropy (combined)
Metrics: IoU per class, mean IoU, F1 per class
"""

from __future__ import annotations
import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR
from typing import Dict, Optional
import logging

log = logging.getLogger(__name__)


# ─────────────────────────────────────────────
# Dice + CE combined loss
# ─────────────────────────────────────────────

class DiceCELoss(nn.Module):
    """Combined Dice + Cross-Entropy loss for multi-class segmentation."""

    def __init__(self, num_classes: int = 4, class_weights: Optional[list] = None,
                 smooth: float = 1.0):
        super().__init__()
        self.num_classes = num_classes
        self.smooth      = smooth
        w = torch.tensor(class_weights, dtype=torch.float32) if class_weights else None
        self.ce = nn.CrossEntropyLoss(weight=w)

    def dice_loss(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        probs   = F.softmax(logits, dim=1)                         # (B,C,H,W)
        targets_oh = F.one_hot(targets, self.num_classes)          # (B,H,W,C)
        targets_oh = targets_oh.permute(0, 3, 1, 2).float()       # (B,C,H,W)

        dims   = (0, 2, 3)
        inter  = (probs * targets_oh).sum(dim=dims)
        union  = probs.sum(dim=dims) + targets_oh.sum(dim=dims)
        dice   = 1.0 - (2.0 * inter + self.smooth) / (union + self.smooth)
        return dice.mean()

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        return self.ce(logits, targets) + self.dice_loss(logits, targets)


# ─────────────────────────────────────────────
# IoU metric
# ─────────────────────────────────────────────

def compute_iou_per_class(preds: torch.Tensor, targets: torch.Tensor,
                           num_classes: int = 4) -> Dict[str, float]:
    iou = {}
    pred_flat   = preds.view(-1)
    target_flat = targets.view(-1)
    for cls in range(num_classes):
        p = (pred_flat == cls)
        t = (target_flat == cls)
        inter = (p & t).sum().float()
        union = (p | t).sum().float()
        iou[f"iou_class_{cls}"] = (inter / (union + 1e-8)).item()
    iou["mean_iou"] = sum(iou.values()) / num_classes
    return iou


# ─────────────────────────────────────────────
# Lightning Module
# ─────────────────────────────────────────────

class CloudDetectionModule(pl.LightningModule):
    """
    Lightning module for training the cloud segmentation model.
    Handles training, validation, logging, and checkpoint saving.
    """

    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.save_hyperparameters(dict(cfg))

        from src.models.cloud_detection.unetplusplus import CloudDetector
        self.model = CloudDetector(
            in_channels = len(cfg.data.bands.optical),
            num_classes = cfg.cloud_detection.num_classes,
            dropout     = cfg.cloud_detection.dropout,
            pretrained  = True,
        )
        self.loss_fn = DiceCELoss(
            num_classes   = cfg.cloud_detection.num_classes,
            class_weights = list(cfg.cloud_detection.training.class_weights),
        )
        self.num_classes = cfg.cloud_detection.num_classes

    def forward(self, x):
        return self.model(x)

    def _shared_step(self, batch: Dict, stage: str) -> torch.Tensor:
        optical    = batch["optical"]     # (B,4,H,W)
        cloud_mask = batch["cloud_mask"]  # (B,H,W) int

        out    = self.model(optical)
        logits = out["logits"]            # (B,4,H,W)
        preds  = out["class_map"]         # (B,H,W)

        loss = self.loss_fn(logits, cloud_mask)
        iou  = compute_iou_per_class(preds, cloud_mask, self.num_classes)

        self.log(f"{stage}/loss",     loss,            prog_bar=True,  sync_dist=True)
        self.log(f"{stage}/mean_iou", iou["mean_iou"], prog_bar=True,  sync_dist=True)
        for k, v in iou.items():
            self.log(f"{stage}/{k}", v, sync_dist=True)

        return loss

    def training_step(self, batch, batch_idx):
        return self._shared_step(batch, "train")

    def validation_step(self, batch, batch_idx):
        return self._shared_step(batch, "val")

    def test_step(self, batch, batch_idx):
        return self._shared_step(batch, "test")

    def configure_optimizers(self):
        cfg = self.cfg.cloud_detection.training
        opt = AdamW(self.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)

        warmup = LinearLR(opt, start_factor=0.1, total_iters=cfg.warmup_epochs)
        cosine = CosineAnnealingLR(opt, T_max=cfg.epochs - cfg.warmup_epochs)
        sched  = SequentialLR(opt, [warmup, cosine],
                               milestones=[cfg.warmup_epochs])
        return {"optimizer": opt, "lr_scheduler": {"scheduler": sched, "interval": "epoch"}}


# ─────────────────────────────────────────────
# Trainer entry point
# ─────────────────────────────────────────────

class CloudDetectionTrainer:
    def __init__(self, cfg, resume: Optional[str] = None, debug: bool = False):
        self.cfg    = cfg
        self.resume = resume
        self.debug  = debug

    def fit(self):
        from src.data.loaders.cloud_removal_dataset import CloudRemovalDataset
        import pytorch_lightning as pl
        from pytorch_lightning.callbacks import (
            ModelCheckpoint, EarlyStopping, LearningRateMonitor, RichProgressBar
        )
        from pytorch_lightning.loggers import WandbLogger

        # Data
        train_ds = CloudRemovalDataset.from_config(self.cfg, "train")
        val_ds   = CloudRemovalDataset.from_config(self.cfg, "val")
        train_dl = train_ds.get_dataloader(
            batch_size  = self.cfg.cloud_detection.training.batch_size,
            shuffle     = True,
            num_workers = self.cfg.hardware.num_workers,
        )
        val_dl = val_ds.get_dataloader(
            batch_size  = self.cfg.cloud_detection.training.batch_size,
            shuffle     = False,
            num_workers = self.cfg.hardware.num_workers,
        )

        # Module
        module = CloudDetectionModule(self.cfg)

        # Callbacks
        callbacks = [
            ModelCheckpoint(
                dirpath    = f"{self.cfg.paths.checkpoints}",
                filename   = "cloud_detection_{epoch:02d}_{val/mean_iou:.4f}",
                monitor    = "val/mean_iou",
                mode       = "max",
                save_top_k = self.cfg.logging.save_top_k,
                verbose    = True,
            ),
            EarlyStopping(monitor="val/mean_iou", patience=15, mode="max"),
            LearningRateMonitor(logging_interval="epoch"),
            RichProgressBar(),
        ]

        # Logger
        logger = None
        if self.cfg.logging.logger == "wandb":
            logger = WandbLogger(
                project = self.cfg.project.name,
                name    = "cloud_detection",
            )

        # Trainer
        trainer = pl.Trainer(
            max_epochs        = 2 if self.debug else self.cfg.cloud_detection.training.epochs,
            accelerator       = "gpu" if torch.cuda.is_available() else "cpu",
            devices           = self.cfg.hardware.gpus,
            precision         = self.cfg.hardware.precision,
            callbacks         = callbacks,
            logger            = logger,
            log_every_n_steps = self.cfg.logging.log_every_n_steps,
            limit_train_batches = 2 if self.debug else 1.0,
            limit_val_batches   = 2 if self.debug else 1.0,
        )
        trainer.fit(module, train_dl, val_dl, ckpt_path=self.resume)
        log.info("Cloud detection training complete.")
