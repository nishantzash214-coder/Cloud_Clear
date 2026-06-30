"""
src/training/train_reconstruction.py

PyTorch Lightning trainer for the full reconstruction pipeline (Layer 4).

Training strategy:
  - Stage A (epochs 0–30):  train with L1 + perceptual only (stable base)
  - Stage B (epochs 30–70): add spectral + physical consistency losses
  - Stage C (epochs 70–100): add temporal consistency + full loss

This curriculum prevents the model from getting stuck optimising
spectral constraints before the reconstruction quality is adequate.

GAN branch uses alternating G/D training with separate optimisers.
"""

from __future__ import annotations
import torch
import torch.nn as nn
import pytorch_lightning as pl
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
from typing import Dict, Optional, List
import logging

log = logging.getLogger(__name__)


# ─────────────────────────────────────────────
# Multi-scale PatchGAN Discriminator
# ─────────────────────────────────────────────

class PatchDiscriminator(nn.Module):
    """Single-scale PatchGAN discriminator for 4-band imagery."""

    def __init__(self, in_ch: int = 4, ndf: int = 64, n_layers: int = 3):
        super().__init__()
        layers = [
            nn.Conv2d(in_ch * 2, ndf, 4, stride=2, padding=1),  # concat pred+target
            nn.LeakyReLU(0.2, True),
        ]
        ch = ndf
        for _ in range(n_layers - 1):
            layers += [
                nn.Conv2d(ch, ch * 2, 4, stride=2, padding=1, bias=False),
                nn.InstanceNorm2d(ch * 2),
                nn.LeakyReLU(0.2, True),
            ]
            ch *= 2
        layers += [
            nn.Conv2d(ch, ch * 2, 4, stride=1, padding=1, bias=False),
            nn.InstanceNorm2d(ch * 2),
            nn.LeakyReLU(0.2, True),
            nn.Conv2d(ch * 2, 1, 4, stride=1, padding=1),
        ]
        self.model = nn.Sequential(*layers)

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        return self.model(torch.cat([pred, target], dim=1))


class MultiScaleDiscriminator(nn.Module):
    """3-scale PatchGAN discriminator (Pix2PixHD style)."""

    def __init__(self, in_ch: int = 4, ndf: int = 64, n_discriminators: int = 3):
        super().__init__()
        self.discriminators = nn.ModuleList([
            PatchDiscriminator(in_ch, ndf, n_layers=3)
            for _ in range(n_discriminators)
        ])
        self.downsample = nn.AvgPool2d(3, stride=2, padding=1, count_include_pad=False)

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> List[torch.Tensor]:
        outputs = []
        for d in self.discriminators:
            outputs.append(d(pred, target))
            pred   = self.downsample(pred)
            target = self.downsample(target)
        return outputs


def gan_loss_d(real_outs: List, fake_outs: List) -> torch.Tensor:
    """Discriminator hinge loss: D should output >0 for real, <0 for fake."""
    loss = 0.0
    for r, f in zip(real_outs, fake_outs):
        loss += torch.relu(1.0 - r).mean() + torch.relu(1.0 + f).mean()
    return loss / len(real_outs)


def gan_loss_g(fake_outs: List) -> torch.Tensor:
    """Generator hinge loss: G should make D output high values for fake."""
    return sum(-f.mean() for f in fake_outs) / len(fake_outs)


# ─────────────────────────────────────────────
# Curriculum loss weighting
# ─────────────────────────────────────────────

def get_loss_weights(epoch: int, cfg) -> Dict[str, float]:
    """
    Curriculum learning: gradually activate scientific loss terms.
    Stage A: 0–30   → L1 + perceptual only
    Stage B: 30–70  → add spectral + physical
    Stage C: 70–100 → full loss including temporal
    """
    w = dict(cfg.losses)

    if epoch < 30:
        w["spectral_consistency"] = 0.0
        w["physical_consistency"] = 0.0
        w["temporal_consistency"] = 0.0
        w["adversarial"]          = 0.0
    elif epoch < 70:
        ramp = (epoch - 30) / 40.0           # 0 → 1 over epochs 30–70
        w["spectral_consistency"] = float(cfg.losses.spectral_consistency) * ramp
        w["physical_consistency"] = float(cfg.losses.physical_consistency) * ramp
        w["temporal_consistency"] = 0.0
        w["adversarial"]          = float(cfg.losses.adversarial) * ramp * 0.5
    else:
        ramp = min((epoch - 70) / 30.0, 1.0)
        w["temporal_consistency"] = float(cfg.losses.temporal_consistency) * ramp

    return w


# ─────────────────────────────────────────────
# Lightning Module
# ─────────────────────────────────────────────

class ReconstructionModule(pl.LightningModule):
    """
    Lightning module for training the full reconstruction pipeline.
    Uses automatic_optimization=False for separate G/D optimiser steps.
    """

    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.save_hyperparameters(dict(cfg))
        self.automatic_optimization = False   # manual for GAN training

        from src.models.reconstruction.pipeline import ReconstructionPipeline
        from src.losses.scientific_losses import CloudRemovalLoss

        self.generator     = ReconstructionPipeline(cfg)
        self.discriminator = MultiScaleDiscriminator(
            in_ch            = len(cfg.data.bands.optical),
            ndf              = cfg.reconstruction.gan.ndf,
            n_discriminators = cfg.reconstruction.gan.discriminators,
        )
        self.loss_fn = CloudRemovalLoss(cfg)

    def forward(self, batch):
        return self.generator(batch)

    def training_step(self, batch: Dict, batch_idx: int):
        opt_g, opt_d = self.optimizers()
        sched_g, sched_d = self.lr_schedulers()

        optical     = batch["optical"]
        target      = batch["target"]
        cloud_mask  = batch["cloud_mask"]
        composite   = batch.get("composite")
        change_prob = None

        w = get_loss_weights(self.current_epoch, self.cfg)

        # ── Generator step ─────────────────────────────────────────
        self.toggle_optimizer(opt_g)
        out       = self.generator(batch)
        pred      = out["reconstructed"]

        # Scientific losses
        mask_float = (cloud_mask > 0).float().unsqueeze(1)
        losses     = self.loss_fn(pred, target, cloud_mask, composite, change_prob)

        # GAN loss (generator side)
        if w["adversarial"] > 0:
            fake_outs = self.discriminator(pred.detach(), target)
            g_adv     = gan_loss_g(fake_outs)
            g_total   = losses["loss"] + w["adversarial"] * g_adv
        else:
            g_total = losses["loss"]

        self.manual_backward(g_total)
        torch.nn.utils.clip_grad_norm_(
            self.generator.parameters(),
            self.cfg.reconstruction.training.gradient_clip,
        )
        opt_g.step()
        opt_g.zero_grad()
        self.untoggle_optimizer(opt_g)

        # ── Discriminator step ─────────────────────────────────────
        if w["adversarial"] > 0:
            self.toggle_optimizer(opt_d)
            real_outs = self.discriminator(target,         target)
            fake_outs = self.discriminator(pred.detach(),  target)
            d_loss    = gan_loss_d(real_outs, fake_outs)
            self.manual_backward(d_loss)
            opt_d.step()
            opt_d.zero_grad()
            self.untoggle_optimizer(opt_d)
            self.log("train/d_loss", d_loss, prog_bar=False)

        # ── Logging ────────────────────────────────────────────────
        self.log("train/loss",       g_total,           prog_bar=True)
        self.log("train/l1",         losses["l1"],      prog_bar=False)
        self.log("train/spectral",   losses["spectral"],prog_bar=False)
        self.log("train/physical",   losses["physical"],prog_bar=False)
        self.log("train/temporal",   losses["temporal"],prog_bar=False)
        self.log("train/perceptual", losses["perceptual"], prog_bar=False)
        # Log branch weight means for interpretability
        bw = out["branch_weights"].mean(dim=(0, 2, 3))
        for i, name in enumerate(["diffusion", "gan", "temporal", "sar"]):
            self.log(f"train/weight_{name}", bw[i].item())

        return g_total

    def validation_step(self, batch: Dict, batch_idx: int):
        optical    = batch["optical"]
        target     = batch["target"]
        cloud_mask = batch["cloud_mask"]
        composite  = batch.get("composite")

        out   = self.generator(batch)
        pred  = out["reconstructed"]

        losses = self.loss_fn(pred, target, cloud_mask, composite)

        # Compute scientific metrics
        from src.utils.metrics import ssim, psnr, rmse
        from src.utils.indices import index_error
        ssim_score = ssim(pred, target)
        psnr_score = psnr(pred, target)
        rmse_score = rmse(pred, target)
        idx_err    = index_error(pred, target)

        self.log("val/loss",        losses["loss"],    prog_bar=True,  sync_dist=True)
        self.log("val/ssim",        ssim_score,        prog_bar=True,  sync_dist=True)
        self.log("val/psnr",        psnr_score,        prog_bar=False, sync_dist=True)
        self.log("val/rmse",        rmse_score,        prog_bar=False, sync_dist=True)
        self.log("val/ndvi_error",  idx_err["ndvi"]["mae"], sync_dist=True)
        self.log("val/ndwi_error",  idx_err["ndwi"]["mae"], sync_dist=True)
        self.log("val/l1",          losses["l1"],      sync_dist=True)
        return losses["loss"]

    def configure_optimizers(self):
        cfg = self.cfg.reconstruction.training
        opt_g = AdamW(self.generator.parameters(),
                      lr=cfg.lr, weight_decay=cfg.weight_decay, betas=(0.9, 0.999))
        opt_d = AdamW(self.discriminator.parameters(),
                      lr=cfg.lr * 0.5, weight_decay=cfg.weight_decay, betas=(0.5, 0.999))

        sched_g = CosineAnnealingWarmRestarts(opt_g, T_0=20, T_mult=2)
        sched_d = CosineAnnealingWarmRestarts(opt_d, T_0=20, T_mult=2)

        return (
            [opt_g, opt_d],
            [
                {"scheduler": sched_g, "interval": "epoch"},
                {"scheduler": sched_d, "interval": "epoch"},
            ]
        )


# ─────────────────────────────────────────────
# Trainer entry point
# ─────────────────────────────────────────────

class ReconstructionTrainer:
    def __init__(self, cfg, resume: Optional[str] = None, debug: bool = False):
        self.cfg    = cfg
        self.resume = resume
        self.debug  = debug

    def fit(self):
        from src.data.loaders.cloud_removal_dataset import CloudRemovalDataset
        from pytorch_lightning.callbacks import (
            ModelCheckpoint, EarlyStopping, LearningRateMonitor,
            RichProgressBar, StochasticWeightAveraging,
        )
        from pytorch_lightning.loggers import WandbLogger

        train_ds = CloudRemovalDataset.from_config(self.cfg, "train")
        val_ds   = CloudRemovalDataset.from_config(self.cfg, "val")
        train_dl = train_ds.get_dataloader(
            batch_size  = self.cfg.reconstruction.training.batch_size,
            num_workers = self.cfg.hardware.num_workers,
        )
        val_dl = val_ds.get_dataloader(
            batch_size  = self.cfg.reconstruction.training.batch_size,
            shuffle     = False,
            num_workers = self.cfg.hardware.num_workers,
        )

        module = ReconstructionModule(self.cfg)

        callbacks = [
            ModelCheckpoint(
                dirpath  = self.cfg.paths.checkpoints,
                filename = "reconstruction_{epoch:02d}_{val/ssim:.4f}",
                monitor  = "val/ssim",
                mode     = "max",
                save_top_k = self.cfg.logging.save_top_k,
                verbose    = True,
            ),
            EarlyStopping(monitor="val/ssim", patience=20, mode="max"),
            LearningRateMonitor(logging_interval="epoch"),
            RichProgressBar(),
            StochasticWeightAveraging(swa_lrs=1e-4, swa_epoch_start=0.8),
        ]

        logger = None
        if self.cfg.logging.logger == "wandb":
            logger = WandbLogger(project=self.cfg.project.name, name="reconstruction")

        trainer = pl.Trainer(
            max_epochs          = 2 if self.debug else self.cfg.reconstruction.training.epochs,
            accelerator         = "gpu" if torch.cuda.is_available() else "cpu",
            devices             = self.cfg.hardware.gpus,
            precision           = self.cfg.hardware.precision,
            callbacks           = callbacks,
            logger              = logger,
            log_every_n_steps   = self.cfg.logging.log_every_n_steps,
            limit_train_batches = 2 if self.debug else 1.0,
            limit_val_batches   = 2 if self.debug else 1.0,
            gradient_clip_val   = self.cfg.reconstruction.training.gradient_clip,
        )
        trainer.fit(module, train_dl, val_dl, ckpt_path=self.resume)
        log.info("Reconstruction training complete.")
