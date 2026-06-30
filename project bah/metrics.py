"""
src/utils/metrics.py
Scientific validation metrics for reconstructed satellite imagery.
All functions accept torch tensors of shape (B, C, H, W) or (C, H, W).
"""

from __future__ import annotations
import torch
import torch.nn.functional as F
import numpy as np
from typing import Optional
from src.utils.indices import index_error


# ─────────────────────────────────────────────
# Basic pixel metrics
# ─────────────────────────────────────────────

def rmse(pred: torch.Tensor, target: torch.Tensor) -> float:
    """Root Mean Squared Error."""
    return torch.sqrt(F.mse_loss(pred, target)).item()


def psnr(pred: torch.Tensor, target: torch.Tensor, data_range: float = 1.0) -> float:
    """Peak Signal-to-Noise Ratio (dB)."""
    mse_val = F.mse_loss(pred, target).item()
    if mse_val == 0:
        return float("inf")
    return 10 * np.log10(data_range**2 / mse_val)


def ssim(
    pred: torch.Tensor,
    target: torch.Tensor,
    window_size: int = 11,
    data_range: float = 1.0,
) -> float:
    """
    Structural Similarity Index (per-image mean over channels and batch).
    Uses a Gaussian sliding window.
    """
    C1 = (0.01 * data_range) ** 2
    C2 = (0.03 * data_range) ** 2

    def _gaussian_window(size: int, sigma: float = 1.5) -> torch.Tensor:
        coords = torch.arange(size, dtype=torch.float32) - size // 2
        g = torch.exp(-(coords**2) / (2 * sigma**2))
        g /= g.sum()
        return g.outer(g).unsqueeze(0).unsqueeze(0)

    if pred.dim() == 3:
        pred, target = pred.unsqueeze(0), target.unsqueeze(0)

    B, C, H, W = pred.shape
    kernel = _gaussian_window(window_size).to(pred.device)
    kernel = kernel.expand(C, 1, window_size, window_size)
    pad = window_size // 2

    mu1 = F.conv2d(pred,   kernel, padding=pad, groups=C)
    mu2 = F.conv2d(target, kernel, padding=pad, groups=C)
    mu1_sq, mu2_sq, mu12 = mu1**2, mu2**2, mu1 * mu2

    sigma1_sq = F.conv2d(pred   * pred,   kernel, padding=pad, groups=C) - mu1_sq
    sigma2_sq = F.conv2d(target * target, kernel, padding=pad, groups=C) - mu2_sq
    sigma12   = F.conv2d(pred   * target, kernel, padding=pad, groups=C) - mu12

    num = (2 * mu12 + C1) * (2 * sigma12 + C2)
    den = (mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2)
    return (num / den).mean().item()


# ─────────────────────────────────────────────
# Spectral metrics
# ─────────────────────────────────────────────

def spectral_angle_mapper(pred: torch.Tensor, target: torch.Tensor) -> float:
    """
    Spectral Angle Mapper (SAM) in radians.
    Measures spectral similarity regardless of illumination magnitude.
    Lower = better; target < 0.10 rad for scientific use.
    """
    if pred.dim() == 3:
        pred, target = pred.unsqueeze(0), target.unsqueeze(0)

    # (B, C, H*W)
    p = pred.flatten(2)
    t = target.flatten(2)

    dot     = (p * t).sum(dim=1)
    norm_p  = p.norm(dim=1).clamp(min=1e-8)
    norm_t  = t.norm(dim=1).clamp(min=1e-8)
    cosine  = (dot / (norm_p * norm_t)).clamp(-1 + 1e-7, 1 - 1e-7)
    return torch.acos(cosine).mean().item()


# ─────────────────────────────────────────────
# Temporal consistency
# ─────────────────────────────────────────────

def temporal_consistency(
    pred: torch.Tensor,
    temporal_stack: torch.Tensor,
    cloud_mask: Optional[torch.Tensor] = None,
) -> float:
    """
    Temporal consistency score: how well the reconstructed image fits
    the ±15-day temporal neighbourhood.

    Args:
        pred:           Reconstructed image  (B, C, H, W)
        temporal_stack: Stack of ±15-day scenes  (B, T, C, H, W)
        cloud_mask:     Optional clear-sky mask for reference pixels  (B, H, W)

    Returns:
        Score in [0, 1]. Higher is better. Target > 0.85.
    """
    B, T, C, H, W = temporal_stack.shape
    temporal_median = temporal_stack.median(dim=1).values  # (B, C, H, W)

    if cloud_mask is not None:
        mask = cloud_mask.unsqueeze(1).float()  # (B, 1, H, W)
        diff = ((pred - temporal_median).abs() * mask).sum() / (mask.sum() * C + 1e-8)
    else:
        diff = (pred - temporal_median).abs().mean()

    return max(0.0, 1.0 - diff.item())


# ─────────────────────────────────────────────
# Full validation report
# ─────────────────────────────────────────────

def compute_all_metrics(
    pred: torch.Tensor,
    target: torch.Tensor,
    temporal_stack: Optional[torch.Tensor] = None,
    cloud_mask: Optional[torch.Tensor] = None,
    targets: Optional[dict] = None,
) -> dict:
    """
    Compute all scientific validation metrics and compare against targets.

    Returns a dict suitable for JSON / PDF report generation.
    """
    default_targets = {
        "ssim":                 0.90,
        "psnr":                 35.0,
        "rmse":                 0.03,
        "ndvi_error":           0.05,
        "ndwi_error":           0.05,
        "sam":                  0.10,
        "temporal_consistency": 0.85,
    }
    t = {**default_targets, **(targets or {})}

    idx_errors = index_error(pred, target)
    metrics = {
        "ssim":      ssim(pred, target),
        "psnr":      psnr(pred, target),
        "rmse":      rmse(pred, target),
        "ndvi_error": idx_errors["ndvi"]["mae"],
        "ndwi_error": idx_errors["ndwi"]["mae"],
        "savi_error": idx_errors["savi"]["mae"],
        "ndbi_error": idx_errors["ndbi"]["mae"],
        "sam":        spectral_angle_mapper(pred, target),
    }
    if temporal_stack is not None:
        metrics["temporal_consistency"] = temporal_consistency(
            pred, temporal_stack, cloud_mask
        )

    # Pass / fail against scientific thresholds
    metrics["validation"] = {
        k: {"value": metrics[k], "target": t[k], "pass": metrics[k] >= t[k]}
        if k in ["ssim", "psnr", "temporal_consistency"]
        else {"value": metrics[k], "target": t[k], "pass": metrics[k] <= t[k]}
        for k in t if k in metrics
    }
    metrics["overall_pass"] = all(v["pass"] for v in metrics["validation"].values())
    return metrics
