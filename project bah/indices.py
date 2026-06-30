"""
src/utils/indices.py
Spectral index computation for multi-band satellite imagery.
All inputs are tensors of shape (B, C, H, W) or (C, H, W).
Band order: [Green=0, Red=1, NIR=2, SWIR=3]
"""

import torch
import torch.nn.functional as F


EPS = 1e-8  # prevent division by zero


def ndvi(x: torch.Tensor) -> torch.Tensor:
    """Normalised Difference Vegetation Index  (NIR - Red) / (NIR + Red)"""
    nir, red = x[..., 2:3, :, :], x[..., 1:2, :, :]
    return (nir - red) / (nir + red + EPS)


def ndwi(x: torch.Tensor) -> torch.Tensor:
    """Normalised Difference Water Index  (Green - NIR) / (Green + NIR)"""
    green, nir = x[..., 0:1, :, :], x[..., 2:3, :, :]
    return (green - nir) / (green + nir + EPS)


def savi(x: torch.Tensor, L: float = 0.5) -> torch.Tensor:
    """Soil-Adjusted Vegetation Index  1.5*(NIR-Red)/(NIR+Red+L)"""
    nir, red = x[..., 2:3, :, :], x[..., 1:2, :, :]
    return 1.5 * (nir - red) / (nir + red + L + EPS)


def ndbi(x: torch.Tensor) -> torch.Tensor:
    """Normalised Difference Built-Up Index  (SWIR - NIR) / (SWIR + NIR)"""
    swir, nir = x[..., 3:4, :, :], x[..., 2:3, :, :]
    return (swir - nir) / (swir + nir + EPS)


def compute_all_indices(x: torch.Tensor) -> dict:
    """Return dict of all four spectral indices for a (B,4,H,W) tensor."""
    return {
        "ndvi": ndvi(x),
        "ndwi": ndwi(x),
        "savi": savi(x),
        "ndbi": ndbi(x),
    }


def index_error(pred: torch.Tensor, target: torch.Tensor) -> dict:
    """
    Compute mean absolute error between predicted and target spectral indices.
    Returns dict with per-index MAE and a boolean pass/fail at 5% threshold.
    """
    pred_idx   = compute_all_indices(pred)
    target_idx = compute_all_indices(target)
    results = {}
    threshold = 0.05
    for name in pred_idx:
        mae = (pred_idx[name] - target_idx[name]).abs().mean().item()
        results[name] = {"mae": mae, "pass": mae < threshold}
    return results
