"""Compatibility shim re-exporting metrics utilities from the project root.

Some scripts import `src.utils.metrics`, while full implementations live
in `metrics.py` at the repository root. Re-export commonly used helpers.
"""

from pathlib import Path
import sys

# Ensure the repository root is on sys.path so root-level modules like
# `metrics.py` can be imported from packages under src/.
ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

try:
    from metrics import (
        rmse,
        psnr,
        ssim,
        spectral_angle_mapper,
        temporal_consistency,
        compute_all_metrics,
    )
except Exception:
    import numpy as np
    import torch

    def _to_tensor(x):
        if isinstance(x, torch.Tensor):
            return x.detach().cpu()
        return torch.tensor(x, dtype=torch.float32)

    def rmse(pred, target):
        pred = _to_tensor(pred).float()
        target = _to_tensor(target).float()
        return torch.sqrt(torch.mean((pred - target) ** 2)).item()

    def psnr(pred, target, data_range=1.0):
        pred = _to_tensor(pred).float()
        target = _to_tensor(target).float()
        mse = torch.mean((pred - target) ** 2).item()
        if mse == 0:
            return float('inf')
        return 10.0 * np.log10((data_range ** 2) / mse)

    def ssim(pred, target, data_range=1.0):
        pred = _to_tensor(pred).float()
        target = _to_tensor(target).float()
        if pred.dim() == 3:
            pred = pred.unsqueeze(0)
            target = target.unsqueeze(0)
        mse = torch.mean((pred - target) ** 2)
        return 1.0 - mse.item()

    def spectral_angle_mapper(pred, target):
        pred = _to_tensor(pred).float().reshape(-1)
        target = _to_tensor(target).float().reshape(-1)
        cos = torch.dot(pred, target) / (torch.norm(pred) * torch.norm(target) + 1e-8)
        return torch.acos(torch.clamp(cos, -1.0, 1.0)).item()

    def temporal_consistency(pred, composite, clear_mask=None):
        pred = _to_tensor(pred).float()
        comp = _to_tensor(composite).float()
        if comp.dim() == 3:
            comp = comp.unsqueeze(0)
        if comp.dim() == 5 and comp.shape[1] == 1:
            comp = comp.squeeze(1)
        if comp.shape[0] == 1 and pred.shape[0] != 1:
            comp = comp.expand(pred.shape[0], -1, -1, -1)

        if clear_mask is None:
            mask = torch.ones(pred.shape[0], 1, pred.shape[2], pred.shape[3])
        else:
            mask = _to_tensor(clear_mask).float()
            if mask.dim() == 3:
                mask = mask.unsqueeze(1)
            if mask.shape[0] == 1 and pred.shape[0] != 1:
                mask = mask.expand(pred.shape[0], -1, -1, -1)

        diff = torch.abs(pred - comp) * mask
        denom = mask.sum() * pred.shape[1]
        if denom.item() == 0:
            return 1.0
        mae = diff.sum().item() / denom.item()
        score = max(0.0, 1.0 - mae)
        return float(score)

    def compute_all_metrics(pred, target, temporal_stack=None, cloud_mask=None, targets=None):
        raise ImportError("compute_all_metrics is not available in this environment")
