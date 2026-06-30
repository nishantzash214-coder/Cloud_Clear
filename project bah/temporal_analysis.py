"""
src/data/preprocessing/temporal_analysis.py

Temporal Analysis Engine (Layer 3).

Builds a ±15-day temporal context from a stack of co-registered scenes:
  1. Temporal median composite  (robust, cloud-free reference)
  2. NDVI / NDWI trend maps     (vegetation + water trajectory)
  3. Change probability map     (flag pixels with unrealistic shifts)
  4. Temporal consistency score (per-pixel confidence for reconstruction)

This module is the primary anti-hallucination mechanism in the pipeline.
"""

from __future__ import annotations
import numpy as np
import torch
from pathlib import Path
from typing import List, Optional, Dict
import logging

from src.utils.indices import ndvi, ndwi

log = logging.getLogger(__name__)


class TemporalAnalyzer:
    """
    Builds temporal context from a stack of ±15-day optical observations.

    Input:  temporal_stack  (T, C, H, W) — T cloud-masked scenes
            cloud_masks     (T, H, W)    — binary: 1=cloudy, 0=clear
    Output: TemporalContext dataclass
    """

    def __init__(self, cfg):
        self.window_days      = getattr(cfg.temporal, 'temporal_window_days', None)
        if self.window_days is None:
            self.window_days = getattr(cfg.data, 'temporal_window_days', None)
        self.composite_method = getattr(cfg.temporal, 'composite_method', 'median')
        self.change_threshold = getattr(cfg.temporal, 'change_threshold', 0.15)
        self.trend_indices    = getattr(cfg.temporal, 'trend_indices', ['ndvi', 'ndwi'])

    # ─────────────────────────────────────────────
    # Main entry point
    # ─────────────────────────────────────────────

    def analyze(
        self,
        temporal_stack: torch.Tensor,   # (T, C, H, W)
        cloud_masks: torch.Tensor,      # (T, H, W)  0=clear 1=cloud
    ) -> Dict:
        """
        Run full temporal analysis and return a context dict consumed by L4.
        """
        if temporal_stack.dim() == 5 and temporal_stack.shape[0] == 1:
            temporal_stack = temporal_stack.squeeze(0)
        if temporal_stack.dim() != 4:
            raise ValueError(
                "temporal_stack must be a 4D tensor (T, C, H, W) or a 5D batch tensor with B=1"
            )

        T, C, H, W = temporal_stack.shape

        # Mask cloudy pixels with NaN before computing statistics
        stack_masked = self._apply_cloud_mask(temporal_stack, cloud_masks)

        # 1. Temporal composite (cloud-free reference)
        composite = self._build_composite(stack_masked)         # (C, H, W)

        # 2. Spectral index trends
        ndvi_trend = self._compute_trend(stack_masked, "ndvi")  # (H, W)
        ndwi_trend = self._compute_trend(stack_masked, "ndwi")  # (H, W)

        # 3. Change probability map
        change_prob = self._change_probability(
            stack_masked, ndvi_trend, self.change_threshold
        )                                                        # (H, W)

        # 4. Per-pixel temporal consistency score
        consistency = self._temporal_consistency_score(
            stack_masked, composite, cloud_masks
        )                                                        # (H, W)

        log.info(
            f"Temporal analysis complete | "
            f"T={T} scenes | "
            f"change coverage: {(change_prob > 0.5).float().mean().item():.2%} | "
            f"mean consistency: {consistency.mean().item():.3f}"
        )

        return {
            "stack":       temporal_stack.unsqueeze(0),   # raw stack (1,T,C,H,W) for transformer
            "composite":   composite,                     # (C,H,W) cloud-free reference
            "ndvi_trend":  ndvi_trend,                    # (H,W) slope of NDVI over time
            "ndwi_trend":  ndwi_trend,                    # (H,W) slope of NDWI over time
            "change_prob": change_prob,                   # (H,W) in [0,1]
            "consistency": consistency,                   # (H,W) in [0,1]
        }

    # ─────────────────────────────────────────────
    # Step 1: Composite
    # ─────────────────────────────────────────────

    def _build_composite(self, stack: torch.Tensor) -> torch.Tensor:
        """
        Build a cloud-free temporal composite from masked stack.
        NaN pixels (clouds) are excluded from the statistic.

        Methods:
          median : robust to outliers, best for mixed land cover
          mean   : smoother but sensitive to residual clouds
          mosaic : newest clear pixel first (best for change detection)
        """
        if self.composite_method == "median":
            # nanmedian not in older torch — use numpy fallback
            np_stack = stack.numpy()                  # (T, C, H, W)
            composite = np.nanmedian(np_stack, axis=0)  # (C, H, W)
            return torch.from_numpy(composite.astype(np.float32))

        elif self.composite_method == "mean":
            np_stack = stack.numpy()
            composite = np.nanmean(np_stack, axis=0)
            return torch.from_numpy(composite.astype(np.float32))

        elif self.composite_method == "mosaic":
            # Use most recent clear pixel (last T is most recent)
            T, C, H, W = stack.shape
            composite = torch.full((C, H, W), float("nan"))
            for t in range(T - 1, -1, -1):
                scene = stack[t]                          # (C, H, W)
                valid = ~torch.isnan(scene[0])            # (H, W)
                composite[:, valid] = scene[:, valid]
            # Fill any remaining NaN with nanmedian
            np_stack  = stack.numpy()
            np_comp   = composite.numpy()
            still_nan = np.isnan(np_comp[0])
            np_comp[:, still_nan] = np.nanmedian(np_stack, axis=0)[:, still_nan]
            return torch.from_numpy(np_comp.astype(np.float32))

        else:
            raise ValueError(f"Unknown composite method: {self.composite_method}")

    def _apply_cloud_mask(
        self,
        stack: torch.Tensor,
        cloud_masks: Optional[torch.Tensor],
    ) -> torch.Tensor:
        """Mask cloudy pixels in the temporal stack with NaN."""
        if cloud_masks is None:
            return stack

        if cloud_masks.dim() == 2:
            cloud_masks = cloud_masks.unsqueeze(0)
        if cloud_masks.dim() == 4 and cloud_masks.shape[1] == 1:
            cloud_masks = cloud_masks.squeeze(1)

        if cloud_masks.shape[0] != stack.shape[0]:
            if cloud_masks.shape[0] == 1:
                cloud_masks = cloud_masks.expand(stack.shape[0], -1, -1)
            else:
                raise ValueError(
                    "cloud_masks must have the same number of time steps as temporal_stack"
                )

        mask = (cloud_masks > 0).unsqueeze(1)
        stack = stack.float().clone()
        return stack.masked_fill(mask, float("nan"))

    # ─────────────────────────────────────────────
    # Step 2: Spectral index trends
    # ─────────────────────────────────────────────

    def _compute_trend(
        self,
        stack: torch.Tensor,   # (T, C, H, W)
        index_name: str,
    ) -> torch.Tensor:
        """
        Compute per-pixel linear trend slope of a spectral index over time.
        Positive slope → index increasing over time window.
        Returns (H, W) slope map.
        """
        T, C, H, W = stack.shape
        index_fn = {"ndvi": ndvi, "ndwi": ndwi}[index_name]

        # Compute index for each time step
        index_maps = []
        for t in range(T):
            idx = index_fn(stack[t].unsqueeze(0))
            idx = torch.from_numpy(np.asarray(idx)).squeeze()
            if idx.ndim != 2:
                raise ValueError(f"Index computation returned unexpected shape: {idx.shape}")
            index_maps.append(idx)
        index_stack = torch.stack(index_maps, dim=0)  # (T, H, W)

        # Linear regression per pixel: slope = (n*sum(t*y) - sum(t)*sum(y)) / denom
        np_index = index_stack.numpy()           # (T, H, W)
        t_vals   = np.arange(T, dtype=np.float32).reshape(T, 1, 1)
        valid    = ~np.isnan(np_index)
        valid_count = valid.sum(axis=0).clip(min=2)

        y_masked = np.where(valid, np_index, 0.0)
        sum_t   = np.sum(t_vals * valid, axis=0)
        sum_y   = np.sum(y_masked, axis=0)
        sum_ty  = np.sum(t_vals * y_masked, axis=0)
        sum_tt  = np.sum(t_vals * t_vals * valid, axis=0)

        denom_np = valid_count * sum_tt - sum_t**2 + 1e-8
        slope    = (valid_count * sum_ty - sum_t * sum_y) / denom_np

        return torch.from_numpy(slope.astype(np.float32))   # (H, W)

    # ─────────────────────────────────────────────
    # Step 3: Change probability
    # ─────────────────────────────────────────────

    def _change_probability(
        self,
        stack: torch.Tensor,       # (T, C, H, W)
        ndvi_trend: torch.Tensor,  # (H, W)
        threshold: float,
    ) -> torch.Tensor:
        """
        Estimate probability that a pixel has undergone real change
        (not cloud contamination) in the temporal window.

        Combines:
          - NDVI trend magnitude
          - Inter-quartile range of pixel values (high IQR = potential change)
          - Abrupt single-step changes (detected via max temporal delta)
        """
        T, C, H, W = stack.shape
        np_stack = stack.numpy()

        # Feature 1: |NDVI trend slope|  (large = changing)
        trend_score = ndvi_trend.abs().numpy()
        trend_score = np.clip(trend_score / (threshold * 2), 0, 1)

        # Feature 2: IQR of NIR band  (high = volatile pixel)
        nir = np_stack[:, 2, :, :]          # (T, H, W)
        q75 = np.nanpercentile(nir, 75, axis=0)
        q25 = np.nanpercentile(nir, 25, axis=0)
        iqr_score = np.clip((q75 - q25) / 0.3, 0, 1)

        # Feature 3: max single-step NDVI delta
        ndvi_maps = np.stack([
            np.asarray(ndvi(np_stack[t:t+1]))[0]
            for t in range(T)
        ], axis=0)                           # (T, H, W) or (T, 1, H, W)
        if ndvi_maps.ndim == 4 and ndvi_maps.shape[1] == 1:
            ndvi_maps = ndvi_maps.squeeze(1)
        deltas     = np.abs(np.diff(ndvi_maps, axis=0))
        max_delta  = np.nanmax(deltas, axis=0)
        delta_score = np.clip(max_delta / threshold, 0, 1)

        # Combine: weighted sum
        change_prob = (0.4 * trend_score +
                       0.3 * iqr_score   +
                       0.3 * delta_score )

        return torch.from_numpy(change_prob.astype(np.float32))   # (H, W)

    # ─────────────────────────────────────────────
    # Step 4: Consistency score
    # ─────────────────────────────────────────────

    def _temporal_consistency_score(
        self,
        stack: torch.Tensor,      # (T, C, H, W)
        composite: torch.Tensor,  # (C, H, W)
        cloud_masks: torch.Tensor # (T, H, W)
    ) -> torch.Tensor:
        """
        Per-pixel consistency score in [0, 1].
        High score = pixel is stable and predictable from temporal context.
        Low score  = pixel is volatile or frequently cloudy.

        Used by L5 verification and L6 uncertainty generation.
        """
        T = stack.shape[0]
        np_stack = stack.numpy()                         # (T,C,H,W)
        np_comp  = composite.numpy()                     # (C,H,W)
        np_masks = cloud_masks.numpy()                   # (T,H,W)

        # 1. Mean absolute deviation from composite (clear pixels only)
        deviations = []
        for t in range(T):
            clear = (np_masks[t] == 0)                  # (H,W) bool
            if clear.sum() == 0:
                continue
            diff = np.abs(np_stack[t] - np_comp)        # (C,H,W)
            mad  = diff.mean(axis=0)                     # (H,W)
            deviations.append(mad)

        if not deviations:
            return torch.ones(stack.shape[2], stack.shape[3])

        mean_mad = np.mean(deviations, axis=0)           # (H,W)

        # 2. Fraction of time steps with clear observations
        clear_fraction = (np_masks == 0).mean(axis=0)   # (H,W) in [0,1]

        # 3. Consistency = low deviation + high clear fraction
        #    Map MAD to [0,1] where 0 MAD → 1.0 score
        mad_score   = np.exp(-mean_mad * 10.0)          # exponential decay
        score       = 0.6 * mad_score + 0.4 * clear_fraction

        return torch.from_numpy(score.clip(0, 1).astype(np.float32))


# ─────────────────────────────────────────────
# Standalone helper for data preparation
# ─────────────────────────────────────────────

def build_temporal_composite(
    temporal_dir: Path,
    method: str = "median",
    bands: int = 4,
) -> np.ndarray:
    """
    Build a cloud-free composite from all GeoTIFFs in temporal_dir.
    Used during scripts/prepare_data.py.

    Returns (C, H, W) float32 composite array.
    """
    from src.utils.geotiff import read_geotiff

    paths  = sorted(temporal_dir.glob("*.tif"))
    scenes = []

    for p in paths:
        arr, _ = read_geotiff(p)
        if arr.shape[0] >= bands:
            scenes.append(arr[:bands])

    if not scenes:
        raise ValueError(f"No valid GeoTIFFs found in {temporal_dir}")

    stack = np.stack(scenes, axis=0)                     # (T,C,H,W)

    if method == "median":
        return np.nanmedian(stack, axis=0).astype(np.float32)
    elif method == "mean":
        return np.nanmean(stack, axis=0).astype(np.float32)
    else:
        raise ValueError(f"Unknown method: {method}")
