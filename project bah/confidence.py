"""
src/inference/confidence.py

Confidence and Uncertainty Map Generator (Layer 6).

For every reconstructed pixel, generates:
  1. Confidence map    (float32, 0–1)
     Aggregated from: verification pass scores + temporal consistency
     + spectral validity + SAR agreement

  2. Uncertainty map   (int8: 0=high, 1=medium, 2=low confidence)
     Thresholded from confidence map

  3. Monte Carlo dropout uncertainty  (optional, more expensive)
     Run N forward passes with dropout enabled → pixel-wise variance

The confidence map is the most important deliverable for end users —
it tells GIS analysts and scientists exactly which pixels to trust.
"""

from __future__ import annotations
import torch
import torch.nn.functional as F
import numpy as np
from typing import Dict, Optional, Tuple
from dataclasses import dataclass
import logging

log = logging.getLogger(__name__)


@dataclass
class ConfidenceMaps:
    confidence:  torch.Tensor    # (B, 1, H, W) float32  [0, 1]
    uncertainty: torch.Tensor    # (B, 1, H, W) int8: 0=high, 1=med, 2=low
    mc_variance: Optional[torch.Tensor]  # (B, 1, H, W) MC dropout variance
    breakdown:   Dict            # per-source confidence components


class ConfidenceMapper:
    """
    Generates per-pixel confidence and uncertainty maps from verification
    pass results and model intrinsics.

    Sources of confidence signal (all normalised to [0,1]):
      S1. Temporal consistency score     (weight 0.30)
      S2. Spectral index error           (weight 0.25)
      S3. SAR edge agreement             (weight 0.20)
      S4. AI artifact probability        (weight 0.15)
      S5. Cloud mask boundary distance   (weight 0.10)
          (pixels near cloud edges are less reliable)
    """

    HIGH_CONF_THRESHOLD = 0.80
    LOW_CONF_THRESHOLD  = 0.50

    WEIGHTS = {
        "temporal":    0.30,
        "spectral":    0.25,
        "sar":         0.20,
        "artifact":    0.15,
        "boundary":    0.10,
    }

    def __init__(self, cfg):
        self.cfg         = cfg
        self.mc_passes   = cfg.confidence.mc_dropout_passes   # 20
        self.high_thresh = cfg.confidence.high_confidence_threshold  # 0.80
        self.low_thresh  = cfg.confidence.low_confidence_threshold   # 0.50

    def generate(
        self,
        reconstructed: torch.Tensor,          # (B, 4, H, W)
        verification_report: Dict,
        cloud_mask: Optional[torch.Tensor] = None,   # (B, H, W)
        temporal_context: Optional[Dict]   = None,
        sar_features: Optional[torch.Tensor] = None, # (B, 4, H, W)
        model: Optional[torch.nn.Module]    = None,  # for MC dropout
    ) -> ConfidenceMaps:
        B, C, H, W = reconstructed.shape
        device = reconstructed.device

        # ── S1: Temporal confidence ───────────────────────────────
        if temporal_context and "consistency" in temporal_context:
            temp_conf = temporal_context["consistency"]   # (B, H, W) or (H, W)
            if temp_conf.dim() == 2:
                temp_conf = temp_conf.unsqueeze(0).expand(B, -1, -1)
            temp_conf = temp_conf.unsqueeze(1).to(device)  # (B,1,H,W)
        else:
            temp_conf = torch.ones(B, 1, H, W, device=device) * 0.7

        # ── S2: Spectral confidence ───────────────────────────────
        # Derive from verification pass 3 (spectral validation score)
        spectral_score = self._get_pass_score(verification_report, pass_id=3)
        spec_conf = torch.full((B, 1, H, W), spectral_score, device=device)

        if temporal_context and temporal_context.get("composite") is not None:
            composite = temporal_context["composite"].to(device)
            if composite.dim() == 3:
                composite = composite.unsqueeze(0).expand(B, -1, -1, -1)
            from src.utils.indices import ndvi
            ndvi_diff = (ndvi(reconstructed) - ndvi(composite)).abs()     # (B,1,H,W)
            spec_conf = torch.clamp(1.0 - ndvi_diff * 10.0, 0.0, 1.0)   # sharper penalty

        # ── S3: SAR confidence ────────────────────────────────────
        sar_score = self._get_pass_score(verification_report, pass_id=2)
        sar_conf  = torch.full((B, 1, H, W), sar_score, device=device)

        if sar_features is not None:
            # Higher confidence near SAR-visible edges (structural certainty)
            sar_edge   = sar_features[:, 3:4]                        # (B,1,H,W)
            sar_conf   = (0.6 + 0.4 * sar_edge).clamp(0, 1)

        # ── S4: Artifact confidence ───────────────────────────────
        artifact_score = self._get_pass_score(verification_report, pass_id=4)
        art_conf       = torch.full((B, 1, H, W), artifact_score, device=device)

        # ── S5: Cloud boundary distance confidence ────────────────
        if cloud_mask is not None:
            bound_conf = self._boundary_confidence(cloud_mask, H, W, device)
        else:
            bound_conf = torch.ones(B, 1, H, W, device=device)

        # ── Weighted combination ──────────────────────────────────
        w = self.WEIGHTS
        confidence = (
            w["temporal"]  * temp_conf +
            w["spectral"]  * spec_conf +
            w["sar"]       * sar_conf  +
            w["artifact"]  * art_conf  +
            w["boundary"]  * bound_conf
        ).clamp(0.0, 1.0)

        # Outside cloud mask → full confidence (original pixels, not reconstructed)
        if cloud_mask is not None:
            is_cloud = (cloud_mask > 0).float().unsqueeze(1).to(device)
            confidence = is_cloud * confidence + (1.0 - is_cloud) * 1.0

        # ── MC Dropout uncertainty ────────────────────────────────
        mc_variance = None
        if model is not None and self.mc_passes > 0:
            mc_variance = self._mc_dropout_uncertainty(
                model, reconstructed, self.mc_passes
            )

        # ── Uncertainty classification ────────────────────────────
        uncertainty = self._classify_uncertainty(confidence)

        log.info(
            f"Confidence map generated | "
            f"high={( confidence > self.high_thresh).float().mean().item():.1%} | "
            f"med={(( confidence >= self.low_thresh) & (confidence <= self.high_thresh)).float().mean().item():.1%} | "
            f"low={( confidence < self.low_thresh ).float().mean().item():.1%}"
        )

        return ConfidenceMaps(
            confidence  = confidence,
            uncertainty = uncertainty,
            mc_variance = mc_variance,
            breakdown   = {
                "temporal":  float(temp_conf.mean()),
                "spectral":  float(spec_conf.mean()),
                "sar":       float(sar_conf.mean()),
                "artifact":  float(art_conf.mean()),
                "boundary":  float(bound_conf.mean()),
            }
        )

    # ─────────────────────────────────────────
    # Helpers
    # ─────────────────────────────────────────

    def _get_pass_score(self, report: Dict, pass_id: int) -> float:
        """Extract score for a specific verification pass from the report."""
        for p in report.get("passes", []):
            if p["id"] == pass_id:
                return float(p["score"])
        return 0.7  # default if pass not found

    def _boundary_confidence(
        self,
        cloud_mask: torch.Tensor,    # (B, H, W)
        H: int, W: int,
        device: torch.device,
    ) -> torch.Tensor:
        """
        Lower confidence near cloud mask boundaries — transition zones
        are harder to reconstruct accurately.
        Distance transform: confidence increases with distance from boundary.
        """
        import scipy.ndimage as ndi
        B = cloud_mask.shape[0]
        conf_list = []

        for b in range(B):
            mask  = (cloud_mask[b] > 0).cpu().numpy().astype(np.uint8)  # (H,W)
            # Distance from cloud boundary (pixels)
            dist  = ndi.distance_transform_edt(mask)          # distance inside cloud
            # Normalise: max distance inside cloud → 1.0 confidence
            max_d = max(dist.max(), 1.0)
            conf  = np.clip(dist / (max_d * 0.3), 0.0, 1.0)  # saturate at 30% of max
            conf_list.append(torch.from_numpy(conf.astype(np.float32)))

        return torch.stack(conf_list, dim=0).unsqueeze(1).to(device)  # (B,1,H,W)

    def _classify_uncertainty(self, confidence: torch.Tensor) -> torch.Tensor:
        """
        Classify confidence into 3 uncertainty levels:
          0 = High confidence  (conf > 0.80)
          1 = Medium           (0.50 ≤ conf ≤ 0.80)
          2 = Low confidence   (conf < 0.50)
        """
        uncertainty = torch.full_like(confidence, 1, dtype=torch.int8)
        uncertainty[confidence > self.high_thresh] = 0
        uncertainty[confidence < self.low_thresh]  = 2
        return uncertainty

    def _mc_dropout_uncertainty(
        self,
        model: torch.nn.Module,
        x: torch.Tensor,
        n_passes: int,
    ) -> torch.Tensor:
        """
        Monte Carlo dropout uncertainty estimate.
        Enable dropout at inference → N stochastic forward passes.
        Returns per-pixel predictive variance.
        """
        model.train()   # enable dropout
        preds = []

        with torch.no_grad():
            for _ in range(n_passes):
                batch = {"optical": x, "cloud_mask": torch.zeros(x.shape[0], *x.shape[2:],
                                                                   device=x.device)}
                out = model(batch)
                preds.append(out["reconstructed"])

        model.eval()
        preds  = torch.stack(preds, dim=0)          # (N, B, C, H, W)
        variance = preds.var(dim=0).mean(dim=1, keepdim=True)  # (B, 1, H, W)
        return variance
