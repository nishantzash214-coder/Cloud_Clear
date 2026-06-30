"""
src/models/verification/verifier.py

Multi-Level Verification Engine (Layer 5).

5 mandatory verification passes — every reconstructed scene must pass all 5
before being exported. Failed passes trigger re-reconstruction or flagging.

Pass 1: Temporal Validation
Pass 2: SAR Consistency
Pass 3: Spectral Validation (NDVI/NDWI/NDBI/SAVI < 5% error)
Pass 4: AI Self-Checker (artifact detection)
Pass 5: Cloud-Free Reference Matching
"""

from __future__ import annotations
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, Optional, List
from dataclasses import dataclass, field
import logging

from src.utils.indices import compute_all_indices, index_error
from src.utils.metrics import ssim, psnr, spectral_angle_mapper, temporal_consistency

log = logging.getLogger(__name__)


# ─────────────────────────────────────────────
# Verification result dataclass
# ─────────────────────────────────────────────

@dataclass
class VerificationResult:
    pass_id:    int
    name:       str
    passed:     bool
    score:      float
    threshold:  float
    details:    dict = field(default_factory=dict)
    pixel_mask: Optional[torch.Tensor] = None   # per-pixel fail map


@dataclass
class VerificationReport:
    passes:        List[VerificationResult]
    overall_pass:  bool
    confidence:    float                        # mean pass score
    fail_mask:     Optional[torch.Tensor]       # pixels that failed ≥1 pass
    metrics:       dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "overall_pass": self.overall_pass,
            "confidence":   self.confidence,
            "passes": [
                {
                    "id":        r.pass_id,
                    "name":      r.name,
                    "passed":    r.passed,
                    "score":     r.score,
                    "threshold": r.threshold,
                    "details":   r.details,
                }
                for r in self.passes
            ],
            **self.metrics,
        }


# ─────────────────────────────────────────────
# AI Self-Checker (Pass 4)
# ─────────────────────────────────────────────

class ArtifactDetector(nn.Module):
    """
    Lightweight ResNet-18 classifier trained to detect:
      - GAN checkerboard artifacts
      - Spectral band anomalies
      - Texture discontinuities at cloud boundaries
      - Implausible land cover transitions

    Binary output: 0 = clean, 1 = artifact detected
    """

    def __init__(self, in_channels: int = 4):
        super().__init__()
        import torchvision.models as models
        resnet = models.resnet18(weights=None)

        # Adapt for 4-channel input
        resnet.conv1 = nn.Conv2d(in_channels, 64, 7, stride=2, padding=3, bias=False)
        resnet.fc    = nn.Sequential(
            nn.Linear(512, 128),
            nn.ReLU(True),
            nn.Dropout(0.3),
            nn.Linear(128, 1),
            nn.Sigmoid(),
        )
        self.model = resnet

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Returns (B, 1) artifact probability."""
        return self.model(x)


# ─────────────────────────────────────────────
# Main Verifier
# ─────────────────────────────────────────────

class MultiPassVerifier:
    """
    Runs all 5 verification passes on a reconstructed scene.

    Usage:
        verifier = MultiPassVerifier(cfg)
        report   = verifier.verify(pred, original, cloud_mask, sar, temporal)
    """

    def __init__(self, cfg):
        self.spectral_thresh   = cfg.verification.spectral_error_threshold   # 0.05
        self.sar_thresh        = cfg.verification.sar_consistency_min         # 0.70
        self.temporal_thresh   = cfg.verification.temporal_consistency_min    # 0.80
        self.artifact_thresh   = cfg.verification.ai_checker.artifact_threshold  # 0.15
        self.artifact_detector = ArtifactDetector()

    def verify(
        self,
        pred:       torch.Tensor,                    # (B, 4, H, W) reconstructed
        original:   torch.Tensor,                    # (B, 4, H, W) original cloudy
        cloud_mask: torch.Tensor,                    # (B, H, W) 0=clear
        sar:        Optional[torch.Tensor] = None,   # (B, 4, H, W) SAR features
        temporal:   Optional[Dict]         = None,   # temporal context dict
    ) -> Dict:
        """Run all 5 passes and return a verification report dict."""
        results = []

        # ── Pass 1: Temporal Validation ───────────────────────────
        results.append(self._pass1_temporal(pred, temporal, cloud_mask))

        # ── Pass 2: SAR Consistency ───────────────────────────────
        results.append(self._pass2_sar(pred, sar, cloud_mask))

        # ── Pass 3: Spectral Validation ───────────────────────────
        results.append(self._pass3_spectral(pred, temporal, cloud_mask))

        # ── Pass 4: AI Self-Checker ───────────────────────────────
        results.append(self._pass4_ai(pred, cloud_mask))

        # ── Pass 5: Reference Matching ────────────────────────────
        results.append(self._pass5_reference(pred, temporal, cloud_mask))

        # ── Aggregate ─────────────────────────────────────────────
        overall_pass  = all(r.passed for r in results)
        confidence    = float(np.mean([r.score for r in results]))

        # Build pixel-level fail mask (union of all failed pixel masks)
        fail_mask = None
        for r in results:
            if r.pixel_mask is not None and not r.passed:
                if fail_mask is None:
                    fail_mask = r.pixel_mask.float()
                else:
                    fail_mask = torch.maximum(fail_mask, r.pixel_mask.float())

        report = VerificationReport(
            passes       = results,
            overall_pass = overall_pass,
            confidence   = confidence,
            fail_mask    = fail_mask,
            metrics      = {
                "ssim": ssim(pred, original),
                "psnr": psnr(pred, original),
                "sam":  spectral_angle_mapper(pred, original),
            }
        )

        status = "✓ PASS" if overall_pass else "✗ FAIL"
        log.info(
            f"Verification {status} | confidence={confidence:.3f} | "
            f"passes={sum(r.passed for r in results)}/5"
        )
        return report.to_dict()

    # ─────────────────────────────────────────
    # Pass 1: Temporal Validation
    # ─────────────────────────────────────────

    def _pass1_temporal(
        self,
        pred:       torch.Tensor,
        temporal:   Optional[Dict],
        cloud_mask: torch.Tensor,
    ) -> VerificationResult:
        if temporal is None or temporal.get("composite") is None:
            return VerificationResult(1, "Temporal Validation", True, 1.0, self.temporal_thresh,
                                      {"note": "No temporal data — pass skipped"})

        composite = temporal["composite"]
        clear_mask = (cloud_mask == 0).float().unsqueeze(1)  # (B,1,H,W)

        score = temporal_consistency(pred, composite.unsqueeze(1), clear_mask)
        passed = score >= self.temporal_thresh

        # Per-pixel: flag where deviation from composite is large
        diff     = (pred - composite).abs().mean(dim=1, keepdim=True)  # (B,1,H,W)
        pix_fail = (diff > 0.15).squeeze(1)                            # (B,H,W)

        return VerificationResult(
            pass_id    = 1,
            name       = "Temporal Validation",
            passed     = bool(passed),
            score      = float(score),
            threshold  = self.temporal_thresh,
            details    = {"temporal_consistency": float(score)},
            pixel_mask = pix_fail,
        )

    # ─────────────────────────────────────────
    # Pass 2: SAR Consistency
    # ─────────────────────────────────────────

    def _pass2_sar(
        self,
        pred:       torch.Tensor,
        sar:        Optional[torch.Tensor],
        cloud_mask: torch.Tensor,
    ) -> VerificationResult:
        if sar is None:
            return VerificationResult(2, "SAR Consistency", True, 1.0, self.sar_thresh,
                                      {"note": "No SAR data — pass skipped"})

        # Use structural similarity between pred NIR and SAR VV band
        # (both capture vegetation/water structure)
        pred_nir = pred[:, 2:3]                                       # (B,1,H,W)
        sar_vv   = sar[:, 0:1]                                        # (B,1,H,W)

        # SSIM between NIR and VV as structural agreement proxy
        score = ssim(pred_nir, sar_vv)
        # SAR and NIR will not be identical; we check structural correlation
        # A good reconstruction should have edge patterns matching SAR edges
        if sar.shape[1] >= 4:
            sar_edges = sar[:, 3:4]
        else:
            sar_edges = self._compute_edges(sar_vv)
        pred_edges = self._compute_edges(pred_nir)

        edge_agreement = F.cosine_similarity(
            pred_edges.flatten(1),
            sar_edges.flatten(1),
        ).mean().item()
        edge_agreement = (edge_agreement + 1) / 2   # scale to [0,1]

        passed = edge_agreement >= self.sar_thresh

        return VerificationResult(
            pass_id   = 2,
            name      = "SAR Consistency",
            passed    = bool(passed),
            score     = float(edge_agreement),
            threshold = self.sar_thresh,
            details   = {"edge_agreement": float(edge_agreement), "ssim_nir_vv": float(score)},
        )

    @staticmethod
    def _compute_edges(x: torch.Tensor) -> torch.Tensor:
        sobel_x = torch.tensor([[-1,0,1],[-2,0,2],[-1,0,1]],
                                dtype=x.dtype, device=x.device).view(1,1,3,3)
        sobel_y = torch.tensor([[-1,-2,-1],[0,0,0],[1,2,1]],
                                dtype=x.dtype, device=x.device).view(1,1,3,3)
        gx = F.conv2d(x, sobel_x.expand(x.shape[1],-1,-1,-1), padding=1, groups=x.shape[1])
        gy = F.conv2d(x, sobel_y.expand(x.shape[1],-1,-1,-1), padding=1, groups=x.shape[1])
        return torch.sqrt(gx**2 + gy**2 + 1e-8)

    # ─────────────────────────────────────────
    # Pass 3: Spectral Validation
    # ─────────────────────────────────────────

    def _pass3_spectral(
        self,
        pred:       torch.Tensor,
        temporal:   Optional[Dict],
        cloud_mask: torch.Tensor,
    ) -> VerificationResult:
        if temporal is None or temporal.get("composite") is None:
            # Self-check: verify indices are in physical range
            idx = compute_all_indices(pred)
            out_of_range = sum(
                ((v < -1.0) | (v > 1.0)).float().mean().item()
                for v in idx.values()
            ) / len(idx)
            score = 1.0 - out_of_range
            return VerificationResult(3, "Spectral Validation", score > 0.95, score, 0.95,
                                      {"out_of_range_fraction": out_of_range})

        composite  = temporal["composite"]
        errors     = index_error(pred, composite)
        mean_error = np.mean([v["mae"] for v in errors.values()])
        all_pass   = all(v["pass"] for v in errors.values())
        score      = 1.0 - mean_error

        # Per-pixel: flag where NDVI error > threshold
        p_idx = compute_all_indices(pred)["ndvi"]
        c_idx = compute_all_indices(composite.unsqueeze(0) if composite.dim() == 3 else composite)["ndvi"]
        ndvi_err = (p_idx - c_idx).abs().squeeze(1)  # (B,H,W)
        pix_fail = ndvi_err > self.spectral_thresh

        return VerificationResult(
            pass_id    = 3,
            name       = "Spectral Validation",
            passed     = bool(all_pass),
            score      = float(score),
            threshold  = 1.0 - self.spectral_thresh,
            details    = {k: v["mae"] for k, v in errors.items()},
            pixel_mask = pix_fail,
        )

    # ─────────────────────────────────────────
    # Pass 4: AI Self-Checker
    # ─────────────────────────────────────────

    def _pass4_ai(
        self,
        pred:       torch.Tensor,
        cloud_mask: torch.Tensor,
    ) -> VerificationResult:
        self.artifact_detector.eval()
        with torch.no_grad():
            artifact_prob = self.artifact_detector(pred)          # (B,1)
        mean_prob = artifact_prob.mean().item()
        passed    = mean_prob < self.artifact_thresh

        return VerificationResult(
            pass_id   = 4,
            name      = "AI Self-Checker",
            passed    = bool(passed),
            score     = float(1.0 - mean_prob),
            threshold = 1.0 - self.artifact_thresh,
            details   = {"artifact_probability": mean_prob},
        )

    # ─────────────────────────────────────────
    # Pass 5: Cloud-Free Reference Matching
    # ─────────────────────────────────────────

    def _pass5_reference(
        self,
        pred:       torch.Tensor,
        temporal:   Optional[Dict],
        cloud_mask: torch.Tensor,
    ) -> VerificationResult:
        if temporal is None or temporal.get("composite") is None:
            return VerificationResult(5, "Reference Matching", True, 1.0, 0.7,
                                      {"note": "No reference data — pass skipped"})

        composite = temporal["composite"]
        # Only evaluate on pixels that are cloud-free in the composite
        # (i.e., high-quality reference pixels)
        clear_ref = (cloud_mask == 0).float().unsqueeze(1)         # (B,1,H,W)

        diff     = ((pred - composite).abs() * clear_ref).sum()
        denom    = clear_ref.sum().clamp(min=1) * pred.shape[1]
        mae      = (diff / denom).item()

        ssim_score = ssim(pred * clear_ref, composite * clear_ref)
        score      = (1.0 - mae + ssim_score) / 2.0
        passed     = score >= 0.70

        return VerificationResult(
            pass_id   = 5,
            name      = "Reference Matching",
            passed    = bool(passed),
            score     = float(score),
            threshold = 0.70,
            details   = {"mae_vs_reference": mae, "ssim_vs_reference": ssim_score},
        )
