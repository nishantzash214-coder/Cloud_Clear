"""
src/losses/scientific_losses.py

Scientific loss functions for physically-constrained cloud removal (Layer 4).

Three custom losses enforce scientific validity beyond pixel-level accuracy:

1. SpectralConsistencyLoss
   Ensures reconstructed NDVI / NDWI / SAVI / NDBI remain within
   scientifically valid ranges. Penalises index deviations > 5%.

2. PhysicalConsistencyLoss
   Enforces surface reflectance physical constraints:
     - Water pixels: low NIR, low SWIR
     - Vegetation pixels: high NIR, low Red
     - Urban pixels: high SWIR, moderate NIR
     - Bare soil: moderate NIR ≈ SWIR
   Also prevents band order inversions that are physically impossible.

3. TemporalConsistencyLoss
   Penalises reconstructed pixels that deviate unrealistically
   from the ±15-day temporal neighbourhood. A field cannot become
   a river between adjacent observations.

Combined loss: L_total = L_pixel + λ_s * L_spectral + λ_p * L_physical + λ_t * L_temporal
"""

from __future__ import annotations
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Dict
from src.utils.indices import ndvi, ndwi, savi, ndbi


# ─────────────────────────────────────────────
# 1. Spectral Consistency Loss
# ─────────────────────────────────────────────

class SpectralConsistencyLoss(nn.Module):
    """
    Penalises deviations in spectral indices between prediction and target.

    For each index I in {NDVI, NDWI, SAVI, NDBI}:
      L_I = mean( max(0, |I_pred - I_target| - threshold) )

    threshold = 0.05 (5% scientific tolerance)

    Only computed inside the cloud mask — clear pixels are not penalised
    (the diffusion model only reconstructs cloud-covered regions).
    """

    def __init__(self, threshold: float = 0.05, weights: Optional[Dict] = None):
        super().__init__()
        self.threshold = threshold
        self.weights   = weights or {
            "ndvi": 1.0,   # most important for agriculture / environment
            "ndwi": 0.8,   # water body accuracy
            "savi": 0.6,   # soil-adjusted (correlated with NDVI)
            "ndbi": 0.6,   # urban mapping
        }

    def forward(
        self,
        pred:       torch.Tensor,              # (B, 4, H, W)
        target:     torch.Tensor,              # (B, 4, H, W)
        cloud_mask: Optional[torch.Tensor],   # (B, 1, H, W) binary
    ) -> Dict[str, torch.Tensor]:
        loss_total = torch.tensor(0.0, device=pred.device)
        per_index  = {}

        index_fns = {"ndvi": ndvi, "ndwi": ndwi, "savi": savi, "ndbi": ndbi}

        for name, fn in index_fns.items():
            idx_pred   = fn(pred)     # (B, 1, H, W)
            idx_target = fn(target)   # (B, 1, H, W)

            error = (idx_pred - idx_target).abs()
            # Hinge: only penalise errors above threshold
            hinge = F.relu(error - self.threshold)

            if cloud_mask is not None:
                # Only penalise cloud-covered pixels
                hinge = hinge * cloud_mask
                denom = cloud_mask.sum().clamp(min=1)
                loss_i = hinge.sum() / denom
            else:
                loss_i = hinge.mean()

            loss_total    = loss_total + self.weights[name] * loss_i
            per_index[name] = loss_i.item()

        return {"total": loss_total, "per_index": per_index}


# ─────────────────────────────────────────────
# 2. Physical Consistency Loss
# ─────────────────────────────────────────────

class PhysicalConsistencyLoss(nn.Module):
    """
    Enforces known physical relationships in surface reflectance.

    Rules (all enforced as soft constraints with ReLU penalty):

    R1: NIR >= Red (almost universally true for land surfaces)
        Violation: Red > NIR + margin (could mean water or deep shadow)
        Exception: water bodies — we detect and exclude them

    R2: Vegetation constraint: if NDVI > 0.4 then NIR > SWIR
        Dense vegetation has high NIR and relatively low SWIR

    R3: Urban/built-up: if NDBI > 0.1 then SWIR > NIR (built-up has high SWIR)

    R4: Water constraint: if NDWI > 0.2 then NIR < 0.15 (water absorbs NIR)

    R5: Band value range [0, 1] — hard clamping shouldn't be relied on
        but soft penalty for negative predicted values during training.

    All penalties are applied only within the cloud mask region.
    """

    def __init__(self, margin: float = 0.02):
        super().__init__()
        self.margin = margin

    def forward(
        self,
        pred:       torch.Tensor,             # (B, 4, H, W)
        cloud_mask: Optional[torch.Tensor],  # (B, 1, H, W)
    ) -> torch.Tensor:
        green = pred[:, 0:1]
        red   = pred[:, 1:2]
        nir   = pred[:, 2:3]
        swir  = pred[:, 3:4]

        eps = 1e-8
        ndvi_map = (nir - red)   / (nir + red   + eps)
        ndwi_map = (green - nir) / (green + nir + eps)
        ndbi_map = (swir - nir)  / (swir + nir  + eps)

        losses = []

        # R1: NIR should generally be >= Red
        r1 = F.relu(red - nir - self.margin)
        # Exclude water (where Red > NIR can be valid)
        water_mask = (ndwi_map > 0.2).float()
        r1 = r1 * (1.0 - water_mask)
        losses.append(r1)

        # R2: Dense vegetation → NIR > SWIR
        veg_mask = (ndvi_map > 0.4).float()
        r2 = F.relu(swir - nir - self.margin) * veg_mask
        losses.append(r2)

        # R3: Urban → SWIR > NIR
        urban_mask = (ndbi_map > 0.1).float()
        r3 = F.relu(nir - swir - self.margin) * urban_mask
        losses.append(r3)

        # R4: Water → low NIR
        water_mask_strict = (ndwi_map > 0.2).float()
        r4 = F.relu(nir - 0.15) * water_mask_strict
        losses.append(r4)

        # R5: Non-negative reflectance
        r5 = F.relu(-pred)
        losses.append(r5)

        total = torch.stack([l.mean() for l in losses]).sum()

        if cloud_mask is not None:
            # Weight by cloud mask — harder penalty inside cloud regions
            weighted = []
            for l in losses:
                if l.shape[1] == 1:
                    weighted.append((l * cloud_mask).mean())
                else:
                    # multi-band loss
                    weighted.append((l * cloud_mask).mean())
            total = torch.stack(weighted).sum()

        return total


# ─────────────────────────────────────────────
# 3. Temporal Consistency Loss
# ─────────────────────────────────────────────

class TemporalConsistencyLoss(nn.Module):
    """
    Penalises reconstructed pixels that deviate unrealistically from
    the temporal median composite.

    L_temporal = mean( max(0, |I_pred - I_composite| - temporal_threshold) )
                 × change_penalty (high for stable pixels, low for change pixels)

    change_penalty = 1 - change_prob
      If change_prob is high (pixel likely changed), we relax the penalty.
      If change_prob is low (pixel is stable), we enforce strict consistency.

    This prevents the model from "hallucinating" changes that didn't happen
    while allowing genuinely changed pixels to be reconstructed freely.
    """

    def __init__(self, threshold: float = 0.10):
        super().__init__()
        self.threshold = threshold

    def forward(
        self,
        pred:        torch.Tensor,             # (B, 4, H, W)
        composite:   torch.Tensor,             # (B, 4, H, W) temporal median
        cloud_mask:  Optional[torch.Tensor],  # (B, 1, H, W)
        change_prob: Optional[torch.Tensor] = None,  # (B, 1, H, W) in [0,1]
    ) -> torch.Tensor:
        diff  = (pred - composite).abs()        # (B, 4, H, W)
        hinge = F.relu(diff - self.threshold)

        # Where change is likely → relax the constraint
        if change_prob is not None:
            stability = (1.0 - change_prob)    # high where pixel is stable
            hinge     = hinge * stability

        # Only apply inside cloud mask
        if cloud_mask is not None:
            hinge = hinge * cloud_mask
            denom = cloud_mask.sum().clamp(min=1) * pred.shape[1]
            return hinge.sum() / denom

        return hinge.mean()


# ─────────────────────────────────────────────
# 4. Perceptual Loss (VGG features)
# ─────────────────────────────────────────────

class PerceptualLoss(nn.Module):
    """
    VGG-based perceptual loss for structural similarity.
    Uses relu3_3 features from VGG-16.
    Adapted for 4-band input by averaging 4 bands → 3 channels.
    """

    def __init__(self):
        super().__init__()
        import torchvision.models as models
        vgg = models.vgg16(weights="IMAGENET1K_V1")
        # Use first 16 layers (up to relu3_3)
        self.features = nn.Sequential(*list(vgg.features)[:16])
        for p in self.parameters():
            p.requires_grad = False

    def _to_rgb(self, x: torch.Tensor) -> torch.Tensor:
        """Convert 4-band to 3-channel by merging NIR into a pseudo-RGB."""
        r   = x[:, 1:2]          # Red
        g   = x[:, 0:1]          # Green
        nir = x[:, 2:3]          # NIR → blue channel (false colour)
        return torch.cat([r, g, nir], dim=1)

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        p = self.features(self._to_rgb(pred))
        t = self.features(self._to_rgb(target))
        return F.mse_loss(p, t)


# ─────────────────────────────────────────────
# Combined Loss
# ─────────────────────────────────────────────

class CloudRemovalLoss(nn.Module):
    """
    Full combined loss for training the reconstruction pipeline.

    L_total = w_l1 * L1
            + w_perceptual * L_perceptual
            + w_spectral   * L_spectral
            + w_physical   * L_physical
            + w_temporal   * L_temporal
    """

    def __init__(self, cfg):
        super().__init__()
        w = cfg.losses
        self.w_l1          = w.pixel_l1
        self.w_perceptual  = w.perceptual
        self.w_spectral    = w.spectral_consistency
        self.w_physical    = w.physical_consistency
        self.w_temporal    = w.temporal_consistency

        self.spectral_loss  = SpectralConsistencyLoss()
        self.physical_loss  = PhysicalConsistencyLoss()
        self.temporal_loss  = TemporalConsistencyLoss()
        self.perceptual_loss = PerceptualLoss()

    def forward(
        self,
        pred:        torch.Tensor,
        target:      torch.Tensor,
        cloud_mask:  torch.Tensor,
        composite:   Optional[torch.Tensor] = None,
        change_prob: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:

        mask = cloud_mask.unsqueeze(1).float() if cloud_mask.dim() == 3 else cloud_mask.float()

        # L1 — only inside cloud mask
        l1 = (pred - target).abs()
        l1 = (l1 * mask).sum() / mask.sum().clamp(min=1) / pred.shape[1]

        # Perceptual
        l_perc = self.perceptual_loss(pred, target)

        # Spectral consistency
        l_spec = self.spectral_loss(pred, target, mask)["total"]

        # Physical consistency
        l_phys = self.physical_loss(pred, mask)

        # Temporal consistency
        l_temp = torch.tensor(0.0, device=pred.device)
        if composite is not None:
            l_temp = self.temporal_loss(pred, composite, mask, change_prob)

        total = (self.w_l1        * l1     +
                 self.w_perceptual * l_perc +
                 self.w_spectral   * l_spec +
                 self.w_physical   * l_phys +
                 self.w_temporal   * l_temp)

        return {
            "loss":       total,
            "l1":         l1.item(),
            "perceptual": l_perc.item(),
            "spectral":   l_spec.item(),
            "physical":   l_phys.item(),
            "temporal":   l_temp.item() if isinstance(l_temp, torch.Tensor) else l_temp,
        }
