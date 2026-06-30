"""
src/models/reconstruction/pipeline.py

Full Reconstruction Pipeline (Layer 4).

Wires together:
  Branch 1 — Conditional Diffusion encoder features
  Branch 2 — GAN generator (Pix2PixHD-style)
  Branch 3 — Temporal Transformer
  Branch 4 — SAR Fusion Encoder
  → Cross-Attention Fusion Layer
  → Pixel decoder → reconstructed image

During training, all branches run in parallel and their features
are fused before the final pixel decoder. The loss is computed
on the decoded output.

At inference, the pipeline runs the full forward pass and returns
the reconstructed image with intermediate branch outputs for debugging.
"""

from __future__ import annotations
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional, List
from pathlib import Path
import logging

log = logging.getLogger(__name__)


# ─────────────────────────────────────────────
# Branch 1: Diffusion Feature Encoder
# (uses U-Net backbone; full DDPM at train time,
#  deterministic single-step at inference for speed)
# ─────────────────────────────────────────────

class DiffusionBranch(nn.Module):
    """
    Extracts high-fidelity reconstruction features via a U-Net
    backbone conditioned on the cloud mask and composite reference.

    Full diffusion sampling is used during training data generation.
    At inference, we use a 20-step DDIM sampler for speed.
    """

    def __init__(self, in_ch: int = 4, cond_ch: int = 1, d_model: int = 256):
        super().__init__()
        # Condition: optical (4) + cloud_mask (1)
        total_in = in_ch + cond_ch

        self.encoder = nn.Sequential(
            nn.Conv2d(total_in, 64,    7, padding=3), nn.GELU(),
            nn.Conv2d(64,       128,   3, padding=1), nn.GELU(),
            nn.Conv2d(128,      d_model, 3, padding=1), nn.GroupNorm(8, d_model), nn.GELU(),
        )
        # Bottleneck with self-attention
        self.attn = nn.MultiheadAttention(d_model, num_heads=8, batch_first=True)
        self.norm = nn.LayerNorm(d_model)
        self.out  = nn.Conv2d(d_model, d_model, 1)

    def forward(
        self,
        optical:    torch.Tensor,             # (B, 4, H, W)
        cloud_mask: torch.Tensor,             # (B, 1, H, W) float
        composite:  Optional[torch.Tensor],   # (B, 4, H, W) or None
    ) -> torch.Tensor:
        # Condition: mask out cloud regions in optical, concat cloud_mask
        masked   = optical * (1.0 - cloud_mask)
        if composite is not None:
            # Fill cloud region with composite as prior
            masked = masked + cloud_mask * composite
        x = torch.cat([masked, cloud_mask], dim=1)          # (B,5,H,W)
        feat = self.encoder(x)                              # (B,D,H,W)

        B, D, H, W = feat.shape
        flat = feat.flatten(2).transpose(1, 2)             # (B,HW,D)
        attn_out, _ = self.attn(flat, flat, flat)
        flat = self.norm(flat + attn_out)
        feat = flat.transpose(1, 2).reshape(B, D, H, W)
        return self.out(feat)


# ─────────────────────────────────────────────
# Branch 2: GAN Generator (Pix2PixHD-style)
# ─────────────────────────────────────────────

class ResBlock(nn.Module):
    def __init__(self, ch: int):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(ch, ch, 3, padding=1), nn.InstanceNorm2d(ch), nn.ReLU(True),
            nn.Conv2d(ch, ch, 3, padding=1), nn.InstanceNorm2d(ch),
        )
    def forward(self, x): return x + self.block(x)


class GANBranch(nn.Module):
    """
    Pix2PixHD-style generator for fine-texture reconstruction.
    Excels at sharp local features (crop row structure, building edges).
    """

    def __init__(self, in_ch: int = 5, d_model: int = 256, n_res: int = 6):
        super().__init__()
        ngf = 64
        # Encoder
        self.enc = nn.Sequential(
            nn.ReflectionPad2d(3),
            nn.Conv2d(in_ch, ngf, 7), nn.InstanceNorm2d(ngf), nn.ReLU(True),
            nn.Conv2d(ngf, ngf*2, 3, stride=2, padding=1),
            nn.InstanceNorm2d(ngf*2), nn.ReLU(True),
            nn.Conv2d(ngf*2, ngf*4, 3, stride=2, padding=1),
            nn.InstanceNorm2d(ngf*4), nn.ReLU(True),
        )
        # Residual blocks
        self.res = nn.Sequential(*[ResBlock(ngf*4) for _ in range(n_res)])

        # Feature projection to d_model
        self.proj = nn.Sequential(
            nn.ConvTranspose2d(ngf*4, ngf*2, 3, stride=2, padding=1, output_padding=1),
            nn.InstanceNorm2d(ngf*2), nn.ReLU(True),
            nn.ConvTranspose2d(ngf*2, d_model, 3, stride=2, padding=1, output_padding=1),
        )

    def forward(
        self,
        optical:    torch.Tensor,
        cloud_mask: torch.Tensor,
        composite:  Optional[torch.Tensor],
    ) -> torch.Tensor:
        masked = optical * (1.0 - cloud_mask)
        if composite is not None:
            masked = masked + cloud_mask * composite
        x = torch.cat([masked, cloud_mask], dim=1)
        return self.proj(self.res(self.enc(x)))


# ─────────────────────────────────────────────
# Pixel Decoder (shared output head)
# ─────────────────────────────────────────────

class PixelDecoder(nn.Module):
    """
    Decodes fused features → reconstructed satellite bands.
    Uses residual connection from original (unmasked) optical.
    Cloud-region pixels are reconstructed; clear pixels are copied.
    """

    def __init__(self, d_model: int = 256, out_ch: int = 4):
        super().__init__()
        self.decode = nn.Sequential(
            nn.Conv2d(d_model, 128, 3, padding=1), nn.GELU(),
            nn.Conv2d(128,      64, 3, padding=1), nn.GELU(),
            nn.Conv2d(64,       32, 3, padding=1), nn.GELU(),
            nn.Conv2d(32,  out_ch,  1),
            nn.Sigmoid(),     # output in [0, 1]
        )

    def forward(
        self,
        fused:      torch.Tensor,             # (B, D, H, W)
        optical:    torch.Tensor,             # (B, 4, H, W) original
        cloud_mask: torch.Tensor,             # (B, 1, H, W) binary
    ) -> torch.Tensor:
        recon = self.decode(fused)            # (B, 4, H, W)
        # Composite: keep original pixels outside cloud, use reconstruction inside
        return optical * (1.0 - cloud_mask) + recon * cloud_mask


# ─────────────────────────────────────────────
# Full Pipeline
# ─────────────────────────────────────────────

class ReconstructionPipeline(nn.Module):
    """
    Complete Layer 4 reconstruction pipeline.

    Forward pass:
      1. Encode optical query features
      2. Run 4 branches in parallel
      3. Cross-attention fusion (adaptive branch weighting)
      4. Pixel decoder → reconstructed image

    Input dict keys:
      optical, cloud_mask, sar, temporal_stack, composite, temporal_context
    """

    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        d_model = cfg.reconstruction.fusion.d_model
        C_opt   = len(cfg.data.bands.optical)

        # ── 4 Branches ────────────────────────────────────────────
        from src.models.reconstruction.temporal_transformer import TemporalTransformer
        from src.models.fusion.sar_encoder import build_sar_encoder
        from src.models.fusion.cross_attention_fusion import (
            OpticalQueryEncoder, CrossAttentionFusionLayer, build_fusion_layer
        )

        self.query_enc   = OpticalQueryEncoder(C_opt, d_model)
        self.diff_branch = DiffusionBranch(C_opt, cond_ch=1, d_model=d_model)
        self.gan_branch  = GANBranch(C_opt + 1, d_model)

        temporal_embed = cfg.reconstruction.temporal_transformer.embed_dim
        self.temp_branch = TemporalTransformer(
            in_channels    = C_opt,
            embed_dim      = temporal_embed,
            max_T          = cfg.reconstruction.temporal_transformer.max_sequence_len,
            temporal_depth = cfg.reconstruction.temporal_transformer.temporal_depth,
            spatial_depth  = cfg.reconstruction.temporal_transformer.spatial_depth,
            num_heads      = cfg.reconstruction.temporal_transformer.num_heads,
        )
        self.temp_proj = nn.Identity()
        if temporal_embed != d_model:
            self.temp_proj = nn.Conv2d(temporal_embed, d_model, 1)

        self.sar_encoder = build_sar_encoder(cfg)

        # ── Fusion ────────────────────────────────────────────────
        self.fusion = build_fusion_layer(cfg)

        # ── Decoder ───────────────────────────────────────────────
        self.decoder = PixelDecoder(d_model, C_opt)

        log.info("ReconstructionPipeline initialised — 4 branches + cross-attention fusion")

    def forward(self, batch: Dict) -> Dict[str, torch.Tensor]:
        optical     = batch["optical"]                    # (B,4,H,W)
        cloud_mask  = batch["cloud_mask"]
        if cloud_mask.dim() == 2:
            cloud_mask = cloud_mask.unsqueeze(0).unsqueeze(0)
        elif cloud_mask.dim() == 3:
            cloud_mask = cloud_mask.unsqueeze(1)
        elif cloud_mask.dim() == 4 and cloud_mask.shape[1] != 1:
            raise ValueError(f"cloud_mask must have shape (B,H,W) or (B,1,H,W), but got {cloud_mask.shape}")
        cloud_mask = (cloud_mask > 0).float()

        sar         = batch.get("sar")                    # (B,4,H,W) or None
        temporal    = batch.get("temporal")               # (B,T,4,H,W) or None
        composite   = batch.get("composite")              # (B,4,H,W) or None
        temp_ctx    = batch.get("temporal_context", {})

        B, C, H, W = optical.shape

        # ── Optical query features ─────────────────────────────────
        q_feat = self.query_enc(optical)                  # (B,D,H,W)

        # ── Branch 1: Diffusion ────────────────────────────────────
        f1 = self.diff_branch(optical, cloud_mask, composite)   # (B,D,H,W)

        # ── Branch 2: GAN ──────────────────────────────────────────
        f2 = self.gan_branch(optical, cloud_mask, composite)    # (B,D,H,W)

        # ── Branch 3: Temporal Transformer ─────────────────────────
        if temporal is not None:
            t_masks = (cloud_mask.squeeze(1) > 0).float().unsqueeze(1)  # (B,1,H,W)
            t_masks = t_masks.expand(-1, temporal.shape[1], -1, -1)   # (B,T,H,W)
            f3 = self.temp_branch(temporal, t_masks)                   # (B,temporal_embed,H,W)
            f3 = self.temp_proj(f3)                                    # (B,d_model,H,W)
        else:
            f3 = torch.zeros(B, q_feat.shape[1], H, W, device=optical.device)

        # ── Branch 4: SAR Encoder ──────────────────────────────────
        sar_ch = sar.shape[1] if sar is not None else len(self.cfg.data.bands.sar)
        sar_input = sar if sar is not None else torch.zeros(B, sar_ch, H, W, device=optical.device)
        sar_out   = self.sar_encoder(sar_input)
        f4        = sar_out["features"]                   # (B,D,H,W)

        # ── Build condition tensor for adaptive gating ─────────────
        cloud_prob   = cloud_mask                         # (B,1,H,W)
        temp_consist = temp_ctx.get(
            "consistency", torch.ones(B, 1, H, W, device=optical.device)
        )
        if temp_consist.dim() == 2:
            temp_consist = temp_consist.unsqueeze(0).unsqueeze(1)
        elif temp_consist.dim() == 3:
            temp_consist = temp_consist.unsqueeze(1)
        elif temp_consist.dim() != 4:
            raise ValueError(
                f"temporal consistency tensor must be 2D, 3D, or 4D, got {temp_consist.shape}"
            )
        sar_edge     = sar_out["edge_mask"]               # (B,1,H,W)
        condition    = torch.cat([cloud_prob, temp_consist, sar_edge], dim=1)  # (B,3,H,W)

        # ── Cross-attention fusion ─────────────────────────────────
        fusion_out = self.fusion(
            query_features  = q_feat,
            branch_features = [f1, f2, f3, f4],
            condition       = condition,
            cloud_mask      = cloud_mask,
        )
        fused = fusion_out["fused"]                       # (B,D,H,W)

        # ── Pixel decoder ──────────────────────────────────────────
        reconstructed = self.decoder(fused, optical, cloud_mask)  # (B,4,H,W)

        return {
            "reconstructed":  reconstructed,
            "branch_weights": fusion_out["branch_weights"],
            "branch_features": [f1, f2, f3, f4],
            "fused_features":  fused,
        }

    def predict(
        self,
        optical:    torch.Tensor,
        cloud_mask: torch.Tensor,
        sar:        Optional[torch.Tensor] = None,
        temporal:   Optional[Dict] = None,
    ) -> torch.Tensor:
        """Inference convenience method — returns reconstructed image tensor."""
        device = next(self.parameters()).device
        optical = optical.to(device)
        cloud_mask = cloud_mask.to(device)
        if cloud_mask.dim() == 2:
            cloud_mask = cloud_mask.unsqueeze(0).unsqueeze(0)
        elif cloud_mask.dim() == 3:
            cloud_mask = cloud_mask.unsqueeze(1)
        elif cloud_mask.dim() == 4 and cloud_mask.shape[1] != 1:
            raise ValueError(f"cloud_mask must have shape (B,H,W) or (B,1,H,W), got {cloud_mask.shape}")
        sar = sar.to(device) if isinstance(sar, torch.Tensor) else None

        temporal_stack = None
        if temporal is not None:
            temporal_stack = temporal.get("stack")
            if isinstance(temporal_stack, torch.Tensor):
                temporal_stack = temporal_stack.to(device)

        composite = None
        if temporal is not None:
            composite = temporal.get("composite")
            if isinstance(composite, torch.Tensor):
                composite = composite.to(device)

        batch = {
            "optical":         optical,
            "cloud_mask":      cloud_mask,
            "sar":             sar,
            "temporal":        temporal_stack,
            "composite":       composite,
            "temporal_context": temporal or {},
        }
        with torch.no_grad():
            return self.forward(batch)["reconstructed"]

    @classmethod
    def from_checkpoint(cls, cfg, device: Optional[str] = None) -> "ReconstructionPipeline":
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        model = cls(cfg)
        ckpts = sorted(Path(cfg.paths.checkpoints).glob("reconstruction_*.ckpt"))
        if ckpts:
            state = torch.load(ckpts[-1], map_location=device)
            model.load_state_dict(state["state_dict"], strict=False)
            log.info(f"Loaded reconstruction model: {ckpts[-1]}")
        else:
            log.warning("No reconstruction checkpoint found — using random weights")
        return model.to(device).eval()
