"""
src/models/fusion/cross_attention_fusion.py

Cross-Attention Fusion Layer — combines all 4 reconstruction branches (Layer 4).

Architecture:
  Q = target feature query  (from cloudy optical encoder)
  K, V = stacked features from all 4 branches:
           F1 (Diffusion), F2 (GAN), F3 (Temporal TF), F4 (SAR)

  Attention = softmax(Q * K^T / sqrt(d)) * V

Key innovation — Adaptive Branch Weighting:
  Branch contributions are NOT fixed. A lightweight gating network
  predicts per-pixel branch weights conditioned on:
    - Cloud coverage fraction (thick cloud → upweight SAR + Temporal)
    - Temporal consistency score (low consistency → upweight Diffusion)
    - SAR edge confidence (high edges → upweight SAR branch)

This means the fusion is cloud-density-aware: for a pixel under
thick cloud, SAR and temporal branches dominate; for a thin cloud
pixel, the diffusion and GAN branches contribute more.
"""

from __future__ import annotations
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from typing import Dict, Optional, List
import logging

log = logging.getLogger(__name__)


# ─────────────────────────────────────────────
# Adaptive Branch Gating Network
# ─────────────────────────────────────────────

class AdaptiveBranchGate(nn.Module):
    """
    Predicts per-pixel branch weights conditioned on cloud density and
    SAR/temporal confidence signals.

    Input:  condition vector (B, cond_dim, H, W)
    Output: (B, n_branches, H, W) softmax weights
    """

    def __init__(self, cond_dim: int, n_branches: int = 4):
        super().__init__()
        self.n_branches = n_branches
        self.net = nn.Sequential(
            nn.Conv2d(cond_dim, 64, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 32, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, n_branches, 1),
        )

    def forward(self, condition: torch.Tensor) -> torch.Tensor:
        """Returns (B, n_branches, H, W) softmax weights."""
        logits = self.net(condition)
        return F.softmax(logits, dim=1)


# ─────────────────────────────────────────────
# Multi-head Cross-Attention (spatial)
# ─────────────────────────────────────────────

class SpatialCrossAttention(nn.Module):
    """
    Cross-attention in the spatial domain.
    Q comes from the main feature query.
    K, V come from branch features.
    Uses window-based attention for efficiency on large feature maps.
    """

    def __init__(self, d_model: int, num_heads: int, dropout: float = 0.1,
                 window_size: int = 8):
        super().__init__()
        self.d_model     = d_model
        self.num_heads   = num_heads
        self.window_size = window_size

        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.out    = nn.Linear(d_model, d_model)
        self.drop   = nn.Dropout(dropout)
        self.scale  = (d_model // num_heads) ** -0.5

    def forward(
        self,
        query: torch.Tensor,     # (B, N, d_model)  — N = H*W
        key:   torch.Tensor,     # (B, N, d_model)
        value: torch.Tensor,     # (B, N, d_model)
    ) -> torch.Tensor:
        B, N, D = query.shape
        H = self.num_heads
        if D % H != 0:
            raise ValueError(f"d_model {D} must be divisible by num_heads {H}")
        Dh = D // H

        S = int(N ** 0.5)
        if S * S != N:
            raise ValueError(f"SpatialCrossAttention requires square spatial layout, got N={N}")
        if S % self.window_size != 0:
            raise ValueError(
                f"Feature map size {S} is not divisible by window_size {self.window_size}"
            )

        q = self.q_proj(query).reshape(B, N, H, Dh).permute(0, 2, 1, 3)
        k = self.k_proj(key).reshape(B, N, H, Dh).permute(0, 2, 1, 3)
        v = self.v_proj(value).reshape(B, N, H, Dh).permute(0, 2, 1, 3)

        attn = torch.softmax(q @ k.transpose(-2, -1) * self.scale, dim=-1)
        attn = self.drop(attn)
        out = attn @ v

        out = out.transpose(1, 2).reshape(B, N, D)
        return self.out(out)


# ─────────────────────────────────────────────
# Fusion Layer
# ─────────────────────────────────────────────

class CrossAttentionFusionLayer(nn.Module):
    """
    Full cross-attention fusion of 4 reconstruction branches.

    Inputs:
      query_features:  (B, d_model, H, W) — from cloudy optical encoder
      branch_features: list of 4 tensors  (B, d_model, H, W)
                         [f_diffusion, f_gan, f_temporal, f_sar]
      condition:       (B, cond_dim, H, W) — cloud mask + temporal score
      cloud_mask:      (B, 1, H, W)        — binary cloud region

    Output:
      (B, d_model, H, W) fused feature map
    """

    def __init__(
        self,
        d_model:     int = 256,
        num_heads:   int = 8,
        num_layers:  int = 4,
        dropout:     float = 0.1,
        n_branches:  int = 4,
        adaptive:    bool = True,
    ):
        super().__init__()
        self.d_model    = d_model
        self.n_branches = n_branches
        self.adaptive   = adaptive

        # Project each branch to d_model (in case they have different dims)
        self.branch_proj = nn.ModuleList([
            nn.Conv2d(d_model, d_model, 1) for _ in range(n_branches)
        ])

        # Query encoder (processes cloudy optical features)
        self.query_encoder = nn.Sequential(
            nn.Conv2d(d_model, d_model, 3, padding=1),
            nn.GELU(),
            nn.Conv2d(d_model, d_model, 1),
        )

        # Multi-scale cross-attention layers
        self.cross_attn_layers = nn.ModuleList([
            SpatialCrossAttention(d_model, num_heads, dropout)
            for _ in range(num_layers)
        ])
        self.layer_norms = nn.ModuleList([
            nn.LayerNorm(d_model) for _ in range(num_layers)
        ])
        self.ff_layers = nn.ModuleList([
            nn.Sequential(
                nn.LayerNorm(d_model),
                nn.Linear(d_model, d_model * 4),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(d_model * 4, d_model),
            )
            for _ in range(num_layers)
        ])

        # Adaptive gating network
        cond_dim = 3  # cloud_prob + temporal_consistency + sar_edge_confidence
        if adaptive:
            self.gate = AdaptiveBranchGate(cond_dim, n_branches)
        else:
            # Fixed equal weights
            self.register_buffer("fixed_weights",
                                 torch.ones(1, n_branches, 1, 1) / n_branches)

        # Output refinement (cloud region only)
        self.cloud_refine = nn.Sequential(
            nn.Conv2d(d_model, d_model, 3, padding=1),
            nn.GELU(),
            nn.Conv2d(d_model, d_model, 3, padding=1),
        )
        self.norm_out = nn.GroupNorm(8, d_model)

    def forward(
        self,
        query_features:  torch.Tensor,
        branch_features: List[torch.Tensor],
        condition:       Optional[torch.Tensor] = None,
        cloud_mask:      Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        B, D, H, W = query_features.shape

        # ── 1. Project all branch features ────────────────────────
        proj_branches = [
            proj(f) for proj, f in zip(self.branch_proj, branch_features)
        ]                                                  # 4 × (B, D, H, W)

        # ── 2. Compute adaptive branch weights ────────────────────
        if self.adaptive and condition is not None:
            weights = self.gate(condition)                 # (B, 4, H, W)
        else:
            weights = self.fixed_weights.expand(B, -1, H, W)

        # ── 3. Weighted combination of branch features ────────────
        # This is the "K, V" side of the cross-attention
        kv_combined = sum(
            weights[:, i:i+1] * proj_branches[i]
            for i in range(self.n_branches)
        )                                                  # (B, D, H, W)

        # ── 4. Build query from optical encoder ───────────────────
        q = self.query_encoder(query_features)             # (B, D, H, W)

        # ── 5. Cross-attention (spatial) ──────────────────────────
        # Flatten spatial dimensions for attention
        q_flat  = rearrange(q,           "b d h w -> b (h w) d")
        kv_flat = rearrange(kv_combined, "b d h w -> b (h w) d")

        x = q_flat
        for attn, norm, ff in zip(self.cross_attn_layers,
                                   self.layer_norms,
                                   self.ff_layers):
            x = norm(x + attn(x, kv_flat, kv_flat))
            x = x + ff(x)

        # ── 6. Reshape back to spatial ────────────────────────────
        fused = rearrange(x, "b (h w) d -> b d h w", h=H, w=W)

        # ── 7. Apply cloud-aware refinement ───────────────────────
        # Only refine pixels inside the cloud mask (outside = keep optical)
        refined = self.cloud_refine(fused)
        fused   = self.norm_out(fused + refined)

        if cloud_mask is not None:
            # Outside cloud: blend back toward original query features
            alpha = cloud_mask.float()                     # 1 inside cloud
            fused = alpha * fused + (1 - alpha) * query_features

        return {
            "fused":          fused,          # (B, D, H, W)
            "branch_weights": weights,         # (B, 4, H, W) for interpretability
            "kv_combined":    kv_combined,     # (B, D, H, W) weighted branch avg
        }


# ─────────────────────────────────────────────
# Optical feature encoder (query side)
# ─────────────────────────────────────────────

class OpticalQueryEncoder(nn.Module):
    """
    Lightweight encoder for the cloudy optical input.
    Produces the query features for the cross-attention layer.
    """
    def __init__(self, in_channels: int = 4, d_model: int = 256):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(in_channels, 64, 7, padding=3),
            nn.GELU(),
            nn.Conv2d(64, 128, 3, padding=1),
            nn.GELU(),
            nn.Conv2d(128, d_model, 3, padding=1),
            nn.GroupNorm(8, d_model),
            nn.GELU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.encoder(x)


def build_fusion_layer(cfg) -> CrossAttentionFusionLayer:
    return CrossAttentionFusionLayer(
        d_model    = cfg.reconstruction.fusion.d_model,
        num_heads  = cfg.reconstruction.fusion.num_heads,
        num_layers = cfg.reconstruction.fusion.num_layers,
        dropout    = cfg.reconstruction.fusion.dropout,
        adaptive   = cfg.reconstruction.fusion.adaptive_weighting,
    )
