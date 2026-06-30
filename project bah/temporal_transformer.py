"""
src/models/reconstruction/temporal_transformer.py

Temporal Transformer — Branch 3 of the reconstruction engine (Layer 4).

Processes the ±15-day stack of co-registered observations to learn
temporal surface dynamics. The transformer reasons across both time
(which observation is most informative?) and space (which pixels are stable?)
to produce temporally grounded features for cloud-covered regions.

Architecture:
  - Spatial patch embedding (16×16 patches, like ViT)
  - Temporal attention across T observations per spatial position
  - Spatial attention across positions within each time step
  - Learned positional encodings for both space and time
  - Output: (B, C_embed, H, W) feature map
"""

from __future__ import annotations
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, repeat
from typing import Optional, Tuple
import logging
import math

log = logging.getLogger(__name__)


# ─────────────────────────────────────────────
# Positional encodings
# ─────────────────────────────────────────────

class SinCos2DPositionEncoding(nn.Module):
    """2D sinusoidal position encoding for spatial patch tokens."""

    def __init__(self, embed_dim: int, max_h: int = 16, max_w: int = 16):
        super().__init__()
        assert embed_dim % 4 == 0

        pe_h = self._sinusoidal(max_h, embed_dim // 2)
        pe_w = self._sinusoidal(max_w, embed_dim // 2)

        # (max_h, max_w, embed_dim)
        pe = torch.cat([
            pe_h.unsqueeze(1).expand(-1, max_w, -1),
            pe_w.unsqueeze(0).expand(max_h, -1, -1),
        ], dim=-1)
        self.register_buffer("pe", pe.reshape(max_h * max_w, embed_dim))

    @staticmethod
    def _sinusoidal(length: int, dim: int) -> torch.Tensor:
        pos  = torch.arange(length, dtype=torch.float32).unsqueeze(1)
        div  = torch.exp(torch.arange(0, dim, 2).float() * -(math.log(10000) / dim))
        enc  = torch.zeros(length, dim)
        enc[:, 0::2] = torch.sin(pos * div)
        enc[:, 1::2] = torch.cos(pos * div)
        return enc

    def forward(self, N: int) -> torch.Tensor:
        return self.pe[:N]


class TemporalPositionEncoding(nn.Module):
    """Learnable temporal position encoding for T time steps."""
    def __init__(self, max_T: int, embed_dim: int):
        super().__init__()
        self.embed = nn.Embedding(max_T, embed_dim)

    def forward(self, T: int) -> torch.Tensor:
        return self.embed(torch.arange(T, device=self.embed.weight.device))


# ─────────────────────────────────────────────
# Transformer blocks
# ─────────────────────────────────────────────

class MultiHeadSelfAttention(nn.Module):
    def __init__(self, embed_dim: int, num_heads: int, dropout: float = 0.1):
        super().__init__()
        self.attn = nn.MultiheadAttention(
            embed_dim, num_heads, dropout=dropout, batch_first=True
        )
        self.norm = nn.LayerNorm(embed_dim)
        self.drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor,
                key_padding_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        attn_out, _ = self.attn(x, x, x, key_padding_mask=key_padding_mask)
        return self.norm(x + self.drop(attn_out))


class FeedForward(nn.Module):
    def __init__(self, embed_dim: int, mlp_ratio: int = 4, dropout: float = 0.1):
        super().__init__()
        hidden = embed_dim * mlp_ratio
        self.ff = nn.Sequential(
            nn.Linear(embed_dim, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, embed_dim),
            nn.Dropout(dropout),
        )
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.norm(x + self.ff(x))


class TemporalAttentionBlock(nn.Module):
    """Attends across T time steps for each spatial position."""
    def __init__(self, embed_dim: int, num_heads: int, dropout: float):
        super().__init__()
        self.attn = MultiHeadSelfAttention(embed_dim, num_heads, dropout)
        self.ff   = FeedForward(embed_dim, mlp_ratio=4, dropout=dropout)

    def forward(self, x: torch.Tensor,
                cloud_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Args:
            x:          (B*N_patches, T, embed_dim)  — N patches, T timesteps
            cloud_mask: (B*N_patches, T) bool — True = cloudy (mask out)
        """
        x = self.attn(x, key_padding_mask=cloud_mask)
        return self.ff(x)


class SpatialAttentionBlock(nn.Module):
    """Attends across spatial patches within each time step."""
    def __init__(self, embed_dim: int, num_heads: int, dropout: float):
        super().__init__()
        self.attn = MultiHeadSelfAttention(embed_dim, num_heads, dropout)
        self.ff   = FeedForward(embed_dim, mlp_ratio=4, dropout=dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B*T, N_patches, embed_dim)
        """
        x = self.attn(x)
        return self.ff(x)


# ─────────────────────────────────────────────
# Full Temporal Transformer
# ─────────────────────────────────────────────

class TemporalTransformer(nn.Module):
    """
    Temporal Transformer for ±15-day observation stack.

    Input:
      temporal_stack: (B, T, C, H, W)  — T multi-temporal optical scenes
      cloud_masks:    (B, T, H, W)     — 0=clear, >0=cloud (optional)

    Output:
      features: (B, embed_dim, H, W)  — temporally grounded feature map
    """

    def __init__(
        self,
        in_channels:   int = 4,
        embed_dim:     int = 256,
        patch_size:    int = 8,
        max_T:         int = 30,
        temporal_depth: int = 4,
        spatial_depth:  int = 4,
        num_heads:     int = 8,
        dropout:       float = 0.1,
        img_size:      int = 256,
    ):
        super().__init__()
        self.patch_size = patch_size
        self.embed_dim  = embed_dim
        n_patches_1d    = img_size // patch_size
        self.n_patches  = n_patches_1d ** 2

        # ── Patch embedding ───────────────────────────────────────
        patch_dim = in_channels * patch_size * patch_size
        self.patch_embed = nn.Sequential(
            nn.LayerNorm(patch_dim),
            nn.Linear(patch_dim, embed_dim),
            nn.LayerNorm(embed_dim),
        )

        # ── Positional encodings ──────────────────────────────────
        self.spatial_pos = SinCos2DPositionEncoding(embed_dim, n_patches_1d, n_patches_1d)
        self.temporal_pos = TemporalPositionEncoding(max_T, embed_dim)

        # ── Transformer layers ────────────────────────────────────
        self.temporal_layers = nn.ModuleList([
            TemporalAttentionBlock(embed_dim, num_heads, dropout)
            for _ in range(temporal_depth)
        ])
        self.spatial_layers = nn.ModuleList([
            SpatialAttentionBlock(embed_dim, num_heads, dropout)
            for _ in range(spatial_depth)
        ])

        # ── Output projection ─────────────────────────────────────
        self.norm = nn.LayerNorm(embed_dim)
        self.out_proj = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.GELU(),
        )

        log.info(
            f"TemporalTransformer: T_max={max_T}, patches={self.n_patches}, "
            f"embed={embed_dim}, temporal_depth={temporal_depth}, "
            f"spatial_depth={spatial_depth}"
        )

    def forward(
        self,
        temporal_stack: torch.Tensor,          # (B, T, C, H, W)
        cloud_masks:    Optional[torch.Tensor] = None,  # (B, T, H, W)
    ) -> torch.Tensor:
        """Returns (B, embed_dim, H, W) temporally-grounded feature map."""
        B, T, C, H, W = temporal_stack.shape
        P = self.patch_size

        # ── Patchify: (B, T, C, H, W) → (B, T, N, patch_dim) ────
        x = rearrange(temporal_stack, "b t c (h p1) (w p2) -> b t (h w) (c p1 p2)",
                       p1=P, p2=P)
        N = x.shape[2]  # number of patches

        # Embed patches
        x = self.patch_embed(x)                            # (B, T, N, D)

        # Add positional encodings
        sp_enc = self.spatial_pos(N).unsqueeze(0).unsqueeze(0)  # (1,1,N,D)
        tp_enc = self.temporal_pos(T).unsqueeze(0).unsqueeze(2) # (1,T,1,D)
        x = x + sp_enc + tp_enc                            # (B, T, N, D)

        # ── Build cloud attention mask ─────────────────────────────
        t_mask = None
        if cloud_masks is not None:
            # Pool cloud mask to patch resolution
            mask_pooled = F.max_pool2d(
                (cloud_masks > 0).float().reshape(B * T, 1, H, W),
                kernel_size=P, stride=P,
            )                                              # (B*T, 1, N^0.5, N^0.5)
            mask_pooled = mask_pooled.reshape(B, T, N)    # (B, T, N)
            # For temporal attention: (B*N, T) — mask cloudy time steps
            t_mask = rearrange(mask_pooled, "b t n -> (b n) t").bool()

        # ── Temporal attention ─────────────────────────────────────
        # Reshape: (B, T, N, D) → (B*N, T, D) for temporal attention
        x = rearrange(x, "b t n d -> (b n) t d")
        for layer in self.temporal_layers:
            x = layer(x, cloud_mask=t_mask)
        x = rearrange(x, "(b n) t d -> b t n d", b=B, n=N)

        # ── Spatial attention ──────────────────────────────────────
        # Reshape: (B, T, N, D) → (B*T, N, D) for spatial attention
        x = rearrange(x, "b t n d -> (b t) n d")
        for layer in self.spatial_layers:
            x = layer(x)
        x = rearrange(x, "(b t) n d -> b t n d", b=B, t=T)

        # ── Temporal pooling: aggregate T → single output ─────────
        # Weight by cloud-free observations (clear pixels have more weight)
        if cloud_masks is not None:
            clear_weight = (cloud_masks == 0).float()          # (B, T, H, W)
            clear_pool   = F.avg_pool2d(
                clear_weight.reshape(B * T, 1, H, W), kernel_size=P, stride=P
            ).reshape(B, T, 1, N)
            clear_pool   = clear_pool + 1e-6
            weights      = clear_pool / clear_pool.sum(dim=1, keepdim=True)
            x = (x * weights.permute(0, 1, 3, 2)).sum(dim=1)  # (B, N, D)
        else:
            x = x.mean(dim=1)                                  # (B, N, D)

        x = self.out_proj(self.norm(x))                        # (B, N, D)

        # ── Reshape patches → spatial feature map ─────────────────
        n1d = int(N ** 0.5)
        x   = rearrange(x, "b (h w) d -> b d h w", h=n1d, w=n1d)

        # Upsample to full (H, W)
        x = F.interpolate(x, size=(H, W), mode="bilinear", align_corners=False)
        return x                                               # (B, embed_dim, H, W)
