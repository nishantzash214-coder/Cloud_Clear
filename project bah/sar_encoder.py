"""
src/models/fusion/sar_encoder.py

SAR Fusion Encoder — Branch 4 of the reconstruction engine (Layer 4).

Encodes 4-channel SAR input (VV, VH, ratio, edges) into a rich
structural feature map that guides the reconstruction branches.

Key property: SAR penetrates clouds, so this branch always sees
the true surface structure even when optical is completely occluded.

Architecture:
  ResNet-34 backbone (lightweight, fast) → FPN neck → (B, out_ch, H/4, W/4)
  Multi-scale outputs fused via lateral connections
"""

from __future__ import annotations
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Tuple
import logging

log = logging.getLogger(__name__)


class ConvBnRelu(nn.Module):
    def __init__(self, in_ch, out_ch, k=3, s=1, p=1):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, k, stride=s, padding=p, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )
    def forward(self, x): return self.block(x)


class FPNLateral(nn.Module):
    """Feature Pyramid Network lateral + top-down pathway."""
    def __init__(self, in_channels: List[int], out_ch: int = 256):
        super().__init__()
        self.lateral = nn.ModuleList([
            nn.Conv2d(c, out_ch, 1) for c in in_channels
        ])
        self.smooth = nn.ModuleList([
            ConvBnRelu(out_ch, out_ch, k=3, p=1) for _ in in_channels
        ])

    def forward(self, feats: List[torch.Tensor]) -> List[torch.Tensor]:
        """feats: [P2, P3, P4, P5] from coarse to fine (reversed)."""
        # Laterals
        laterals = [lat(f) for lat, f in zip(self.lateral, feats)]

        # Top-down fusion (coarse → fine)
        for i in range(len(laterals) - 2, -1, -1):
            up = F.interpolate(laterals[i+1], size=laterals[i].shape[2:],
                               mode="nearest")
            laterals[i] = laterals[i] + up

        return [smooth(lat) for smooth, lat in zip(self.smooth, laterals)]


class SARFusionEncoder(nn.Module):
    """
    SAR Fusion Encoder Branch.

    Input:   (B, 4, H, W)  — VV, VH, ratio, edge_magnitude
    Output:  (B, out_ch, H, W)  — structural feature map at full resolution

    The output is consumed by the Cross-Attention Fusion Layer as K, V keys.
    """

    def __init__(
        self,
        in_channels: int = 4,
        out_channels: int = 256,
        pretrained:  bool = True,
    ):
        super().__init__()
        self.out_channels = out_channels

        # ── Backbone: ResNet-34 ───────────────────────────────────
        import torchvision.models as models
        resnet = models.resnet34(weights="IMAGENET1K_V1" if pretrained else None)

        # Adapt first conv for multi-channel SAR input
        old_w = resnet.conv1.weight                     # (64, 3, 7, 7)
        new_conv = nn.Conv2d(in_channels, 64, 7, stride=2, padding=3, bias=False)
        with torch.no_grad():
            if in_channels == 3:
                new_conv.weight.copy_(old_w)
            elif in_channels == 4:
                new_conv.weight[:, :3] = old_w
                new_conv.weight[:, 3] = old_w.mean(dim=1)
            elif in_channels == 2:
                new_conv.weight[:, 0] = old_w[:, 0]
                new_conv.weight[:, 1] = old_w[:, 1]
            else:
                new_conv.weight[:, :old_w.shape[1]] = old_w
                if in_channels > old_w.shape[1]:
                    extra = in_channels - old_w.shape[1]
                    new_conv.weight[:, old_w.shape[1]:] = old_w.mean(dim=1, keepdim=True).repeat(1, extra, 1, 1)
        resnet.conv1 = new_conv

        self.stem  = nn.Sequential(resnet.conv1, resnet.bn1, resnet.relu,
                                   resnet.maxpool)         # /4,  ch=64
        self.layer1 = resnet.layer1                        # /4,  ch=64
        self.layer2 = resnet.layer2                        # /8,  ch=128
        self.layer3 = resnet.layer3                        # /16, ch=256
        self.layer4 = resnet.layer4                        # /32, ch=512

        # ── FPN neck ──────────────────────────────────────────────
        self.fpn = FPNLateral(
            in_channels=[64, 128, 256, 512],
            out_ch=out_channels,
        )

        # ── Merge multi-scale FPN outputs → full resolution ───────
        self.merge = nn.Sequential(
            ConvBnRelu(out_channels * 4, out_channels * 2),
            ConvBnRelu(out_channels * 2, out_channels),
        )

        # ── SAR-specific attention: highlight structural edges ─────
        self.edge_attention = nn.Sequential(
            nn.Conv2d(out_channels, 1, 1),
            nn.Sigmoid(),
        )

        log.info(f"SARFusionEncoder initialised: {in_channels}ch → {out_channels}ch")

    def forward(self, sar: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Args:
            sar: (B, 4, H, W) preprocessed SAR features

        Returns dict with:
            features:   (B, out_ch, H, W)  — main structural features
            edge_mask:  (B, 1, H, W)       — structural edge attention map
            fpn_scales: list of 4 feature maps at different scales
        """
        B, C, H, W = sar.shape

        # Backbone
        s0 = self.stem(sar)    # /4
        p1 = self.layer1(s0)   # /4,  (B, 64,  H/4, W/4)
        p2 = self.layer2(p1)   # /8,  (B, 128, H/8, W/8)
        p3 = self.layer3(p2)   # /16, (B, 256, H/16,W/16)
        p4 = self.layer4(p3)   # /32, (B, 512, H/32,W/32)

        # FPN: produces 4 feature maps at scales /4, /8, /16, /32
        fpn_outs = self.fpn([p1, p2, p3, p4])

        # Upsample all FPN outputs to H/4 (finest scale)
        target_size = fpn_outs[0].shape[2:]
        upsampled   = [
            F.interpolate(f, size=target_size, mode="bilinear", align_corners=False)
            for f in fpn_outs
        ]
        merged = self.merge(torch.cat(upsampled, dim=1))  # (B, out_ch, H/4, W/4)

        # Upsample to full resolution
        features = F.interpolate(merged, size=(H, W), mode="bilinear",
                                  align_corners=False)

        edge_mask = self.edge_attention(features)

        return {
            "features":   features,
            "edge_mask":  edge_mask,
            "fpn_scales": fpn_outs,
        }


class ZeroSAREncoder(nn.Module):
    """
    Fallback when SAR data is unavailable.
    Returns zero features of the correct shape so the fusion layer
    receives a consistent input regardless of data availability.
    """
    def __init__(self, out_channels: int = 256):
        super().__init__()
        self.out_channels = out_channels

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        B, C, H, W = x.shape
        dev = x.device
        return {
            "features":   torch.zeros(B, self.out_channels, H, W, device=dev),
            "edge_mask":  torch.zeros(B, 1, H, W, device=dev),
            "fpn_scales": [],
        }


def build_sar_encoder(cfg) -> nn.Module:
    """Factory: return real or zero SAR encoder based on config."""
    if cfg.reconstruction.sar_encoder.enabled:
        return SARFusionEncoder(
            in_channels  = len(cfg.data.bands.sar),
            out_channels = cfg.reconstruction.fusion.d_model,
            pretrained   = True,
        )
    return ZeroSAREncoder(out_channels=cfg.reconstruction.fusion.d_model)
