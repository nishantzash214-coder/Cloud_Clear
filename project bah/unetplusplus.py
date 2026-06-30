"""
src/models/cloud_detection/unetplusplus.py

U-Net++ based cloud segmentation model (Layer 2).

Classes:
  0 = clear
  1 = thin cloud / haze
  2 = thick cloud
  3 = cloud shadow

Outputs:
  - logits:       (B, 4, H, W)   raw class logits
  - cloud_prob:   (B, 1, H, W)   P(any cloud) = 1 - P(clear)
  - confidence:   (B, 1, H, W)   max softmax probability (detection confidence)

Architecture:
  Encoder: ResNet-50 (ImageNet pretrained, adapted for 4-band input)
  Decoder: U-Net++ dense skip connections
  Head:    1×1 conv → 4 classes
"""

from __future__ import annotations
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional, Dict
import logging

log = logging.getLogger(__name__)


# ─────────────────────────────────────────────
# Building blocks
# ─────────────────────────────────────────────

class ConvBnRelu(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, kernel: int = 3, padding: int = 1):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel, padding=padding, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.block(x)


class DenseBlock(nn.Module):
    """U-Net++ dense node: aggregates multiple skip connections."""
    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.conv = nn.Sequential(
            ConvBnRelu(in_ch, out_ch),
            ConvBnRelu(out_ch, out_ch),
        )

    def forward(self, *inputs):
        x = torch.cat(inputs, dim=1)
        return self.conv(x)


class AttentionGate(nn.Module):
    """
    Additive attention gate — focuses decoder on relevant spatial regions.
    Helps the model attend to cloud boundaries.
    """
    def __init__(self, g_ch: int, x_ch: int, inter_ch: int):
        super().__init__()
        self.W_g = nn.Conv2d(g_ch, inter_ch, 1)
        self.W_x = nn.Conv2d(x_ch, inter_ch, 1)
        self.psi = nn.Sequential(
            nn.Conv2d(inter_ch, 1, 1),
            nn.Sigmoid(),
        )

    def forward(self, g: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        g_up = F.interpolate(g, size=x.shape[2:], mode="bilinear", align_corners=False)
        psi  = F.relu(self.W_g(g_up) + self.W_x(x))
        alpha = self.psi(psi)
        return g_up * alpha


# ─────────────────────────────────────────────
# U-Net++ Encoder
# ─────────────────────────────────────────────

class Encoder(nn.Module):
    """
    4-band adapted ResNet-50 encoder.
    First conv layer is re-initialised for 4 input channels (G,R,NIR,SWIR).
    Pretrained RGB weights are averaged across the 3 channels and replicated
    for the 4th band — preserves feature learning while using pretrained init.
    """

    def __init__(self, in_channels: int = 4, pretrained: bool = True):
        super().__init__()
        import torchvision.models as models
        resnet = models.resnet50(weights="IMAGENET1K_V2" if pretrained else None)

        # Adapt first conv for 4-band input
        old_conv  = resnet.conv1
        new_conv  = nn.Conv2d(in_channels, 64, kernel_size=7, stride=2,
                               padding=3, bias=False)
        if pretrained:
            with torch.no_grad():
                # Average RGB weights → replicate for 4th band
                rgb_mean = old_conv.weight.mean(dim=1, keepdim=True)
                new_conv.weight[:, :3] = old_conv.weight
                new_conv.weight[:,  3] = rgb_mean.squeeze(1)
        resnet.conv1 = new_conv

        self.enc0 = nn.Sequential(resnet.conv1, resnet.bn1, resnet.relu)  # /2
        self.pool = resnet.maxpool                                          # /4
        self.enc1 = resnet.layer1    # /4,  ch=256
        self.enc2 = resnet.layer2    # /8,  ch=512
        self.enc3 = resnet.layer3    # /16, ch=1024
        self.enc4 = resnet.layer4    # /32, ch=2048

        self.out_channels = [64, 256, 512, 1024, 2048]

    def forward(self, x: torch.Tensor):
        e0 = self.enc0(x)                  # (B,  64, H/2,  W/2)
        e1 = self.enc1(self.pool(e0))      # (B, 256, H/4,  W/4)
        e2 = self.enc2(e1)                 # (B, 512, H/8,  W/8)
        e3 = self.enc3(e2)                 # (B,1024, H/16, W/16)
        e4 = self.enc4(e3)                 # (B,2048, H/32, W/32)
        return e0, e1, e2, e3, e4


# ─────────────────────────────────────────────
# U-Net++ Decoder
# ─────────────────────────────────────────────

class UNetPlusPlusDecoder(nn.Module):
    """
    U-Net++ decoder with dense skip connections.
    Each node X^{i,j} aggregates all X^{i, <j} nodes at the same scale
    and the upsampled output from X^{i+1, j-1}.
    """

    def __init__(self, enc_channels, decoder_ch=256, dropout=0.2):
        super().__init__()
        ch = decoder_ch
        e0, e1, e2, e3, e4 = enc_channels  # [64, 256, 512, 1024, 2048]

        # ── Row 0 (full resolution) ───────────────────────────────
        self.node_0_1 = DenseBlock(e0 + e1,          ch)
        self.node_0_2 = DenseBlock(e0 + ch * 2,      ch)
        self.node_0_3 = DenseBlock(e0 + ch * 3,      ch)
        self.node_0_4 = DenseBlock(e0 + ch * 4,      ch)

        # ── Row 1 ─────────────────────────────────────────────────
        self.node_1_1 = DenseBlock(e1 + e2,          ch)
        self.node_1_2 = DenseBlock(e1 + ch * 2,      ch)
        self.node_1_3 = DenseBlock(e1 + ch * 3,      ch)

        # ── Row 2 ─────────────────────────────────────────────────
        self.node_2_1 = DenseBlock(e2 + e3,          ch)
        self.node_2_2 = DenseBlock(e2 + ch * 2,      ch)

        # ── Row 3 ─────────────────────────────────────────────────
        self.node_3_1 = DenseBlock(e3 + e4,          ch)

        # ── Attention gates ───────────────────────────────────────
        self.att_0 = AttentionGate(ch, e0, ch // 2)

        self.dropout = nn.Dropout2d(dropout)

        self.up = lambda x, ref: F.interpolate(
            x, size=ref.shape[2:], mode="bilinear", align_corners=False
        )

    def forward(self, feats):
        e0, e1, e2, e3, e4 = feats

        # Row 3
        x3_1 = self.node_3_1(e3, self.up(e4, e3))

        # Row 2
        x2_1 = self.node_2_1(e2, self.up(e3, e2))
        x2_2 = self.node_2_2(e2, self.up(x3_1, e2), x2_1)

        # Row 1
        x1_1 = self.node_1_1(e1, self.up(e2, e1))
        x1_2 = self.node_1_2(e1, self.up(x2_1, e1), x1_1)
        x1_3 = self.node_1_3(e1, self.up(x2_2, e1), x1_1, x1_2)

        # Row 0
        x0_1 = self.node_0_1(e0, self.up(e1, e0))
        x0_2 = self.node_0_2(e0, self.up(x1_1, e0), x0_1)
        x0_3 = self.node_0_3(e0, self.up(x1_2, e0), x0_1, x0_2)
        x0_4 = self.node_0_4(e0, self.up(x1_3, e0), x0_1, x0_2, x0_3)

        # Attention + dropout on final feature map
        x0_4 = self.att_0(x0_4, e0)
        return self.dropout(x0_4)


# ─────────────────────────────────────────────
# Full model
# ─────────────────────────────────────────────

class CloudDetector(nn.Module):
    """
    U-Net++ cloud segmentation model.

    Inputs:  (B, 4, H, W) normalised optical imagery
    Outputs: dict with logits, cloud_prob, confidence
    """

    def __init__(
        self,
        in_channels:  int  = 4,
        num_classes:  int  = 4,
        decoder_ch:   int  = 256,
        dropout:      float = 0.2,
        pretrained:   bool  = True,
    ):
        super().__init__()
        self.encoder = Encoder(in_channels, pretrained)
        self.decoder = UNetPlusPlusDecoder(
            self.encoder.out_channels, decoder_ch, dropout
        )
        self.head = nn.Sequential(
            nn.Conv2d(decoder_ch, decoder_ch // 2, 3, padding=1),
            nn.BatchNorm2d(decoder_ch // 2),
            nn.ReLU(inplace=True),
            nn.Conv2d(decoder_ch // 2, num_classes, 1),
        )
        self.num_classes = num_classes

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        feats  = self.encoder(x)
        dec    = self.decoder(feats)
        logits = self.head(dec)                             # (B,4,H,W)

        # Upsample to original resolution if needed
        if logits.shape[2:] != x.shape[2:]:
            logits = F.interpolate(logits, size=x.shape[2:],
                                   mode="bilinear", align_corners=False)

        probs      = F.softmax(logits, dim=1)               # (B,4,H,W)
        cloud_prob = 1.0 - probs[:, 0:1]                   # P(not clear)
        confidence = probs.max(dim=1, keepdim=True).values  # max class prob

        return {
            "logits":     logits,
            "cloud_prob": cloud_prob,
            "confidence": confidence,
            "class_map":  logits.argmax(dim=1),             # (B,H,W) int
        }

    def predict(self, x: torch.Tensor) -> torch.Tensor:
        """Convenience: return (B, H, W) integer class map."""
        with torch.no_grad():
            return self.forward(x)["class_map"]

    @classmethod
    def from_checkpoint(cls, cfg, device: Optional[str] = None) -> "CloudDetector":
        import os
        from pathlib import Path
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        model = cls(
            in_channels = len(cfg.data.bands.optical),
            num_classes = cfg.cloud_detection.num_classes,
            dropout     = cfg.cloud_detection.dropout,
            pretrained  = False,
        )
        ckpt_dir = Path(cfg.paths.checkpoints)
        ckpts    = sorted(ckpt_dir.glob("cloud_detection_*.ckpt"))
        if ckpts:
            state = torch.load(ckpts[-1], map_location=device)
            model.load_state_dict(state["state_dict"], strict=False)
            log.info(f"Loaded cloud detector: {ckpts[-1]}")
        else:
            log.warning("No cloud detection checkpoint found — using random weights")
        return model.to(device).eval()
