"""
src/visualization/visualizer.py

Visualization utilities for cloud removal outputs.

Generates:
  1. RGB false-colour composites (NIR-Red-Green)
  2. Side-by-side comparison: cloudy | reconstructed | confidence
  3. Spectral index comparison maps
  4. Branch weight heatmaps (interpretability)
  5. Verification pass summary charts
  6. Temporal consistency maps

All outputs saved as PNG for report embedding and web display.
"""

from __future__ import annotations
import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")   # non-interactive backend
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.gridspec import GridSpec
from pathlib import Path
from typing import Optional, Dict, List, Tuple
import logging

log = logging.getLogger(__name__)


# ─────────────────────────────────────────────
# Colour maps
# ─────────────────────────────────────────────

NDVI_CMAP   = mcolors.LinearSegmentedColormap.from_list(
    "ndvi", ["#8B4513", "#FFFF00", "#006400"], N=256
)
CONF_CMAP   = mcolors.LinearSegmentedColormap.from_list(
    "conf", ["#CC0000", "#FF8800", "#00AA00"], N=256
)
CLOUD_COLORS = {0: (0.9, 0.95, 1.0), 1: (0.8, 0.85, 0.9),
                2: (1.0, 1.0, 1.0),  3: (0.2, 0.2, 0.3)}

CLASS_LABELS = ["Clear", "Thin cloud", "Thick cloud", "Cloud shadow"]


# ─────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────

def tensor_to_rgb(x: torch.Tensor, gamma: float = 0.7) -> np.ndarray:
    """
    Convert 4-band (G,R,NIR,SWIR) tensor to false-colour RGB (NIR,R,G).
    Returns (H,W,3) uint8.
    """
    if x.dim() == 4:
        x = x[0]
    nir  = x[2].cpu().numpy()
    red  = x[1].cpu().numpy()
    green= x[0].cpu().numpy()

    rgb = np.stack([nir, red, green], axis=-1)
    rgb = np.clip(rgb, 0, 1) ** gamma
    return (rgb * 255).astype(np.uint8)


def apply_percentile_stretch(arr: np.ndarray, p: Tuple = (2, 98)) -> np.ndarray:
    """Percentile stretch for display."""
    lo = np.percentile(arr, p[0])
    hi = np.percentile(arr, p[1])
    return np.clip((arr - lo) / (hi - lo + 1e-8), 0, 1)


# ─────────────────────────────────────────────
# Main visualizer
# ─────────────────────────────────────────────

class SceneVisualizer:
    """Generates all visualisation panels for a reconstructed scene."""

    def __init__(self, output_dir: Path, dpi: int = 150):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.dpi = dpi

    def plot_comparison(
        self,
        scene_name:    str,
        cloudy:        torch.Tensor,      # (1,4,H,W)
        reconstructed: torch.Tensor,      # (1,4,H,W)
        cloud_mask:    torch.Tensor,      # (1,H,W) int
        confidence:    torch.Tensor,      # (1,1,H,W)
    ) -> Path:
        """
        3-panel comparison: cloudy | reconstructed | confidence.
        """
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        fig.patch.set_facecolor("#1A1A2E")

        cloudy_rgb = tensor_to_rgb(cloudy)
        recon_rgb  = tensor_to_rgb(reconstructed)
        conf_map   = confidence[0, 0].cpu().numpy()
        mask_map   = cloud_mask[0].cpu().numpy()

        # Panel 1: Cloudy input
        axes[0].imshow(cloudy_rgb)
        # Overlay cloud mask as semi-transparent hatching
        cloud_overlay = np.zeros((*mask_map.shape, 4), dtype=np.float32)
        cloud_overlay[mask_map == 2] = [1.0, 1.0, 1.0, 0.5]   # thick cloud
        cloud_overlay[mask_map == 1] = [0.8, 0.8, 0.8, 0.3]   # thin cloud
        cloud_overlay[mask_map == 3] = [0.1, 0.1, 0.3, 0.4]   # shadow
        axes[0].imshow(cloud_overlay)
        axes[0].set_title("Cloudy Input (NIR-R-G)", color="white", fontsize=11, pad=8)
        axes[0].axis("off")

        # Panel 2: Reconstructed
        axes[1].imshow(recon_rgb)
        axes[1].set_title("Reconstructed (Cloud-Free)", color="white", fontsize=11, pad=8)
        axes[1].axis("off")

        # Panel 3: Confidence map
        im = axes[2].imshow(conf_map, cmap=CONF_CMAP, vmin=0, vmax=1)
        axes[2].set_title("Reconstruction Confidence", color="white", fontsize=11, pad=8)
        axes[2].axis("off")
        cbar = fig.colorbar(im, ax=axes[2], fraction=0.046, pad=0.04)
        cbar.ax.yaxis.set_tick_params(color="white")
        plt.setp(cbar.ax.yaxis.get_ticklabels(), color="white", fontsize=8)
        cbar.set_label("Confidence", color="white", fontsize=9)

        for ax in axes:
            ax.set_facecolor("#1A1A2E")

        plt.suptitle(f"Cloud Removal — {scene_name}", color="white",
                     fontsize=13, fontweight="bold", y=1.01)
        plt.tight_layout()

        path = self.output_dir / f"{scene_name}_comparison.png"
        plt.savefig(path, dpi=self.dpi, bbox_inches="tight",
                    facecolor=fig.get_facecolor())
        plt.close(fig)
        log.info(f"  ✓ Comparison plot → {path.name}")
        return path

    def plot_indices(
        self,
        scene_name:    str,
        cloudy:        torch.Tensor,      # (1,4,H,W)
        reconstructed: torch.Tensor,      # (1,4,H,W)
    ) -> Path:
        """4-panel spectral index comparison: NDVI/NDWI for cloudy vs reconstructed."""
        from src.utils.indices import ndvi, ndwi

        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        fig.patch.set_facecolor("#1A1A2E")

        panels = [
            ("NDVI — Cloudy",        ndvi(cloudy)[0,0].cpu().numpy(),        NDVI_CMAP, -1, 1),
            ("NDVI — Reconstructed", ndvi(reconstructed)[0,0].cpu().numpy(), NDVI_CMAP, -1, 1),
            ("NDWI — Cloudy",        ndwi(cloudy)[0,0].cpu().numpy(),        "RdYlBu", -1, 1),
            ("NDWI — Reconstructed", ndwi(reconstructed)[0,0].cpu().numpy(), "RdYlBu", -1, 1),
        ]

        for ax, (title, data, cmap, vmin, vmax) in zip(axes.flat, panels):
            ax.set_facecolor("#1A1A2E")
            im = ax.imshow(data, cmap=cmap, vmin=vmin, vmax=vmax)
            ax.set_title(title, color="white", fontsize=10, pad=6)
            ax.axis("off")
            cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
            plt.setp(cbar.ax.yaxis.get_ticklabels(), color="white", fontsize=7)

        plt.suptitle(f"Spectral Index Comparison — {scene_name}",
                     color="white", fontsize=12, fontweight="bold")
        plt.tight_layout()

        path = self.output_dir / f"{scene_name}_indices.png"
        plt.savefig(path, dpi=self.dpi, bbox_inches="tight",
                    facecolor=fig.get_facecolor())
        plt.close(fig)
        log.info(f"  ✓ Index comparison → {path.name}")
        return path

    def plot_branch_weights(
        self,
        scene_name:     str,
        branch_weights: torch.Tensor,     # (1,4,H,W)
    ) -> Path:
        """Visualise per-pixel contribution of each reconstruction branch."""
        BRANCH_NAMES = ["Diffusion", "GAN", "Temporal TF", "SAR Encoder"]
        BRANCH_CMAPS = ["Oranges", "Purples", "Greens", "Blues"]

        fig, axes = plt.subplots(1, 4, figsize=(16, 4))
        fig.patch.set_facecolor("#1A1A2E")

        for i, (ax, name, cmap) in enumerate(zip(axes, BRANCH_NAMES, BRANCH_CMAPS)):
            data = branch_weights[0, i].cpu().numpy()
            im   = ax.imshow(data, cmap=cmap, vmin=0, vmax=1)
            ax.set_title(f"Branch {i+1}\n{name}", color="white", fontsize=9, pad=6)
            ax.axis("off")
            ax.set_facecolor("#1A1A2E")
            cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
            plt.setp(cbar.ax.yaxis.get_ticklabels(), color="white", fontsize=7)

        plt.suptitle(f"Adaptive Branch Weights — {scene_name}",
                     color="white", fontsize=12, fontweight="bold")
        plt.tight_layout()

        path = self.output_dir / f"{scene_name}_branch_weights.png"
        plt.savefig(path, dpi=self.dpi, bbox_inches="tight",
                    facecolor=fig.get_facecolor())
        plt.close(fig)
        log.info(f"  ✓ Branch weight map → {path.name}")
        return path

    def plot_verification_summary(
        self,
        scene_name: str,
        report:     Dict,
    ) -> Path:
        """Horizontal bar chart of verification pass scores vs thresholds."""
        passes = report.get("passes", [])
        if not passes:
            return None

        names      = [p["name"] for p in passes]
        scores     = [p["score"] for p in passes]
        thresholds = [p["threshold"] for p in passes]
        colours    = ["#00AA00" if p["passed"] else "#CC0000" for p in passes]

        fig, ax = plt.subplots(figsize=(10, 5))
        fig.patch.set_facecolor("#1A1A2E")
        ax.set_facecolor("#1A1A2E")

        y_pos = range(len(names))
        bars  = ax.barh(y_pos, scores, color=colours, height=0.5, alpha=0.85)
        for i, (thr, bar) in enumerate(zip(thresholds, bars)):
            ax.axvline(x=thr, ymin=(i / len(names)) + 0.05,
                       ymax=((i + 1) / len(names)) - 0.05,
                       color="white", linewidth=1.5, linestyle="--", alpha=0.7)

        ax.set_yticks(list(y_pos))
        ax.set_yticklabels([f"P{p['id']}: {p['name']}" for p in passes],
                            color="white", fontsize=9)
        ax.set_xlabel("Score", color="white", fontsize=10)
        ax.set_xlim(0, 1)
        ax.tick_params(axis="x", colors="white")
        ax.spines["bottom"].set_color("#444")
        ax.spines["left"].set_color("#444")
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

        overall = "PASS ✓" if report.get("overall_pass") else "FAIL ✗"
        colour  = "#00AA00" if report.get("overall_pass") else "#CC0000"
        ax.set_title(f"Verification Summary — {scene_name}  [{overall}]",
                     color=colour, fontsize=12, fontweight="bold")

        plt.tight_layout()
        path = self.output_dir / f"{scene_name}_verification.png"
        plt.savefig(path, dpi=self.dpi, bbox_inches="tight",
                    facecolor=fig.get_facecolor())
        plt.close(fig)
        log.info(f"  ✓ Verification chart → {path.name}")
        return path

    def plot_temporal_context(
        self,
        scene_name:     str,
        temporal_stack: torch.Tensor,   # (1,T,4,H,W)
        cloud_masks:    torch.Tensor,   # (1,T,H,W)
        n_show:         int = 5,
    ) -> Path:
        """Show a sample of T temporal observations with cloud coverage."""
        T    = temporal_stack.shape[1]
        show = min(n_show, T)
        idxs = np.linspace(0, T - 1, show).astype(int)

        fig, axes = plt.subplots(2, show, figsize=(show * 3.5, 7))
        fig.patch.set_facecolor("#1A1A2E")

        for col, t in enumerate(idxs):
            scene_t = temporal_stack[0, t]                          # (4,H,W)
            mask_t  = cloud_masks[0, t].cpu().numpy() if cloud_masks is not None else None
            rgb     = tensor_to_rgb(scene_t.unsqueeze(0))

            axes[0, col].set_facecolor("#1A1A2E")
            axes[0, col].imshow(rgb)
            axes[0, col].set_title(f"t={t}", color="white", fontsize=9)
            axes[0, col].axis("off")

            # Cloud coverage fraction
            if mask_t is not None:
                cloud_frac = (mask_t > 0).mean()
                axes[0, col].set_title(
                    f"t={t}  ({cloud_frac:.0%} cloud)",
                    color="white", fontsize=8
                )

            # NDVI heatmap
            from src.utils.indices import ndvi
            ndvi_t = ndvi(scene_t.unsqueeze(0))[0, 0].cpu().numpy()
            axes[1, col].set_facecolor("#1A1A2E")
            im = axes[1, col].imshow(ndvi_t, cmap=NDVI_CMAP, vmin=-1, vmax=1)
            axes[1, col].axis("off")
            if col == show - 1:
                cbar = fig.colorbar(im, ax=axes[1, col], fraction=0.046)
                plt.setp(cbar.ax.yaxis.get_ticklabels(), color="white", fontsize=7)

        axes[0, 0].set_ylabel("RGB (NIR-R-G)", color="white", fontsize=9)
        axes[1, 0].set_ylabel("NDVI",          color="white", fontsize=9)

        plt.suptitle(f"Temporal Context (±15 days) — {scene_name}",
                     color="white", fontsize=11, fontweight="bold")
        plt.tight_layout()

        path = self.output_dir / f"{scene_name}_temporal_context.png"
        plt.savefig(path, dpi=self.dpi, bbox_inches="tight",
                    facecolor=fig.get_facecolor())
        plt.close(fig)
        log.info(f"  ✓ Temporal context → {path.name}")
        return path
