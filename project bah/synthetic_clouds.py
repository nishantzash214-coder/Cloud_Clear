"""
src/data/augmentation/synthetic_clouds.py

Synthetic Cloud Generator for training data creation (Layer training strategy).

Applied to cloud-free images to generate (cloudy_input, clear_target) pairs
with perfect ground truth — eliminating the need for paired cloudy/clear scenes.

Generates three types:
  1. Thick cloud   — white/grey, opaque, spectrally bright
  2. Thin cloud    — semi-transparent, transmittance-weighted blending
  3. Cloud shadow  — dark, slightly bluish, spatially offset from cloud

Physics motivation:
  - Thin clouds are modelled as a transmittance layer:
      I_observed = tau * I_surface + (1 - tau) * I_cloud
    where tau ∈ (0.3, 0.8) — the key realism challenge
  - Thick clouds use full replacement with realistic cloud spectra
  - Shadows apply a multiplicative darkening with atmospheric scattering offset
"""

from __future__ import annotations
import numpy as np
import torch
import torch.nn.functional as F
from typing import Tuple, Optional, List
from dataclasses import dataclass
import logging

log = logging.getLogger(__name__)


@dataclass
class AugmentedPatch:
    """A training pair: cloudy input + clear target + cloud mask."""
    cloudy:     np.ndarray    # (C, H, W) — model input
    clear:      np.ndarray    # (C, H, W) — reconstruction target
    cloud_mask: np.ndarray    # (H, W)    — 0=clear,1=thin,2=thick,3=shadow
    metadata:   dict

    def save(self, path):
        np.savez_compressed(
            path,
            cloudy     = self.cloudy,
            clear      = self.clear,
            cloud_mask = self.cloud_mask,
            **self.metadata,
        )

    @classmethod
    def load(cls, path) -> "AugmentedPatch":
        d = np.load(path)
        return cls(
            cloudy     = d["cloudy"],
            clear      = d["clear"],
            cloud_mask = d["cloud_mask"],
            metadata   = {k: d[k] for k in d.files
                          if k not in ("cloudy", "clear", "cloud_mask")},
        )


class SyntheticCloudGenerator:
    """
    Generates physically-motivated synthetic clouds on clear satellite patches.

    Each call to augment() returns multiple augmented copies per input patch.
    """

    def __init__(self, cfg=None, copies_per_patch: int = 3, seed: int = 42):
        self.copies_per_patch = copies_per_patch
        self.rng = np.random.default_rng(seed)

        # Cloud spectral signatures (normalised [0,1] for 4 bands: G,R,NIR,SWIR)
        # Typical optically thick cloud reflectance (bright, spectrally flat)
        self.thick_cloud_spectrum = np.array([0.82, 0.80, 0.78, 0.65],
                                             dtype=np.float32)
        self.thin_cloud_spectrum  = np.array([0.55, 0.52, 0.50, 0.40],
                                             dtype=np.float32)

    # ─────────────────────────────────────────────
    # Public API
    # ─────────────────────────────────────────────

    def augment(self, patches: list) -> List[AugmentedPatch]:
        """
        Generate augmented (cloudy, clear, mask) pairs.
        patches: list of (C, H, W) clear numpy arrays.
        """
        results = []
        for patch in patches:
            for _ in range(self.copies_per_patch):
                cloud_type = self.rng.choice(
                    ["thick", "thin", "mixed"],
                    p=[0.35, 0.35, 0.30],
                )
                cloudy, mask = self._apply(patch, cloud_type)
                results.append(AugmentedPatch(
                    cloudy     = cloudy,
                    clear      = patch,
                    cloud_mask = mask,
                    metadata   = {"cloud_type": cloud_type},
                ))
        log.debug(f"Generated {len(results)} augmented patches from {len(patches)} inputs")
        return results

    def augment_tensor(
        self,
        x: torch.Tensor,   # (B, C, H, W) clear batch
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        GPU-compatible batch augmentation.
        Returns (cloudy, mask) tensors of same shape as input.
        """
        B, C, H, W = x.shape
        cloudy_list, mask_list = [], []

        for b in range(B):
            arr    = x[b].cpu().numpy()
            cloud_type = self.rng.choice(["thick", "thin", "mixed"])
            cloudy, mask = self._apply(arr, cloud_type)
            cloudy_list.append(torch.from_numpy(cloudy))
            mask_list.append(torch.from_numpy(mask.astype(np.int64)))

        return torch.stack(cloudy_list).to(x.device), \
               torch.stack(mask_list).to(x.device)

    # ─────────────────────────────────────────────
    # Cloud type dispatch
    # ─────────────────────────────────────────────

    def _apply(
        self,
        clear: np.ndarray,   # (C, H, W)
        cloud_type: str,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Return (cloudy array, cloud mask)."""
        if cloud_type == "thick":
            return self._thick_cloud(clear)
        elif cloud_type == "thin":
            return self._thin_cloud(clear)
        elif cloud_type == "mixed":
            return self._mixed_cloud(clear)
        else:
            raise ValueError(f"Unknown cloud type: {cloud_type}")

    # ─────────────────────────────────────────────
    # Thick cloud (opaque)
    # ─────────────────────────────────────────────

    def _thick_cloud(
        self, clear: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        C, H, W = clear.shape
        mask_2d = self._generate_cloud_shape(H, W, density=0.40)
        mask_class = np.where(mask_2d > 0.5, 2, 0).astype(np.int8)

        # Also add shadow (spatially offset from cloud)
        shadow = self._generate_shadow(mask_2d, H, W)
        mask_class = np.where((shadow > 0) & (mask_class == 0), 3, mask_class)

        cloudy = clear.copy()
        cloud_region = mask_2d > 0.5

        # Replace with cloud spectrum + per-pixel brightness variation
        noise = self.rng.normal(0, 0.03, (C, H, W)).astype(np.float32)
        for c in range(C):
            cloud_val = self.thick_cloud_spectrum[c] + noise[c]
            cloudy[c] = np.where(cloud_region,
                                  np.clip(cloud_val, 0, 1),
                                  cloudy[c])

        # Apply shadow darkening
        shadow_region = shadow > 0
        for c in range(C):
            darkening = self.rng.uniform(0.4, 0.65)
            scatter   = self.rng.uniform(0.00, 0.02)
            cloudy[c] = np.where(shadow_region,
                                  np.clip(cloudy[c] * darkening + scatter, 0, 1),
                                  cloudy[c])

        return cloudy.astype(np.float32), mask_class

    # ─────────────────────────────────────────────
    # Thin cloud (semi-transparent)
    # ─────────────────────────────────────────────

    def _thin_cloud(
        self, clear: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Physically correct thin cloud model:
          I_obs = tau * I_surface + (1 - tau) * I_cloud

        tau (transmittance) varies spatially within [0.3, 0.8].
        Lower tau = denser cloud = less surface visible.
        """
        C, H, W = clear.shape
        mask_2d  = self._generate_cloud_shape(H, W, density=0.45, smoothness=0.6)
        mask_class = np.where(mask_2d > 0.2, 1, 0).astype(np.int8)

        # Spatially varying transmittance
        tau_center = self.rng.uniform(0.3, 0.7)
        tau = tau_center * mask_2d   # thin at edges, dense at center
        tau = np.clip(tau, 0.15, 0.85)

        cloudy = clear.copy()
        for c in range(C):
            cloud_signal = (self.thin_cloud_spectrum[c] +
                           self.rng.normal(0, 0.02, (H, W))).clip(0, 1)
            cloudy[c] = (tau * clear[c] +
                         (1.0 - tau) * cloud_signal).clip(0, 1).astype(np.float32)

        return cloudy, mask_class

    # ─────────────────────────────────────────────
    # Mixed cloud (realistic combination)
    # ─────────────────────────────────────────────

    def _mixed_cloud(
        self, clear: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Realistic scene: thick cloud core + thin halo + shadow.
        Most common in Indian monsoon imagery.
        """
        C, H, W = clear.shape
        thick_mask = self._generate_cloud_shape(H, W, density=0.25, smoothness=0.4)
        thin_halo  = self._generate_cloud_shape(H, W, density=0.45, smoothness=0.7)
        thin_halo  = np.where(thick_mask > 0.5, 0, thin_halo)

        mask_class = np.zeros((H, W), dtype=np.int8)
        mask_class[thin_halo  > 0.25] = 1
        mask_class[thick_mask > 0.50] = 2

        shadow = self._generate_shadow(thick_mask, H, W)
        mask_class = np.where((shadow > 0) & (mask_class == 0), 3, mask_class)

        cloudy = clear.copy()

        # Thin halo
        tau = np.clip(0.5 * thin_halo, 0.15, 0.85)
        for c in range(C):
            cloud_signal = self.thin_cloud_spectrum[c] + self.rng.normal(0, 0.02, (H, W))
            cloudy[c] = np.where(
                thin_halo > 0.25,
                (tau * clear[c] + (1 - tau) * cloud_signal).clip(0, 1),
                cloudy[c]
            ).astype(np.float32)

        # Thick core (overwrites thin halo)
        noise = self.rng.normal(0, 0.03, (C, H, W)).astype(np.float32)
        for c in range(C):
            cloud_val = self.thick_cloud_spectrum[c] + noise[c]
            cloudy[c] = np.where(thick_mask > 0.5,
                                  cloud_val.clip(0, 1),
                                  cloudy[c])

        # Shadow
        for c in range(C):
            cloudy[c] = np.where(shadow > 0,
                                  (cloudy[c] * 0.5).clip(0, 1),
                                  cloudy[c])

        return cloudy.astype(np.float32), mask_class

    # ─────────────────────────────────────────────
    # Shape generators
    # ─────────────────────────────────────────────

    def _generate_cloud_shape(
        self,
        H: int, W: int,
        density: float = 0.4,
        smoothness: float = 0.5,
    ) -> np.ndarray:
        """
        Generate a random cloud-shaped binary mask using smoothed Perlin-like noise.
        Returns (H, W) float32 in [0, 1].
        """
        # Start with random noise at multiple scales
        shape = np.zeros((H, W), dtype=np.float32)
        for scale in [4, 8, 16, 32]:
            coarse = self.rng.random((H // scale + 1, W // scale + 1)).astype(np.float32)
            # Upsample using torch for quality interpolation
            t = torch.from_numpy(coarse).unsqueeze(0).unsqueeze(0)
            up = F.interpolate(t, size=(H, W), mode="bilinear",
                               align_corners=False).squeeze().numpy()
            shape += up * (1.0 / scale)

        # Normalise and threshold
        shape = (shape - shape.min()) / (shape.max() - shape.min() + 1e-8)

        # Smooth edges for realistic cloud boundaries
        sigma = int(smoothness * min(H, W) * 0.05) + 1
        from scipy.ndimage import gaussian_filter
        shape = gaussian_filter(shape, sigma=sigma)
        shape = (shape - shape.min()) / (shape.max() - shape.min() + 1e-8)

        # Random placement: move cloud centroid
        threshold = 1.0 - density
        return np.clip((shape - threshold) / (1.0 - threshold + 1e-8), 0, 1)

    def _generate_shadow(
        self,
        cloud_mask: np.ndarray,  # (H, W) float
        H: int, W: int,
    ) -> np.ndarray:
        """
        Generate cloud shadow offset from the cloud mask.
        Shadow direction and distance simulates solar illumination angle.
        Typical offset: 5–25% of image size in SE direction.
        """
        # Random solar geometry (offset in pixels)
        dx = int(self.rng.uniform(0.05, 0.20) * W)
        dy = int(self.rng.uniform(0.05, 0.15) * H)

        shadow = np.zeros((H, W), dtype=np.float32)
        thick  = (cloud_mask > 0.5).astype(np.float32)

        # Shift cloud footprint to get shadow location
        y_src_max = max(0, H - dy)
        x_src_max = max(0, W - dx)
        shadow[dy:, dx:] = thick[:y_src_max, :x_src_max]

        # Soften shadow edges
        from scipy.ndimage import gaussian_filter
        shadow = gaussian_filter(shadow, sigma=3)
        return np.clip(shadow, 0, 1)
