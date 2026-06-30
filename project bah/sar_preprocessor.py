"""
src/data/preprocessing/sar_preprocessor.py

SAR (Synthetic Aperture Radar) preprocessing for RISAT-1 / Sentinel-1 data.

Pipeline:
  1. Load VV + VH polarisation bands
  2. Convert DN → sigma-naught (backscatter coefficient) in dB
  3. Lee speckle filter (reduces multiplicative noise)
  4. Co-register to optical scene extent and resolution
  5. Normalise to [0, 1]
  6. Extract structural features (edge map, roughness proxy)
"""

from __future__ import annotations
import numpy as np
import torch
import torch.nn.functional as F
from pathlib import Path
from typing import Tuple, Optional
import logging

log = logging.getLogger(__name__)


class SARPreprocessor:
    """
    End-to-end SAR preprocessing for cloud-penetrating structural guidance.

    Output bands (C=4):
      0: VV  backscatter (normalised)
      1: VH  backscatter (normalised)
      2: VV/VH ratio    (surface roughness proxy)
      3: Edge magnitude  (structural boundary detector)
    """

    def __init__(self, cfg):
        self.target_resolution = 30.0     # metres — match LISS-IV
        self.speckle_window    = 7        # Lee filter window size
        self.db_min            = -30.0    # dB clip minimum
        self.db_max            =   5.0    # dB clip maximum

    # ─────────────────────────────────────────────
    # Public API
    # ─────────────────────────────────────────────

    def process(
        self,
        sar_path: str | Path,
        optical_meta: dict,
    ) -> Tuple[np.ndarray, dict]:
        """
        Full SAR preprocessing pipeline.

        Args:
            sar_path:     Path to SAR GeoTIFF (2-band: VV, VH).
            optical_meta: rasterio metadata of the reference optical scene
                          (used for co-registration).

        Returns:
            array: (4, H, W) float32 — VV, VH, ratio, edges
            meta:  updated rasterio metadata
        """
        from src.utils.geotiff import read_geotiff
        raw, sar_meta = read_geotiff(sar_path)  # (2, H_sar, W_sar)

        # Step 1: DN → dB (sigma-naught)
        db = self._to_db(raw)

        # Step 2: Lee speckle filter per band
        filtered = np.stack([
            self._lee_filter(db[0], self.speckle_window),
            self._lee_filter(db[1], self.speckle_window),
        ], axis=0)

        # Step 3: Co-register to optical grid
        coregistered = self._coregister(filtered, sar_meta, optical_meta)

        # Step 4: Normalise to [0, 1]
        vv = self._normalise_db(coregistered[0])
        vh = self._normalise_db(coregistered[1])

        # Step 5: Derived features
        ratio = self._vv_vh_ratio(coregistered[0], coregistered[1])
        edges = self._edge_magnitude(vv)

        result = np.stack([vv, vh, ratio, edges], axis=0).astype(np.float32)
        log.info(f"SAR preprocessing complete: {result.shape}")
        return result, optical_meta

    # ─────────────────────────────────────────────
    # Processing steps
    # ─────────────────────────────────────────────

    def _to_db(self, raw: np.ndarray) -> np.ndarray:
        """Convert linear DN to sigma-naught dB. Handles zeros safely."""
        linear = np.where(raw > 0, raw.astype(np.float32), 1e-10)
        db = 10.0 * np.log10(linear)
        return np.clip(db, self.db_min, self.db_max)

    def _lee_filter(self, band: np.ndarray, window: int = 7) -> np.ndarray:
        """
        Lee speckle filter — reduces multiplicative noise in SAR.
        Uses local mean and variance within a sliding window.
        """
        from scipy.ndimage import uniform_filter

        mean    = uniform_filter(band,    size=window)
        sq_mean = uniform_filter(band**2, size=window)
        variance = sq_mean - mean**2

        # Estimate noise variance from the overall image
        overall_var = np.var(band)
        if overall_var < 1e-8:
            return band

        weight  = variance / (variance + overall_var + 1e-8)
        filtered = mean + weight * (band - mean)
        return filtered.astype(np.float32)

    def _coregister(
        self,
        sar: np.ndarray,
        sar_meta: dict,
        optical_meta: dict,
    ) -> np.ndarray:
        """
        Resample SAR array to optical scene dimensions using bilinear interpolation.
        Uses rasterio's reproject for proper geospatial alignment.
        """
        import rasterio
        from rasterio.warp import reproject, Resampling

        C, H_sar, W_sar = sar.shape
        H_opt = optical_meta["height"]
        W_opt = optical_meta["width"]

        if H_sar == H_opt and W_sar == W_opt:
            return sar  # already aligned

        output = np.zeros((C, H_opt, W_opt), dtype=np.float32)
        for c in range(C):
            reproject(
                source            = sar[c],
                destination       = output[c],
                src_transform     = sar_meta["transform"],
                src_crs           = sar_meta["crs"],
                dst_transform     = optical_meta["transform"],
                dst_crs           = optical_meta["crs"],
                resampling        = Resampling.bilinear,
            )
        log.debug(f"SAR co-registered: {sar.shape} → {output.shape}")
        return output

    def _normalise_db(self, db_band: np.ndarray) -> np.ndarray:
        """Scale dB values from [db_min, db_max] → [0, 1]."""
        return ((db_band - self.db_min) / (self.db_max - self.db_min + 1e-8)
                ).clip(0, 1).astype(np.float32)

    def _vv_vh_ratio(self, vv_db: np.ndarray, vh_db: np.ndarray) -> np.ndarray:
        """
        VV/VH ratio — proxy for surface roughness and volume scattering.
        High ratio → smooth surface (water, roads).
        Low ratio  → rough/vegetated surface.
        """
        ratio = vv_db - vh_db   # dB subtraction = linear division
        # Typical range: -10 to +15 dB → normalise
        return ((ratio + 10.0) / 25.0).clip(0, 1).astype(np.float32)

    def _edge_magnitude(self, band: np.ndarray) -> np.ndarray:
        """
        Sobel edge magnitude — highlights structural boundaries
        (roads, field edges, building outlines) visible in SAR.
        """
        t = torch.from_numpy(band).unsqueeze(0).unsqueeze(0)  # (1,1,H,W)

        sobel_x = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]],
                                dtype=torch.float32).view(1, 1, 3, 3)
        sobel_y = torch.tensor([[-1,-2,-1], [ 0, 0, 0], [ 1, 2, 1]],
                                dtype=torch.float32).view(1, 1, 3, 3)

        gx = F.conv2d(t, sobel_x, padding=1)
        gy = F.conv2d(t, sobel_y, padding=1)
        mag = torch.sqrt(gx**2 + gy**2).squeeze().numpy()

        # Normalise edge magnitude
        if mag.max() > 0:
            mag = (mag / mag.max()).astype(np.float32)
        return mag
