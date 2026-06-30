"""Radiometric normalisation utilities for data preparation."""

from pathlib import Path
import numpy as np
from src.utils.geotiff import read_geotiff


class RadiometricNormaliser:
    def __init__(self, cfg):
        self.cfg = cfg

    def process(self, path: Path) -> np.ndarray:
        """Read a GeoTIFF and apply simple radiometric normalisation.

        Returns a numpy array (C, H, W) in float32 with values in [0, 1].
        """
        arr, meta = read_geotiff(path)
        arr = arr.astype(np.float32)
        # If values appear scaled by 10000, rescale to [0,1]
        if arr.max() > 2.0:
            arr = arr / 10000.0
        return arr


__all__ = ["RadiometricNormaliser"]
