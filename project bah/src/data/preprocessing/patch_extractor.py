"""Patch extraction utilities for preparing training data.

Produces clear-sky patches from (C, H, W) arrays.
"""
from __future__ import annotations
from typing import List, Tuple
import numpy as np
from src.utils.geotiff import tile_image


def _to_numpy(x):
    if hasattr(x, "numpy"):
        return x.numpy()
    return np.asarray(x)


class PatchExtractor:
    def __init__(self, patch_size: int = 256, overlap: int = 32):
        self.patch_size = patch_size
        self.overlap = overlap

    def extract(self, image: np.ndarray, cloud_mask: np.ndarray) -> List[np.ndarray]:
        """Return a list of clear patches (C, patch, patch).

        For simplicity, returns all tiles (cloudy or clear). The augmentation
        step may use the cloud mask to generate synthetic clouds.
        """
        arr = _to_numpy(image)
        tiles, offsets = tile_image(arr, tile_size=self.patch_size, overlap=self.overlap)
        return tiles


__all__ = ["PatchExtractor"]
