"""
src/utils/geotiff.py
GeoTIFF read / write / tile / merge utilities using rasterio.
All arrays are returned as (C, H, W) float32 numpy or torch tensors.
"""

from __future__ import annotations
import numpy as np
import torch
from pathlib import Path
from typing import Optional, Tuple, List
import rasterio
from rasterio.windows import Window
from rasterio.transform import from_bounds
from rasterio.crs import CRS
from rasterio.merge import merge as rio_merge
import logging

log = logging.getLogger(__name__)


# ─────────────────────────────────────────────
# Reading
# ─────────────────────────────────────────────

def read_geotiff(
    path: str | Path,
    bands: Optional[List[int]] = None,
    as_tensor: bool = False,
) -> Tuple[np.ndarray, dict]:
    """
    Read a GeoTIFF into a (C, H, W) float32 array.

    Args:
        path:       Path to GeoTIFF file.
        bands:      1-based band indices to read (default: all bands).
        as_tensor:  If True, return torch.Tensor instead of numpy array.

    Returns:
        array:  (C, H, W) float32
        meta:   rasterio metadata dict (crs, transform, nodata, etc.)
    """
    with rasterio.open(path) as src:
        meta = src.meta.copy()
        idx = bands if bands else list(range(1, src.count + 1))
        data = src.read(idx).astype(np.float32)
        # Replace nodata with NaN
        if src.nodata is not None:
            data[data == src.nodata] = np.nan
    if as_tensor:
        return torch.from_numpy(data), meta
    return data, meta


# ─────────────────────────────────────────────
# Writing
# ─────────────────────────────────────────────

def write_geotiff(
    path: str | Path,
    array: np.ndarray | torch.Tensor,
    meta: dict,
    nodata: float = -9999.0,
    compress: str = "lzw",
) -> None:
    """
    Write a (C, H, W) array to a GeoTIFF.

    Args:
        path:     Output path.
        array:    (C, H, W) array — numpy or torch.
        meta:     rasterio metadata dict with crs and transform.
        nodata:   Value to mark missing pixels.
        compress: Compression algorithm (lzw | deflate | none).
    """
    if isinstance(array, torch.Tensor):
        array = array.cpu().numpy()

    array = array.astype(np.float32)
    array = np.nan_to_num(array, nan=nodata)

    if array.ndim == 2:
        array = array[np.newaxis, ...]
    elif array.ndim != 3:
        raise ValueError(f"GeoTIFF input must be 2D or 3D, got shape {array.shape}")

    out_meta = meta.copy()
    out_meta.update({
        "driver":   "GTiff",
        "dtype":    "float32",
        "count":    array.shape[0],
        "height":   array.shape[1],
        "width":    array.shape[2],
        "nodata":   nodata,
        "compress": compress,
    })

    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with rasterio.open(path, "w", **out_meta) as dst:
        dst.write(array)
    log.info(f"Saved GeoTIFF → {path}")


# ─────────────────────────────────────────────
# Tiling
# ─────────────────────────────────────────────

def tile_image(
    array: np.ndarray,
    tile_size: int = 256,
    overlap: int = 32,
) -> Tuple[List[np.ndarray], List[Tuple[int, int]]]:
    """
    Split a (C, H, W) array into overlapping tiles for inference.

    Returns:
        tiles:    list of (C, tile_size, tile_size) patches
        offsets:  list of (row, col) top-left pixel offsets
    """
    _, H, W = array.shape
    stride = tile_size - overlap
    tiles, offsets = [], []

    for r in range(0, H - overlap, stride):
        for c in range(0, W - overlap, stride):
            r_end = min(r + tile_size, H)
            c_end = min(c + tile_size, W)
            r_start = max(0, r_end - tile_size)
            c_start = max(0, c_end - tile_size)
            tiles.append(array[:, r_start:r_end, c_start:c_end])
            offsets.append((r_start, c_start))

    return tiles, offsets


def merge_tiles(
    tiles: List[np.ndarray],
    offsets: List[Tuple[int, int]],
    output_shape: Tuple[int, int, int],
    overlap: int = 32,
) -> np.ndarray:
    """
    Blend overlapping tiles back into a full (C, H, W) array.
    Uses linear blending in the overlap regions.
    """
    C, H, W = output_shape
    output  = np.zeros((C, H, W), dtype=np.float32)
    weights = np.zeros((1, H, W), dtype=np.float32)

    tile_size = tiles[0].shape[1]
    half = overlap // 2

    for tile, (r, c) in zip(tiles, offsets):
        th, tw = tile.shape[1], tile.shape[2]
        output[:, r:r+th, c:c+tw] += tile
        weights[0, r:r+th, c:c+tw] += 1.0

    weights = np.clip(weights, 1, None)
    return output / weights


# ─────────────────────────────────────────────
# Multi-file merge
# ─────────────────────────────────────────────

def merge_geotiffs(paths: List[str | Path], output_path: str | Path) -> None:
    """Mosaic multiple GeoTIFFs into a single output file."""
    datasets = [rasterio.open(p) for p in paths]
    mosaic, transform = rio_merge(datasets)
    meta = datasets[0].meta.copy()
    meta.update({
        "height":    mosaic.shape[1],
        "width":     mosaic.shape[2],
        "transform": transform,
    })
    for ds in datasets:
        ds.close()
    write_geotiff(output_path, mosaic, meta)
