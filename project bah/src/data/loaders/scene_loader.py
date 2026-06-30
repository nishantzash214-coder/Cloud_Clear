"""SceneLoader implementation for inference.

Loads optical, SAR, and temporal GeoTIFF data into torch tensors and
returns the metadata needed by the pipeline.
"""
from __future__ import annotations
from types import SimpleNamespace
from pathlib import Path
from typing import Optional

import torch
from src.utils.geotiff import read_geotiff


class SceneLoader:
    def __init__(self, cfg):
        self.cfg = cfg

    def load(
        self,
        optical_path: str,
        sar_path: Optional[str] = None,
        temporal_path: Optional[str] = None,
    ):
        """Load scene assets and return a runtime scene object."""
        optical_arr, meta = read_geotiff(optical_path)
        optical_arr = optical_arr.astype("float32")
        if optical_arr.max() > 2.0:
            optical_arr = optical_arr / 10000.0

        optical = torch.from_numpy(optical_arr).float()
        if optical.ndim == 3:
            optical = optical.unsqueeze(0)
        if optical.ndim != 4:
            raise ValueError(f"Unexpected optical tensor shape: {optical.shape}")

        sar = None
        if sar_path is not None:
            sar_arr, _ = read_geotiff(sar_path)
            sar_arr = sar_arr.astype("float32")
            if sar_arr.max() > 2.0:
                sar_arr = sar_arr / 10000.0
            sar = torch.from_numpy(sar_arr).float()
            if sar.ndim == 3:
                sar = sar.unsqueeze(0)
            if sar.ndim != 4:
                raise ValueError(f"Unexpected SAR tensor shape: {sar.shape}")

        temporal_stack = None
        if temporal_path is not None:
            temp_arr, _ = read_geotiff(temporal_path)
            temp_arr = temp_arr.astype("float32")
            if temp_arr.max() > 2.0:
                temp_arr = temp_arr / 10000.0
            temp = torch.from_numpy(temp_arr).float()
            if temp.ndim == 3:
                band_count = optical.shape[1]
                if temp.shape[0] % band_count != 0:
                    raise ValueError(
                        f"Temporal GeoTIFF band count {temp.shape[0]} is not a multiple of optical bands {band_count}"
                    )
                t = temp.shape[0] // band_count
                temp = temp.reshape(t, band_count, temp.shape[1], temp.shape[2])
            elif temp.ndim == 4 and temp.shape[0] == 1 and temp.shape[1] == optical.shape[1]:
                temp = temp.squeeze(0)
            if temp.ndim != 4:
                raise ValueError(f"Unexpected temporal tensor shape: {temp.shape}")
            temporal_stack = temp

        scene = SimpleNamespace()
        scene.optical = optical
        scene.sar = sar
        scene.temporal_stack = temporal_stack
        scene.meta = meta
        scene.meta["source_paths"] = {
            "optical": str(optical_path),
            "sar": str(sar_path) if sar_path is not None else None,
            "temporal": str(temporal_path) if temporal_path is not None else None,
        }
        return scene


__all__ = ["SceneLoader"]
