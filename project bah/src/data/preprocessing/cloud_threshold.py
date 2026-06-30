"""Simple threshold-based cloud mask used during data preparation."""

import numpy as np


class ThresholdCloudMask:
    def __init__(self, cfg=None, threshold: float = 0.2):
        # Try to read threshold from config if available
        try:
            self.threshold = float(cfg.data.cloud_threshold)
        except Exception:
            self.threshold = threshold

    def predict(self, image: np.ndarray) -> np.ndarray:
        """Return a basic cloud mask (H, W) with 0=clear, 1=cloud.

        Uses a simple brightness threshold on the green+red+nir mean.
        """
        # image: (C, H, W)
        if image is None:
            return np.zeros((0, 0), dtype=np.uint8)
        mean_band = np.nanmean(image[[0, 1, 2], :, :], axis=0)
        mask = (mean_band > self.threshold).astype(np.uint8)
        return mask


__all__ = ["ThresholdCloudMask"]
