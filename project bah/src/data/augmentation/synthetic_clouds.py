from pathlib import Path
import numpy as np


class SyntheticCloudGenerator:
    def __init__(self, seed: int = 42):
        self.seed = seed

    def _apply(self, clear: np.ndarray, cloud_type: str):
        rng = np.random.default_rng(self.seed)
        cloudy = clear.copy()
        mask = np.zeros(clear.shape[1:], dtype=np.uint8)

        if cloud_type == "thick":
            size = 64
            y = rng.integers(0, clear.shape[1] - size)
            x = rng.integers(0, clear.shape[2] - size)
            mask[y:y+size, x:x+size] = 2
        elif cloud_type == "thin":
            size = 32
            y = rng.integers(0, clear.shape[1] - size)
            x = rng.integers(0, clear.shape[2] - size)
            mask[y:y+size, x:x+size] = 1
        else:
            size = 48
            y = rng.integers(0, clear.shape[1] - size)
            x = rng.integers(0, clear.shape[2] - size)
            mask[y:y+size, x:x+size] = 2

        return cloudy, mask

    def augment(self, patches: list) -> list:
        """Create synthetic cloudy/clear pairs from input clear patches.

        Input: list of clear patches as numpy arrays (C, H, W)
        Output: list of Patch-like objects with a `.save(path)` method
        """
        augmented = []
        class PatchObj:
            def __init__(self, clear, cloudy, mask):
                self.clear = clear
                self.cloudy = cloudy
                self.mask = mask

            def save(self, path):
                import numpy as _np
                from pathlib import Path as _P
                _P(path).parent.mkdir(parents=True, exist_ok=True)
                _np.savez(path, clear=self.clear, cloudy=self.cloudy, mask=self.mask)

        rng = __import__("numpy").random.default_rng(self.seed)
        for clear in patches:
            # pick random cloud type
            t = rng.choice(["thin", "thick", "patchy"]) 
            cloudy, mask = self._apply(clear, t)
            augmented.append(PatchObj(clear, cloudy, mask))
        return augmented
