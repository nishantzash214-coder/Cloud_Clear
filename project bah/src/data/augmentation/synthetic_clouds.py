"""Compatibility wrapper for synthetic cloud generation.

Provides: `from src.data.augmentation.synthetic_clouds import SyntheticCloudGenerator, AugmentedPatch`
"""
from pathlib import Path
import sys

# Ensure root is in sys.path
ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from synthetic_clouds import SyntheticCloudGenerator, AugmentedPatch

__all__ = ["SyntheticCloudGenerator", "AugmentedPatch"]
