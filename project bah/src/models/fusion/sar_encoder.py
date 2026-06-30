"""Compatibility wrapper for SAR encoder.

Re-exports symbols from top-level `sar_encoder.py` so imports like
`from src.models.fusion.sar_encoder import build_sar_encoder` work.
"""

from sar_encoder import SARFusionEncoder, ZeroSAREncoder, build_sar_encoder

__all__ = ["SARFusionEncoder", "ZeroSAREncoder", "build_sar_encoder"]
