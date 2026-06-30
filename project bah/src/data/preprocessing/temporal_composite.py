"""Compatibility wrapper exposing `build_temporal_composite`.

Re-exports the function defined in `temporal_analysis.py`.
"""

from temporal_analysis import build_temporal_composite

__all__ = ["build_temporal_composite"]
