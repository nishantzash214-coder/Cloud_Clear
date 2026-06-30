"""Wrapper module to expose the top-level `earth_engine` SceneDownloader

This file allows existing imports like:
    from src.data.downloaders.earth_engine import SceneDownloader

to continue working while the real implementation lives in `earth_engine.py`.
"""

from earth_engine import SceneDownloader  # re-export

__all__ = ["SceneDownloader"]
