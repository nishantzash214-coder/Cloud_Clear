"""Compatibility shim re-exporting geotiff utilities from the project root.

Some scripts import `src.utils.geotiff`, while full implementations live
in `geotiff.py` at the repository root. Re-export commonly used helpers.
"""

from pathlib import Path
import sys

# Ensure the repository root is on sys.path so root-level modules like
# `geotiff.py` can be imported from packages under src/.
ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

try:
    from geotiff import (
        read_geotiff,
        write_geotiff,
        tile_image,
        merge_tiles,
        merge_geotiffs,
    )
except Exception:
    # Minimal fallbacks if the top-level module isn't available
    from pathlib import Path
    import numpy as np

    def write_geotiff(path, array, meta, nodata=-9999.0, compress='lzw'):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        return path

    def read_geotiff(path, bands=None, as_tensor=False):
        raise ImportError("read_geotiff is not available in this environment")

    def tile_image(array, tile_size=256, overlap=32):
        raise ImportError("tile_image is not available in this environment")

    def merge_tiles(tiles, offsets, output_shape, overlap=32):
        raise ImportError("merge_tiles is not available in this environment")

    def merge_geotiffs(paths, output_path):
        raise ImportError("merge_geotiffs is not available in this environment")
