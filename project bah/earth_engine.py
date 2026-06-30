"""
src/data/downloaders/earth_engine.py

Automated data downloader for:
  - Sentinel-2 MSI (optical — 4 bands: B3, B4, B8, B11)
  - Sentinel-1 SAR (VV + VH backscatter)
  - Temporal stacks (±15 days around a target date)

Usage:
    python scripts/download_data.py \
        --aoi     "77.5,28.5,78.5,29.5"  \   # lon_min,lat_min,lon_max,lat_max
        --date    "2024-08-15"             \   # target date
        --output  data/raw/                \
        --cloud-threshold 20               \   # max cloud % for optical
        --temporal-days   15                   # ± days for temporal stack

Authentication:
    earthengine authenticate   (one-time)
    or set GOOGLE_APPLICATION_CREDENTIALS env var for service account
"""

from __future__ import annotations
import os
import time
import logging
from datetime import datetime, timedelta
from pathlib import Path
from typing import List, Optional, Tuple

log = logging.getLogger(__name__)


# ─────────────────────────────────────────────
# Initialise Earth Engine
# ─────────────────────────────────────────────

def init_ee(project: Optional[str] = None):
    """Initialise Earth Engine. Call once before any EE operations."""
    import ee
    try:
        if project:
            ee.Initialize(project=project)
        else:
            ee.Initialize()
        log.info("Earth Engine initialised successfully")
    except Exception as e:
        log.error(f"EE initialisation failed: {e}")
        log.error("Run: earthengine authenticate")
        raise


# ─────────────────────────────────────────────
# AOI helper
# ─────────────────────────────────────────────

def bbox_to_ee_geometry(bbox: str):
    """Convert 'lon_min,lat_min,lon_max,lat_max' string to ee.Geometry.Rectangle."""
    import ee
    coords = [float(x) for x in bbox.split(",")]
    return ee.Geometry.Rectangle(coords)


# ─────────────────────────────────────────────
# Sentinel-2 downloader
# ─────────────────────────────────────────────

class Sentinel2Downloader:
    """
    Downloads Sentinel-2 MSI Level-2A (surface reflectance) imagery.

    Bands exported:
      B3  (Green)  → band 0
      B4  (Red)    → band 1
      B8  (NIR)    → band 2
      B11 (SWIR)   → band 3

    All bands resampled to 10 m resolution.
    Values scaled: reflectance × 10000 → divide by 10000 after download.
    """

    BANDS      = ["B3", "B4", "B8", "B11"]
    BAND_NAMES = ["green", "red", "nir", "swir"]
    SCALE      = 10   # metres (native S2 resolution)
    COLLECTION = "COPERNICUS/S2_SR_HARMONIZED"

    def __init__(self, cloud_threshold: int = 20, scale: int = 10):
        self.cloud_threshold = cloud_threshold
        self.scale           = scale

    def get_collection(
        self,
        aoi:        object,    # ee.Geometry
        start_date: str,       # "YYYY-MM-DD"
        end_date:   str,
    ):
        """Return filtered + cloud-masked Sentinel-2 collection."""
        import ee

        collection = (
            ee.ImageCollection(self.COLLECTION)
            .filterBounds(aoi)
            .filterDate(start_date, end_date)
            .filter(ee.Filter.lt("CLOUDY_PIXEL_PERCENTAGE", self.cloud_threshold))
            .select(self.BANDS, self.BAND_NAMES)
            .map(self._mask_clouds)
            .map(self._scale_reflectance)
        )
        log.info(f"S2 collection: {start_date} → {end_date} | "
                 f"cloud < {self.cloud_threshold}% | "
                 f"count = {collection.size().getInfo()}")
        return collection

    def download_scene(
        self,
        aoi:        object,
        date:       str,
        output_dir: Path,
        scene_name: str,
        temporal_days: int = 0,
    ) -> Optional[Path]:
        """
        Download the best (lowest cloud) scene around target date.
        If temporal_days > 0, download the full ±temporal_days stack.
        """
        import ee
        from datetime import datetime, timedelta

        dt    = datetime.strptime(date, "%Y-%m-%d")
        start = (dt - timedelta(days=temporal_days)).strftime("%Y-%m-%d")
        end   = (dt + timedelta(days=temporal_days + 1)).strftime("%Y-%m-%d")

        collection = self.get_collection(aoi, start, end)
        size       = collection.size().getInfo()

        if size == 0:
            log.warning(f"No S2 scenes found for {date} ± {temporal_days} days")
            return None

        if temporal_days == 0:
            # Single best scene (lowest cloud cover)
            image     = collection.sort("CLOUDY_PIXEL_PERCENTAGE").first()
            out_path  = output_dir / "optical" / f"{scene_name}.tif"
            out_path.parent.mkdir(parents=True, exist_ok=True)
            self._export_image(image, aoi, str(out_path))
            return out_path
        else:
            # Full temporal stack — export each scene
            stack_dir = output_dir / "temporal" / scene_name
            stack_dir.mkdir(parents=True, exist_ok=True)
            images    = collection.toList(collection.size())
            n         = size

            for i in range(n):
                img  = ee.Image(images.get(i))
                date_str = img.date().format("YYYY-MM-dd").getInfo()
                out  = stack_dir / f"{scene_name}_t{i:02d}_{date_str}.tif"
                self._export_image(img, aoi, str(out))
                log.info(f"  Downloaded temporal scene {i+1}/{n}: {date_str}")

            log.info(f"Temporal stack → {stack_dir} ({n} scenes)")
            return stack_dir

    def download_composite(
        self,
        aoi:        object,
        year:       int,
        month:      int,
        output_dir: Path,
        scene_name: str,
    ) -> Path:
        """
        Download a cloud-free monthly median composite for the archive.
        Used to build the historical cloud-free reference library.
        """
        import ee
        import calendar

        _, last_day = calendar.monthrange(year, month)
        start = f"{year}-{month:02d}-01"
        end   = f"{year}-{month:02d}-{last_day}"

        collection = self.get_collection(aoi, start, end)
        composite  = collection.median().clip(aoi)

        out_path = output_dir / "archive" / f"{scene_name}_{year}_{month:02d}_composite.tif"
        out_path.parent.mkdir(parents=True, exist_ok=True)
        self._export_image(composite, aoi, str(out_path))
        return out_path

    # ─────────────────────────────────────────
    # Image processing
    # ─────────────────────────────────────────

    @staticmethod
    def _mask_clouds(image):
        """Apply S2 cloud mask using the QA60 bitmask band."""
        import ee
        qa        = image.select("QA60")
        cloud_bit = 1 << 10
        cirrus_bit= 1 << 11
        mask      = (
            qa.bitwiseAnd(cloud_bit ).eq(0)
             .And(qa.bitwiseAnd(cirrus_bit).eq(0))
        )
        return image.updateMask(mask)

    @staticmethod
    def _scale_reflectance(image):
        """Scale S2 reflectance from [0, 10000] → [0, 1]."""
        return image.divide(10000.0).copyProperties(image, ["system:time_start"])

    def _export_image(self, image, aoi, out_path: str):
        """
        Export an EE image to a local GeoTIFF via ee.data.getPixels.
        For large areas, use Drive export instead.
        """
        import ee
        import numpy as np
        import rasterio
        from rasterio.transform import from_bounds

        # Get image info for dimensions
        info      = image.getInfo()
        proj      = image.projection()
        crs       = proj.getInfo()["crs"]

        # Download as numpy array
        region    = aoi.bounds().getInfo()["coordinates"][0]
        lon_vals  = [p[0] for p in region]
        lat_vals  = [p[1] for p in region]
        bbox      = [min(lon_vals), min(lat_vals), max(lon_vals), max(lat_vals)]

        data = image.sampleRectangle(
            region         = aoi,
            defaultValue   = 0,
        ).getInfo()

        # Extract band arrays
        arrays = []
        for band in self.BAND_NAMES:
            if band in data["properties"]:
                arr = np.array(data["properties"][band], dtype=np.float32)
                arrays.append(arr)

        if not arrays:
            log.warning(f"No data extracted for {out_path}")
            return

        stack     = np.stack(arrays, axis=0)   # (C, H, W)
        H, W      = stack.shape[1], stack.shape[2]
        transform = from_bounds(*bbox, W, H)

        Path(out_path).parent.mkdir(parents=True, exist_ok=True)
        with rasterio.open(
            out_path, "w",
            driver    = "GTiff",
            dtype     = "float32",
            count     = len(arrays),
            height    = H,
            width     = W,
            crs       = "EPSG:4326",
            transform = transform,
            compress  = "lzw",
        ) as dst:
            dst.write(stack)

        log.info(f"  Saved: {Path(out_path).name}  ({H}×{W} px, {len(arrays)} bands)")


# ─────────────────────────────────────────────
# Sentinel-1 SAR downloader
# ─────────────────────────────────────────────

class Sentinel1Downloader:
    """
    Downloads Sentinel-1 SAR GRD (Ground Range Detected) imagery.

    Bands exported:
      VV  → backscatter in dB
      VH  → backscatter in dB

    Uses IW (Interferometric Wide) swath mode.
    Speckle filtering is applied server-side via GEE.
    """

    COLLECTION = "COPERNICUS/S1_GRD"
    BANDS      = ["VV", "VH"]

    def __init__(self, scale: int = 10):
        self.scale = scale

    def get_collection(self, aoi, start_date: str, end_date: str):
        """Return filtered Sentinel-1 IW collection."""
        import ee

        collection = (
            ee.ImageCollection(self.COLLECTION)
            .filterBounds(aoi)
            .filterDate(start_date, end_date)
            .filter(ee.Filter.listContains("transmitterReceiverPolarisation", "VV"))
            .filter(ee.Filter.listContains("transmitterReceiverPolarisation", "VH"))
            .filter(ee.Filter.eq("instrumentMode", "IW"))
            .select(self.BANDS)
            .map(self._apply_speckle_filter)
        )
        log.info(f"S1 collection count: {collection.size().getInfo()}")
        return collection

    def download_scene(
        self,
        aoi:        object,
        date:       str,
        output_dir: Path,
        scene_name: str,
        window_days: int = 6,
    ) -> Optional[Path]:
        """
        Download SAR scene closest to target date (within ±window_days).
        S1 revisit is ~6 days so window_days=6 ensures coverage.
        """
        import ee
        from datetime import datetime, timedelta

        dt    = datetime.strptime(date, "%Y-%m-%d")
        start = (dt - timedelta(days=window_days)).strftime("%Y-%m-%d")
        end   = (dt + timedelta(days=window_days + 1)).strftime("%Y-%m-%d")

        collection = self.get_collection(aoi, start, end)
        size       = collection.size().getInfo()

        if size == 0:
            log.warning(f"No S1 scenes for {date} ± {window_days} days")
            return None

        # Use median composite of available scenes (reduces speckle further)
        composite = collection.median().clip(aoi)
        out_path  = output_dir / "sar" / f"{scene_name}_sar.tif"
        out_path.parent.mkdir(parents=True, exist_ok=True)

        self._export_sar(composite, aoi, str(out_path))
        return out_path

    @staticmethod
    def _apply_speckle_filter(image):
        """
        Boxcar (mean) speckle filter applied server-side on GEE.
        A 3×3 kernel reduces speckle noise while preserving edges.
        """
        import ee
        kernel   = ee.Kernel.square(radius=1.5, units="pixels")
        filtered = image.convolve(kernel)
        return filtered.copyProperties(image, ["system:time_start"])

    def _export_sar(self, image, aoi, out_path: str):
        """Export SAR image — similar to optical export."""
        import ee
        import numpy as np
        import rasterio
        from rasterio.transform import from_bounds

        data = image.sampleRectangle(region=aoi, defaultValue=-9999).getInfo()

        arrays = []
        for band in self.BANDS:
            if band in data["properties"]:
                arr = np.array(data["properties"][band], dtype=np.float32)
                # Replace nodata
                arr[arr == -9999] = np.nan
                arrays.append(arr)

        if not arrays:
            log.warning(f"No SAR data for {out_path}")
            return

        stack     = np.stack(arrays, axis=0)
        H, W      = stack.shape[1], stack.shape[2]
        region    = aoi.bounds().getInfo()["coordinates"][0]
        lon_vals  = [p[0] for p in region]
        lat_vals  = [p[1] for p in region]
        bbox      = [min(lon_vals), min(lat_vals), max(lon_vals), max(lat_vals)]
        transform = rasterio.transform.from_bounds(*bbox, W, H)

        Path(out_path).parent.mkdir(parents=True, exist_ok=True)
        with rasterio.open(
            out_path, "w",
            driver="GTiff", dtype="float32", count=2,
            height=H, width=W, crs="EPSG:4326",
            transform=transform, compress="lzw", nodata=np.nan,
        ) as dst:
            dst.write(stack)
        log.info(f"  SAR saved: {Path(out_path).name}")


# ─────────────────────────────────────────────
# Orchestrated downloader
# ─────────────────────────────────────────────

class SceneDownloader:
    """
    Orchestrates Sentinel-2 + Sentinel-1 + temporal stack download
    for a given AOI and date. Single entry point for scripts/download_data.py.
    """

    def __init__(self, output_dir: str, cloud_threshold: int = 20,
                 temporal_days: int = 15, ee_project: Optional[str] = None):
        self.output_dir      = Path(output_dir)
        self.temporal_days   = temporal_days
        self.s2              = Sentinel2Downloader(cloud_threshold)
        self.s1              = Sentinel1Downloader()
        init_ee(ee_project)

    def download(self, bbox: str, date: str, scene_name: Optional[str] = None) -> dict:
        """
        Full download: optical + SAR + temporal stack for one scene.

        Args:
            bbox:       "lon_min,lat_min,lon_max,lat_max"
            date:       target date "YYYY-MM-DD"
            scene_name: optional name prefix (defaults to date)

        Returns:
            dict of {type: path}
        """
        import ee
        if scene_name is None:
            scene_name = date.replace("-", "")

        aoi = bbox_to_ee_geometry(bbox)
        paths = {}

        log.info(f"Downloading scene '{scene_name}' | AOI: {bbox} | Date: {date}")
        log.info("─" * 55)

        # ── Optical: target scene ─────────────────────────────────
        log.info("[1/3] Downloading Sentinel-2 optical scene...")
        p = self.s2.download_scene(aoi, date, self.output_dir, scene_name, temporal_days=0)
        if p: paths["optical"] = str(p)

        # ── SAR ───────────────────────────────────────────────────
        log.info("[2/3] Downloading Sentinel-1 SAR scene...")
        p = self.s1.download_scene(aoi, date, self.output_dir, scene_name)
        if p: paths["sar"] = str(p)

        # ── Temporal stack ────────────────────────────────────────
        log.info(f"[3/3] Downloading temporal stack (±{self.temporal_days} days)...")
        p = self.s2.download_scene(
            aoi, date, self.output_dir, scene_name,
            temporal_days=self.temporal_days,
        )
        if p: paths["temporal"] = str(p)

        log.info("─" * 55)
        log.info(f"Download complete: {len(paths)} datasets saved to {self.output_dir}")
        return paths

    def build_archive(
        self,
        bbox:        str,
        years:       List[int],
        months:      List[int],
        scene_prefix: str = "india",
    ):
        """
        Build a historical cloud-free archive for the given AOI.
        Downloads monthly median composites for each year-month combination.
        Used to populate data/archive/ for the historical reference library.
        """
        import ee
        aoi = bbox_to_ee_geometry(bbox)
        log.info(f"Building archive: {len(years)*len(months)} monthly composites")

        paths = []
        for year in years:
            for month in months:
                name = f"{scene_prefix}_{year}_{month:02d}"
                try:
                    p = self.s2.download_composite(
                        aoi, year, month, self.output_dir, name
                    )
                    paths.append(p)
                    log.info(f"  ✓ {year}-{month:02d} composite")
                except Exception as e:
                    log.warning(f"  ✗ {year}-{month:02d} failed: {e}")
                time.sleep(1)   # rate limit

        log.info(f"Archive complete: {len(paths)} composites saved")
        return paths
