#!/usr/bin/env python3
"""
scripts/download_data.py

Download Sentinel-2 + Sentinel-1 data for a given AOI and date.

Usage examples:
  # Single scene (Delhi region, 2024-08-15)
  python scripts/download_data.py \
      --bbox  "76.8,28.4,77.4,29.0" \
      --date  "2024-08-15" \
      --output data/raw/

  # Build monthly archive (2022–2024 for central India)
  python scripts/download_data.py \
      --bbox    "76.8,28.4,77.4,29.0" \
      --archive \
      --years   2022 2023 2024 \
      --months  1 2 3 4 5 6 7 8 9 10 11 12 \
      --output  data/

  # Batch: multiple dates from a CSV
  python scripts/download_data.py \
      --csv    scripts/scenes.csv \
      --output data/raw/

AOI presets for Indian regions:
  Delhi:     76.8,28.4,77.4,29.0
  Mumbai:    72.7,18.8,73.1,19.2
  Chennai:   79.9,12.8,80.4,13.3
  Bangalore: 77.4,12.8,77.8,13.1
  Bhopal:    77.2,23.1,77.6,23.5
"""

import argparse
import logging
import csv
from pathlib import Path

logging.basicConfig(
    level  = logging.INFO,
    format = "%(asctime)s [%(levelname)s] %(message)s",
)
log = logging.getLogger(__name__)


def parse_args():
    p = argparse.ArgumentParser(description="Download satellite data via Earth Engine")
    p.add_argument("--bbox",    default="77.2,23.1,77.6,23.5",
                   help="lon_min,lat_min,lon_max,lat_max  (default: Bhopal)")
    p.add_argument("--date",    default=None,    help="Target date YYYY-MM-DD")
    p.add_argument("--output",  default="data/", help="Output root directory")
    p.add_argument("--cloud-threshold", type=int, default=20,
                   help="Max cloud %% for S2 scenes (default 20)")
    p.add_argument("--temporal-days",   type=int, default=15,
                   help="±days for temporal stack (default 15)")
    p.add_argument("--scene-name",      default=None)
    p.add_argument("--ee-project",      default=None,
                   help="GEE project ID (for service account auth)")

    # Archive mode
    p.add_argument("--archive",  action="store_true", help="Build monthly archive")
    p.add_argument("--years",    nargs="+", type=int, default=[2023])
    p.add_argument("--months",   nargs="+", type=int,
                   default=list(range(1, 13)))
    p.add_argument("--prefix",   default="india")

    # Batch mode
    p.add_argument("--csv", default=None,
                   help="CSV file with columns: bbox,date,scene_name")

    return p.parse_args()


def main():
    args = parse_args()

    from src.data.downloaders.earth_engine import SceneDownloader

    downloader = SceneDownloader(
        output_dir       = args.output,
        cloud_threshold  = args.cloud_threshold,
        temporal_days    = args.temporal_days,
        ee_project       = args.ee_project,
    )

    if args.archive:
        # ── Archive mode ──────────────────────────────────────────
        log.info(f"Building archive for AOI: {args.bbox}")
        downloader.build_archive(
            bbox         = args.bbox,
            years        = args.years,
            months       = args.months,
            scene_prefix = args.prefix,
        )

    elif args.csv:
        # ── Batch mode ────────────────────────────────────────────
        with open(args.csv) as f:
            reader = csv.DictReader(f)
            rows   = list(reader)

        log.info(f"Batch download: {len(rows)} scenes from {args.csv}")
        for i, row in enumerate(rows):
            log.info(f"Scene {i+1}/{len(rows)}: {row.get('scene_name', row['date'])}")
            try:
                downloader.download(
                    bbox       = row["bbox"],
                    date       = row["date"],
                    scene_name = row.get("scene_name"),
                )
            except Exception as e:
                log.error(f"  Failed: {e}")

    else:
        # ── Single scene mode ─────────────────────────────────────
        if not args.date:
            log.error("--date is required for single scene download")
            return 1

        paths = downloader.download(
            bbox       = args.bbox,
            date       = args.date,
            scene_name = args.scene_name,
        )

        log.info("\nDownloaded files:")
        for k, v in paths.items():
            log.info(f"  {k:<12} → {v}")

    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())
