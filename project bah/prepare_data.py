#!/usr/bin/env python3
"""
scripts/prepare_data.py
Prepare raw satellite data for training:
  1. Radiometric normalisation
  2. Cloud detection (initial run using threshold-based method)
  3. Temporal composite generation
  4. Patch extraction (256×256) with synthetic cloud augmentation
  5. Train / val / test split

Usage:
    python scripts/prepare_data.py --config configs/base.yaml
"""

import argparse
import logging
import random
from pathlib import Path
from omegaconf import OmegaConf

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--config", required=True)
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--dry-run", action="store_true",
                   help="List files to process without executing.")
    return p.parse_args()


def main():
    args = parse_args()
    cfg  = OmegaConf.load(args.config)

    raw_dir  = Path(cfg.paths.raw_optical)
    proc_dir = Path(cfg.paths.processed)
    proc_dir.mkdir(parents=True, exist_ok=True)

    scenes = sorted(raw_dir.glob("*.tif"))
    log.info(f"Found {len(scenes)} raw optical scenes.")

    if args.dry_run:
        for s in scenes:
            log.info(f"  [dry-run] Would process: {s.name}")
        return

    from src.data.preprocessing.radiometry import RadiometricNormaliser
    from src.data.preprocessing.cloud_threshold import ThresholdCloudMask
    from src.data.preprocessing.temporal_composite import build_temporal_composite
    from src.data.preprocessing.patch_extractor import PatchExtractor
    from src.data.augmentation.synthetic_clouds import SyntheticCloudGenerator

    normaliser     = RadiometricNormaliser(cfg)
    cloud_thres    = ThresholdCloudMask(cfg)
    patch_extractor = PatchExtractor(patch_size=cfg.data.patch_size,
                                     overlap=cfg.data.overlap)
    cloud_gen      = SyntheticCloudGenerator(cfg)

    all_patches = []

    for scene_path in scenes:
        log.info(f"Processing: {scene_path.name}")

        # Step 1: Normalise
        normalised = normaliser.process(scene_path)

        # Step 2: Initial cloud mask (threshold-based for data prep)
        cloud_mask = cloud_thres.predict(normalised)

        # Step 3: Temporal composite (looks for matching temporal scenes)
        temporal_dir = Path(cfg.paths.raw_temporal) / scene_path.stem
        if temporal_dir.exists():
            composite = build_temporal_composite(
                temporal_dir, method=cfg.temporal.composite_method
            )
        else:
            composite = None
            log.warning(f"  No temporal data found for {scene_path.stem}")

        # Step 4: Extract clear-sky patches
        patches = patch_extractor.extract(normalised, cloud_mask)

        # Step 5: Synthetic cloud augmentation → perfect ground truth pairs
        augmented = cloud_gen.augment(patches)
        all_patches.extend(augmented)

    log.info(f"Total training patches generated: {len(all_patches)}")

    # Step 6: Train / val / test split
    random.seed(cfg.project.seed)
    random.shuffle(all_patches)
    n     = len(all_patches)
    n_tr  = int(n * cfg.data.split.train)
    n_val = int(n * cfg.data.split.val)

    splits = {
        "train": all_patches[:n_tr],
        "val":   all_patches[n_tr:n_tr+n_val],
        "test":  all_patches[n_tr+n_val:],
    }

    for split_name, split_patches in splits.items():
        split_dir = proc_dir / "patches" / split_name
        split_dir.mkdir(parents=True, exist_ok=True)
        for i, patch in enumerate(split_patches):
            patch.save(split_dir / f"{split_name}_{i:06d}.npz")
        log.info(f"  {split_name}: {len(split_patches)} patches → {split_dir}")

    log.info("Data preparation complete.")


if __name__ == "__main__":
    main()
