#!/usr/bin/env python3
"""
scripts/train.py
Entry point for training any stage of the cloud removal pipeline.

Usage:
    python scripts/train.py --stage cloud_detection --config configs/base.yaml
    python scripts/train.py --stage reconstruction   --config configs/base.yaml
"""

import argparse
import logging
from pathlib import Path
from omegaconf import OmegaConf

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)


def parse_args():
    p = argparse.ArgumentParser(description="Train cloud removal pipeline stages.")
    p.add_argument("--stage",  required=True,
                   choices=["cloud_detection", "reconstruction", "verification"],
                   help="Which pipeline stage to train.")
    p.add_argument("--config", required=True, help="Path to YAML config file.")
    p.add_argument("--resume", default=None, help="Checkpoint path to resume from.")
    p.add_argument("--debug",  action="store_true", help="Quick debug run (2 batches).")
    return p.parse_args()


def main():
    args = parse_args()
    cfg  = OmegaConf.load(args.config)
    log.info(f"Starting training — stage: {args.stage}")
    log.info(f"Config: {args.config}")

    if args.stage == "cloud_detection":
        from src.training.train_cloud_detection import CloudDetectionTrainer
        trainer = CloudDetectionTrainer(cfg, resume=args.resume, debug=args.debug)

    elif args.stage == "reconstruction":
        from src.training.train_reconstruction import ReconstructionTrainer
        trainer = ReconstructionTrainer(cfg, resume=args.resume, debug=args.debug)

    elif args.stage == "verification":
        from src.training.train_verification import VerificationTrainer
        trainer = VerificationTrainer(cfg, resume=args.resume, debug=args.debug)

    trainer.fit()
    log.info("Training complete.")


if __name__ == "__main__":
    main()
