#!/usr/bin/env python3
"""
scripts/infer.py
Run the full 7-layer cloud removal pipeline on a single input scene.

Usage:
    python scripts/infer.py \
        --input  data/raw/optical/scene.tif \
        --sar    data/raw/sar/scene_sar.tif \
        --temporal data/raw/temporal/scene_stack.tif \
        --config configs/base.yaml \
        --output outputs/predictions/
"""

import argparse
import logging
from pathlib import Path

import torch
from omegaconf import OmegaConf

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)


def parse_args():
    p = argparse.ArgumentParser(description="End-to-end cloud removal inference.")
    p.add_argument("--input",    required=True, help="Input optical GeoTIFF (4-band).")
    p.add_argument("--sar",      default=None,  help="SAR GeoTIFF (VV, VH).")
    p.add_argument("--temporal", default=None,  help="Temporal stack GeoTIFF (T*4 bands).")
    p.add_argument("--config",   required=True, help="Path to YAML config file.")
    p.add_argument("--output",   default="outputs/predictions", help="Output directory.")
    p.add_argument("--no-report", action="store_true", help="Skip validation report generation.")
    p.add_argument("--fast-export", action="store_true", help="Skip expensive optional export outputs for faster runs.")
    return p.parse_args()


def main():
    args = parse_args()
    cfg  = OmegaConf.load(args.config)
    out  = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)

    scene_name = Path(args.input).stem

    log.info("=" * 60)
    log.info("Cloud Removal Pipeline — Full Inference")
    log.info("=" * 60)

    # ── L1: Load data ────────────────────────────────────────────
    log.info("[L1] Loading multi-source data...")
    from src.data.loaders.scene_loader import SceneLoader
    loader = SceneLoader(cfg)
    scene  = loader.load(
        optical_path  = args.input,
        sar_path      = args.sar,
        temporal_path = args.temporal,
    )

    # ── L2: Cloud detection ──────────────────────────────────────
    device = "cuda" if torch.cuda.is_available() else "cpu"
    log.info(f"Using device: {device}")
    log.info("[L2] Running cloud detection...")
    from src.models.cloud_detection.detector import CloudDetector
    detector   = CloudDetector.from_checkpoint(cfg, device=device)
    cloud_mask = detector.predict(scene.optical)
    # cloud_mask: (H, W) values 0=clear, 1=thin, 2=thick, 3=shadow

    # ── L3: Temporal analysis ────────────────────────────────────
    log.info("[L3] Running temporal analysis engine...")
    from src.data.preprocessing.temporal_analysis import TemporalAnalyzer
    analyzer = TemporalAnalyzer(cfg)
    if scene.temporal_stack is not None:
        cloud_mask_stack = cloud_mask
        if cloud_mask_stack.dim() == 3 and cloud_mask_stack.shape[0] == 1:
            cloud_mask_stack = cloud_mask_stack.expand(scene.temporal_stack.shape[0], -1, -1)
        temporal = analyzer.analyze(scene.temporal_stack, cloud_mask_stack)
    else:
        temporal = {
            "stack": None,
            "composite": None,
            "consistency": torch.ones_like(cloud_mask.float()),
        }
    # temporal: {composite, trend_maps, consistency_score}

    # ── L4: Reconstruction ───────────────────────────────────────
    log.info("[L4] Running hybrid AI reconstruction...")
    from src.models.reconstruction.pipeline import ReconstructionPipeline
    reconstructor = ReconstructionPipeline.from_checkpoint(cfg, device=device)
    reconstructed = reconstructor.predict(
        optical       = scene.optical,
        cloud_mask    = cloud_mask,
        sar           = scene.sar,
        temporal      = temporal,
    )

    # ── L5: Verification ─────────────────────────────────────────
    log.info("[L5] Running 5-pass verification engine...")
    from src.models.verification.verifier import MultiPassVerifier
    verifier = MultiPassVerifier(cfg)
    report   = verifier.verify(
        pred          = reconstructed,
        original      = scene.optical,
        cloud_mask    = cloud_mask,
        sar           = scene.sar,
        temporal      = temporal,
    )

    # ── L6: Confidence maps ──────────────────────────────────────
    log.info("[L6] Generating confidence and uncertainty maps...")
    from src.inference.confidence import ConfidenceMapper
    conf_mapper  = ConfidenceMapper(cfg)
    conf_maps    = conf_mapper.generate(
        reconstructed,
        report,
        cloud_mask,
        temporal,
        sar_features=scene.sar,
    )
    # conf_maps: {confidence (0-1), uncertainty (high/med/low)}

    # ── L7: Export outputs ───────────────────────────────────────
    log.info("[L7] Exporting scientific outputs...")
    try:
        from src.inference.exporter import SceneExporter
    except ModuleNotFoundError:
        from exporter import SceneExporter
    exporter = SceneExporter(cfg, out)
    if args.fast_export:
        cfg.export.fast_mode = True
        cfg.export.generate_pdf = False
        cfg.export.write_indices = False
        cfg.export.write_branch_weights = False

    exporter.export(
        scene_name    = scene_name,
        reconstructed = reconstructed,
        cloud_mask    = cloud_mask,
        conf_maps     = conf_maps,
        meta          = scene.meta,
        report        = report,
        generate_pdf  = not args.no_report,
    )

    log.info("=" * 60)
    log.info(f"Done. Outputs saved to: {out}")
    log.info(f"Overall validation: {'PASS' if report.get('overall_pass') else 'FAIL'}")
    log.info("=" * 60)


if __name__ == "__main__":
    main()
