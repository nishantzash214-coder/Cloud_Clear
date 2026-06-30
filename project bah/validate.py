#!/usr/bin/env python3
"""
scripts/validate.py

Compute full scientific validation metrics for a reconstructed scene
against a known cloud-free reference (ground truth).

Usage:
    python scripts/validate.py \
        --prediction  outputs/predictions/scene_reconstructed.tif \
        --reference   data/archive/scene_cloudfree.tif \
        --cloud-mask  outputs/predictions/scene_cloud_mask.tif \
        --output-dir  outputs/reports/
"""

import argparse
import json
import logging
from pathlib import Path
import torch

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--prediction",  required=True)
    p.add_argument("--reference",   required=True)
    p.add_argument("--cloud-mask",  default=None)
    p.add_argument("--temporal",    default=None,
                   help="Temporal stack GeoTIFF for temporal consistency metric")
    p.add_argument("--output-dir",  default="outputs/reports")
    return p.parse_args()


def main():
    args     = parse_args()
    out_dir  = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    scene    = Path(args.prediction).stem.replace("_reconstructed", "")

    from src.utils.geotiff import read_geotiff
    from src.utils.metrics import compute_all_metrics

    log.info("Loading prediction and reference...")
    pred_arr, _  = read_geotiff(args.prediction)
    ref_arr,  _  = read_geotiff(args.reference)

    pred  = torch.from_numpy(pred_arr).unsqueeze(0)  # (1,C,H,W)
    ref   = torch.from_numpy(ref_arr).unsqueeze(0)

    # Align band counts
    C = min(pred.shape[1], ref.shape[1], 4)
    pred, ref = pred[:, :C], ref[:, :C]

    cloud_mask = None
    if args.cloud_mask:
        cm, _ = read_geotiff(args.cloud_mask)
        cloud_mask = torch.from_numpy(cm[0]).unsqueeze(0)  # (1,H,W)

    temporal_stack = None
    if args.temporal:
        t_arr, _ = read_geotiff(args.temporal)
        T = t_arr.shape[0] // C
        t_stack  = t_arr[:T*C].reshape(T, C, *t_arr.shape[1:])
        temporal_stack = torch.from_numpy(t_stack).unsqueeze(0)  # (1,T,C,H,W)

    log.info("Computing scientific metrics...")
    metrics = compute_all_metrics(
        pred           = pred,
        target         = ref,
        temporal_stack = temporal_stack,
        cloud_mask     = cloud_mask,
    )

    # ── Print summary ─────────────────────────────────────────────
    log.info("=" * 55)
    log.info(f"{'Metric':<30} {'Value':>10}  {'Target':>10}  Status")
    log.info("-" * 55)
    for metric, info in metrics.get("validation", {}).items():
        val    = info["value"]
        tgt    = info["target"]
        status = "✓ PASS" if info["pass"] else "✗ FAIL"
        log.info(f"  {metric:<28} {val:>10.4f}  {tgt:>10.4f}  {status}")
    log.info("=" * 55)
    overall = "✓ OVERALL PASS" if metrics.get("overall_pass") else "✗ OVERALL FAIL"
    log.info(f"  {overall}")
    log.info("=" * 55)

    # ── Save JSON report ──────────────────────────────────────────
    report_path = out_dir / f"{scene}_validation.json"
    with open(report_path, "w") as f:
        json.dump(metrics, f, indent=2, default=str)
    log.info(f"Validation report saved → {report_path}")

    return 0 if metrics.get("overall_pass") else 1


if __name__ == "__main__":
    import sys
    sys.exit(main())
