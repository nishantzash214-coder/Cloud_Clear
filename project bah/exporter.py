"""
src/inference/exporter.py

Scene Exporter (Layer 7).

Exports all scientific deliverables for a reconstructed scene:
  1. Reconstructed image         → {scene}_reconstructed.tif
  2. Cloud mask                  → {scene}_cloud_mask.tif
  3. Confidence map              → {scene}_confidence.tif
  4. Uncertainty map             → {scene}_uncertainty.tif
  5. Spectral index maps         → {scene}_ndvi.tif, _ndwi.tif, _savi.tif, _ndbi.tif
  6. Branch weight maps          → {scene}_branch_weights.tif
  7. Validation report           → {scene}_report.json + {scene}_report.pdf

All GeoTIFFs are Cloud-Optimised GeoTIFF (COG) compatible,
georeferenced, and use the same CRS/transform as the input scene.
"""

from __future__ import annotations
import torch
import numpy as np
import json
from pathlib import Path
from typing import Dict, Optional
import logging

try:
    from src.utils.geotiff import write_geotiff
    from src.utils.indices import ndvi, ndwi, savi, ndbi
except ModuleNotFoundError:
    from geotiff import write_geotiff
    from indices import ndvi, ndwi, savi, ndbi

log = logging.getLogger(__name__)


class SceneExporter:
    """
    Exports all Layer 7 outputs to disk in GeoTIFF + report formats.
    """

    BAND_NAMES = ["green", "red", "nir", "swir"]

    def __init__(self, cfg, output_dir: Path):
        self.cfg        = cfg
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.nodata     = cfg.export.nodata       # -9999
        self.compress   = "lzw"
        self.fast_mode  = getattr(cfg.export, "fast_mode", False)
        self.write_indices = getattr(cfg.export, "write_indices", not self.fast_mode)
        self.write_branch_weights = getattr(cfg.export, "write_branch_weights", not self.fast_mode)
        self.generate_pdf = getattr(cfg.export, "generate_pdf", True)

    def export(
        self,
        scene_name:    str,
        reconstructed: torch.Tensor,              # (B, 4, H, W)
        cloud_mask:    torch.Tensor,              # (B, H, W)
        conf_maps,                                # ConfidenceMaps dataclass
        meta:          dict,                      # rasterio metadata
        report:        Dict,
        branch_weights: Optional[torch.Tensor] = None,  # (B, 4, H, W)
        generate_pdf:  bool = True,
    ) -> Dict[str, Path]:
        """
        Export all outputs. Returns dict of {output_name: path}.
        """
        log.info(f"Exporting scene: {scene_name}")
        paths = {}

        def _first_item(x):
            if isinstance(x, torch.Tensor):
                if x.dim() >= 4 and x.shape[0] == 1:
                    return x[0].cpu()
                if x.dim() == 3 and x.shape[0] == 1:
                    return x[0].cpu()
                if x.dim() >= 1 and x.shape[0] != 1:
                    return x[0].cpu()
                return x.cpu()
            return x

        # Use first batch item (inference is single-scene)
        recon = _first_item(reconstructed)
        mask  = _first_item(cloud_mask)
        if isinstance(conf_maps, dict):
            confidence = conf_maps.get("confidence")
            uncertainty = conf_maps.get("uncertainty")
            mc_variance = conf_maps.get("mc_variance")
            breakdown = conf_maps.get("breakdown", {})
        else:
            confidence = getattr(conf_maps, "confidence", None)
            uncertainty = getattr(conf_maps, "uncertainty", None)
            mc_variance = getattr(conf_maps, "mc_variance", None)
            breakdown = getattr(conf_maps, "breakdown", {})
        conf = _first_item(confidence)
        uncert = _first_item(uncertainty)

        def _normalize_export_tensor(x):
            if isinstance(x, torch.Tensor):
                x = x.cpu()
                while x.dim() > 3 and x.shape[0] == 1:
                    x = x[0]
                if x.ndim == 2:
                    x = x.unsqueeze(0)
                return x
            if isinstance(x, np.ndarray):
                if x.ndim == 2:
                    x = x[np.newaxis, ...]
            return x

        recon = _normalize_export_tensor(recon)
        mask  = _normalize_export_tensor(mask)

        # ── 1. Reconstructed image ────────────────────────────────
        p = self.output_dir / f"{scene_name}_reconstructed.tif"
        write_geotiff(p, recon.numpy(), meta, self.nodata, self.compress)
        paths["reconstructed"] = p
        log.info(f"  ✓ Reconstructed image    → {p.name}")

        # ── 2. Cloud mask ─────────────────────────────────────────
        p = self.output_dir / f"{scene_name}_cloud_mask.tif"
        write_geotiff(p, mask.numpy().astype(np.float32), meta,
                      self.nodata, self.compress)
        paths["cloud_mask"] = p
        log.info(f"  ✓ Cloud mask             → {p.name}")

        # ── 3. Confidence map ─────────────────────────────────────
        conf = _normalize_export_tensor(conf)
        uncert = _normalize_export_tensor(uncert)

        p = self.output_dir / f"{scene_name}_confidence.tif"
        write_geotiff(p, conf.numpy(), meta, self.nodata, self.compress)
        paths["confidence"] = p
        log.info(f"  ✓ Confidence map         → {p.name}")

        # ── 4. Uncertainty map ────────────────────────────────────
        p = self.output_dir / f"{scene_name}_uncertainty.tif"
        write_geotiff(p, uncert.numpy().astype(np.float32), meta,
                      self.nodata, self.compress)
        paths["uncertainty"] = p
        log.info(f"  ✓ Uncertainty map        → {p.name}")

        # ── 5. Spectral index maps ────────────────────────────────
        if self.write_indices and not self.fast_mode:
            recon_t = recon.unsqueeze(0)              # (1,4,H,W)
            for name, fn in [("ndvi", ndvi), ("ndwi", ndwi),
                             ("savi", savi), ("ndbi", ndbi)]:
                idx = fn(recon_t).squeeze(0).numpy()  # (1,H,W)
                p   = self.output_dir / f"{scene_name}_{name}.tif"
                write_geotiff(p, idx, meta, self.nodata, self.compress)
                paths[name] = p
            log.info(f"  ✓ Spectral index maps    → ndvi/ndwi/savi/ndbi")
        elif self.fast_mode:
            log.info("  ↷ Fast mode enabled — skipping spectral index exports")

        # ── 6. Branch weight maps ─────────────────────────────────
        if self.write_branch_weights and branch_weights is not None and not self.fast_mode:
            bw = branch_weights[0].cpu().numpy()  # (4, H, W)
            p  = self.output_dir / f"{scene_name}_branch_weights.tif"
            bw_meta = meta.copy()
            bw_meta["count"] = 4
            write_geotiff(p, bw, bw_meta, self.nodata, self.compress)
            paths["branch_weights"] = p
            log.info(f"  ✓ Branch weight maps     → {p.name}")
        elif self.fast_mode:
            log.info("  ↷ Fast mode enabled — skipping branch-weight exports")

        # ── 7. MC variance map (if available) ────────────────────
        if mc_variance is not None:
            mc = _first_item(mc_variance).numpy()
            p  = self.output_dir / f"{scene_name}_mc_uncertainty.tif"
            write_geotiff(p, mc, meta, self.nodata, self.compress)
            paths["mc_uncertainty"] = p

        # ── 8. Validation report (JSON) ───────────────────────────
        report_data = {
            "scene":    scene_name,
            "outputs":  {k: str(v) for k, v in paths.items()},
            "confidence_breakdown": breakdown,
            **report,
        }
        p_json = self.output_dir / f"{scene_name}_report.json"
        with open(p_json, "w") as f:
            json.dump(report_data, f, indent=2, default=str)
        paths["report_json"] = p_json
        log.info(f"  ✓ Validation report JSON → {p_json.name}")

        # ── 9. PDF validation report ──────────────────────────────
        pdf_enabled = generate_pdf and self.generate_pdf and not self.fast_mode
        if pdf_enabled:
            try:
                p_pdf = self._generate_pdf_report(scene_name, report_data)
                paths["report_pdf"] = p_pdf
                log.info(f"  ✓ Validation report PDF  → {p_pdf.name}")
            except Exception as e:
                log.warning(f"PDF generation failed: {e}")
        elif self.fast_mode:
            log.info("  ↷ Fast mode enabled — skipping PDF report generation")

        log.info(f"Export complete — {len(paths)} files written to {self.output_dir}")
        return paths

    def _generate_pdf_report(self, scene_name: str, report: Dict) -> Path:
        """Generate a PDF validation report using ReportLab."""
        from reportlab.lib.pagesizes import A4
        from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
        from reportlab.lib.units import cm
        from reportlab.lib import colors
        from reportlab.platypus import (
            SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, HRFlowable
        )

        path = self.output_dir / f"{scene_name}_report.pdf"
        doc  = SimpleDocTemplate(str(path), pagesize=A4,
                                  leftMargin=2*cm, rightMargin=2*cm,
                                  topMargin=2*cm, bottomMargin=2*cm)
        styles = getSampleStyleSheet()

        header_style = ParagraphStyle("header", parent=styles["Heading1"],
                                       fontSize=16, spaceAfter=6)
        sub_style    = ParagraphStyle("sub",    parent=styles["Heading2"],
                                       fontSize=12, spaceAfter=4)
        body_style   = styles["Normal"]

        elements = []

        # Title
        elements.append(Paragraph(
            "AI-Powered Cloud Removal — Validation Report", header_style
        ))
        elements.append(Paragraph(f"Scene: {scene_name}", sub_style))
        elements.append(HRFlowable(width="100%", thickness=1, color=colors.grey))
        elements.append(Spacer(1, 0.3*cm))

        # Overall result
        overall = "✓ PASS" if report.get("overall_pass") else "✗ FAIL"
        conf    = report.get("confidence", 0.0)
        elements.append(Paragraph(
            f"Overall Result: <b>{overall}</b> | System Confidence: <b>{conf:.1%}</b>",
            body_style
        ))
        elements.append(Spacer(1, 0.3*cm))

        # Verification passes table
        elements.append(Paragraph("Verification Passes (Layer 5)", sub_style))
        pass_data = [["Pass", "Name", "Score", "Threshold", "Status"]]
        for p in report.get("passes", []):
            status = "PASS" if p["passed"] else "FAIL"
            pass_data.append([
                str(p["id"]),
                p["name"],
                f"{p['score']:.3f}",
                f"{p['threshold']:.3f}",
                status,
            ])

        table = Table(pass_data, colWidths=[1.2*cm, 5.5*cm, 2*cm, 2.5*cm, 2*cm])
        table.setStyle(TableStyle([
            ("BACKGROUND",  (0, 0), (-1, 0),  colors.HexColor("#2B4C7E")),
            ("TEXTCOLOR",   (0, 0), (-1, 0),  colors.white),
            ("FONTNAME",    (0, 0), (-1, 0),  "Helvetica-Bold"),
            ("FONTSIZE",    (0, 0), (-1, -1), 9),
            ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#F2F2F2")]),
            ("GRID",        (0, 0), (-1, -1), 0.5, colors.grey),
            ("ALIGN",       (2, 0), (-1, -1), "CENTER"),
        ]))
        elements.append(table)
        elements.append(Spacer(1, 0.4*cm))

        # Pixel metrics table
        elements.append(Paragraph("Pixel-Level Metrics", sub_style))
        metric_targets = {
            "ssim": (0.90, "higher"),
            "psnr": (35.0, "higher"),
            "rmse": (0.03, "lower"),
            "sam":  (0.10, "lower"),
        }
        met_data = [["Metric", "Value", "Target", "Status"]]
        for k, (tgt, direction) in metric_targets.items():
            val = report.get("metrics", {}).get(k, report.get(k))
            if val is None:
                continue
            if direction == "higher":
                ok = val >= tgt
            else:
                ok = val <= tgt
            met_data.append([
                k.upper(),
                f"{val:.4f}",
                f"{tgt}",
                "PASS" if ok else "FAIL",
            ])

        if len(met_data) > 1:
            mtable = Table(met_data, colWidths=[3*cm, 3*cm, 3*cm, 3*cm])
            mtable.setStyle(TableStyle([
                ("BACKGROUND",  (0, 0), (-1, 0),  colors.HexColor("#2B4C7E")),
                ("TEXTCOLOR",   (0, 0), (-1, 0),  colors.white),
                ("FONTNAME",    (0, 0), (-1, 0),  "Helvetica-Bold"),
                ("FONTSIZE",    (0, 0), (-1, -1), 9),
                ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#F2F2F2")]),
                ("GRID",        (0, 0), (-1, -1), 0.5, colors.grey),
                ("ALIGN",       (1, 0), (-1, -1), "CENTER"),
            ]))
            elements.append(mtable)
            elements.append(Spacer(1, 0.4*cm))

        # Confidence breakdown
        elements.append(Paragraph("Confidence Map Breakdown (Layer 6)", sub_style))
        bd = report.get("confidence_breakdown", {})
        for source, val in bd.items():
            elements.append(Paragraph(
                f"  {source.capitalize()}: {float(val):.1%}", body_style
            ))
        elements.append(Spacer(1, 0.4*cm))

        # Footer
        elements.append(HRFlowable(width="100%", thickness=0.5, color=colors.grey))
        elements.append(Paragraph(
            "Generated by AI-Powered Scientific Cloud Removal Framework | "
            "Compatible with QGIS · ArcGIS · Bhuvan",
            ParagraphStyle("footer", parent=body_style, fontSize=8,
                            textColor=colors.grey)
        ))

        doc.build(elements)
        return path
