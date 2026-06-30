"""
src/api/main.py

FastAPI backend for the cloud removal system.

Endpoints:
  POST /infer          — submit a satellite scene for cloud removal
  GET  /status/{job}   — poll job status and progress
  GET  /result/{job}   — download reconstructed GeoTIFF
  GET  /report/{job}   — download validation report (PDF/JSON)
  GET  /health         — service health check
  GET  /metrics        — recent job statistics

Designed for deployment on AWS EC2 with GPU, behind a load balancer.
Results stored in S3 (or local disk for development).
"""

from __future__ import annotations
import uuid
import asyncio
import logging
from datetime import datetime
from pathlib import Path
from typing import Optional, Dict
from enum import Enum

import torch

from fastapi import FastAPI, UploadFile, File, HTTPException, BackgroundTasks
from fastapi.responses import FileResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

log = logging.getLogger(__name__)

app = FastAPI(
    title      = "AI Cloud Removal API",
    description = "Scientific cloud removal for LISS-IV / Resourcesat imagery",
    version    = "1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins  = ["*"],
    allow_methods  = ["*"],
    allow_headers  = ["*"],
)


# ─────────────────────────────────────────────
# Job state store (in-memory for dev, Redis for prod)
# ─────────────────────────────────────────────

class JobStatus(str, Enum):
    QUEUED      = "queued"
    PROCESSING  = "processing"
    VERIFYING   = "verifying"
    COMPLETE    = "complete"
    FAILED      = "failed"


class Job(BaseModel):
    job_id:     str
    status:     JobStatus
    progress:   int          # 0–100
    stage:      str          # current pipeline stage
    created_at: str
    outputs:    Dict[str, str] = Field(default_factory=dict)
    error:      Optional[str]  = None
    validation: Optional[Dict] = None


_jobs: Dict[str, Job] = {}
_output_base = Path("outputs/predictions")
# Ensure outputs directory exists
_output_base.mkdir(parents=True, exist_ok=True)


# ─────────────────────────────────────────────
# Endpoints
# ─────────────────────────────────────────────

@app.get("/health")
async def health():
    """Service health check."""
    import torch
    return {
        "status":       "healthy",
        "gpu_available": torch.cuda.is_available(),
        "gpu_name":     torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "active_jobs":  sum(1 for j in _jobs.values() if j.status == JobStatus.PROCESSING),
        "timestamp":    datetime.utcnow().isoformat(),
    }


@app.post("/infer", response_model=Job)
async def submit_inference(
    background_tasks: BackgroundTasks,
    optical: UploadFile  = File(..., description="4-band optical GeoTIFF"),
    sar:     Optional[UploadFile] = File(None, description="SAR GeoTIFF (optional)"),
    temporal: Optional[UploadFile] = File(None, description="Temporal stack GeoTIFF (optional)"),
):
    """
    Submit a satellite scene for cloud removal.
    Returns a job ID for polling.
    """
    job_id = str(uuid.uuid4())[:8]
    job_dir = _output_base / job_id
    job_dir.mkdir(parents=True, exist_ok=True)

    # Save uploaded files
    opt_path = job_dir / "input_optical.tif"
    with open(opt_path, "wb") as f:
        f.write(await optical.read())

    sar_path = None
    if sar:
        sar_path = job_dir / "input_sar.tif"
        with open(sar_path, "wb") as f:
            f.write(await sar.read())

    temp_path = None
    if temporal:
        temp_path = job_dir / "input_temporal.tif"
        with open(temp_path, "wb") as f:
            f.write(await temporal.read())

    job = Job(
        job_id     = job_id,
        status     = JobStatus.QUEUED,
        progress   = 0,
        stage      = "queued",
        created_at = datetime.utcnow().isoformat(),
    )
    _jobs[job_id] = job

    background_tasks.add_task(
        run_pipeline, job_id, opt_path, sar_path, temp_path, job_dir
    )

    log.info(f"Job {job_id} queued — optical={optical.filename}")
    return job


@app.get("/status/{job_id}", response_model=Job)
async def get_status(job_id: str):
    """Poll job status and progress."""
    if job_id not in _jobs:
        raise HTTPException(404, f"Job {job_id} not found")
    return _jobs[job_id]


@app.get("/result/{job_id}")
async def get_result(job_id: str, output: str = "reconstructed"):
    """Download a specific output file."""
    job = _jobs.get(job_id)
    if not job:
        raise HTTPException(404, f"Job {job_id} not found")
    if job.status != JobStatus.COMPLETE:
        raise HTTPException(400, f"Job {job_id} is {job.status}, not complete")

    file_path = job.outputs.get(output)
    if not file_path or not Path(file_path).exists():
        raise HTTPException(404, f"Output '{output}' not found for job {job_id}")

    suffix  = Path(file_path).suffix
    media   = "image/tiff" if suffix == ".tif" else "application/octet-stream"
    return FileResponse(file_path, media_type=media,
                        filename=Path(file_path).name)


@app.get("/report/{job_id}")
async def get_report(job_id: str, format: str = "json"):
    """Download validation report (json or pdf)."""
    job = _jobs.get(job_id)
    if not job:
        raise HTTPException(404, f"Job {job_id} not found")
    if job.status != JobStatus.COMPLETE:
        raise HTTPException(400, f"Job {job_id} not complete yet")

    key = "report_pdf" if format == "pdf" else "report_json"
    p   = job.outputs.get(key)
    if not p or not Path(p).exists():
        raise HTTPException(404, f"Report format '{format}' not available")

    media = "application/pdf" if format == "pdf" else "application/json"
    return FileResponse(p, media_type=media, filename=Path(p).name)


@app.get("/metrics")
async def get_metrics():
    """Summary statistics across recent jobs."""
    total    = len(_jobs)
    complete = sum(1 for j in _jobs.values() if j.status == JobStatus.COMPLETE)
    failed   = sum(1 for j in _jobs.values() if j.status == JobStatus.FAILED)
    active   = sum(1 for j in _jobs.values() if j.status == JobStatus.PROCESSING)

    # Collect validation scores from completed jobs
    ssim_scores  = []
    pass_rates   = []
    for j in _jobs.values():
        if j.validation:
            if "ssim" in j.validation.get("metrics", {}):
                ssim_scores.append(j.validation["metrics"]["ssim"])
            pass_rates.append(1 if j.validation.get("overall_pass") else 0)

    return {
        "total_jobs":       total,
        "complete":         complete,
        "failed":           failed,
        "active":           active,
        "mean_ssim":        sum(ssim_scores) / max(len(ssim_scores), 1),
        "overall_pass_rate": sum(pass_rates) / max(len(pass_rates), 1),
    }


# ─────────────────────────────────────────────
# Background pipeline runner
# ─────────────────────────────────────────────

async def run_pipeline(
    job_id:    str,
    opt_path:  Path,
    sar_path:  Optional[Path],
    temp_path: Optional[Path],
    job_dir:   Path,
):
    """Execute the full 7-layer pipeline as a background task."""
    job = _jobs[job_id]

    def update(status: JobStatus, progress: int, stage: str):
        job.status   = status
        job.progress = progress
        job.stage    = stage
        log.info(f"[{job_id}] {stage} ({progress}%)")

    try:
        from omegaconf import OmegaConf
        cfg_path = Path("configs/base.yaml")
        if not cfg_path.exists():
            cfg_path = Path("base.yaml")
        cfg = OmegaConf.load(cfg_path)

        update(JobStatus.PROCESSING, 10, "L1: Loading data")
        from src.data.loaders.scene_loader import SceneLoader
        loader = SceneLoader(cfg)
        scene  = loader.load(opt_path, sar_path, temp_path)

        device = "cuda" if torch.cuda.is_available() else "cpu"
        log.info(f"Using device: {device}")
        if device == "cpu":
            _, _, H, W = scene.optical.shape
            max_pixels = 256 * 256
            if H * W > max_pixels:
                raise RuntimeError(
                    f"Scene too large for CPU inference: {H}x{W} ({H*W} pixels). "
                    "Use a smaller input tile such as data/raw/optical/test_scene_64.tif, "
                    "or run the API on a GPU-enabled machine."
                )
        update(JobStatus.PROCESSING, 25, "L2: Cloud detection")
        from src.models.cloud_detection.unetplusplus import CloudDetector
        detector   = CloudDetector.from_checkpoint(cfg, device=device)
        cloud_mask = detector.predict(scene.optical)

        update(JobStatus.PROCESSING, 40, "L3: Temporal analysis")
        from src.data.preprocessing.temporal_analysis import TemporalAnalyzer
        analyzer = TemporalAnalyzer(cfg)

        if scene.temporal_stack is not None:
            temp_stack = scene.temporal_stack
            if temp_stack.dim() == 5 and temp_stack.shape[0] == 1:
                temp_stack = temp_stack.squeeze(0)
            cloud_mask_stack = cloud_mask
            if cloud_mask_stack.dim() == 3 and cloud_mask_stack.shape[0] == 1:
                cloud_mask_stack = cloud_mask_stack.expand(temp_stack.shape[0], -1, -1)
            temp_ctx = analyzer.analyze(temp_stack, cloud_mask_stack)
        else:
            temp_ctx = {
                "stack": None,
                "composite": None,
                "consistency": torch.ones_like(cloud_mask.float()),
            }

        update(JobStatus.PROCESSING, 60, "L4: Reconstruction")
        from src.models.reconstruction.pipeline import ReconstructionPipeline
        pipeline      = ReconstructionPipeline.from_checkpoint(cfg, device=device)
        reconstructed = pipeline.predict(scene.optical, cloud_mask,
                                          scene.sar, temp_ctx)

        update(JobStatus.VERIFYING, 75, "L5: Verification")
        from src.models.verification.verifier import MultiPassVerifier
        verifier = MultiPassVerifier(cfg)
        report   = verifier.verify(reconstructed, scene.optical,
                                    cloud_mask, scene.sar, temp_ctx)

        update(JobStatus.VERIFYING, 85, "L6: Confidence maps")
        from src.inference.confidence import ConfidenceMapper
        mapper    = ConfidenceMapper(cfg)
        conf_maps = mapper.generate(
            reconstructed,
            report,
            cloud_mask,
            temp_ctx,
            sar_features=scene.sar,
        )

        update(JobStatus.VERIFYING, 92, "L7: Exporting outputs")
        try:
            from src.inference.exporter import SceneExporter
        except ModuleNotFoundError:
            from exporter import SceneExporter
        exporter = SceneExporter(cfg, job_dir)
        paths    = exporter.export(
            scene_name    = job_id,
            reconstructed = reconstructed,
            cloud_mask    = cloud_mask,
            conf_maps     = conf_maps,
            meta          = scene.meta,
            report        = report,
        )

        if not paths:
            raise RuntimeError("SceneExporter.export returned no outputs")

        missing_outputs = [k for k, v in paths.items() if not Path(v).exists()]
        if missing_outputs:
            raise RuntimeError(
                f"Exporter returned missing files: {', '.join(missing_outputs)}"
            )

        job.outputs    = {k: str(v) for k, v in paths.items()}
        job.validation = report
        update(JobStatus.COMPLETE, 100, "Complete")

    except Exception as e:
        log.exception(f"[{job_id}] Pipeline failed: {e}")
        job.status = JobStatus.FAILED
        err_text = str(e)
        if "not enough memory" in err_text or "DefaultCPUAllocator" in err_text:
            job.error = (
                "Inference failed due to insufficient memory. "
                "Please use a smaller GeoTIFF, a GPU-enabled machine, or preprocess the scene into smaller tiles."
            )
        else:
            job.error = err_text
