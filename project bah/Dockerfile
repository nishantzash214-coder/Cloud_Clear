# ============================================================
#  Cloud Removal Framework — Dockerfile
#  Multi-stage build: base → training → inference
# ============================================================

FROM pytorch/pytorch:2.2.0-cuda12.1-cudnn8-runtime AS base

WORKDIR /app

# System dependencies for GDAL / rasterio
RUN apt-get update && apt-get install -y --no-install-recommends \
    gdal-bin \
    libgdal-dev \
    libspatialindex-dev \
    libgl1-mesa-glx \
    git \
    && rm -rf /var/lib/apt/lists/*

ENV GDAL_VERSION=3.8
ENV CPLUS_INCLUDE_PATH=/usr/include/gdal
ENV C_INCLUDE_PATH=/usr/include/gdal

# Python deps
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .
RUN pip install -e . --no-deps

# ── Inference image ───────────────────────────────────────────
FROM base AS inference

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=10s --start-period=60s \
    CMD curl -f http://localhost:8000/health || exit 1

CMD ["uvicorn", "src.api.main:app", \
     "--host", "0.0.0.0", \
     "--port", "8000", \
     "--workers", "1", \
     "--log-level", "info"]

# ── Training image ────────────────────────────────────────────
FROM base AS training

# Install additional training deps
RUN pip install --no-cache-dir wandb rich

CMD ["python", "scripts/train.py", \
     "--stage", "reconstruction", \
     "--config", "configs/base.yaml"]
