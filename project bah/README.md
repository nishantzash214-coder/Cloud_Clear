# AI-Powered Scientific Cloud Removal Framework
### High-Resolution Indian Remote Sensing Imagery (LISS-IV / Resourcesat)

A physics-aware, temporally-grounded, scientifically validated cloud removal system
producing reconstructed imagery suitable for government GIS, agriculture monitoring,
disaster management, and environmental assessment.

---

## Architecture Overview

```
L1  Multi-Source Data Acquisition  (Optical + SAR + Temporal ± 15 days + Archive)
L2  Intelligent Cloud Detection    (U-Net++ / DeepLabV3+, 4-class segmentation)
L3  Temporal Analysis Engine       (NDVI/NDWI trend, change probability maps)
L4  Scientific Reconstruction      (Diffusion + GAN + Temporal TF + SAR → Cross-Attention)
L5  Multi-Level Verification       (5 mandatory passes — temporal, SAR, spectral, AI, reference)
L6  Confidence & Uncertainty       (per-pixel 0-100% confidence + H/M/L uncertainty map)
L7  Output Generation              (GeoTIFF + cloud mask + validation report)
```

## Project Structure

```
cloud_removal/
├── data/                        # All data (raw, processed, archive)
│   ├── raw/
│   │   ├── optical/             # LISS-IV, Resourcesat, Sentinel-2 scenes
│   │   ├── sar/                 # RISAT / Sentinel-1 acquisitions
│   │   └── temporal/            # ±15-day stacks per scene
│   ├── processed/
│   │   ├── cloud_masks/         # Detected cloud masks (4-class)
│   │   ├── composites/          # Temporal median composites
│   │   └── patches/             # Training patches (256×256)
│   └── archive/                 # Historical cloud-free reference scenes
├── src/
│   ├── data/
│   │   ├── loaders/             # Dataset classes (PyTorch)
│   │   ├── preprocessing/       # Radiometric correction, co-registration, band math
│   │   └── augmentation/        # Spectral perturbation, rotation, seasonal sim
│   ├── models/
│   │   ├── cloud_detection/     # U-Net++, DeepLabV3+ segmentation
│   │   ├── reconstruction/      # Diffusion, GAN, Temporal Transformer
│   │   ├── fusion/              # SAR encoder + Cross-Attention fusion layer
│   │   └── verification/        # AI self-checker (secondary model)
│   ├── losses/                  # Spectral, physical, temporal consistency losses
│   ├── training/                # Trainer, callbacks, schedulers
│   ├── inference/               # End-to-end inference pipeline
│   ├── utils/                   # Metrics, GeoTIFF I/O, index computation
│   └── visualization/           # Confidence maps, validation report generation
├── configs/                     # YAML configs for each experiment
├── scripts/                     # CLI entry points (train, infer, validate, export)
├── tests/                       # Unit + integration tests
├── notebooks/                   # EDA, result analysis
├── outputs/
│   ├── checkpoints/             # Saved model weights
│   ├── logs/                    # TensorBoard / W&B logs
│   ├── predictions/             # Inference outputs (GeoTIFF)
│   └── reports/                 # Validation reports (PDF/JSON)
└── docs/                        # Architecture docs, API reference
```

## Quick Start

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Set up configuration
cp configs/base.yaml configs/my_experiment.yaml

# 3. Prepare data (runs cloud masking + temporal compositing)
python scripts/prepare_data.py --config configs/my_experiment.yaml

# 4. Train cloud detection (L2)
python scripts/train.py --stage cloud_detection --config configs/my_experiment.yaml

# 5. Train reconstruction (L4)
python scripts/train.py --stage reconstruction --config configs/my_experiment.yaml

# 6. Run full inference pipeline (L1–L7)
python scripts/infer.py --input data/raw/optical/scene.tif --config configs/my_experiment.yaml

# 7. Generate validation report
python scripts/validate.py --prediction outputs/predictions/scene_reconstructed.tif
```

## Spectral Bands

| Band   | Wavelength     | Use                              |
|--------|----------------|----------------------------------|
| Green  | 0.52–0.59 µm   | Vegetation, water clarity        |
| Red    | 0.62–0.68 µm   | Vegetation health (NDVI)         |
| NIR    | 0.77–0.86 µm   | Biomass, vegetation stress       |
| SWIR   | 1.55–1.75 µm   | Moisture, geology, cloud shadow  |

## Spectral Indices

| Index  | Formula                        | Purpose                    |
|--------|--------------------------------|----------------------------|
| NDVI   | (NIR-Red)/(NIR+Red)            | Vegetation health          |
| NDWI   | (Green-NIR)/(Green+NIR)        | Water bodies               |
| SAVI   | 1.5*(NIR-Red)/(NIR+Red+0.5)    | Soil-adjusted vegetation   |
| NDBI   | (SWIR-NIR)/(SWIR+NIR)          | Built-up / urban index     |

## Validation Thresholds

| Metric                   | Target        |
|--------------------------|---------------|
| SSIM                     | > 0.90        |
| PSNR                     | > 35 dB       |
| RMSE                     | < 0.03        |
| NDVI Error               | < 5%          |
| NDWI Error               | < 5%          |
| Spectral Angle Mapper    | < 0.10 rad    |
| Temporal Consistency     | > 0.85        |
