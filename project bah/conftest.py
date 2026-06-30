import pytest
import torch
from omegaconf import OmegaConf

@pytest.fixture
def cfg():
    return OmegaConf.create({
        "project": {"name": "test", "seed": 42},
        "data": {
            "patch_size": 64,
            "overlap": 8,
            "temporal_window_days": 15,
            "normalization": "percentile",
            "percentile_clip": [2, 98],
            "bands": {"optical": ["green","red","nir","swir"], "sar": ["VV","VH"]},
            "split": {"train": 0.7, "val": 0.15, "test": 0.15},
            "augmentation": {
                "horizontal_flip": True,
                "vertical_flip": True,
                "rotation_90": True,
                "spectral_jitter": 0.05,
                "seasonal_sim": True,
            },
        },
        "cloud_detection": {
            "num_classes": 4,
            "dropout": 0.2,
            "model": "unetplusplus",
            "training": {
                "epochs": 2, "batch_size": 2, "lr": 1e-4,
                "weight_decay": 1e-4, "warmup_epochs": 1,
                "class_weights": [0.5,1.5,2.0,1.8],
            },
        },
        "temporal": {
            "composite_method": "median",
            "trend_indices": ["ndvi","ndwi"],
            "change_threshold": 0.15,
            "consistency_weight": 0.3,
        },
        "reconstruction": {
            "fusion": {
                "d_model": 64,
                "num_heads": 4,
                "num_layers": 2,
                "dropout": 0.1,
                "adaptive_weighting": True,
            },
            "diffusion": {"enabled": True},
            "gan": {"enabled": True, "discriminators": 2, "ngf": 32, "ndf": 32},
            "temporal_transformer": {
                "enabled": True,
                "temporal_depth": 2,
                "spatial_depth": 2,
                "num_heads": 4,
                "max_sequence_len": 5,
            },
            "sar_encoder": {"enabled": True},
            "training": {
                "epochs": 2, "batch_size": 2, "lr": 2e-4,
                "weight_decay": 1e-4, "gradient_clip": 1.0,
            },
        },
        "losses": {
            "pixel_l1": 1.0, "perceptual": 0.0,
            "spectral_consistency": 2.0, "physical_consistency": 1.5,
            "temporal_consistency": 1.0, "adversarial": 0.0,
        },
        "verification": {
            "spectral_error_threshold": 0.05,
            "sar_consistency_min": 0.70,
            "temporal_consistency_min": 0.80,
            "ai_checker": {"artifact_threshold": 0.15},
        },
        "confidence": {
            "mc_dropout_passes": 2,
            "high_confidence_threshold": 0.80,
            "low_confidence_threshold": 0.50,
        },
        "export": {"nodata": -9999, "crs": "EPSG:32643", "formats": ["geotiff"]},
        "paths": {
            "checkpoints": "/tmp/test_ckpts",
            "processed": "/tmp/test_data",
        },
        "hardware": {"gpus": 1, "precision": "32", "num_workers": 0, "pin_memory": False},
        "logging": {"logger": "csv", "log_every_n_steps": 1, "save_top_k": 1,
                    "monitor": "val/ssim", "mode": "max"},
    })

@pytest.fixture
def small_optical():
    """Small (B=1, 4, 64, 64) optical tensor."""
    return torch.rand(1, 4, 64, 64)

@pytest.fixture
def small_sar():
    return torch.rand(1, 4, 64, 64)

@pytest.fixture
def small_cloud_mask():
    """Cloud mask with ~40% coverage."""
    mask = torch.zeros(1, 64, 64, dtype=torch.long)
    mask[0, 20:45, 20:45] = 2   # thick cloud square
    mask[0, 10:20, 10:20] = 1   # thin cloud
    return mask

@pytest.fixture
def small_temporal():
    return torch.rand(1, 5, 4, 64, 64)

# Fixtures for metrics tests compatibility
@pytest.fixture
def optical_batch():
    return torch.rand(2, 4, 256, 256)

@pytest.fixture
def temporal_stack(optical_batch):
    # Make it consistent with optical_batch to pass temporal consistency check
    B, C, H, W = optical_batch.shape
    T = 5
    stack = optical_batch.unsqueeze(1).expand(-1, T, -1, -1, -1).clone()
    stack += torch.randn_like(stack) * 0.01
    return stack.clamp(0, 1)

@pytest.fixture
def cloud_mask():
    return torch.zeros(2, 256, 256, dtype=torch.long)
