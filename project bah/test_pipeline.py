"""
tests/integration/test_pipeline.py

End-to-end integration tests for the full 7-layer pipeline.
Uses synthetic random data — no real satellite imagery required.

Run:
    pytest tests/integration/ -v
"""

import pytest
import torch
import numpy as np
from types import SimpleNamespace
from unittest.mock import MagicMock, patch
from omegaconf import OmegaConf


# ─────────────────────────────────────────────
# Minimal config for testing
# ─────────────────────────────────────────────

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


# ─────────────────────────────────────────────
# Fixtures
# ─────────────────────────────────────────────

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


# ─────────────────────────────────────────────
# Fast export mode
# ─────────────────────────────────────────────

def test_fast_export_skips_optional_outputs(tmp_path, cfg):
    from exporter import SceneExporter

    cfg.export.fast_mode = True
    cfg.export.generate_pdf = False
    cfg.export.write_indices = False
    cfg.export.write_branch_weights = False

    exporter = SceneExporter(cfg, tmp_path)
    reconstructed = torch.rand(1, 4, 8, 8)
    cloud_mask = torch.zeros(1, 8, 8, dtype=torch.long)
    conf_maps = SimpleNamespace(
        confidence=torch.rand(1, 1, 8, 8),
        uncertainty=torch.rand(1, 1, 8, 8),
        mc_variance=None,
        breakdown={"model_a": 0.8},
    )

    with patch("exporter.write_geotiff") as mock_write:
        paths = exporter.export(
            scene_name="scene",
            reconstructed=reconstructed,
            cloud_mask=cloud_mask,
            conf_maps=conf_maps,
            meta={"driver": "GTiff", "crs": "EPSG:32643", "transform": None},
            report={"overall_pass": True, "passes": []},
            generate_pdf=True,
        )

    assert "report_pdf" not in paths
    assert "ndvi" not in paths
    assert "branch_weights" not in paths
    assert mock_write.call_count == 4


# ─────────────────────────────────────────────
# L2: Cloud detection
# ─────────────────────────────────────────────

class TestCloudDetection:
    def test_unetplusplus_forward(self, small_optical):
        from src.models.cloud_detection.unetplusplus import CloudDetector
        model = CloudDetector(in_channels=4, num_classes=4, pretrained=False)
        model.eval()
        with torch.no_grad():
            out = model(small_optical)
        assert "logits"     in out
        assert "cloud_prob" in out
        assert "class_map"  in out
        assert out["logits"].shape    == (1, 4, 64, 64)
        assert out["cloud_prob"].shape == (1, 1, 64, 64)
        assert out["class_map"].shape  == (1, 64, 64)
        # class_map should be integers 0–3
        assert out["class_map"].min() >= 0
        assert out["class_map"].max() <= 3

    def test_cloud_prob_range(self, small_optical):
        from src.models.cloud_detection.unetplusplus import CloudDetector
        model = CloudDetector(pretrained=False)
        model.eval()
        with torch.no_grad():
            out = model(small_optical)
        prob = out["cloud_prob"]
        assert prob.min() >= 0.0
        assert prob.max() <= 1.0


# ─────────────────────────────────────────────
# L3: Temporal analysis
# ─────────────────────────────────────────────

class TestTemporalAnalysis:
    def test_temporal_analyzer(self, cfg, small_temporal, small_cloud_mask):
        from src.data.preprocessing.temporal_analysis import TemporalAnalyzer
        analyzer = TemporalAnalyzer(cfg)
        stack    = small_temporal[0]               # (5,4,64,64)
        masks    = small_cloud_mask.expand(5,-1,-1) # (5,64,64)
        result   = analyzer.analyze(stack, masks)

        assert "composite"   in result
        assert "ndvi_trend"  in result
        assert "change_prob" in result
        assert "consistency" in result
        assert result["composite"].shape   == (4, 64, 64)
        assert result["ndvi_trend"].shape  == (64, 64)
        assert result["change_prob"].shape == (64, 64)
        assert result["consistency"].min() >= 0.0
        assert result["consistency"].max() <= 1.0


# ─────────────────────────────────────────────
# L4: Reconstruction branches
# ─────────────────────────────────────────────

class TestSAREncoder:
    def test_sar_encoder_forward(self, small_sar):
        from src.models.fusion.sar_encoder import SARFusionEncoder
        enc = SARFusionEncoder(in_channels=4, out_channels=64, pretrained=False)
        enc.eval()
        with torch.no_grad():
            out = enc(small_sar)
        assert "features"  in out
        assert "edge_mask" in out
        assert out["features"].shape  == (1, 64, 64, 64)
        assert out["edge_mask"].shape == (1, 1, 64, 64)
        assert out["edge_mask"].min() >= 0.0
        assert out["edge_mask"].max() <= 1.0


class TestTemporalTransformer:
    def test_forward(self, small_temporal, small_cloud_mask):
        from src.models.reconstruction.temporal_transformer import TemporalTransformer
        model = TemporalTransformer(
            in_channels=4, embed_dim=32, patch_size=8,
            max_T=5, temporal_depth=1, spatial_depth=1, num_heads=4, img_size=64,
        )
        model.eval()
        masks = small_cloud_mask.expand(1, 5, -1, -1).float()
        with torch.no_grad():
            out = model(small_temporal, masks)
        assert out.shape == (1, 32, 64, 64)


class TestFusion:
    def test_cross_attention_fusion(self, cfg, small_optical):
        from src.models.fusion.cross_attention_fusion import (
            CrossAttentionFusionLayer, OpticalQueryEncoder
        )
        d = 64
        enc     = OpticalQueryEncoder(4, d)
        fusion  = CrossAttentionFusionLayer(d_model=d, num_heads=4, num_layers=2)
        q_feat  = enc(small_optical)
        branches = [torch.rand(1, d, 64, 64) for _ in range(4)]
        cond     = torch.rand(1, 3, 64, 64)
        cloud_m  = (small_optical[:, 0:1] > 0.5).float()
        out      = fusion(q_feat, branches, cond, cloud_m)
        assert "fused"          in out
        assert "branch_weights" in out
        assert out["fused"].shape          == (1, d, 64, 64)
        assert out["branch_weights"].shape == (1, 4, 64, 64)
        # Branch weights should sum to 1 per pixel
        w_sum = out["branch_weights"].sum(dim=1)
        assert torch.allclose(w_sum, torch.ones_like(w_sum), atol=1e-5)


# ─────────────────────────────────────────────
# Loss functions
# ─────────────────────────────────────────────

class TestLosses:
    def test_spectral_consistency_loss(self, small_optical, small_cloud_mask):
        from src.losses.scientific_losses import SpectralConsistencyLoss
        loss_fn = SpectralConsistencyLoss(threshold=0.05)
        cloud_b = (small_cloud_mask > 0).float().unsqueeze(1)
        result  = loss_fn(small_optical, small_optical, cloud_b)
        # Identical pred/target → loss should be ~0
        assert result["total"].item() < 1e-4

    def test_physical_consistency_loss(self, small_optical, small_cloud_mask):
        from src.losses.scientific_losses import PhysicalConsistencyLoss
        loss_fn = PhysicalConsistencyLoss()
        cloud_b = (small_cloud_mask > 0).float().unsqueeze(1)
        loss    = loss_fn(small_optical, cloud_b)
        assert loss.item() >= 0.0

    def test_temporal_consistency_loss(self, small_optical, small_cloud_mask):
        from src.losses.scientific_losses import TemporalConsistencyLoss
        loss_fn   = TemporalConsistencyLoss(threshold=0.10)
        composite = small_optical.clone()
        cloud_b   = (small_cloud_mask > 0).float().unsqueeze(1)
        loss      = loss_fn(small_optical, composite, cloud_b)
        # Same pred as composite → loss should be ~0
        assert loss.item() < 1e-4


# ─────────────────────────────────────────────
# L5: Verification
# ─────────────────────────────────────────────

class TestVerifier:
    def test_all_passes(self, cfg, small_optical, small_sar, small_cloud_mask, small_temporal):
        from src.models.verification.verifier import MultiPassVerifier
        verifier = MultiPassVerifier(cfg)
        composite = small_optical.clone()
        temporal_ctx = {
            "composite": composite[0],
            "stack": small_temporal[0],
        }
        report = verifier.verify(
            pred       = small_optical,
            original   = small_optical,
            cloud_mask = small_cloud_mask,
            sar        = small_sar,
            temporal   = temporal_ctx,
        )
        assert "passes"       in report
        assert "overall_pass" in report
        assert "confidence"   in report
        assert len(report["passes"]) == 5
        for p in report["passes"]:
            assert "id"        in p
            assert "name"      in p
            assert "passed"    in p
            assert "score"     in p
            assert "threshold" in p


# ─────────────────────────────────────────────
# Synthetic cloud generator
# ─────────────────────────────────────────────

class TestSyntheticClouds:
    def test_augmented_patch_shapes(self):
        from src.data.augmentation.synthetic_clouds import SyntheticCloudGenerator
        gen    = SyntheticCloudGenerator(copies_per_patch=2)
        clear  = np.random.rand(4, 64, 64).astype(np.float32)
        result = gen.augment([clear])

        assert len(result) == 2    # 2 copies
        for patch in result:
            assert patch.cloudy.shape     == (4, 64, 64)
            assert patch.clear.shape      == (4, 64, 64)
            assert patch.cloud_mask.shape == (64, 64)
            assert patch.cloudy.min()  >= 0.0
            assert patch.cloudy.max()  <= 1.0
            assert set(np.unique(patch.cloud_mask)).issubset({0, 1, 2, 3})

    def test_thin_cloud_transmittance(self):
        """Thin clouds should be semi-transparent — surface still visible."""
        from src.data.augmentation.synthetic_clouds import SyntheticCloudGenerator
        gen   = SyntheticCloudGenerator()
        clear = np.ones((4, 64, 64), dtype=np.float32) * 0.3  # uniform surface
        cloudy, mask = gen._thin_cloud(clear)
        # Thin cloud pixels should be brighter than clear surface (cloud adds signal)
        thin_px = mask == 1
        if thin_px.any():
            # Mean cloudy value under thin cloud should be higher than clear (0.3)
            assert cloudy[:, thin_px].mean() > 0.30

    @pytest.mark.parametrize("cloud_type", ["thick", "thin", "mixed"])
    def test_all_cloud_types(self, cloud_type):
        from src.data.augmentation.synthetic_clouds import SyntheticCloudGenerator
        gen   = SyntheticCloudGenerator()
        clear = np.random.rand(4, 64, 64).astype(np.float32)
        cloudy, mask = gen._apply(clear, cloud_type)
        assert cloudy.shape == (4, 64, 64)
        assert mask.shape   == (64, 64)
        assert cloudy.min() >= 0.0
        assert cloudy.max() <= 1.0
