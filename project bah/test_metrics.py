"""
tests/unit/test_metrics.py
"""
import torch
import pytest
from src.utils.indices import ndvi, ndwi, savi, ndbi, index_error
from src.utils.metrics import rmse, psnr, ssim, spectral_angle_mapper, compute_all_metrics


class TestSpectralIndices:
    def test_ndvi_range(self, optical_batch):
        out = ndvi(optical_batch)
        assert out.shape == (2, 1, 256, 256)
        assert out.min() >= -1.0 and out.max() <= 1.0

    def test_ndwi_range(self, optical_batch):
        out = ndwi(optical_batch)
        assert out.min() >= -1.0 and out.max() <= 1.0

    def test_index_error_identical(self, optical_batch):
        errors = index_error(optical_batch, optical_batch)
        for name, result in errors.items():
            assert result["mae"] < 1e-5, f"{name} error should be ~0 for identical inputs"
            assert result["pass"] is True

    def test_index_error_threshold(self, optical_batch):
        noisy = optical_batch + torch.randn_like(optical_batch) * 0.5
        errors = index_error(optical_batch, noisy)
        # With high noise, at least some indices should fail the 5% threshold
        assert any(not v["pass"] for v in errors.values())


class TestMetrics:
    def test_rmse_identical(self, optical_batch):
        assert rmse(optical_batch, optical_batch) < 1e-6

    def test_psnr_identical(self, optical_batch):
        assert psnr(optical_batch, optical_batch) == float("inf")

    def test_psnr_noisy(self, optical_batch):
        noisy = (optical_batch + torch.randn_like(optical_batch) * 0.01).clamp(0, 1)
        score = psnr(optical_batch, noisy)
        assert score > 30.0  # should be decent with low noise

    def test_ssim_identical(self, optical_batch):
        score = ssim(optical_batch, optical_batch)
        assert score > 0.99

    def test_ssim_range(self, optical_batch):
        noisy = (optical_batch + torch.randn_like(optical_batch) * 0.2).clamp(0, 1)
        score = ssim(optical_batch, noisy)
        assert 0.0 <= score <= 1.0

    def test_sam_identical(self, optical_batch):
        score = spectral_angle_mapper(optical_batch, optical_batch)
        assert score < 1e-4  # should be ~0

    def test_sam_range(self, optical_batch):
        noisy = (optical_batch + 0.3).clamp(0, 1)
        score = spectral_angle_mapper(optical_batch, noisy)
        assert 0.0 <= score <= torch.pi / 2

    def test_compute_all_metrics(self, optical_batch, temporal_stack, cloud_mask):
        result = compute_all_metrics(optical_batch, optical_batch,
                                     temporal_stack, (cloud_mask == 0).float())
        assert "ssim" in result
        assert "psnr" in result
        assert "rmse" in result
        assert "sam" in result
        assert "temporal_consistency" in result
        assert "overall_pass" in result
        assert result["overall_pass"] is True  # identical pred/target should pass
