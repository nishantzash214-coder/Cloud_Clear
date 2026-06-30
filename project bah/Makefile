# ============================================================
#  Makefile — Cloud Removal Framework
# ============================================================

PYTHON   = python
CONFIG   = configs/base.yaml
DEVICE   = cuda

.PHONY: help install prepare train-detection train-reconstruction infer validate test docker-build docker-up clean

help:
	@echo ""
	@echo "  AI-Powered Cloud Removal Framework"
	@echo "  ─────────────────────────────────"
	@echo "  make install              Install all dependencies"
	@echo "  make prepare              Prepare training data patches"
	@echo "  make train-detection      Train cloud detection model (L2)"
	@echo "  make train-reconstruction Train reconstruction pipeline (L4)"
	@echo "  make infer SCENE=<path>   Run full inference on a scene"
	@echo "  make validate             Validate latest prediction"
	@echo "  make test                 Run all tests"
	@echo "  make test-unit            Run unit tests only"
	@echo "  make test-integration     Run integration tests only"
	@echo "  make docker-build         Build Docker images"
	@echo "  make docker-up            Start API server via Docker Compose"
	@echo "  make clean                Remove outputs and cache"
	@echo ""

install:
	pip install -r requirements.txt
	pip install -e .

prepare:
	$(PYTHON) scripts/prepare_data.py --config $(CONFIG) --workers 8

train-detection:
	$(PYTHON) scripts/train.py \
		--stage cloud_detection \
		--config $(CONFIG)

train-reconstruction:
	$(PYTHON) scripts/train.py \
		--stage reconstruction \
		--config $(CONFIG)

train-all: train-detection train-reconstruction

infer:
	$(PYTHON) scripts/infer.py \
		--input  $(SCENE) \
		--config $(CONFIG) \
		--output outputs/predictions/

validate:
	$(PYTHON) scripts/validate.py \
		--prediction outputs/predictions/scene_reconstructed.tif \
		--reference  data/archive/scene_cloudfree.tif \
		--output-dir outputs/reports/

test:
	pytest tests/ -v --cov=src --cov-report=term-missing

test-unit:
	pytest tests/unit/ -v

test-integration:
	pytest tests/integration/ -v

debug-train:
	$(PYTHON) scripts/train.py \
		--stage reconstruction \
		--config $(CONFIG) \
		--debug

docker-build:
	docker compose build

docker-up:
	docker compose up api postgis

docker-train:
	docker compose --profile training run trainer

clean:
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null; true
	find . -name "*.pyc" -delete
	rm -rf outputs/predictions/* outputs/logs/* .pytest_cache/
	@echo "Cleaned."

# ── Earth Engine authentication ───────────────────────────────
ee-auth:
	earthengine authenticate

# ── Quick smoke test (no real data needed) ───────────────────
smoke-test:
	$(PYTHON) -c "\
import torch; \
from src.models.cloud_detection.unetplusplus import CloudDetector; \
m = CloudDetector(pretrained=False).eval(); \
x = torch.rand(1,4,256,256); \
out = m(x); \
print('Cloud detector OK —', out['class_map'].shape); \
from src.models.reconstruction.pipeline import ReconstructionPipeline; \
print('All imports OK'); \
"
