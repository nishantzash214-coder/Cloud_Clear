# CloudClear — Deployment Runbook
### AI-Powered Scientific Cloud Removal Framework

---

## Prerequisites

| Tool         | Version  | Install |
|---|---|---|
| Terraform    | ≥ 1.5    | `brew install terraform` |
| AWS CLI      | ≥ 2.0    | `pip install awscli` |
| Docker       | ≥ 24     | docker.com/get-docker |
| kubectl      | ≥ 1.28   | `brew install kubectl` |
| Python       | 3.10+    | python.org |

---

## 1. First-Time Setup

### 1.1 AWS credentials
```bash
aws configure
# AWS Access Key ID:     <your key>
# AWS Secret Access Key: <your secret>
# Default region:        ap-south-1
# Default output format: json
```

### 1.2 Terraform state bucket (one-time)
```bash
aws s3 mb s3://cloud-removal-tf-state --region ap-south-1
aws s3api put-bucket-versioning \
    --bucket cloud-removal-tf-state \
    --versioning-configuration Status=Enabled
```

### 1.3 Deploy infrastructure
```bash
cd deploy/terraform/
terraform init
terraform plan -var="db_password=YOUR_SECURE_PASSWORD"
terraform apply -var="db_password=YOUR_SECURE_PASSWORD"
# Note the outputs: api_endpoint, ssh_command, s3_data_bucket
```

### 1.4 GitHub secrets (for CI/CD)
Set these in GitHub → Settings → Secrets:
```
AWS_ACCESS_KEY_ID       → your IAM key
AWS_SECRET_ACCESS_KEY   → your IAM secret
EC2_INSTANCE_ID         → from terraform output
EC2_PUBLIC_IP           → from terraform output
SLACK_WEBHOOK_URL       → optional
```

---

## 2. Initial Model Training

### 2.1 SSH into GPU instance
```bash
ssh -i ~/.ssh/id_rsa ubuntu@<EC2_PUBLIC_IP>
cd /app
```

### 2.2 Authenticate Earth Engine
```bash
earthengine authenticate
# Follow browser link, paste token
```

### 2.3 Download training data
```bash
# Single AOI (Bhopal) — start small
python scripts/download_data.py \
    --bbox  "77.2,23.1,77.6,23.5" \
    --date  "2024-07-01" \
    --output data/raw/

# Build archive (2022–2024)
python scripts/download_data.py \
    --bbox    "77.2,23.1,77.6,23.5" \
    --archive \
    --years   2022 2023 2024 \
    --output  data/
```

### 2.4 Prepare data patches
```bash
python scripts/prepare_data.py \
    --config  configs/base.yaml \
    --workers 8
# Expected: ~50,000 training patches from 100 scenes
```

### 2.5 Train cloud detection (Stage 1, ~4 hours on T4)
```bash
python scripts/train.py \
    --stage  cloud_detection \
    --config configs/base.yaml
# Monitor: wandb.ai or tensorboard --logdir outputs/logs
```

### 2.6 Train reconstruction (Stage 2, ~12 hours on T4)
```bash
python scripts/train.py \
    --stage  reconstruction \
    --config configs/base.yaml
```

### 2.7 Upload checkpoints to S3
```bash
aws s3 sync outputs/checkpoints/ \
    s3://$(terraform output -raw s3_model_bucket)/checkpoints/
```

---

## 3. Running Inference

### 3.1 Single scene (CLI)
```bash
python scripts/infer.py \
    --input   data/raw/optical/my_scene.tif \
    --sar     data/raw/sar/my_scene_sar.tif \
    --temporal data/raw/temporal/my_scene/ \
    --config  configs/base.yaml \
    --output  outputs/predictions/
```

### 3.2 Via API
```bash
curl -X POST http://localhost:8000/infer \
    -F "optical=@data/raw/optical/scene.tif" \
    -F "sar=@data/raw/sar/scene_sar.tif"
# Returns: {"job_id": "abc123", "status": "queued", ...}

# Poll status
curl http://localhost:8000/status/abc123

# Download result
curl -o reconstructed.tif \
    "http://localhost:8000/result/abc123?output=reconstructed"
```

### 3.3 Frontend
```bash
# Local
open frontend/index.html

# Serve via nginx (already configured on EC2)
# Access: http://<EC2_PUBLIC_IP>
```

---

## 4. Monitoring

### 4.1 GPU utilisation
```bash
# On EC2
nvidia-smi -l 5   # refresh every 5s

# CloudWatch dashboard
aws cloudwatch get-metric-statistics \
    --namespace AWS/EC2 \
    --metric-name GPUUtilization \
    --dimensions Name=InstanceId,Value=<INSTANCE_ID> \
    --start-time $(date -u -d '1 hour ago' +%FT%TZ) \
    --end-time   $(date -u +%FT%TZ) \
    --period 300 --statistics Average
```

### 4.2 API health
```bash
curl http://localhost:8000/health
curl http://localhost:8000/metrics
```

### 4.3 Database
```bash
psql $DATABASE_URL -c "SELECT * FROM v_monthly_performance;"
psql $DATABASE_URL -c "SELECT * FROM v_job_summary LIMIT 20;"
```

### 4.4 Docker logs
```bash
docker logs cloudclear-api -f --tail 200
```

---

## 5. Scaling

### 5.1 Upgrade instance type (more GPU VRAM)
```bash
# Edit deploy/terraform/main.tf
# Change: variable "gpu_instance" { default = "g4dn.2xlarge" }
terraform apply -var="db_password=..."
```

### 5.2 Multiple GPUs (multi-scene parallel processing)
```bash
# In configs/base.yaml
hardware:
  gpus: 4
  precision: bf16  # on A100
```

### 5.3 Spot instances (80% cost reduction)
```bash
# Add to aws_instance resource in main.tf:
instance_market_options {
  market_type = "spot"
  spot_options {
    max_price = "0.50"   # max $/hr
    spot_instance_type = "persistent"
  }
}
```

---

## 6. Costs (ap-south-1 Mumbai)

| Resource           | Type          | Est. Cost/month |
|---|---|---|
| EC2 g4dn.xlarge    | GPU inference | ~$180           |
| RDS db.t3.medium   | PostGIS       | ~$55            |
| S3 (500 GB)        | Data storage  | ~$12            |
| Data transfer      | Outputs       | ~$10            |
| **Total**          |               | **~$257/month** |

With spot instances: **~$95/month**

---

## 7. Maintenance

### Weekly
```bash
# Sync outputs to S3
aws s3 sync outputs/predictions/ s3://$S3_BUCKET/outputs/

# Check disk usage
df -h /app/
docker system df
```

### Monthly
```bash
# Retrain with new data
python scripts/download_data.py --archive --years $(date +%Y) --months $(date +%m)
python scripts/prepare_data.py --config configs/base.yaml
python scripts/train.py --stage reconstruction --resume outputs/checkpoints/latest.ckpt
```

### Backup
```bash
# Database
pg_dump $DATABASE_URL | gzip > backup_$(date +%Y%m%d).sql.gz
aws s3 cp backup_$(date +%Y%m%d).sql.gz s3://$S3_BUCKET/backups/

# Checkpoints (auto via cron, manual:)
aws s3 sync outputs/checkpoints/ s3://$S3_BUCKET/checkpoints/
```

---

## 8. Troubleshooting

### GPU out of memory
```bash
# Reduce batch size in configs/base.yaml
reconstruction:
  training:
    batch_size: 4   # from 8

# Or use gradient checkpointing
hardware:
  precision: 16-mixed
```

### EE quota exceeded
```bash
# Use smaller AOI or fewer dates
# Or switch to offline mode with pre-downloaded data
```

### API not starting
```bash
docker logs cloudclear-api --tail 50
# Check GPU driver: nvidia-smi
# Check disk: df -h
# Check memory: free -h
```

### Verification failures (SSIM < 0.90)
```bash
# 1. Check cloud mask quality
python -c "
from src.models.cloud_detection.unetplusplus import CloudDetector
# ... inspect cloud_prob outputs
"
# 2. Verify temporal stack has sufficient clear scenes
# 3. Check SAR co-registration accuracy
# 4. Increase training epochs for reconstruction model
```
