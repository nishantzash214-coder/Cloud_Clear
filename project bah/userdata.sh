#!/bin/bash
# deploy/terraform/userdata.sh
# Bootstraps EC2 GPU instance with the cloud removal API

set -e
exec > >(tee /var/log/userdata.log) 2>&1
echo "=== CloudClear bootstrap started $(date) ==="

# ── System packages ───────────────────────────────────────────
apt-get update -y
apt-get install -y gdal-bin libgdal-dev postgresql-client docker.io awscli nginx

# ── Docker GPU support ────────────────────────────────────────
curl -fsSL https://nvidia-container-toolkit.github.io/libnvidia-container/gpgkey \
  | gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg
systemctl enable docker
systemctl start docker
usermod -aG docker ubuntu

# ── AWS configuration ─────────────────────────────────────────
aws configure set region ${aws_region}

# ── Pull model weights from S3 ────────────────────────────────
mkdir -p /app/outputs/checkpoints
aws s3 sync s3://${s3_bucket}/checkpoints/ /app/outputs/checkpoints/ || echo "No checkpoints yet"

# ── Login to ECR and pull image ───────────────────────────────
aws ecr get-login-password --region ${aws_region} \
  | docker login --username AWS --password-stdin ${ecr_registry}

docker pull ${ecr_registry}:latest || echo "No image yet — will use local build"

# ── Environment file ──────────────────────────────────────────
cat > /app/.env <<EOF
DATABASE_URL=postgresql://cr_user:${db_password}@${db_endpoint}/cloud_removal
S3_DATA_BUCKET=${s3_bucket}
AWS_DEFAULT_REGION=${aws_region}
CUDA_VISIBLE_DEVICES=0
EOF

# ── Run PostGIS schema ────────────────────────────────────────
# Wait for RDS to be available
for i in {1..20}; do
  pg_isready -h $(echo ${db_endpoint} | cut -d: -f1) -p 5432 && break
  echo "Waiting for DB... ($i)"
  sleep 10
done

PGPASSWORD=${db_password} psql \
  -h $(echo ${db_endpoint} | cut -d: -f1) \
  -U cr_user -d cloud_removal \
  -f /app/database/schema.sql \
  2>/dev/null || echo "Schema already applied"

# ── Nginx reverse proxy ───────────────────────────────────────
cat > /etc/nginx/sites-available/cloudclear <<'NGINX'
server {
    listen 80;
    client_max_body_size 2G;
    proxy_read_timeout 600;

    location / {
        proxy_pass         http://localhost:8000;
        proxy_set_header   Host $host;
        proxy_set_header   X-Real-IP $remote_addr;
        proxy_set_header   X-Forwarded-For $proxy_add_x_forwarded_for;
    }

    location /static/ {
        root /app/frontend;
    }
}
NGINX
ln -sf /etc/nginx/sites-available/cloudclear /etc/nginx/sites-enabled/
nginx -t && systemctl restart nginx

# ── Start API container ───────────────────────────────────────
docker run -d \
  --name cloudclear-api \
  --gpus all \
  --restart unless-stopped \
  -p 8000:8000 \
  --env-file /app/.env \
  -v /app/data:/app/data \
  -v /app/outputs:/app/outputs \
  -v /app/configs:/app/configs \
  ${ecr_registry}:latest

# ── Auto-sync outputs to S3 (every hour) ─────────────────────
cat > /etc/cron.hourly/sync-outputs <<'CRON'
#!/bin/bash
aws s3 sync /app/outputs/predictions/ s3://${s3_bucket}/outputs/ \
    --exclude "*.tmp" --quiet
CRON
chmod +x /etc/cron.hourly/sync-outputs

echo "=== Bootstrap complete $(date) ==="
echo "API running at http://$(curl -s http://169.254.169.254/latest/meta-data/public-ipv4):8000"
