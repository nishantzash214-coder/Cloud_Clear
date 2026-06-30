# ============================================================
#  deploy/terraform/main.tf
#  AWS infrastructure for Cloud Removal Framework
#  Resources: VPC, EC2 GPU, S3, RDS PostGIS, ECR, ALB
# ============================================================

terraform {
  required_version = ">= 1.5"
  required_providers {
    aws = {
      source  = "hashicorp/aws"
      version = "~> 5.0"
    }
  }
  backend "s3" {
    bucket = "cloud-removal-tf-state"
    key    = "prod/terraform.tfstate"
    region = "ap-south-1"
  }
}

provider "aws" {
  region = var.aws_region
}

# ── Variables ─────────────────────────────────────────────────
variable "aws_region"     { default = "ap-south-1" }   # Mumbai — closest to India
variable "project_name"   { default = "cloud-removal" }
variable "environment"    { default = "production" }
variable "gpu_instance"   { default = "g4dn.xlarge" }   # T4 GPU, 16GB VRAM
variable "db_password"    { sensitive = true }
variable "allowed_cidr"   { default = "0.0.0.0/0" }

# ── VPC ───────────────────────────────────────────────────────
resource "aws_vpc" "main" {
  cidr_block           = "10.0.0.0/16"
  enable_dns_hostnames = true
  enable_dns_support   = true
  tags = { Name = "${var.project_name}-vpc" }
}

resource "aws_subnet" "public" {
  count                   = 2
  vpc_id                  = aws_vpc.main.id
  cidr_block              = "10.0.${count.index}.0/24"
  availability_zone       = data.aws_availability_zones.available.names[count.index]
  map_public_ip_on_launch = true
  tags = { Name = "${var.project_name}-public-${count.index}" }
}

resource "aws_subnet" "private" {
  count             = 2
  vpc_id            = aws_vpc.main.id
  cidr_block        = "10.0.${count.index + 10}.0/24"
  availability_zone = data.aws_availability_zones.available.names[count.index]
  tags = { Name = "${var.project_name}-private-${count.index}" }
}

resource "aws_internet_gateway" "igw" {
  vpc_id = aws_vpc.main.id
  tags   = { Name = "${var.project_name}-igw" }
}

resource "aws_route_table" "public" {
  vpc_id = aws_vpc.main.id
  route {
    cidr_block = "0.0.0.0/0"
    gateway_id = aws_internet_gateway.igw.id
  }
}

resource "aws_route_table_association" "public" {
  count          = 2
  subnet_id      = aws_subnet.public[count.index].id
  route_table_id = aws_route_table.public.id
}

data "aws_availability_zones" "available" { state = "available" }

# ── Security Groups ────────────────────────────────────────────
resource "aws_security_group" "api" {
  name   = "${var.project_name}-api-sg"
  vpc_id = aws_vpc.main.id

  ingress {
    from_port   = 8000
    to_port     = 8000
    protocol    = "tcp"
    cidr_blocks = [var.allowed_cidr]
    description = "FastAPI"
  }
  ingress {
    from_port   = 22
    to_port     = 22
    protocol    = "tcp"
    cidr_blocks = [var.allowed_cidr]
    description = "SSH"
  }
  egress {
    from_port   = 0
    to_port     = 0
    protocol    = "-1"
    cidr_blocks = ["0.0.0.0/0"]
  }
  tags = { Name = "${var.project_name}-api-sg" }
}

resource "aws_security_group" "rds" {
  name   = "${var.project_name}-rds-sg"
  vpc_id = aws_vpc.main.id
  ingress {
    from_port       = 5432
    to_port         = 5432
    protocol        = "tcp"
    security_groups = [aws_security_group.api.id]
    description     = "PostGIS from API"
  }
  tags = { Name = "${var.project_name}-rds-sg" }
}

# ── EC2 GPU Instance ──────────────────────────────────────────
data "aws_ami" "deep_learning" {
  most_recent = true
  owners      = ["amazon"]
  filter {
    name   = "name"
    values = ["Deep Learning OSS Nvidia Driver AMI GPU PyTorch*Ubuntu*"]
  }
}

resource "aws_iam_role" "ec2" {
  name = "${var.project_name}-ec2-role"
  assume_role_policy = jsonencode({
    Version = "2012-10-17"
    Statement = [{
      Action    = "sts:AssumeRole"
      Effect    = "Allow"
      Principal = { Service = "ec2.amazonaws.com" }
    }]
  })
}

resource "aws_iam_role_policy_attachment" "s3" {
  role       = aws_iam_role.ec2.name
  policy_arn = "arn:aws:iam::aws:policy/AmazonS3FullAccess"
}

resource "aws_iam_role_policy_attachment" "ecr" {
  role       = aws_iam_role.ec2.name
  policy_arn = "arn:aws:iam::aws:policy/AmazonEC2ContainerRegistryReadOnly"
}

resource "aws_iam_instance_profile" "ec2" {
  name = "${var.project_name}-ec2-profile"
  role = aws_iam_role.ec2.name
}

resource "aws_instance" "gpu" {
  ami                    = data.aws_ami.deep_learning.id
  instance_type          = var.gpu_instance
  subnet_id              = aws_subnet.public[0].id
  vpc_security_group_ids = [aws_security_group.api.id]
  iam_instance_profile   = aws_iam_instance_profile.ec2.name
  key_name               = aws_key_pair.deploy.key_name

  root_block_device {
    volume_size = 100
    volume_type = "gp3"
    iops        = 3000
  }

  user_data = base64encode(templatefile("${path.module}/userdata.sh", {
    s3_bucket    = aws_s3_bucket.data.bucket
    db_endpoint  = aws_db_instance.postgis.endpoint
    db_password  = var.db_password
    ecr_registry = aws_ecr_repository.api.repository_url
    aws_region   = var.aws_region
  }))

  tags = {
    Name        = "${var.project_name}-gpu"
    Environment = var.environment
  }
}

resource "aws_key_pair" "deploy" {
  key_name   = "${var.project_name}-key"
  public_key = file("~/.ssh/id_rsa.pub")
}

resource "aws_eip" "gpu" {
  instance = aws_instance.gpu.id
  domain   = "vpc"
  tags     = { Name = "${var.project_name}-eip" }
}

# ── S3 Buckets ────────────────────────────────────────────────
resource "aws_s3_bucket" "data" {
  bucket        = "${var.project_name}-data-${random_id.suffix.hex}"
  force_destroy = false
  tags          = { Name = "${var.project_name}-data" }
}

resource "aws_s3_bucket" "models" {
  bucket        = "${var.project_name}-models-${random_id.suffix.hex}"
  force_destroy = false
  tags          = { Name = "${var.project_name}-models" }
}

resource "aws_s3_bucket_versioning" "models" {
  bucket = aws_s3_bucket.models.id
  versioning_configuration { status = "Enabled" }
}

resource "aws_s3_bucket_lifecycle_configuration" "data" {
  bucket = aws_s3_bucket.data.id
  rule {
    id     = "archive-old-outputs"
    status = "Enabled"
    filter { prefix = "outputs/" }
    transition {
      days          = 30
      storage_class = "STANDARD_IA"
    }
    transition {
      days          = 90
      storage_class = "GLACIER"
    }
  }
}

resource "random_id" "suffix" { byte_length = 4 }

# ── RDS PostGIS ───────────────────────────────────────────────
resource "aws_db_subnet_group" "postgis" {
  name       = "${var.project_name}-db-subnet"
  subnet_ids = aws_subnet.private[*].id
}

resource "aws_db_instance" "postgis" {
  identifier              = "${var.project_name}-postgis"
  engine                  = "postgres"
  engine_version          = "16.1"
  instance_class          = "db.t3.medium"
  allocated_storage       = 50
  max_allocated_storage   = 200
  storage_type            = "gp3"
  db_name                 = "cloud_removal"
  username                = "cr_user"
  password                = var.db_password
  db_subnet_group_name    = aws_db_subnet_group.postgis.name
  vpc_security_group_ids  = [aws_security_group.rds.id]
  backup_retention_period = 7
  deletion_protection     = true
  skip_final_snapshot     = false
  final_snapshot_identifier = "${var.project_name}-final"

  tags = { Name = "${var.project_name}-postgis" }
}

# ── ECR ───────────────────────────────────────────────────────
resource "aws_ecr_repository" "api" {
  name                 = "${var.project_name}/api"
  image_tag_mutability = "MUTABLE"
  image_scanning_configuration { scan_on_push = true }
}

# ── CloudWatch monitoring ─────────────────────────────────────
resource "aws_cloudwatch_metric_alarm" "gpu_util" {
  alarm_name          = "${var.project_name}-gpu-util-low"
  comparison_operator = "LessThanThreshold"
  evaluation_periods  = 2
  metric_name         = "GPUUtilization"
  namespace           = "AWS/EC2"
  period              = 300
  statistic           = "Average"
  threshold           = 5
  alarm_description   = "GPU utilisation below 5% for 10 min — consider stopping"
  dimensions          = { InstanceId = aws_instance.gpu.id }
}

# ── Outputs ───────────────────────────────────────────────────
output "api_endpoint"   { value = "http://${aws_eip.gpu.public_ip}:8000" }
output "ssh_command"    { value = "ssh -i ~/.ssh/id_rsa ubuntu@${aws_eip.gpu.public_ip}" }
output "s3_data_bucket" { value = aws_s3_bucket.data.bucket }
output "s3_model_bucket"{ value = aws_s3_bucket.models.bucket }
output "db_endpoint"    { value = aws_db_instance.postgis.endpoint }
output "ecr_registry"   { value = aws_ecr_repository.api.repository_url }
