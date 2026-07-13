# =============================================================================
# Deploy da API de recomendação — ECR + EC2 (Free Tier)
#
# Por que EC2 e não App Runner/Fargate: a conta usada está no plano gratuito da
# AWS, que não habilita App Runner (SubscriptionRequiredException) nem Fargate.
# Uma instância t3.micro é elegível ao Free Tier (750h/mês) e roda o mesmo
# container publicado no ECR, expondo a API em um IP público estável (EIP).
#
# Uso:
#   terraform init
#   terraform apply                                  # cria o ECR
#   ./scripts/push_image.sh                          # build + push da imagem
#   terraform apply -var deploy_service=true         # sobe a EC2 com a API
#   terraform output api_url
#   terraform destroy                                # remove tudo
# =============================================================================

terraform {
  required_version = ">= 1.5"

  required_providers {
    aws = {
      source  = "hashicorp/aws"
      version = "~> 5.0"
    }
  }
}

provider "aws" {
  region = var.aws_region

  default_tags {
    tags = {
      Project   = "tech-challenge-02"
      ManagedBy = "terraform"
    }
  }
}

data "aws_vpc" "default" {
  default = true
}

data "aws_subnets" "default" {
  filter {
    name   = "vpc-id"
    values = [data.aws_vpc.default.id]
  }
}

# Amazon Linux 2023 — traz suporte a container e é Free Tier elegível.
data "aws_ami" "al2023" {
  most_recent = true
  owners      = ["amazon"]

  filter {
    name   = "name"
    values = ["al2023-ami-2023.*-x86_64"]
  }
}

# -----------------------------------------------------------------------------
# Repositório de imagens
# -----------------------------------------------------------------------------
resource "aws_ecr_repository" "api" {
  name                 = var.service_name
  image_tag_mutability = "MUTABLE"
  force_delete         = true

  image_scanning_configuration {
    scan_on_push = true
  }
}

# Imagens sem tag ficam órfãs a cada novo push de `latest` e acumulam custo.
resource "aws_ecr_lifecycle_policy" "api" {
  repository = aws_ecr_repository.api.name

  policy = jsonencode({
    rules = [{
      rulePriority = 1
      description  = "Expira imagens sem tag apos 1 dia"
      selection = {
        tagStatus   = "untagged"
        countType   = "sinceImagePushed"
        countUnit   = "days"
        countNumber = 1
      }
      action = { type = "expire" }
    }]
  })
}

# -----------------------------------------------------------------------------
# IAM — a instância precisa apenas de leitura no ECR para puxar a imagem
# -----------------------------------------------------------------------------
resource "aws_iam_role" "ec2_ecr_read" {
  name = "${var.service_name}-ec2-ecr-read"

  assume_role_policy = jsonencode({
    Version = "2012-10-17"
    Statement = [{
      Effect    = "Allow"
      Action    = "sts:AssumeRole"
      Principal = { Service = "ec2.amazonaws.com" }
    }]
  })
}

resource "aws_iam_role_policy_attachment" "ecr_read_only" {
  role       = aws_iam_role.ec2_ecr_read.name
  policy_arn = "arn:aws:iam::aws:policy/AmazonEC2ContainerRegistryReadOnly"
}

resource "aws_iam_instance_profile" "api" {
  name = "${var.service_name}-profile"
  role = aws_iam_role.ec2_ecr_read.name
}

# -----------------------------------------------------------------------------
# Rede — expõe apenas a porta da API
# -----------------------------------------------------------------------------
resource "aws_security_group" "api" {
  name        = "${var.service_name}-sg"
  description = "Permite acesso HTTP publico a API de recomendacao"
  vpc_id      = data.aws_vpc.default.id

  ingress {
    description = "API de recomendacao"
    from_port   = var.api_port
    to_port     = var.api_port
    protocol    = "tcp"
    cidr_blocks = ["0.0.0.0/0"]
  }

  egress {
    description = "Saida liberada (pull do ECR)"
    from_port   = 0
    to_port     = 0
    protocol    = "-1"
    cidr_blocks = ["0.0.0.0/0"]
  }
}

# -----------------------------------------------------------------------------
# Instância que serve a API
# Criada apenas quando `deploy_service = true` (depois que a imagem existe).
# -----------------------------------------------------------------------------
resource "aws_instance" "api" {
  count = var.deploy_service ? 1 : 0

  ami                    = data.aws_ami.al2023.id
  instance_type          = var.instance_type
  subnet_id              = data.aws_subnets.default.ids[0]
  vpc_security_group_ids = [aws_security_group.api.id]
  iam_instance_profile   = aws_iam_instance_profile.api.name

  # A imagem tem ~1,5 GB e o PyTorch consome memória no import; o Free Tier
  # oferece 1 GB de RAM, então o swap evita OOM durante o startup do container.
  user_data = templatefile("${path.module}/user_data.sh.tftpl", {
    ecr_image   = "${aws_ecr_repository.api.repository_url}:${var.image_tag}"
    aws_region  = var.aws_region
    api_port    = var.api_port
    swap_size_m = 2048
  })

  # Recria a instância quando a imagem alvo muda.
  user_data_replace_on_change = true

  root_block_device {
    volume_size = 20 # GB — cabe a imagem de ~1,5 GB com folga (Free Tier: 30 GB)
    volume_type = "gp3"
  }

  tags = { Name = var.service_name }
}

# IP público estável: sem EIP, o IP muda a cada stop/start da instância.
resource "aws_eip" "api" {
  count = var.deploy_service ? 1 : 0

  instance = aws_instance.api[0].id
  domain   = "vpc"

  tags = { Name = "${var.service_name}-eip" }
}
