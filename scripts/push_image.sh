#!/usr/bin/env bash
# =============================================================================
# Build da imagem da API e push para o ECR.
#
# Pré-requisitos:
#   - AWS CLI autenticado (aws sts get-caller-identity)
#   - ECR já criado (terraform -chdir=infra/aws apply)
#
# Uso:
#   ./scripts/push_image.sh [tag]      # tag default: latest
# =============================================================================
set -euo pipefail

TAG="${1:-latest}"
REGION="${AWS_REGION:-us-east-1}"
SERVICE_NAME="${SERVICE_NAME:-tc02-api}"

ACCOUNT_ID="$(aws sts get-caller-identity --query Account --output text)"
ECR_URL="${ACCOUNT_ID}.dkr.ecr.${REGION}.amazonaws.com"
IMAGE="${ECR_URL}/${SERVICE_NAME}:${TAG}"

echo "→ Build da imagem (stage api)"
docker build --target api -t "${SERVICE_NAME}:${TAG}" .

echo "→ Login no ECR (${ECR_URL})"
aws ecr get-login-password --region "${REGION}" \
  | docker login --username AWS --password-stdin "${ECR_URL}"

echo "→ Push de ${IMAGE}"
docker tag "${SERVICE_NAME}:${TAG}" "${IMAGE}"
docker push "${IMAGE}"

echo "✅ Imagem publicada: ${IMAGE}"
echo "   Suba o serviço com:"
echo "   terraform -chdir=infra/aws apply -var deploy_service=true"
