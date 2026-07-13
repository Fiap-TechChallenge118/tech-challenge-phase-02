variable "aws_region" {
  description = "Região AWS onde o ECR e a instância são criados."
  type        = string
  default     = "us-east-1"
}

variable "service_name" {
  description = "Nome do repositório ECR e prefixo dos recursos."
  type        = string
  default     = "tc02-api"
}

variable "image_tag" {
  description = "Tag da imagem no ECR que a instância deve servir."
  type        = string
  default     = "latest"
}

variable "deploy_service" {
  description = <<-EOT
    Sobe a instância EC2 que serve a API. Deixe `false` no primeiro `apply`
    (cria apenas o ECR, pois a instância falharia sem imagem publicada), envie a
    imagem com scripts/push_image.sh e então rode novamente com `true`.
  EOT
  type        = bool
  default     = false
}

variable "instance_type" {
  description = <<-EOT
    Tipo da instância. `t3.micro` é elegível ao Free Tier (750h/mês nos
    primeiros 12 meses). Se a API sofrer OOM, subir para `t3.small` (2 GB)
    — fora do Free Tier, ~US$ 15/mês.
  EOT
  type        = string
  default     = "t3.micro"
}

variable "api_port" {
  description = "Porta pública onde a API responde."
  type        = number
  default     = 8000
}
