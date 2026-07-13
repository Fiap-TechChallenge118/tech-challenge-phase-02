output "ecr_repository_url" {
  description = "URL do repositório ECR (destino do docker push)."
  value       = aws_ecr_repository.api.repository_url
}

output "api_url" {
  description = "URL pública da API (vazia enquanto deploy_service = false)."
  value       = var.deploy_service ? "http://${aws_eip.api[0].public_ip}:${var.api_port}" : ""
}

output "health_check" {
  description = "Comando para verificar se a API subiu."
  value = var.deploy_service ? (
    "curl http://${aws_eip.api[0].public_ip}:${var.api_port}/health"
  ) : "deploy_service = false"
}

output "example_request" {
  description = "Exemplo de chamada ao endpoint de recomendação."
  value = var.deploy_service ? (
    "curl 'http://${aws_eip.api[0].public_ip}:${var.api_port}/recommend/1?k=5'"
  ) : "deploy_service = false"
}
