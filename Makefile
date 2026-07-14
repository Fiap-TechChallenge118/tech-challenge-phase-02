.DEFAULT_GOAL := help

.PHONY: help setup install lint lint-fix format check test test-cov test-integration \
        validate ci repro preprocess train evaluate register download-data \
        dvc-pull dvc-push dvc-status dvc-dag mlflow-ui

DVC := uv run dvc
RUFF := uv run ruff
PYTEST := uv run pytest

# -----------------------------------------------------------------------------
# Ajuda
# -----------------------------------------------------------------------------
help: ## Mostra os comandos disponíveis
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | \
	  awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-18s\033[0m %s\n", $$1, $$2}'

# -----------------------------------------------------------------------------
# Ambiente
# -----------------------------------------------------------------------------
setup: ## Configura o ambiente do zero (rodar uma vez ao clonar)
	uv sync
	cp -n .env.example .env || true
	@echo "✅ Setup completo. Edite o .env com seus valores."

install: ## Instala/atualiza dependências via uv
	uv sync

# -----------------------------------------------------------------------------
# Qualidade de código
# -----------------------------------------------------------------------------
lint: ## Verifica erros de linting (sem corrigir)
	$(RUFF) check .

lint-fix: ## Corrige erros de linting automaticamente
	$(RUFF) check . --fix

format: ## Formata o código
	$(RUFF) format .

check: ## Lint + format check (usado no CI)
	$(RUFF) check .
	$(RUFF) format . --check

# -----------------------------------------------------------------------------
# Testes
# -----------------------------------------------------------------------------
test: ## Executa todos os testes unitários
	$(PYTEST) tests/ -v

test-cov: ## Executa testes com cobertura
	$(PYTEST) tests/ -v --cov-report=term-missing

test-unit: ## Executa testes com cobertura
	$(PYTEST) tests/ -v --tb=short

test-integration: ## Executa testes de integração (requer artefatos do DVC)
	$(PYTEST) tests/integration/ -v --tb=short

# -----------------------------------------------------------------------------
# Validação
# -----------------------------------------------------------------------------
validate: ## Valida o ambiente antes de rodar o pipeline
	uv run python scripts/validate_env.py

# -----------------------------------------------------------------------------
# CI — sequência completa executada pelo GitHub Actions
# -----------------------------------------------------------------------------
ci: check test-unit dvc-dag ## Sequência completa do CI: lint + testes + validação do DAG

# -----------------------------------------------------------------------------
# Pipeline DVC — execução completa e por stage
# -----------------------------------------------------------------------------
repro: ## Reproduz o pipeline DVC completo
	$(DVC) repro

preprocess: ## Executa apenas o stage de pré-processamento
	$(DVC) repro preprocess

train: ## Executa apenas o stage de treino
	$(DVC) repro train

evaluate: ## Executa apenas o stage de avaliação
	$(DVC) repro evaluate

register: ## Executa apenas o stage de registro no MLflow
	$(DVC) repro register

# -----------------------------------------------------------------------------
# Dados brutos — caminho público, sem AWS
# O Instacart é um dataset público do Kaggle: qualquer pessoa reproduz o pipeline
# sem credencial do remote DVC.
# -----------------------------------------------------------------------------
download-data: ## Baixa o dataset Instacart do Kaggle para data/raw/
	uv run python scripts/download_dataset.py

# -----------------------------------------------------------------------------
# DVC — dados e versionamento
# Uso com argumentos: make dvc-push ARGS="data/raw.dvc"
#                     make dvc-pull ARGS="data/processed models"
#
# Atenção: dados rastreados diretamente pelo DVC (ex: data/raw) devem ser
# referenciados pelo arquivo .dvc correspondente: data/raw.dvc
# Outputs de stages do dvc.yaml (ex: models, data/processed) usam o caminho normal.
# -----------------------------------------------------------------------------
dvc-pull: ## Baixa artefatos do remote DVC (ex: make dvc-pull ARGS="data/raw.dvc")
	$(DVC) pull $(ARGS)

dvc-push: ## Envia artefatos para o remote DVC (ex: make dvc-push ARGS="data/raw.dvc")
	$(DVC) push $(ARGS)

dvc-status: ## Exibe o status do pipeline e dos dados rastreados
	$(DVC) status

dvc-dag: ## Exibe o grafo de dependências do pipeline
	$(DVC) dag

# -----------------------------------------------------------------------------
# MLflow
# -----------------------------------------------------------------------------
mlflow-ui: ## Sobe a UI do MLflow (acesse http://localhost:5000)
	uv run mlflow ui --backend-store-uri mlflow.db
