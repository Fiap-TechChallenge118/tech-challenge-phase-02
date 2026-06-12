.DEFAULT_GOAL := help

.PHONY: help setup lint lint-fix format check test

help: ## Mostra os comandos disponíveis
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-15s\033[0m %s\n", $$1, $$2}'

setup: ## Configura o ambiente do zero (rodar uma vez ao clonar)
	uv sync
	cp -n .env.example .env || true
	@echo "✅ Setup completo. Edite o .env com seus valores."

lint: ## Verifica erros de linting (sem corrigir)
	uv run ruff check .

lint-fix: ## Corrige erros de linting automaticamente
	uv run ruff check . --fix

format: ## Formata o código
	uv run ruff format .

check: ## Lint + format check (usado no CI)
	uv run ruff check .
	uv run ruff format . --check
