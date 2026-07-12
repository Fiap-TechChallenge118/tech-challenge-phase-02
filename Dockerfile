# =============================================================================
# Dockerfile multi-stage — Tech Challenge 02
#
# Stage builder : instala dependências de produção com uv
# Stage runtime : imagem enxuta com apenas o necessário para executar o pipeline
#
# Build:
#   docker build -t tc02-trainer .
#
# Smoke test:
#   docker run --rm tc02-trainer python -c "import src; print('OK')"
# =============================================================================

# -----------------------------------------------------------------------------
# Stage 1 — builder
# Responsável por instalar as dependências de produção e gerar o virtualenv.
# -----------------------------------------------------------------------------
FROM python:3.12-slim AS builder

WORKDIR /app

# Instalar uv diretamente do binário oficial
COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv

# Copiar apenas os arquivos de dependências (cache de layer otimizado)
COPY pyproject.toml uv.lock ./

# Instalar dependências de produção (sem grupo dev) em /app/.venv
RUN uv sync --frozen --no-dev --no-install-project

# -----------------------------------------------------------------------------
# Stage 2 — runtime
# Imagem final enxuta: apenas o venv e o código-fonte da aplicação.
# -----------------------------------------------------------------------------
FROM python:3.12-slim AS runtime

WORKDIR /app

# Copiar o virtualenv gerado no builder
COPY --from=builder /app/.venv /app/.venv

# Copiar código-fonte e configurações
COPY src/ src/
COPY configs/ configs/
COPY scripts/ scripts/

# Garantir que o venv esteja no PATH
ENV PATH="/app/.venv/bin:$PATH"
ENV PYTHONPATH="/app"
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# Usuário não-root para segurança
RUN useradd --create-home --shell /bin/bash appuser
USER appuser

# Ponto de entrada padrão: executa o pipeline DVC completo
CMD ["python", "-m", "src.training.trainer"]
