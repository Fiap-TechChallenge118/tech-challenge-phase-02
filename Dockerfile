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
# Stage 2 — api
# Serve o modelo promovido a Production via HTTP (FastAPI + uvicorn).
# O checkpoint é embutido na imagem: o container sobe autossuficiente, sem
# depender do MLflow ou do remote DVC em runtime.
#
# Precisa de --target: o stage default do arquivo é o `runtime` (treino).
#
# Build:
#   docker build --target api -t tc02-api .
#
# Execução local:
#   docker run --rm -p 8000:8000 tc02-api
#   curl http://localhost:8000/health
#   curl "http://localhost:8000/recommend/1?k=5"
# -----------------------------------------------------------------------------
FROM python:3.12-slim AS api

WORKDIR /app

COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv
COPY pyproject.toml uv.lock ./

# Dependências de produção + grupo `api` (FastAPI/uvicorn), sem o grupo dev.
RUN uv sync --frozen --no-dev --no-install-project --group api

COPY src/ src/
COPY configs/ configs/

# Modelo treinado (versionado via DVC) embutido na imagem.
COPY models/model.pt models/model.pt

ENV PATH="/app/.venv/bin:$PATH"
ENV PYTHONPATH="/app"
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV MODEL_PATH="/app/models/model.pt"
ENV PORT=8000

RUN useradd --create-home --shell /bin/bash appuser
USER appuser

EXPOSE 8000

CMD ["sh", "-c", "uvicorn src.api.main:app --host 0.0.0.0 --port ${PORT}"]

# -----------------------------------------------------------------------------
# Stage 3 — runtime (DEFAULT)
# Imagem de treino: apenas o venv e o código-fonte da aplicação.
#
# É o último stage de propósito — `docker build .` sem --target produz esta
# imagem, que é a usada pelo docker-compose para rodar o pipeline.
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

# Ponto de entrada padrão: executa o pipeline de treino
CMD ["python", "-m", "src.training.trainer"]
