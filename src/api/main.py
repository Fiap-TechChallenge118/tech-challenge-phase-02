"""API HTTP de recomendação de produtos.

Serve o modelo promovido a Production, embutido na imagem em ``models/model.pt``.

Endpoints:
    ``GET /health``            — liveness/readiness (usado pelo App Runner).
    ``GET /recommend/{user_id}`` — top-K produtos para um usuário.

Execução local::

    uv run --group api uvicorn src.api.main:app --reload
"""

from __future__ import annotations

import logging
import os
from contextlib import asynccontextmanager
from pathlib import Path
from typing import AsyncIterator

from fastapi import FastAPI, HTTPException, Query
from pydantic import BaseModel, Field

from src.api.model_service import ModelService

logger = logging.getLogger(__name__)

# Caminho do checkpoint dentro do container (sobrescrevível para rodar local).
MODEL_PATH = Path(os.getenv("MODEL_PATH", "models/model.pt"))

# Preenchido no startup e reutilizado por todas as requisições.
_service: ModelService | None = None


class RecommendationOut(BaseModel):
    """Um produto recomendado."""

    product_id: int = Field(..., description="product_id original do Instacart.")
    score: float = Field(..., description="Probabilidade prevista de compra.")


class RecommendationsOut(BaseModel):
    """Resposta do endpoint de recomendação."""

    user_id: int
    count: int
    recommendations: list[RecommendationOut]


class HealthOut(BaseModel):
    """Resposta do healthcheck."""

    status: str
    model_loaded: bool
    n_users: int
    n_items: int


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    """Carrega o modelo uma única vez, no startup do processo."""
    global _service
    logging.basicConfig(level=logging.INFO, format="%(levelname)-8s %(message)s")
    logger.info("Carregando modelo de %s ...", MODEL_PATH)
    _service = ModelService(MODEL_PATH)
    logger.info("API pronta: %d usuários, %d itens", _service.n_users, _service.n_items)
    yield
    _service = None


app = FastAPI(
    title="Tech Challenge 02 — API de Recomendação",
    description="Recomenda produtos do Instacart via MLP treinada em PyTorch.",
    version="1.0.0",
    lifespan=lifespan,
)


def _get_service() -> ModelService:
    """Retorna o serviço carregado.

    Raises:
        HTTPException: 503 se o modelo ainda não subiu.
    """
    if _service is None:
        raise HTTPException(status_code=503, detail="Modelo ainda não carregado.")
    return _service


@app.get("/health", response_model=HealthOut, tags=["infra"])
def health() -> HealthOut:
    """Reporta se o modelo está carregado e pronto para servir."""
    if _service is None:
        return HealthOut(status="starting", model_loaded=False, n_users=0, n_items=0)
    return HealthOut(
        status="ok",
        model_loaded=True,
        n_users=_service.n_users,
        n_items=_service.n_items,
    )


@app.get(
    "/recommend/{user_id}",
    response_model=RecommendationsOut,
    tags=["recomendação"],
)
def recommend(
    user_id: int,
    k: int = Query(default=10, ge=1, le=50, description="Número de recomendações."),
) -> RecommendationsOut:
    """Retorna os ``k`` produtos mais prováveis para o usuário.

    Args:
        user_id: ``user_id`` original do dataset Instacart.
        k: Quantidade de recomendações (1 a 50).

    Returns:
        Lista de produtos ordenada por score decrescente.

    Raises:
        HTTPException: 404 se o usuário não existir no vocabulário (cold start).
    """
    service = _get_service()
    try:
        results = service.recommend(user_id=user_id, k=k)
    except KeyError:
        raise HTTPException(
            status_code=404,
            detail=(
                f"Usuário {user_id} não existe no vocabulário do modelo "
                "(cold start não suportado)."
            ),
        ) from None

    return RecommendationsOut(
        user_id=user_id,
        count=len(results),
        recommendations=[
            RecommendationOut(product_id=r.product_id, score=r.score) for r in results
        ],
    )
