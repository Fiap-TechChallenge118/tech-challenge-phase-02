"""Arquiteturas de redes neurais para recomendação (padrão Factory).

Exporta os modelos disponíveis e a factory de instanciação por nome a partir
de ``configs/config.yaml``.
"""

from src.models.factory import ModelFactory
from src.models.mlp import RecommendationMLP

__all__ = [
    "ModelFactory",
    "RecommendationMLP",
]
