"""Pré-processadores intercambiáveis (padrão Strategy) para o dataset Instacart.

Exporta a interface, as estratégias concretas e a factory de seleção por nome.
"""

from src.features.preprocessing.aggregated_features import AggregatedFeaturesStrategy
from src.features.preprocessing.base import (
    InstacartFrames,
    NotFittedError,
    PreprocessorStrategy,
)
from src.features.preprocessing.enums import PreprocessorStrategyName
from src.features.preprocessing.factory import PreprocessorFactory
from src.features.preprocessing.interaction_pairs import InteractionPairsStrategy
from src.features.preprocessing.user_item_matrix import (
    UserItemMatrix,
    UserItemMatrixStrategy,
)

__all__ = [
    "AggregatedFeaturesStrategy",
    "InstacartFrames",
    "InteractionPairsStrategy",
    "NotFittedError",
    "PreprocessorFactory",
    "PreprocessorStrategy",
    "PreprocessorStrategyName",
    "UserItemMatrix",
    "UserItemMatrixStrategy",
]
