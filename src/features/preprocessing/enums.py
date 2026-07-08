"""Enums do módulo de pré-processamento.

Módulo sem dependências internas de ``src`` para evitar importações circulares.
Pode ser importado com segurança por ``src.settings`` antes que o restante
do pacote esteja inicializado.
"""

from enum import StrEnum


class PreprocessorStrategyName(StrEnum):
    """Nomes válidos de estratégias registradas no ``PreprocessorFactory``.

    Usar este enum evita erros de typo e garante validação em tempo de parse
    pelo Pydantic Settings.

    Exemplo::

        from src.features.preprocessing.enums import PreprocessorStrategyName

        name = PreprocessorStrategyName.INTERACTION_PAIRS
        # name.value == "interaction_pairs"
    """

    INTERACTION_PAIRS = "interaction_pairs"
    USER_ITEM_MATRIX = "user_item_matrix"
    AGGREGATED_FEATURES = "aggregated_features"
