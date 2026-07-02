"""Factory (registry) para seleção de estratégias de pré-processamento.

Combina os padrões Factory e Strategy: o cliente escolhe a representação por nome
(ex.: vindo de ``config.yaml``) sem conhecer as classes concretas. Registrar uma nova
estratégia não exige alterar o código cliente (princípio Aberto/Fechado).
"""

from src.features.preprocessing.aggregated_features import AggregatedFeaturesStrategy
from src.features.preprocessing.base import PreprocessorStrategy
from src.features.preprocessing.interaction_pairs import InteractionPairsStrategy
from src.features.preprocessing.user_item_matrix import UserItemMatrixStrategy


class PreprocessorFactory:
    """Cria estratégias de pré-processamento a partir de um nome registrado."""

    _registry: dict[str, type[PreprocessorStrategy]] = {
        "interaction_pairs": InteractionPairsStrategy,
        "user_item_matrix": UserItemMatrixStrategy,
        "aggregated_features": AggregatedFeaturesStrategy,
    }

    @classmethod
    def create(cls, name: str, **kwargs: object) -> PreprocessorStrategy:
        """Instancia a estratégia registrada sob ``name``.

        Args:
            name: Identificador da estratégia (ex.: ``"interaction_pairs"``).
            **kwargs: Argumentos repassados ao construtor da estratégia.

        Returns:
            A instância da estratégia solicitada.

        Raises:
            ValueError: Se ``name`` não estiver registrado.
        """
        try:
            strategy_cls = cls._registry[name]
        except KeyError:
            options = ", ".join(cls.available())
            msg = f"Estratégia {name!r} desconhecida. Disponíveis: {options}."
            raise ValueError(msg) from None
        return strategy_cls(**kwargs)  # type: ignore[arg-type]

    @classmethod
    def register(
        cls, name: str, strategy_cls: type[PreprocessorStrategy]
    ) -> None:
        """Registra uma nova estratégia sob ``name``.

        Args:
            name: Identificador da estratégia.
            strategy_cls: Classe concreta que herda de ``PreprocessorStrategy``.
        """
        cls._registry[name] = strategy_cls

    @classmethod
    def available(cls) -> list[str]:
        """Retorna os nomes de estratégias registradas, em ordem alfabética."""
        return sorted(cls._registry)
