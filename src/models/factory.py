"""Factory (registry) para instanciação de modelos.

Combina os padrões Factory e Strategy: o cliente escolhe o modelo por nome
(ex.: ``"mlp"`` vindo de ``configs/config.yaml``) sem conhecer as classes
concretas. Registrar um novo modelo não exige alterar código cliente.
"""

import torch.nn as nn

from src.models.mlp import RecommendationMLP


class ModelFactory:
    """Cria modelos de recomendação a partir de um nome registrado."""

    _registry: dict[str, type[nn.Module]] = {
        "mlp": RecommendationMLP,
    }

    @classmethod
    def create(cls, name: str, **kwargs: object) -> nn.Module:
        """Instancia o modelo registrado sob ``name``.

        Args:
            name: Identificador do modelo (ex.: ``"mlp"``).
            **kwargs: Argumentos repassados ao construtor do modelo.

        Returns:
            Instância do modelo solicitado.

        Raises:
            ValueError: Se ``name`` não estiver registrado.
        """
        try:
            model_cls = cls._registry[name]
        except KeyError:
            options = ", ".join(cls.available())
            msg = f"Modelo {name!r} desconhecido. Disponíveis: {options}."
            raise ValueError(msg) from None
        return model_cls(**kwargs)  # type: ignore[arg-type]

    @classmethod
    def register(cls, name: str, model_cls: type[nn.Module]) -> None:
        """Registra um novo modelo sob ``name``.

        Args:
            name: Identificador do modelo.
            model_cls: Classe concreta que herda de ``nn.Module``.
        """
        cls._registry[name] = model_cls

    @classmethod
    def available(cls) -> list[str]:
        """Retorna os nomes de modelos registrados, em ordem alfabética."""
        return sorted(cls._registry)
