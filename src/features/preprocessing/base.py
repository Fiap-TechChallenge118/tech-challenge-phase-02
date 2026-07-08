"""Interface Strategy para pré-processadores do dataset Instacart.

Define o contrato comum (``fit`` / ``transform`` / ``fit_transform``) que todas as
estratégias de pré-processamento devem seguir. O código cliente depende apenas desta
abstração, permitindo trocar a representação dos dados sem alterações (princípio
Aberto/Fechado). As estratégias recebem os DataFrames já carregados em memória
(``InstacartFrames``) para não se acoplarem ao caminho físico dos arquivos.

Uso::

    from src.features.preprocessing import InteractionPairsStrategy, InstacartFrames

    frames = InstacartFrames(orders=..., order_products=..., products=..., ...)
    pairs = InteractionPairsStrategy(n_negatives=1).fit_transform(frames)
"""

import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Generic, Self, TypeVar

import pandas as pd

logger = logging.getLogger(__name__)


# * Tipo de saída de cada estratégia (matriz esparsa, DataFrame de pares, etc.)
TransformOutput = TypeVar("TransformOutput")


class NotFittedError(RuntimeError):
    """Levantado quando ``transform`` é chamado antes de ``fit``."""


@dataclass(frozen=True)
class InstacartFrames:
    """Agrupa os DataFrames brutos do Instacart consumidos pelas estratégias.

    Attributes:
        orders: Pedidos (order_id, user_id, order_number, order_dow, ...).
        order_products: Itens por pedido (order_id, product_id, add_to_cart_order,
            reordered). Pode conter ``prior`` e ``train`` já concatenados.
        products: Catálogo de produtos (product_id, product_name, aisle_id, ...).
        aisles: Corredores (aisle_id, aisle).
        departments: Departamentos (department_id, department).
    """

    orders: pd.DataFrame
    order_products: pd.DataFrame
    products: pd.DataFrame
    aisles: pd.DataFrame
    departments: pd.DataFrame


class PreprocessorStrategy(ABC, Generic[TransformOutput]):
    """Contrato comum das estratégias de pré-processamento do Instacart."""

    #: Marca se ``fit`` já foi executado. Subclasses definem ``True`` ao final do fit.
    _is_fitted: bool = False
    _logger: logging.Logger = logger

    @property
    def is_fitted(self) -> bool:
        """Indica se a estratégia já aprendeu seu estado interno via ``fit``."""
        return self._is_fitted

    @abstractmethod
    def fit(self, data: InstacartFrames) -> Self:
        """Aprende o estado necessário (mapeamentos, estatísticas, scalers).

        Args:
            data: DataFrames brutos do Instacart.

        Returns:
            A própria instância, para encadeamento fluente.
        """

    @abstractmethod
    def transform(self, data: InstacartFrames) -> TransformOutput:
        """Aplica a transformação, exigindo ``fit`` prévio.

        Args:
            data: DataFrames brutos do Instacart.

        Returns:
            A representação produzida pela estratégia.

        Raises:
            NotFittedError: Se chamado antes de ``fit``.
        """

    def fit_transform(self, data: InstacartFrames) -> TransformOutput:
        """Executa ``fit`` seguido de ``transform`` sobre os mesmos dados.

        Args:
            data: DataFrames brutos do Instacart.

        Returns:
            A representação produzida pela estratégia.
        """
        return self.fit(data).transform(data)

    def _ensure_fitted(self) -> None:
        """Garante que ``fit`` foi chamado antes de ``transform``.

        Raises:
            NotFittedError: Se a estratégia ainda não foi ajustada.
        """
        if not self._is_fitted:
            msg = f"{type(self).__name__} exige fit() antes de transform()."
            raise NotFittedError(msg)
