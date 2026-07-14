"""Estratégia de matriz esparsa user-item para filtragem colaborativa clássica.

Constrói uma ``scipy.sparse.csr_matrix`` de interações user × item, aplicando filtros
de cold-start (``min_user_interactions`` / ``min_item_interactions``). Serve de baseline
de CF e de insumo para modelos que consomem a matriz densa/esparsa diretamente.
"""

from dataclasses import dataclass
from typing import Self

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix

from src.features.preprocessing.base import InstacartFrames, PreprocessorStrategy
from src.features.preprocessing.schemas import validate_frames


@dataclass(frozen=True)
class UserItemMatrix:
    """Resultado da estratégia: matriz esparsa e seus mapeamentos de índice.

    Attributes:
        matrix: Matriz esparsa (n_users × n_items) de interações.
        user_to_idx: Mapeamento ``user_id`` → linha.
        item_to_idx: Mapeamento ``product_id`` → coluna.
    """

    matrix: csr_matrix
    user_to_idx: dict[int, int]
    item_to_idx: dict[int, int]


class UserItemMatrixStrategy(PreprocessorStrategy[UserItemMatrix]):
    """Monta a matriz esparsa user-item filtrada por interações mínimas."""

    def __init__(
        self,
        min_user_interactions: int = 1,
        min_item_interactions: int = 1,
        binary: bool = False,
    ) -> None:
        """Inicializa a estratégia.

        Args:
            min_user_interactions: Mínimo de interações para manter um usuário.
            min_item_interactions: Mínimo de interações para manter um item.
            binary: Se ``True``, usa valores 1/0; senão, contagem de interações.
        """
        self.min_user_interactions = min_user_interactions
        self.min_item_interactions = min_item_interactions
        self.binary = binary
        self._user_to_idx: dict[int, int] = {}
        self._item_to_idx: dict[int, int] = {}

    def fit(self, data: InstacartFrames) -> Self:
        """Aprende mapeamentos contíguos a partir das interações filtradas."""
        validate_frames(data)
        interactions = self._filtered_interactions(data)
        users = sorted(interactions["user_id"].unique())
        items = sorted(interactions["product_id"].unique())
        self._user_to_idx = {user: idx for idx, user in enumerate(users)}
        self._item_to_idx = {item: idx for idx, item in enumerate(items)}
        self._is_fitted = True
        return self

    def transform(self, data: InstacartFrames) -> UserItemMatrix:
        """Constrói a matriz esparsa restrita aos usuários e itens conhecidos."""
        self._ensure_fitted()
        interactions = self._filtered_interactions(data)
        known = interactions["user_id"].isin(self._user_to_idx) & interactions[
            "product_id"
        ].isin(self._item_to_idx)
        return self._build_matrix(interactions[known])

    @staticmethod
    def _interactions(data: InstacartFrames) -> pd.DataFrame:
        """Agrega contagem de interações por par ``(user_id, product_id)``."""
        merged = data.order_products.merge(
            data.orders[["order_id", "user_id"]], on="order_id", how="inner"
        )
        return (
            merged.groupby(["user_id", "product_id"]).size().reset_index(name="count")
        )

    def _filtered_interactions(self, data: InstacartFrames) -> pd.DataFrame:
        """Remove usuários e itens abaixo dos limiares de interação mínima."""
        interactions = self._interactions(data)
        user_counts = interactions.groupby("user_id")["count"].sum()
        item_counts = interactions.groupby("product_id")["count"].sum()
        valid_users = user_counts[user_counts >= self.min_user_interactions].index
        valid_items = item_counts[item_counts >= self.min_item_interactions].index
        keep = interactions["user_id"].isin(valid_users) & interactions[
            "product_id"
        ].isin(valid_items)
        return interactions[keep]

    def _build_matrix(self, interactions: pd.DataFrame) -> UserItemMatrix:
        """Materializa a ``csr_matrix`` a partir das interações mapeadas."""
        rows = interactions["user_id"].map(self._user_to_idx).to_numpy()
        cols = interactions["product_id"].map(self._item_to_idx).to_numpy()
        values = (
            np.ones(len(interactions))
            if self.binary
            else interactions["count"].to_numpy()
        )
        shape = (len(self._user_to_idx), len(self._item_to_idx))
        matrix = csr_matrix((values, (rows, cols)), shape=shape)
        return UserItemMatrix(matrix, dict(self._user_to_idx), dict(self._item_to_idx))
