"""Estratégia de features agregadas por usuário para baselines Scikit-Learn.

Sumariza o comportamento de cada usuário (nº de pedidos, taxa de reorder, hábitos de
horário/dia, etc.) em um vetor de features tabular. Opcionalmente padroniza as colunas
com ``StandardScaler`` ajustado no ``fit`` (estado reaproveitado no transform).
"""

from typing import Self

import pandas as pd
from sklearn.preprocessing import StandardScaler

from src.features.preprocessing.base import InstacartFrames, PreprocessorStrategy
from src.features.preprocessing.schemas import validate_frames


class AggregatedFeaturesStrategy(PreprocessorStrategy[pd.DataFrame]):
    """Gera features agregadas por usuário, opcionalmente padronizadas."""

    def __init__(self, scale: bool = True) -> None:
        """Inicializa a estratégia.

        Args:
            scale: Se ``True``, padroniza as features com ``StandardScaler``.
        """
        self.scale = scale
        self._scaler = StandardScaler()
        self._feature_names: list[str] = []

    def fit(self, data: InstacartFrames) -> Self:
        """Descobre as colunas de feature e ajusta o scaler quando habilitado."""
        validate_frames(data)
        features = self._build_features(data)
        self._feature_names = [c for c in features.columns if c != "user_id"]
        if self.scale:
            self._scaler.fit(features[self._feature_names])
        self._is_fitted = True
        return self

    def transform(self, data: InstacartFrames) -> pd.DataFrame:
        """Retorna as features por usuário, padronizadas se ``scale=True``."""
        self._ensure_fitted()
        features = self._build_features(data)
        if self.scale:
            features[self._feature_names] = self._scaler.transform(
                features[self._feature_names]
            )
        return features

    @staticmethod
    def _build_features(data: InstacartFrames) -> pd.DataFrame:
        """Agrega estatísticas comportamentais por ``user_id``."""
        merged = data.order_products.merge(data.orders, on="order_id", how="inner")
        grouped = merged.groupby("user_id")
        return grouped.agg(
            n_orders=("order_id", "nunique"),
            n_items=("product_id", "count"),
            n_unique_items=("product_id", "nunique"),
            reorder_rate=("reordered", "mean"),
            avg_add_to_cart=("add_to_cart_order", "mean"),
            avg_order_dow=("order_dow", "mean"),
            avg_order_hour=("order_hour_of_day", "mean"),
        ).reset_index()
