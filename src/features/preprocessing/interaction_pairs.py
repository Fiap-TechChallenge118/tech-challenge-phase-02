"""Estratégia de pares de interação user-item para a MLP/embedding.

Produz um ``DataFrame`` de triplas ``(user_idx, item_idx, label)`` combinando pares
positivos (produtos efetivamente comprados) com negativos amostrados. Os índices são
contíguos a partir de 0, prontos para as camadas de embedding do PyTorch. O sampling é
determinístico, controlado por ``random_seed`` (requisito de reprodutibilidade).
"""

from typing import Self

import numpy as np
import pandas as pd

from src.features.preprocessing.base import InstacartFrames, PreprocessorStrategy
from src.features.preprocessing.schemas import validate_frames
from src.settings import get_settings


class InteractionPairsStrategy(PreprocessorStrategy[pd.DataFrame]):
    """Gera pares positivos + negativos rotulados para treino da rede neural."""

    def __init__(self, n_negatives: int = 1, seed: int | None = None) -> None:
        """Inicializa a estratégia.

        Args:
            n_negatives: Nº de negativos amostrados por par positivo.
            seed: Semente do sampling. Se ``None``, usa ``settings.random_seed``.
        """
        self.n_negatives = n_negatives
        self.seed = seed if seed is not None else get_settings().random_seed
        self._user_to_idx: dict[int, int] = {}
        self._item_to_idx: dict[int, int] = {}

    @property
    def user_to_idx(self) -> dict[int, int]:
        """Mapeamento ``user_id`` → índice contíguo aprendido no fit."""
        return dict(self._user_to_idx)

    @property
    def item_to_idx(self) -> dict[int, int]:
        """Mapeamento ``product_id`` → índice contíguo aprendido no fit."""
        return dict(self._item_to_idx)

    def fit(self, data: InstacartFrames) -> Self:
        """Constrói mapeamentos contíguos de usuários e itens observados."""
        validate_frames(data)
        interactions = self._interactions(data)
        users = sorted(interactions["user_id"].unique())
        items = sorted(interactions["product_id"].unique())
        self._user_to_idx = {user: idx for idx, user in enumerate(users)}
        self._item_to_idx = {item: idx for idx, item in enumerate(items)}
        self._is_fitted = True
        return self

    def transform(self, data: InstacartFrames) -> pd.DataFrame:
        """Retorna os pares ``(user_idx, item_idx, label)`` positivos e negativos."""
        self._ensure_fitted()
        interactions = self._interactions(data)
        positives = self._positive_pairs(interactions)
        negatives = self._negative_pairs(interactions)
        return pd.concat([positives, negatives], ignore_index=True)

    @staticmethod
    def _interactions(data: InstacartFrames) -> pd.DataFrame:
        """Retorna pares distintos ``(user_id, product_id)`` observados."""
        merged = data.order_products.merge(
            data.orders[["order_id", "user_id"]], on="order_id", how="inner"
        )
        return merged[["user_id", "product_id"]].drop_duplicates()

    def _positive_pairs(self, interactions: pd.DataFrame) -> pd.DataFrame:
        """Converte interações observadas em pares positivos (``label = 1``)."""
        pairs = list(interactions.itertuples(index=False, name=None))
        return self._to_idx_frame(pairs, label=1)

    def _negative_pairs(self, interactions: pd.DataFrame) -> pd.DataFrame:
        """Amostra pares não observados por usuário (``label = 0``)."""
        rng = np.random.default_rng(self.seed)
        all_items = np.array(sorted(self._item_to_idx), dtype=np.int64)
        rows: list[tuple[int, int]] = []
        positives_by_user = interactions.groupby("user_id")["product_id"].agg(set)
        for user, pos_items in positives_by_user.items():
            candidates = np.setdiff1d(all_items, list(pos_items), assume_unique=True)
            if candidates.size == 0:
                continue
            size = self.n_negatives * len(pos_items)
            sampled = rng.choice(candidates, size=size, replace=candidates.size < size)
            rows.extend((int(user), int(item)) for item in sampled)
        return self._to_idx_frame(rows, label=0)

    def _to_idx_frame(
        self, pairs: list[tuple[int, int]], label: int
    ) -> pd.DataFrame:
        """Mapeia pares de ids para índices contíguos e anexa o rótulo."""
        frame = pd.DataFrame(pairs, columns=["user_id", "product_id"])
        frame["user_idx"] = frame["user_id"].map(self._user_to_idx)
        frame["item_idx"] = frame["product_id"].map(self._item_to_idx)
        frame["label"] = label
        return frame[["user_idx", "item_idx", "label"]]
