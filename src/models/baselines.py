"""Baselines não-neurais para comparação com a MLP.

Implementa dois baselines exigidos pelo Tech Challenge:

- **PopularityBaseline**: recomenda os itens mais comprados globalmente.
  Não depende de features do usuário — mesma lista para todos.
- **SklearnBaseline**: regressão logística treinada sobre features agregadas
  por usuário (frequência de compra, diversidade, etc.).

Ambos expõem ``fit`` / ``predict`` para a avaliação comparativa.
"""

from typing import Self

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler


class PopularityBaseline:
    """Recomenda os itens mais populares globalmente (frequência de compra).

    Não possui treino — apenas conta as interações observadas. Útil como
    baseline mínimo: se o modelo não superar popularidade, não aprendeu nada.

    Attributes:
        item_scores: Series com score de popularidade por ``item_idx``.
    """

    def __init__(self) -> None:
        self.item_scores: pd.Series | None = None

    def fit(self, pairs: pd.DataFrame) -> Self:
        """Aprende a popularidade de cada item pelas interações positivas.

        Args:
            pairs: DataFrame com colunas ``item_idx`` e ``label``.

        Returns:
            A própria instância.
        """
        positives = pairs[pairs["label"] == 1]
        self.item_scores = positives["item_idx"].value_counts(normalize=False)
        return self

    def predict(self, user_idx: np.ndarray, top_k: int = 10) -> np.ndarray:
        """Retorna os ``top_k`` itens mais populares para cada usuário.

        Args:
            user_idx: Array de índices de usuário (usado apenas para
                determinar quantas listas gerar).
            top_k: Número de recomendações por usuário.

        Returns:
            Array ``(n_users, top_k)`` com ``item_idx`` recomendados.
        """
        if self.item_scores is None:
            msg = "PopularityBaseline exige fit() antes de predict()."
            raise RuntimeError(msg)
        top_items = self.item_scores.head(top_k).index.to_numpy(dtype=np.int64)
        return np.tile(top_items, (len(user_idx), 1))


class SklearnBaseline:
    """Regressão logística sobre features agregadas por usuário.

    Treina um classificador binário para prever a probabilidade de interação
    e ordena os itens por score decrescente.

    Attributes:
        model: Instância de ``LogisticRegression`` treinada.
        scaler: ``StandardScaler`` ajustado às features de treino.
    """

    def __init__(self, seed: int = 42) -> None:
        self.model = LogisticRegression(
            max_iter=1000,
            random_state=seed,
            class_weight="balanced",
        )
        self.scaler = StandardScaler()
        self._all_items: np.ndarray | None = None

    def fit(self, features: pd.DataFrame, pairs: pd.DataFrame) -> Self:
        """Treina o classificador sobre features agregadas por usuário.

        Args:
            features: DataFrame com features por usuário (coluna ``user_idx``
                + features numéricas).
            pairs: DataFrame com colunas ``user_idx``, ``item_idx``, ``label``.

        Returns:
            A própria instância.
        """
        self._all_items = np.sort(pairs["item_idx"].unique())

        # Junta features do usuário com cada interação
        merged = pairs.merge(features, on="user_idx", how="inner")
        feature_cols = [c for c in features.columns if c != "user_idx"]
        if not feature_cols:
            msg = "Features não contém colunas além de user_idx."
            raise ValueError(msg)

        x_np = merged[feature_cols].to_numpy(dtype=np.float64)
        y = merged["label"].to_numpy(dtype=np.int64)

        x_scaled = self.scaler.fit_transform(x_np)
        self.model.fit(x_scaled, y)
        return self

    def predict(
        self, features: pd.DataFrame, top_k: int = 10
    ) -> np.ndarray:
        """Gera top-K recomendações por usuário.

        Para cada usuário, prediz o score de todos os itens do catálogo
        usando a mesma feature row repetida e seleciona os top-K.

        Args:
            features: DataFrame com uma linha por usuário (``user_idx`` +
                features numéricas).
            top_k: Número de recomendações por usuário.

        Returns:
            Array ``(n_users, top_k)`` com ``item_idx`` recomendados.
        """
        if self._all_items is None:
            msg = "SklearnBaseline exige fit() antes de predict()."
            raise RuntimeError(msg)

        feature_cols = [c for c in features.columns if c != "user_idx"]
        recommendations: list[np.ndarray] = []

        for _, row in features.iterrows():
            user_feat = row[feature_cols].to_numpy(dtype=np.float64).reshape(1, -1)
            feats_scaled = self.scaler.transform(user_feat)
            feats_scaled = np.repeat(feats_scaled, len(self._all_items), axis=0)
            scores = self.model.predict_proba(feats_scaled)[:, 1]
            top_idx = np.argsort(scores)[-top_k:][::-1]
            recommendations.append(self._all_items[top_idx])

        return np.array(recommendations)
