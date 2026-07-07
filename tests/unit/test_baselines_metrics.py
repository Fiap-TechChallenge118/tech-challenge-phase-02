"""Testes unitários dos baselines e das métricas de avaliação."""

import numpy as np
import pandas as pd
import pytest

from src.models.baselines import PopularityBaseline, SklearnBaseline
from src.training.metrics import (
    _average_precision_at_k,
    _ndcg_at_k,
    _precision_at_k,
    _recall_at_k,
    build_test_pairs,
    evaluate_recommendations,
)

# * ---------------------------------------------------------------------------
# * Métricas
# * ---------------------------------------------------------------------------


class TestMetrics:
    """Cobertura das 4 métricas de ranking."""

    @pytest.fixture
    def perfect_recs(self) -> np.ndarray:
        """Recomendações perfeitas: itens relevantes primeiro."""
        return np.array([1, 2, 3, 10, 11])

    @pytest.fixture
    def positives(self) -> set[int]:
        """Itens relevantes."""
        return {1, 2, 3}

    def test_precision_at_k_perfect(self, perfect_recs, positives) -> None:
        """Precision@3 perfeita deve ser 1.0."""
        assert _precision_at_k(perfect_recs, positives, k=3) == 1.0

    def test_precision_at_k_partial(self, perfect_recs, positives) -> None:
        """Precision@5 com 3 de 5 relevantes = 0.6."""
        result = _precision_at_k(perfect_recs, positives, k=5)
        assert result == pytest.approx(3 / 5)

    def test_recall_at_k_perfect(self, perfect_recs, positives) -> None:
        """Recall@3 perfeita deve ser 1.0."""
        assert _recall_at_k(perfect_recs, positives, k=3) == 1.0

    def test_recall_at_k_partial(self, perfect_recs, positives) -> None:
        """Recall@1 deve ser 1/3 (1 de 3 relevantes)."""
        assert _recall_at_k(perfect_recs, positives, k=1) == pytest.approx(1 / 3)

    def test_ndcg_at_k_perfect(self, perfect_recs, positives) -> None:
        """NDCG@3 perfeita deve ser 1.0."""
        assert _ndcg_at_k(perfect_recs, positives, k=3) == pytest.approx(1.0)

    def test_ndcg_at_k_worst(self) -> None:
        """NDCG com itens relevantes no final deve ser < 1."""
        recs = np.array([10, 11, 1, 2, 3])
        ndcg = _ndcg_at_k(recs, {1, 2, 3}, k=5)
        assert ndcg > 0.0
        assert ndcg < 1.0

    def test_map_at_k_perfect(self, perfect_recs, positives) -> None:
        """MAP@3 perfeita deve ser 1.0."""
        result = _average_precision_at_k(perfect_recs, positives, k=3)
        assert result == pytest.approx(1.0)

    def test_map_at_k_worst(self) -> None:
        """MAP com nenhum relevante = 0."""
        assert _average_precision_at_k(np.array([10, 11]), {1, 2}, k=2) == 0.0

    def test_empty_positives(self) -> None:
        """Usuário sem itens positivos = métricas 0."""
        recs = np.array([1, 2, 3])
        empty = set()
        assert _precision_at_k(recs, empty, k=3) == 0.0
        assert _recall_at_k(recs, empty, k=3) == 0.0
        assert _ndcg_at_k(recs, empty, k=3) == 0.0
        assert _average_precision_at_k(recs, empty, k=3) == 0.0

    def test_k_zero(self) -> None:
        """K=0 deve retornar 0 em todas as métricas."""
        recs = np.array([1, 2, 3])
        assert _precision_at_k(recs, {1}, k=0) == 0.0
        assert _recall_at_k(recs, {1}, k=0) == 0.0
        assert _ndcg_at_k(recs, {1}, k=0) == 0.0
        assert _average_precision_at_k(recs, {1}, k=0) == 0.0


class TestEvaluateRecommendations:
    """Cobertura da função evaluate_recommendations."""

    def test_multi_user_average(self) -> None:
        """Métricas devem ser a média sobre os usuários."""
        recs = np.array([
            [1, 2, 3],   # usuário a: 2 de 3 relevantes
            [10, 11, 12], # usuário b: 0 de 3 relevantes
        ])
        test_pairs = [
            (0, {1, 2}),
            (1, {20, 21}),
        ]
        metrics = evaluate_recommendations(recs, test_pairs, k=3)
        # usuário a: P=2/3=0.667, R=2/2=1.0
        # usuário b: P=0/3=0.0, R=0/2=0.0
        # média: P=0.333, R=0.5
        assert metrics["precision_at_k"] == pytest.approx(0.3333, abs=1e-3)
        assert metrics["recall_at_k"] == pytest.approx(0.5)
        assert "ndcg_at_k" in metrics
        assert "map_at_k" in metrics


class TestBuildTestPairs:
    """Cobertura da função build_test_pairs."""

    def test_groups_positives_by_user(self) -> None:
        """Deve agrupar apenas itens positivos por usuário."""
        df = pd.DataFrame({
            "user_idx": [0, 0, 0, 1, 1],
            "item_idx": [10, 20, 30, 10, 40],
            "label": [1, 0, 1, 1, 0],
        })
        result = build_test_pairs(df)
        pairs_by_user = {uid: items for uid, items in result}
        assert pairs_by_user[0] == {10, 30}
        assert pairs_by_user[1] == {10}


# * ---------------------------------------------------------------------------
# * Baselines
# * ---------------------------------------------------------------------------


class TestPopularityBaseline:
    """Cobertura do baseline de popularidade."""

    @pytest.fixture
    def train_pairs(self) -> pd.DataFrame:
        """Dados de treino: item 1 comprado 10x, item 2 comprado 5x."""
        rows = []
        for _ in range(10):
            rows.append({"user_idx": 0, "item_idx": 1, "label": 1})
        for _ in range(5):
            rows.append({"user_idx": 0, "item_idx": 2, "label": 1})
        for _ in range(8):
            rows.append({"user_idx": 1, "item_idx": 3, "label": 0})
        return pd.DataFrame(rows)

    def test_fit_counts_positives(self, train_pairs) -> None:
        """fit deve contar apenas interações positivas."""
        pop = PopularityBaseline().fit(train_pairs)
        # item 1 = 10, item 2 = 5, item 3 não aparece (label=0)
        assert pop.item_scores is not None
        assert pop.item_scores.loc[1] == 10
        assert pop.item_scores.loc[2] == 5
        assert 3 not in pop.item_scores.index

    def test_predict_returns_top_k(self, train_pairs) -> None:
        """predict deve retornar os K itens mais populares."""
        pop = PopularityBaseline().fit(train_pairs)
        recs = pop.predict(np.array([0, 1]), top_k=1)
        assert recs.shape == (2, 1)
        assert recs[0, 0] == 1  # item mais popular

    def test_predict_before_fit_raises(self) -> None:
        """Chamar predict antes de fit deve levantar RuntimeError."""
        with pytest.raises(RuntimeError, match="fit"):
            PopularityBaseline().predict(np.array([0]))


class TestSklearnBaseline:
    """Cobertura do baseline sklearn."""

    @pytest.fixture
    def features(self) -> pd.DataFrame:
        """Features de usuário sintéticas."""
        return pd.DataFrame({
            "user_idx": [0, 1, 2],
            "n_items": [5, 2, 8],
            "n_orders": [3, 1, 5],
        })

    @pytest.fixture
    def pairs(self) -> pd.DataFrame:
        """Pares de treino sintéticos."""
        return pd.DataFrame({
            "user_idx": [0, 0, 1, 1, 2, 2],
            "item_idx": [10, 20, 10, 30, 20, 40],
            "label": [1, 0, 1, 0, 0, 1],
        })

    def test_fit_completes(self, features, pairs) -> None:
        """fit deve completar sem erro."""
        sk = SklearnBaseline(seed=42).fit(features, pairs)
        assert sk.model is not None

    def test_predict_shape(self, features, pairs) -> None:
        """predict deve retornar (n_users, min(top_k, n_itens))."""
        sk = SklearnBaseline(seed=42).fit(features, pairs)
        # Catálogo tem 4 itens, top_k=5 → retorna os 4 disponíveis
        recs = sk.predict(features, top_k=5)
        assert recs.shape == (3, 4)
        # Com top_k=3, retorna exatamente 3
        recs2 = sk.predict(features, top_k=3)
        assert recs2.shape == (3, 3)

    def test_predict_before_fit_raises(self, features) -> None:
        """Chamar predict antes de fit deve levantar RuntimeError."""
        with pytest.raises(RuntimeError, match="fit"):
            SklearnBaseline().predict(features)
