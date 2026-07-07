"""Testes unitários das estratégias de pré-processamento (padrão Strategy).

Coberturas:
- InteractionPairsStrategy ponta a ponta: shape, positivos preservados, negative
  sampling determinístico, índices contíguos e erro sem fit.
- UserItemMatrixStrategy: shape da matriz coerente com os mapeamentos e filtros.
- AggregatedFeaturesStrategy: features por usuário geradas corretamente.
- PreprocessorFactory: criação por nome, registro e nomes desconhecidos.

As fixtures constroem ``InstacartFrames`` sintéticos pequenos, sem ler CSV do disco.
"""

import pandas as pd
import pytest

from src.features.preprocessing import (
    AggregatedFeaturesStrategy,
    InstacartFrames,
    InteractionPairsStrategy,
    NotFittedError,
    PreprocessorFactory,
    PreprocessorStrategy,
    UserItemMatrixStrategy,
)

# * ---------------------------------------------------------------------------
# * Fixture sintética
# * ---------------------------------------------------------------------------


@pytest.fixture
def frames() -> InstacartFrames:
    """Dataset Instacart mínimo com 3 usuários, 3 pedidos e 4 produtos.

    Interações (user → produtos):
      - user 1 (pedido 10): produtos 100, 101
      - user 2 (pedido 20): produto 100
      - user 3 (pedido 30): produtos 101, 102, 103
    """
    orders = pd.DataFrame(
        {
            "order_id": [10, 20, 30],
            "user_id": [1, 2, 3],
            "order_number": [1, 1, 1],
            "order_dow": [2, 3, 1],
            "order_hour_of_day": [8, 10, 14],
            "days_since_prior_order": [None, None, None],
        }
    )
    order_products = pd.DataFrame(
        {
            "order_id": [10, 10, 20, 30, 30, 30],
            "product_id": [100, 101, 100, 101, 102, 103],
            "add_to_cart_order": [1, 2, 1, 1, 2, 3],
            "reordered": [0, 1, 0, 1, 0, 0],
        }
    )
    products = pd.DataFrame(
        {
            "product_id": [100, 101, 102, 103],
            "product_name": ["a", "b", "c", "d"],
            "aisle_id": [1, 1, 2, 2],
            "department_id": [1, 1, 2, 2],
        }
    )
    aisles = pd.DataFrame({"aisle_id": [1, 2], "aisle": ["x", "y"]})
    departments = pd.DataFrame({"department_id": [1, 2], "department": ["p", "q"]})
    return InstacartFrames(
        orders=orders,
        order_products=order_products,
        products=products,
        aisles=aisles,
        departments=departments,
    )


# * ---------------------------------------------------------------------------
# * InteractionPairsStrategy (ponta a ponta)
# * ---------------------------------------------------------------------------


class TestInteractionPairsStrategy:
    """Cobertura ponta a ponta da estratégia principal da rede neural."""

    def test_output_columns(self, frames: InstacartFrames) -> None:
        """A saída deve conter exatamente user_idx, item_idx e label."""
        result = InteractionPairsStrategy(n_negatives=1, seed=42).fit_transform(frames)
        assert list(result.columns) == ["user_idx", "item_idx", "label"]

    def test_positive_pairs_preserved(self, frames: InstacartFrames) -> None:
        """Todos os 6 pares comprados devem aparecer com label == 1."""
        strategy = InteractionPairsStrategy(n_negatives=1, seed=42).fit(frames)
        result = strategy.transform(frames)
        positives = result[result["label"] == 1]
        assert len(positives) == 6

    def test_negative_count_matches_ratio(self, frames: InstacartFrames) -> None:
        """Com n_negatives=2, o nº de negativos = 2 × nº de positivos."""
        result = InteractionPairsStrategy(n_negatives=2, seed=42).fit_transform(frames)
        n_pos = int((result["label"] == 1).sum())
        n_neg = int((result["label"] == 0).sum())
        assert n_neg == 2 * n_pos

    def test_negative_sampling_deterministic(self, frames: InstacartFrames) -> None:
        """Mesma seed deve produzir exatamente a mesma saída."""
        first = InteractionPairsStrategy(n_negatives=1, seed=7).fit_transform(frames)
        second = InteractionPairsStrategy(n_negatives=1, seed=7).fit_transform(frames)
        pd.testing.assert_frame_equal(first, second)

    def test_index_mapping_is_contiguous(self, frames: InstacartFrames) -> None:
        """Índices de usuário e item devem ser contíguos a partir de 0."""
        strategy = InteractionPairsStrategy(seed=42).fit(frames)
        assert sorted(strategy.user_to_idx.values()) == [0, 1, 2]
        assert sorted(strategy.item_to_idx.values()) == [0, 1, 2, 3]

    def test_transform_before_fit_raises(self, frames: InstacartFrames) -> None:
        """Chamar transform sem fit deve levantar NotFittedError."""
        with pytest.raises(NotFittedError):
            InteractionPairsStrategy().transform(frames)

    def test_negatives_are_not_positives(self, frames: InstacartFrames) -> None:
        """Nenhum par negativo pode coincidir com um par positivo do mesmo usuário."""
        result = InteractionPairsStrategy(n_negatives=2, seed=1).fit_transform(frames)
        pos_frame = result[result["label"] == 1][["user_idx", "item_idx"]]
        neg_frame = result[result["label"] == 0][["user_idx", "item_idx"]]
        positives = set(map(tuple, pos_frame.to_numpy()))
        assert all(tuple(pair) not in positives for pair in neg_frame.to_numpy())


# * ---------------------------------------------------------------------------
# * UserItemMatrixStrategy
# * ---------------------------------------------------------------------------


class TestUserItemMatrixStrategy:
    """Cobertura da estratégia de matriz esparsa user-item."""

    def test_matrix_shape_matches_index(self, frames: InstacartFrames) -> None:
        """Shape da matriz deve ser (n_users, n_items) dos mapeamentos."""
        result = UserItemMatrixStrategy().fit_transform(frames)
        assert result.matrix.shape == (
            len(result.user_to_idx),
            len(result.item_to_idx),
        )

    def test_matrix_nonzero_equals_interactions(
        self, frames: InstacartFrames
    ) -> None:
        """Sem filtros, o nº de células não nulas = nº de pares distintos (6)."""
        result = UserItemMatrixStrategy().fit_transform(frames)
        assert result.matrix.nnz == 6

    def test_min_item_filter_drops_items(self, frames: InstacartFrames) -> None:
        """min_item_interactions=2 mantém apenas produtos 100 e 101."""
        result = UserItemMatrixStrategy(min_item_interactions=2).fit_transform(frames)
        assert set(result.item_to_idx) == {100, 101}

    def test_binary_matrix_has_unit_values(self, frames: InstacartFrames) -> None:
        """Com binary=True, todos os valores não nulos devem ser 1."""
        result = UserItemMatrixStrategy(binary=True).fit_transform(frames)
        assert set(result.matrix.data.tolist()) == {1.0}


# * ---------------------------------------------------------------------------
# * AggregatedFeaturesStrategy
# * ---------------------------------------------------------------------------


class TestAggregatedFeaturesStrategy:
    """Cobertura da estratégia de features agregadas por usuário."""

    def test_one_row_per_user(self, frames: InstacartFrames) -> None:
        """Deve haver uma linha por usuário (3 usuários)."""
        result = AggregatedFeaturesStrategy(scale=False).fit_transform(frames)
        assert len(result) == 3
        assert set(result["user_id"]) == {1, 2, 3}

    def test_unscaled_counts_are_correct(self, frames: InstacartFrames) -> None:
        """Sem scaling, n_items do usuário 3 deve ser 3."""
        result = AggregatedFeaturesStrategy(scale=False).fit_transform(frames)
        user3 = result[result["user_id"] == 3].iloc[0]
        assert user3["n_items"] == 3
        assert user3["n_unique_items"] == 3

    def test_scaling_changes_values(self, frames: InstacartFrames) -> None:
        """Com scaling ativo, a média de cada feature deve ficar próxima de 0."""
        result = AggregatedFeaturesStrategy(scale=True).fit_transform(frames)
        feature_cols = [c for c in result.columns if c != "user_id"]
        assert result[feature_cols].mean().abs().max() < 1e-9


# * ---------------------------------------------------------------------------
# * PreprocessorFactory
# * ---------------------------------------------------------------------------


class TestPreprocessorFactory:
    """Cobertura da factory de seleção de estratégias."""

    def test_create_known_strategy(self) -> None:
        """create() deve instanciar a classe correta por nome."""
        strategy = PreprocessorFactory.create("interaction_pairs", n_negatives=3)
        assert isinstance(strategy, InteractionPairsStrategy)
        assert strategy.n_negatives == 3

    def test_create_unknown_raises(self) -> None:
        """Nome desconhecido deve levantar ValueError."""
        with pytest.raises(ValueError, match="desconhecida"):
            PreprocessorFactory.create("inexistente")

    def test_available_lists_all(self) -> None:
        """available() deve listar as três estratégias padrão."""
        assert set(PreprocessorFactory.available()) >= {
            "interaction_pairs",
            "user_item_matrix",
            "aggregated_features",
        }

    def test_register_new_strategy(self, frames: InstacartFrames) -> None:
        """register() deve permitir adicionar e criar uma nova estratégia."""

        class _Dummy(PreprocessorStrategy[int]):
            def fit(self, data: InstacartFrames) -> "_Dummy":
                self._is_fitted = True
                return self

            def transform(self, data: InstacartFrames) -> int:
                self._ensure_fitted()
                return len(data.orders)

        PreprocessorFactory.register("dummy", _Dummy)
        strategy = PreprocessorFactory.create("dummy")
        assert strategy.fit_transform(frames) == 3
