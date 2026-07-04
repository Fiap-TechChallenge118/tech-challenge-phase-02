"""Smoke tests do modelo RecommendationMLP e ModelFactory.

Verificações mínimas: forward pass com tensores aleatórios, shape do output
e instanciação via factory.
"""

import pytest
import torch

from src.models import ModelFactory, RecommendationMLP


class TestRecommendationMLP:
    """Smoke tests da arquitetura RecommendationMLP."""

    @pytest.fixture
    def model(self) -> RecommendationMLP:
        """Instancia um modelo pequeno com dimensões fixas para teste."""
        return RecommendationMLP(
            n_users=100,
            n_items=500,
            embedding_dim=32,
            hidden_dims=[64, 32],
            dropout=0.2,
            activation="relu",
            batch_norm=True,
        )

    def test_forward_output_shape(self, model: RecommendationMLP) -> None:
        """O output deve ter shape (batch_size, 1) com valores em [0, 1]."""
        batch_size = 16
        user_idx = torch.randint(0, 100, (batch_size,))
        item_idx = torch.randint(0, 500, (batch_size,))
        output = model(user_idx, item_idx)
        assert output.shape == (batch_size, 1)
        assert output.min() >= 0.0
        assert output.max() <= 1.0

    def test_forward_different_batch_sizes(self, model: RecommendationMLP) -> None:
        """A MLP deve aceitar batches de tamanhos variados."""
        for bs in [8, 64, 128]:
            user_idx = torch.randint(0, 100, (bs,))
            item_idx = torch.randint(0, 500, (bs,))
            output = model(user_idx, item_idx)
            assert output.shape == (bs, 1)

    def test_embedding_dimensions(self) -> None:
        """As camadas de embedding devem ter as dimensões corretas."""
        model = RecommendationMLP(n_users=10, n_items=20, embedding_dim=16)
        assert model.user_embedding.weight.shape == (10, 16)
        assert model.item_embedding.weight.shape == (20, 16)

    def test_batch_norm_with_single_sample_eval(self) -> None:
        """BatchNorm + batch=1 em modo eval não deve gerar erro."""
        model = RecommendationMLP(
            n_users=50,
            n_items=100,
            embedding_dim=8,
            hidden_dims=[16],
            batch_norm=True,
        )
        model.eval()
        with torch.no_grad():
            output = model(torch.tensor([3]), torch.tensor([7]))
        assert output.shape == (1, 1)

    def test_activation_options(self) -> None:
        """As três funções de ativação devem ser aceitas sem erro."""
        for activation in ("relu", "gelu", "tanh"):
            model = RecommendationMLP(
                n_users=10, n_items=10, embedding_dim=4,
                hidden_dims=[8], activation=activation,
                batch_norm=False,
            )
            output = model(torch.tensor([0]), torch.tensor([0]))
            assert output.shape == (1, 1)

    def test_deterministic_with_seed(self) -> None:
        """Mesma seed do PyTorch deve produzir o mesmo output."""
        torch.manual_seed(42)
        model1 = RecommendationMLP(n_users=10, n_items=10, embedding_dim=4)
        out1 = model1(torch.tensor([0, 1]), torch.tensor([2, 3]))

        torch.manual_seed(42)
        model2 = RecommendationMLP(n_users=10, n_items=10, embedding_dim=4)
        out2 = model2(torch.tensor([0, 1]), torch.tensor([2, 3]))

        assert torch.equal(out1, out2)


class TestModelFactory:
    """Smoke tests da factory de modelos."""

    def test_create_mlp(self) -> None:
        """Factory deve instanciar RecommendationMLP por nome."""
        model = ModelFactory.create(
            "mlp",
            n_users=50,
            n_items=200,
            embedding_dim=32,
            hidden_dims=[64, 32],
        )
        assert isinstance(model, RecommendationMLP)
        assert model.n_users == 50
        assert model.n_items == 200

    def test_create_unknown_raises(self) -> None:
        """Nome desconhecido deve levantar ValueError."""
        with pytest.raises(ValueError, match="desconhecido"):
            ModelFactory.create("transformer")

    def test_available_lists_mlp(self) -> None:
        """available() deve incluir 'mlp'."""
        assert "mlp" in ModelFactory.available()

    def test_register_new_model(self) -> None:
        """register() deve permitir adicionar e criar um novo modelo."""

        class _DummyModel(torch.nn.Module):
            def __init__(self, hidden: int = 10):
                super().__init__()
                self.linear = torch.nn.Linear(hidden, 1)

            def forward(self, x: torch.Tensor) -> torch.Tensor:
                return self.linear(x)

        ModelFactory.register("dummy", _DummyModel)
        model = ModelFactory.create("dummy", hidden=20)
        assert isinstance(model, _DummyModel)
        assert model.linear.in_features == 20
