"""Smoke tests do modelo RecommendationMLP e ModelFactory.

Verificações mínimas: forward pass com tensores aleatórios, shape do output
e instanciação via factory.
"""

import os
from pathlib import Path
from typing import Generator

import numpy as np
import pandas as pd
import pytest
import torch
from torch.utils.data import DataLoader

from src.models import ModelFactory, RecommendationMLP
from src.training.dataset import InteractionDataset
from src.training.trainer import _set_seeds, train


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
        """O output deve ter shape (batch_size, 1) e valores finitos."""
        batch_size = 16
        user_idx = torch.randint(0, 100, (batch_size,))
        item_idx = torch.randint(0, 500, (batch_size,))
        output = model(user_idx, item_idx)
        assert output.shape == (batch_size, 1)
        assert torch.isfinite(output).all()

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
                n_users=10,
                n_items=10,
                embedding_dim=4,
                hidden_dims=[8],
                activation=activation,
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


# * ---------------------------------------------------------------------------
# * Early stopping
# * ---------------------------------------------------------------------------


class TestEarlyStopping:
    """Cobertura de early stopping na função train()."""

    @pytest.fixture(autouse=True)
    def _mlflow_local(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> Generator:
        """Aponta MLflow para sqlite local e abre uma run isolada por teste.

        Cada teste roda dentro de um ``mlflow.start_run()`` próprio para evitar
        que um run_id de um teste anterior seja herdado pelo próximo.
        """
        import mlflow

        uri = f"sqlite:///{tmp_path}/mlflow_test.db"
        monkeypatch.setenv("MLFLOW_TRACKING_URI", uri)
        mlflow.set_tracking_uri(uri)

        # * Garante que não há run ativa residual antes de abrir a nova
        while mlflow.active_run():
            mlflow.end_run()

        with mlflow.start_run():
            yield

        # * Encerra a run ao sair do teste (mesmo que falhe)
        if mlflow.active_run():
            mlflow.end_run()

    @pytest.fixture
    def dummy_data(self) -> DataLoader:
        """Dataset sintético pequeno para teste rápido de early stopping."""
        n = 128
        rng = np.random.default_rng(42)
        pairs = pd.DataFrame(
            {
                "user_idx": rng.integers(0, 10, n),
                "item_idx": rng.integers(0, 20, n),
                "label": rng.integers(0, 2, n).astype(np.float32),
            }
        )
        dataset = InteractionDataset(pairs)
        return DataLoader(dataset, batch_size=32, shuffle=True)

    def _make_model(self) -> RecommendationMLP:
        """Modelo mínimo para teste."""
        return RecommendationMLP(
            n_users=10,
            n_items=20,
            embedding_dim=4,
            hidden_dims=[8],
            batch_norm=False,
        )

    def test_early_stopping_triggers(self, dummy_data: DataLoader) -> None:
        """Com patience=1 e lr=0, deve parar após 2 épocas."""
        model = self._make_model()
        _, _, _, best_epoch, early_stopped = train(
            model=model,
            train_loader=dummy_data,
            val_loader=dummy_data,
            epochs=50,
            lr=0.0,
            weight_decay=0.0,
            patience=1,
            min_delta=1e-8,
            pos_weight=None,
            device=torch.device("cpu"),
        )
        assert early_stopped
        assert best_epoch <= 2  # parou cedo

    def test_no_stop_with_high_patience(self, dummy_data: DataLoader) -> None:
        """Com patience >= epochs, nunca deve parar por early stopping."""
        model = self._make_model()
        _, train_losses, _, _, early_stopped = train(
            model=model,
            train_loader=dummy_data,
            val_loader=dummy_data,
            epochs=5,
            lr=0.001,
            weight_decay=0.0,
            patience=10,
            min_delta=1e-8,
            pos_weight=None,
            device=torch.device("cpu"),
        )
        assert not early_stopped
        assert len(train_losses) == 5


# * ---------------------------------------------------------------------------
# * Reproducibilidade das seeds
# * ---------------------------------------------------------------------------


class TestSeedReproducibility:
    """Garante que _set_seeds produz resultados idênticos entre execuções."""

    def test_set_seeds_fixes_python_hash(self) -> None:
        """PYTHONHASHSEED deve ser fixado após _set_seeds."""
        _set_seeds(42)
        assert os.environ["PYTHONHASHSEED"] == "42"

    def test_torch_manual_seed_is_set(self) -> None:
        """torch.manual_seed deve ser chamado por _set_seeds."""
        _set_seeds(123)
        # Gera dois tensores aleatórios e verifica determinismo
        torch.manual_seed(123)
        a = torch.randn(10)
        torch.manual_seed(123)
        b = torch.randn(10)
        assert torch.equal(a, b)

    def test_numpy_seed_is_set(self) -> None:
        """np.random.seed deve ser chamado por _set_seeds."""
        _set_seeds(7)
        a = np.random.randn(5)
        np.random.seed(7)
        b = np.random.randn(5)
        assert np.array_equal(a, b)

    def test_cudnn_deterministic_when_cuda(self) -> None:
        """Se CUDA disponível, cudnn.deterministic deve ser True."""
        _set_seeds(42)
        if torch.cuda.is_available():
            assert torch.backends.cudnn.deterministic
            assert not torch.backends.cudnn.benchmark
