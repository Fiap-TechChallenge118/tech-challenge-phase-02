"""Loop de treino da rede neural de recomendação.

Módulo executável que orquestra o treino ponta a ponta:

1. Carrega os CSVs brutos do Instacart (``data/raw/``)
2. Constrói pares user-item via ``InteractionPairsStrategy``
3. Divide em treino (80%) / validação (10%) / teste (10%)
4. Instancia ``RecommendationMLP`` via ``ModelFactory``
5. Treina com ``BCEWithLogitsLoss`` + ``Adam``
6. Salva checkpoint + test set para avaliação futura

Uso::

    # Treino padrão (20% dos dados, 10 épocas)
    uv run python -m src.training.trainer

    # Treino completo (100% dos dados, 50 épocas)
    uv run python -m src.training.trainer --frac 1.0

    # Ajustar fração e épocas
    uv run python -m src.training.trainer --frac 0.3 --epochs 15
"""

import argparse
import logging
import random
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import yaml
from torch.utils.data import DataLoader

from src.features.preprocessing import InstacartFrames, InteractionPairsStrategy
from src.models.factory import ModelFactory
from src.settings import get_settings
from src.training.dataset import InteractionDataset

logger = logging.getLogger(__name__)


def _load_frames(data_dir: Path) -> InstacartFrames:
    """Carrega os CSVs brutos do Instacart e retorna os DataFrames agrupados.

    Args:
        data_dir: Caminho para ``data/raw/``.

    Returns:
        DataFrames prontos para o pré-processamento.
    """
    orders = pd.read_csv(data_dir / "orders.csv")
    prior = pd.read_csv(data_dir / "order_products__prior.csv")
    train_orders = pd.read_csv(data_dir / "order_products__train.csv")
    products = pd.read_csv(data_dir / "products.csv")
    aisles = pd.read_csv(data_dir / "aisles.csv")
    departments = pd.read_csv(data_dir / "departments.csv")

    order_products = pd.concat([prior, train_orders], ignore_index=True)
    logger.info(
        "Dados carregados: %d pedidos, %d interações, %d produtos",
        len(orders),
        len(order_products),
        len(products),
    )

    return InstacartFrames(
        orders=orders,
        order_products=order_products,
        products=products,
        aisles=aisles,
        departments=departments,
    )


def _split_train_val_test(
    n: int, val_frac: float, test_frac: float, seed: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Divide índices em treino, validação e teste deterministicamente.

    Args:
        n: Número total de amostras.
        val_frac: Fração reservada para validação.
        test_frac: Fração reservada para teste (holdout, nunca visto no treino).
        seed: Semente para reprodutibilidade.

    Returns:
        Tupla ``(train_indices, val_indices, test_indices)``.
    """
    rng = np.random.default_rng(seed)
    indices = rng.permutation(n)
    test_end = int(n * test_frac)
    val_end = test_end + int(n * val_frac)
    test_idx = indices[:test_end]
    val_idx = indices[test_end:val_end]
    train_idx = indices[val_end:]
    return train_idx, val_idx, test_idx


def _set_seeds(seed: int) -> None:
    """Fixa as sementes de PyTorch, NumPy e Python para reprodutibilidade.

    Args:
        seed: Valor inteiro da semente.
    """
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)


def _load_config(path: Path) -> dict:
    """Carrega o arquivo YAML de configuração.

    Args:
        path: Caminho para ``configs/config.yaml``.

    Returns:
        Dicionário com os valores do YAML.
    """
    with path.open(encoding="utf-8") as fh:
        return yaml.safe_load(fh)


def train(
    model: torch.nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    epochs: int,
    lr: float,
    weight_decay: float,
    pos_weight: torch.Tensor | None,
    device: torch.device,
) -> tuple[torch.nn.Module, list[float], list[float]]:
    """Executa o loop de treino com validação por época.

    Args:
        model: Modelo PyTorch a ser treinado.
        train_loader: DataLoader com os dados de treino.
        val_loader: DataLoader com os dados de validação.
        epochs: Número máximo de épocas.
        lr: Taxa de aprendizado inicial do optimizer.
        weight_decay: Fator de regularização L2.
        pos_weight: Peso da classe positiva para BCEWithLogitsLoss.
        device: Dispositivo onde o treino será executado.

    Returns:
        Tupla ``(modelo_treinado, train_losses, val_losses)``.
    """
    model.to(device)
    criterion = torch.nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

    train_losses: list[float] = []
    val_losses: list[float] = []

    for epoch in range(1, epochs + 1):
        t0 = time.perf_counter()

        model.train()
        total_loss = 0.0
        for user, item, label in train_loader:
            user, item, label = user.to(device), item.to(device), label.to(device)
            optimizer.zero_grad()
            logits = model(user, item)
            loss = criterion(logits, label)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        avg_train = total_loss / len(train_loader)
        train_losses.append(avg_train)

        model.eval()
        total_val = 0.0
        with torch.no_grad():
            for user, item, label in val_loader:
                user, item, label = user.to(device), item.to(device), label.to(device)
                logits = model(user, item)
                total_val += criterion(logits, label).item()
        avg_val = total_val / len(val_loader)
        val_losses.append(avg_val)

        epoch_time = time.perf_counter() - t0
        logger.info(
            "Época %3d/%d | train_loss: %.4f | val_loss: %.4f | %.1fs",
            epoch,
            epochs,
            avg_train,
            avg_val,
            epoch_time,
        )

    return model, train_losses, val_losses


def _build_dataloaders(
    dataset: InteractionDataset,
    train_idx: np.ndarray,
    val_idx: np.ndarray,
    batch_size: int,
    num_workers: int = 4,
) -> tuple[DataLoader, DataLoader]:
    """Cria DataLoaders de treino e validação a partir de índices pré-calculados.

    Args:
        dataset: Dataset com todos os pares rotulados.
        train_idx: Índices de treino.
        val_idx: Índices de validação.
        batch_size: Tamanho do batch.
        num_workers: Workers para carregamento paralelo.

    Returns:
        Tupla ``(train_loader, val_loader)``.
    """
    pin_memory = torch.cuda.is_available()
    train_loader = DataLoader(
        torch.utils.data.Subset(dataset, train_idx.tolist()),
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=num_workers > 0,
    )
    val_loader = DataLoader(
        torch.utils.data.Subset(dataset, val_idx.tolist()),
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=num_workers > 0,
    )
    return train_loader, val_loader


def main() -> None:
    """Ponto de entrada do script de treino."""
    parser = argparse.ArgumentParser(description="Treinar MLP de recomendação.")
    parser.add_argument(
        "--frac",
        type=float,
        default=0.2,
        help="Fração dos dados a usar (default: 0.2 = 20%%).",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=None,
        help="Número de épocas (default: valor do config.yaml).",
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(levelname)-8s %(message)s",
    )

    settings = get_settings()
    cfg = _load_config(Path("configs/config.yaml"))
    _set_seeds(settings.random_seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("Dispositivo: %s | Seed: %d", device, settings.random_seed)

    # ---- Pré-processamento ------------------------------------------------
    frames = _load_frames(Path(settings.data_raw_dir))
    strategy = InteractionPairsStrategy(seed=settings.random_seed)
    pairs = strategy.fit_transform(frames)
    logger.info(
        "Pares gerados: %d (usuários=%d, itens=%d)",
        len(pairs),
        len(strategy.user_to_idx),
        len(strategy.item_to_idx),
    )

    # ---- Subset para treino rápido ----------------------------------------
    if args.frac < 1.0:
        pairs = pairs.sample(frac=args.frac, random_state=settings.random_seed)
        logger.info("Usando %.0f%% dos dados = %d pares", args.frac * 100, len(pairs))

    # ---- Split triplo: treino / validação / teste -------------------------
    dataset = InteractionDataset(pairs)
    train_idx, val_idx, test_idx = _split_train_val_test(
        len(dataset),
        settings.validation_split,
        settings.test_split,
        settings.random_seed,
    )
    logger.info(
        "Split: %d treino / %d validação / %d teste (holdout)",
        len(train_idx),
        len(val_idx),
        len(test_idx),
    )

    # ---- Salvar test set para avaliação futura (Card 3) -------------------
    processed_dir = Path(settings.data_processed_dir)
    processed_dir.mkdir(parents=True, exist_ok=True)
    test_pairs = pairs.iloc[test_idx].copy()
    test_pairs.to_parquet(processed_dir / "test_pairs.parquet", index=False)
    logger.info(
        "Test set salvo em %s (%d pares)",
        processed_dir / "test_pairs.parquet",
        len(test_pairs),
    )

    # ---- DataLoaders ------------------------------------------------------
    train_loader, val_loader = _build_dataloaders(
        dataset, train_idx, val_idx, settings.batch_size
    )

    # ---- Modelo -----------------------------------------------------------
    model_cfg = cfg["model"]
    train_cfg = cfg["training"]
    epochs = args.epochs if args.epochs is not None else train_cfg["max_epochs"]
    hidden_dims = list(model_cfg.get("hidden_dims", [256, 128, 64]))

    model = ModelFactory.create(
        model_cfg["type"],
        n_users=len(strategy.user_to_idx),
        n_items=len(strategy.item_to_idx),
        embedding_dim=model_cfg.get("embedding_dim", 64),
        hidden_dims=hidden_dims,
        dropout=model_cfg.get("dropout", 0.3),
        activation=model_cfg.get("activation", "relu"),
        batch_norm=model_cfg.get("batch_norm", True),
    )
    total_params = sum(p.numel() for p in model.parameters())
    logger.info(
        "Modelo: %s | hidden=%s | Parâmetros: %d",
        model_cfg["type"],
        hidden_dims,
        total_params,
    )

    # ---- Treino -----------------------------------------------------------
    pos_weight = train_cfg.get("pos_weight")
    pos_weight_tensor = (
        torch.tensor([pos_weight], device=device) if pos_weight else None
    )

    t0 = time.perf_counter()
    model, train_losses, val_losses = train(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        epochs=epochs,
        lr=train_cfg["learning_rate"],
        weight_decay=train_cfg["weight_decay"],
        pos_weight=pos_weight_tensor,
        device=device,
    )
    elapsed = time.perf_counter() - t0
    logger.info("Treino concluído em %.1f min (%d épocas)", elapsed / 60, epochs)

    # ---- Salvamento -------------------------------------------------------
    models_dir = Path("models")
    models_dir.mkdir(exist_ok=True)
    checkpoint = {
        "model_state_dict": model.state_dict(),
        "n_users": len(strategy.user_to_idx),
        "n_items": len(strategy.item_to_idx),
        "embedding_dim": model_cfg.get("embedding_dim", 64),
        "hidden_dims": hidden_dims,
        "user_to_idx": strategy.user_to_idx,
        "item_to_idx": strategy.item_to_idx,
        "seed": settings.random_seed,
        "train_losses": train_losses,
        "val_losses": val_losses,
    }
    torch.save(checkpoint, models_dir / "model.pt")
    logger.info("Checkpoint salvo em models/model.pt")
    logger.info(
        "Loss final — train: %.4f | val: %.4f", train_losses[-1], val_losses[-1]
    )


if __name__ == "__main__":
    main()
