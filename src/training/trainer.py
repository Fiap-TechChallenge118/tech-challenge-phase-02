"""Loop de treino da rede neural de recomendação.

Módulo executável que orquestra o treino ponta a ponta:

1. Carrega ``data/processed/interactions.parquet`` (produzido pelo stage preprocess)
2. Carrega ``data/processed/mappings.json`` (user_to_idx / item_to_idx)
3. Divide em treino (80%) / validação (10%) / teste (10%)
4. Instancia ``RecommendationMLP`` via ``ModelFactory``
5. Treina com ``BCEWithLogitsLoss`` + ``Adam``
6. Loga parâmetros, métricas e artefatos no MLflow Tracking
7. Salva checkpoint + train_pairs.parquet + test_pairs.parquet para avaliação futura

Uso::

    # Treino padrão (20% dos dados, 10 épocas)
    uv run python -m src.training.trainer

    # Treino completo (100% dos dados, 50 épocas)
    uv run python -m src.training.trainer --frac 1.0

    # Ajustar fração e épocas
    uv run python -m src.training.trainer --frac 0.3 --epochs 15
"""

import argparse
import copy
import logging
import os
import random
import time
from pathlib import Path

import mlflow
import numpy as np
import torch
from torch.utils.data import DataLoader

from src.models.factory import ModelFactory
from src.settings import get_settings
from src.training.dataset import InteractionDataset
from src.training.io import load_config, load_mappings, load_pairs, split_train_val_test

logger = logging.getLogger(__name__)


def _set_seeds(seed: int) -> None:
    """Fixa todas as sementes para reprodutibilidade determinística.

    Cobre PyTorch (CPU e CUDA), NumPy, Python e hash de strings.
    Deve ser chamada antes de qualquer operação aleatória.

    Args:
        seed: Valor inteiro da semente.
    """
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def train(
    model: torch.nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    epochs: int,
    lr: float,
    weight_decay: float,
    patience: int,
    min_delta: float,
    pos_weight: torch.Tensor | None,
    device: torch.device,
) -> tuple[torch.nn.Module, list[float], list[float], int, bool]:
    """Executa o loop de treino com early stopping.

    Args:
        model: Modelo PyTorch a ser treinado.
        train_loader: DataLoader com os dados de treino.
        val_loader: DataLoader com os dados de validação.
        epochs: Número máximo de épocas.
        lr: Taxa de aprendizado inicial do optimizer.
        weight_decay: Fator de regularização L2.
        patience: Épocas sem melhora na val_loss antes de parar.
        min_delta: Melhora mínima da val_loss para considerar progresso.
        pos_weight: Peso da classe positiva para BCEWithLogitsLoss.
        device: Dispositivo onde o treino será executado.

    Returns:
        Tupla ``(modelo, train_losses, val_losses, best_epoch, early_stopped)``.
        O modelo retornado é o de melhor val_loss (restaurado do checkpoint).
    """
    model.to(device)
    criterion = torch.nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

    train_losses: list[float] = []
    val_losses: list[float] = []
    best_val = float("inf")
    best_state = copy.deepcopy(model.state_dict())
    best_epoch = 0
    stale = 0

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
        improved = ""
        if avg_val < best_val - min_delta:
            best_val = avg_val
            best_state = copy.deepcopy(model.state_dict())
            best_epoch = epoch
            stale = 0
            improved = " *"
        else:
            stale += 1

        # * Loga métricas por época no MLflow
        mlflow.log_metrics(
            {
                "train_loss": avg_train,
                "val_loss": avg_val,
                "epoch_time_s": epoch_time,
            },
            step=epoch,
        )

        logger.info(
            "Época %3d/%d | train_loss: %.4f | val_loss: %.4f | %.1fs%s",
            epoch,
            epochs,
            avg_train,
            avg_val,
            epoch_time,
            improved,
        )

        if stale >= patience:
            logger.info(
                "Early stopping: val_loss sem melhora por %d épocas. "
                "Melhor: época %d (val_loss=%.4f).",
                patience,
                best_epoch,
                best_val,
            )
            break

    early_stopped = stale >= patience
    model.load_state_dict(best_state)
    return model, train_losses, val_losses, best_epoch, early_stopped


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


def _save_split_pairs(
    pairs,
    train_idx: np.ndarray,
    test_idx: np.ndarray,
    processed_dir: Path,
) -> None:
    """Persiste os subsets de treino e teste como parquet para os stages seguintes.

    Args:
        pairs: DataFrame completo de pares.
        train_idx: Índices do split de treino.
        test_idx: Índices do split de teste.
        processed_dir: Diretório de saída (``data/processed/``).
    """
    train_pairs = pairs.iloc[train_idx].copy()
    train_pairs.to_parquet(processed_dir / "train_pairs.parquet", index=False)
    logger.info(
        "train_pairs.parquet salvo em %s (%d pares)",
        processed_dir / "train_pairs.parquet",
        len(train_pairs),
    )

    test_pairs = pairs.iloc[test_idx].copy()
    test_pairs.to_parquet(processed_dir / "test_pairs.parquet", index=False)
    logger.info(
        "test_pairs.parquet salvo em %s (%d pares)",
        processed_dir / "test_pairs.parquet",
        len(test_pairs),
    )


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
    cfg = load_config(Path("configs/config.yaml"))
    _set_seeds(settings.random_seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("Dispositivo: %s | Seed: %d", device, settings.random_seed)

    # * ---- MLflow setup --------------------------------------------------
    mlflow.set_tracking_uri(settings.mlflow_tracking_uri)
    mlflow.set_experiment(settings.mlflow_experiment_name)
    while mlflow.active_run():
        mlflow.end_run()

    logger.info(
        "MLflow: tracking_uri=%s | experiment=%s",
        settings.mlflow_tracking_uri,
        settings.mlflow_experiment_name,
    )

    # * ---- Carregar dados processados (sem reprocessar CSVs brutos) ------
    processed_dir = Path(settings.data_processed_dir)
    pairs = load_pairs(processed_dir / "interactions.parquet")
    user_to_idx, item_to_idx = load_mappings(processed_dir / "mappings.json")
    logger.info(
        "Dados carregados: %d pares (usuários=%d, itens=%d)",
        len(pairs),
        len(user_to_idx),
        len(item_to_idx),
    )

    # * ---- Subset para treino rápido -------------------------------------
    if args.frac < 1.0:
        pairs = pairs.sample(frac=args.frac, random_state=settings.random_seed)
        logger.info("Usando %.0f%% dos dados = %d pares", args.frac * 100, len(pairs))

    # * ---- Split triplo: treino / validação / teste ----------------------
    dataset = InteractionDataset(pairs)
    train_idx, val_idx, test_idx = split_train_val_test(
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

    # * ---- Persistir splits para os stages seguintes ---------------------
    processed_dir.mkdir(parents=True, exist_ok=True)
    _save_split_pairs(pairs, train_idx, test_idx, processed_dir)

    # * ---- DataLoaders ---------------------------------------------------
    train_loader, val_loader = _build_dataloaders(
        dataset, train_idx, val_idx, settings.batch_size
    )

    # * ---- Modelo --------------------------------------------------------
    model_cfg = cfg["model"]
    train_cfg = cfg["training"]
    epochs = args.epochs if args.epochs is not None else train_cfg["max_epochs"]
    hidden_dims = list(model_cfg.get("hidden_dims", [256, 128, 64]))

    model = ModelFactory.create(
        model_cfg["type"],
        n_users=len(user_to_idx),
        n_items=len(item_to_idx),
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

    # * ---- Treino (com MLflow tracking) ----------------------------------
    pos_weight = train_cfg.get("pos_weight")
    pos_weight_tensor = (
        torch.tensor([pos_weight], device=device) if pos_weight else None
    )

    with mlflow.start_run() as run:
        run_id = run.info.run_id
        logger.info("MLflow run: %s", run_id)

        mlflow_tags = cfg.get("mlflow", {}).get("tags", {})
        mlflow.set_tags(mlflow_tags)

        mlflow.log_params(
            {
                "model_type": model_cfg["type"],
                "n_users": len(user_to_idx),
                "n_items": len(item_to_idx),
                "embedding_dim": model_cfg.get("embedding_dim", 64),
                "hidden_dims": str(hidden_dims),
                "dropout": model_cfg.get("dropout", 0.3),
                "activation": model_cfg.get("activation", "relu"),
                "batch_norm": model_cfg.get("batch_norm", True),
                "total_params": total_params,
                "learning_rate": train_cfg["learning_rate"],
                "weight_decay": train_cfg["weight_decay"],
                "batch_size": settings.batch_size,
                "epochs": epochs,
                "patience": train_cfg.get("patience", 5),
                "min_delta": train_cfg.get("min_delta", 1e-4),
                "pos_weight": pos_weight,
                "data_frac": args.frac,
                "random_seed": settings.random_seed,
                "device": str(device),
            }
        )
        mlflow.log_param("train_pairs", len(train_idx))
        mlflow.log_param("val_pairs", len(val_idx))

        t0 = time.perf_counter()
        patience = train_cfg.get("patience", 5)
        min_delta = train_cfg.get("min_delta", 1e-4)
        model, train_losses, val_losses, best_epoch, early_stopped = train(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            epochs=epochs,
            lr=train_cfg["learning_rate"],
            weight_decay=train_cfg["weight_decay"],
            patience=patience,
            min_delta=min_delta,
            pos_weight=pos_weight_tensor,
            device=device,
        )
        elapsed = time.perf_counter() - t0

        mlflow.log_metrics(
            {
                "best_val_loss": float(val_losses[best_epoch - 1]),
                "final_train_loss": train_losses[-1],
                "best_epoch": best_epoch,
                "total_epochs": len(train_losses),
                "early_stopped": early_stopped,
                "training_time_min": elapsed / 60,
            }
        )

        logger.info(
            "Treino concluído em %.1f min (%d épocas, best=%d)",
            elapsed / 60,
            len(train_losses),
            best_epoch,
        )

        # * ---- Salvamento ------------------------------------------------
        models_dir = Path("models")
        models_dir.mkdir(exist_ok=True)
        checkpoint = {
            "model_state_dict": model.state_dict(),
            "n_users": len(user_to_idx),
            "n_items": len(item_to_idx),
            "embedding_dim": model_cfg.get("embedding_dim", 64),
            "hidden_dims": hidden_dims,
            "user_to_idx": user_to_idx,
            "item_to_idx": item_to_idx,
            "seed": settings.random_seed,
            "train_losses": train_losses,
            "val_losses": val_losses,
            "best_epoch": best_epoch,
            "early_stopped": early_stopped,
        }
        torch.save(checkpoint, models_dir / "model.pt")
        logger.info("Checkpoint salvo em models/model.pt")

        sample_user = torch.tensor([0])
        sample_item = torch.tensor([0])
        mlflow.pytorch.log_model(
            model,
            artifact_path="model",
            registered_model_name=settings.mlflow_registered_model_name,
            input_example=(sample_user, sample_item),
            serialization_format="pickle",
        )
        logger.info(
            "Modelo logado no MLflow: %s (run %s)",
            settings.mlflow_registered_model_name,
            run_id,
        )

        logger.info(
            "Loss final — train: %.4f | val: %.4f",
            train_losses[-1],
            val_losses[-1],
        )


if __name__ == "__main__":
    main()
