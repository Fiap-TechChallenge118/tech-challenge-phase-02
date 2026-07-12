"""Stage de feature engineering do pipeline DVC.

Lê ``data/processed/interactions.parquet`` (produzido pelo stage preprocess),
aplica fração opcional e divide em três splits com seed fixa:

- ``data/processed/train_pairs.parquet`` — pares de treino
- ``data/processed/val_pairs.parquet``   — pares de validação (early stopping)
- ``data/processed/test_pairs.parquet``  — pares de teste (holdout final)

Os splits são consumidos pelo stage ``train`` (train + val)
e ``evaluate`` (test + train).
O contrato de colunas é: ``user_idx``, ``item_idx``, ``label`` (int64, int64, float32).

Uso::

    uv run python -m src.features.feature_eng
    uv run python -m src.features.feature_eng --frac 0.2
"""

import argparse
import logging
from pathlib import Path

from src.settings import get_settings
from src.training.io import load_mappings, load_pairs, split_train_val_test

logger = logging.getLogger(__name__)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Stage feature_eng: interactions.parquet → train/val/test splits"
    )
    parser.add_argument(
        "--frac",
        type=float,
        default=0.2,
        help="Fração dos pares a usar (default: 0.2 = 20%%).",
    )
    return parser.parse_args()


def main() -> None:
    """Ponto de entrada do stage de feature engineering."""
    args = _parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(levelname)-8s %(message)s",
    )

    settings = get_settings()
    processed_dir = Path(settings.data_processed_dir)
    processed_dir.mkdir(parents=True, exist_ok=True)

    # * 1 — Carregar pares e mapeamentos do stage preprocess
    pairs = load_pairs(processed_dir / "interactions.parquet")
    load_mappings(processed_dir / "mappings.json")  # valida existência

    logger.info("Total de pares carregados: %d", len(pairs))

    # * 2 — Aplicar fração opcional (reprodutível via seed)
    if args.frac < 1.0:
        pairs = pairs.sample(frac=args.frac, random_state=settings.random_seed)
        pairs = pairs.reset_index(drop=True)
        logger.info(
            "Fração %.0f%% aplicada → %d pares",
            args.frac * 100,
            len(pairs),
        )

    # * 3 — Split treino / validação / teste com seed fixa
    train_idx, val_idx, test_idx = split_train_val_test(
        n=len(pairs),
        val_frac=settings.validation_split,
        test_frac=settings.test_split,
        seed=settings.random_seed,
    )

    logger.info(
        "Split: %d treino / %d validação / %d teste",
        len(train_idx),
        len(val_idx),
        len(test_idx),
    )

    # * 4 — Salvar os três splits para os stages seguintes
    train_pairs = pairs.iloc[train_idx].copy()
    val_pairs = pairs.iloc[val_idx].copy()
    test_pairs = pairs.iloc[test_idx].copy()

    train_pairs.to_parquet(processed_dir / "train_pairs.parquet", index=False)
    val_pairs.to_parquet(processed_dir / "val_pairs.parquet", index=False)
    test_pairs.to_parquet(processed_dir / "test_pairs.parquet", index=False)

    logger.info(
        "Splits salvos em %s: train=%d | val=%d | test=%d",
        processed_dir,
        len(train_pairs),
        len(val_pairs),
        len(test_pairs),
    )


if __name__ == "__main__":
    main()
