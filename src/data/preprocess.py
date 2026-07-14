"""Stage de pré-processamento do pipeline DVC.

Lê os CSVs brutos do Instacart (``data/raw/``), instancia a estratégia de
pré-processamento definida em ``configs/config.yaml`` via ``PreprocessorFactory``
e salva dois artefatos em ``data/processed/``:

- ``interactions.parquet`` — pares ``(user_idx, item_idx, label)`` prontos para treino.
- ``mappings.json`` — mapeamentos ``user_to_idx`` / ``item_to_idx`` aprendidos no fit,
  consumidos pelo stage ``train`` e ``evaluate`` sem precisar reprocessar os CSVs.

A estratégia padrão é ``interaction_pairs``, que gera triplas
``(user_idx, item_idx, label)`` com negative sampling. Trocar a estratégia
não exige alterar este módulo — basta mudar ``data.strategy`` no config.

Este módulo é o ponto de entrada do primeiro stage do ``dvc.yaml``.

Uso::

    uv run python -m src.data.preprocess
    uv run python -m src.data.preprocess --frac 0.2
"""

import argparse
import json
import logging
from pathlib import Path

import pandas as pd

from src.features.preprocessing import InstacartFrames, PreprocessorFactory
from src.features.preprocessing.enums import PreprocessorStrategyName
from src.settings import get_settings

logger = logging.getLogger(__name__)


def _load_frames(raw_dir: Path) -> InstacartFrames:
    """Carrega os CSVs brutos do Instacart.

    Args:
        raw_dir: Caminho para o diretório ``data/raw/``.

    Returns:
        ``InstacartFrames`` com todos os DataFrames carregados.
    """
    logger.info("Carregando CSVs de %s ...", raw_dir)
    orders = pd.read_csv(raw_dir / "orders.csv")
    prior = pd.read_csv(raw_dir / "order_products__prior.csv")
    train_orders = pd.read_csv(raw_dir / "order_products__train.csv")
    products = pd.read_csv(raw_dir / "products.csv")
    aisles = pd.read_csv(raw_dir / "aisles.csv")
    departments = pd.read_csv(raw_dir / "departments.csv")

    order_products = pd.concat([prior, train_orders], ignore_index=True)
    logger.info(
        "Dados carregados: %d pedidos | %d interações | %d produtos",
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


def _parse_args() -> argparse.Namespace:
    """Define os argumentos de linha de comando do stage."""
    parser = argparse.ArgumentParser(
        description="Stage preprocess: raw CSVs → interactions.parquet"
    )
    parser.add_argument(
        "--frac",
        type=float,
        default=1.0,
        help="Fração dos pares a manter (default: 1.0 = 100%%).",
    )
    return parser.parse_args()


def main() -> None:
    """Ponto de entrada do stage de pré-processamento."""
    args = _parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(levelname)-8s %(message)s",
    )

    settings = get_settings()
    raw_dir = Path(settings.data_raw_dir)
    processed_dir = Path(settings.data_processed_dir)
    processed_dir.mkdir(parents=True, exist_ok=True)

    # * 1 — Carregar frames brutos
    frames = _load_frames(raw_dir)

    # * 2 — Instanciar estratégia via Factory (nome vem do config.yaml via settings)
    strategy_name = settings.preprocessing_strategy
    strategy_kwargs: dict = {"seed": settings.random_seed}

    if strategy_name == PreprocessorStrategyName.INTERACTION_PAIRS:
        strategy_kwargs["n_negatives"] = settings.n_negatives

    logger.info(
        "Estratégia selecionada: %r (kwargs=%s)",
        strategy_name,
        strategy_kwargs,
    )
    strategy = PreprocessorFactory.create(strategy_name, **strategy_kwargs)
    logger.info("Estratégia instanciada: %r", strategy)
    # * 3 — Ajustar e transformar
    pairs = strategy.fit_transform(frames)
    logger.info("Pares gerados: %d", len(pairs))

    # * 4 — Aplicar fração opcional (para experimentos rápidos)
    if args.frac < 1.0:
        pairs = pairs.sample(frac=args.frac, random_state=settings.random_seed)
        logger.info(
            "Fração %.0f%% aplicada → %d pares",
            args.frac * 100,
            len(pairs),
        )

    # * 5 — Salvar artefato principal
    out_path = processed_dir / "interactions.parquet"
    pairs.to_parquet(out_path, index=False)
    logger.info(
        "interactions.parquet salvo em %s (%d linhas)",
        out_path,
        len(pairs),
    )

    # * 6 — Salvar mapeamentos para reuso nos stages train e evaluate
    # * Evita que trainer/evaluate reprocessem os CSVs brutos só para obter os índices
    if hasattr(strategy, "user_to_idx") and hasattr(strategy, "item_to_idx"):
        mappings = {
            "user_to_idx": {str(k): v for k, v in strategy.user_to_idx.items()},
            "item_to_idx": {str(k): v for k, v in strategy.item_to_idx.items()},
        }
        mappings_path = processed_dir / "mappings.json"
        mappings_path.write_text(
            json.dumps(mappings, ensure_ascii=False), encoding="utf-8"
        )
        logger.info(
            "mappings.json salvo em %s (usuários=%d, itens=%d)",
            mappings_path,
            len(strategy.user_to_idx),
            len(strategy.item_to_idx),
        )


if __name__ == "__main__":
    main()
