"""Utilitários de I/O compartilhados entre os módulos de training.

Centraliza funções que eram duplicadas entre ``trainer.py`` e ``evaluate.py``:

- :func:`load_pairs` — lê um parquet de pares ``(user_idx, item_idx, label)``.
- :func:`load_mappings` — lê ``mappings.json`` com ``user_to_idx``/``item_to_idx``.
- :func:`split_train_val_test` — divide índices em treino/validação/teste.
- :func:`load_config` — lê ``configs/config.yaml``.

Nenhuma função deste módulo lê CSVs brutos — os dados brutos são
responsabilidade exclusiva do stage ``preprocess`` (``src/data/preprocess.py``).
"""

import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

logger = logging.getLogger(__name__)


def load_pairs(path: Path) -> pd.DataFrame:
    """Lê um arquivo parquet de pares ``(user_idx, item_idx, label)``.

    Args:
        path: Caminho para o arquivo ``.parquet``.

    Returns:
        DataFrame com colunas ``user_idx``, ``item_idx`` e ``label``.

    Raises:
        FileNotFoundError: Se o arquivo não existir.
    """
    if not path.exists():
        msg = f"Arquivo de pares não encontrado: {path}. Execute o stage anterior."
        raise FileNotFoundError(msg)
    pairs = pd.read_parquet(path)
    logger.info("Pares carregados de %s (%d linhas)", path, len(pairs))
    return pairs


def load_mappings(path: Path) -> tuple[dict[int, int], dict[int, int]]:
    """Lê os mapeamentos ``user_to_idx`` e ``item_to_idx`` de um arquivo JSON.

    O arquivo é gerado pelo stage ``preprocess`` (``src/data/preprocess.py``)
    e garante que trainer e evaluate usem os mesmos índices contíguos.

    Args:
        path: Caminho para ``mappings.json``.

    Returns:
        Tupla ``(user_to_idx, item_to_idx)`` com chaves e valores como ``int``.

    Raises:
        FileNotFoundError: Se o arquivo não existir.
    """
    if not path.exists():
        msg = f"Mappings não encontrados: {path}. Execute o stage preprocess."
        raise FileNotFoundError(msg)
    with path.open(encoding="utf-8") as fh:
        data = json.load(fh)
    # * JSON serializa chaves como string — converte de volta para int
    user_to_idx: dict[int, int] = {
        int(k): int(v) for k, v in data["user_to_idx"].items()
    }
    item_to_idx: dict[int, int] = {
        int(k): int(v) for k, v in data["item_to_idx"].items()
    }
    logger.info(
        "Mappings carregados: %d usuários, %d itens",
        len(user_to_idx),
        len(item_to_idx),
    )
    return user_to_idx, item_to_idx


def split_train_val_test(
    n: int, val_frac: float, test_frac: float, seed: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Divide índices em treino, validação e teste deterministicamente.

    Args:
        n: Número total de amostras.
        val_frac: Fração reservada para validação.
        test_frac: Fração reservada para teste (holdout final).
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


def load_config(path: Path) -> dict:
    """Carrega o arquivo YAML de configuração do pipeline.

    Args:
        path: Caminho para ``configs/config.yaml``.

    Returns:
        Dicionário com os valores do YAML.

    Raises:
        FileNotFoundError: Se o arquivo não existir.
    """
    if not path.exists():
        msg = f"Arquivo de configuração não encontrado: {path}"
        raise FileNotFoundError(msg)
    with path.open(encoding="utf-8") as fh:
        return yaml.safe_load(fh)
