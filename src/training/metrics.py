"""Métricas de avaliação de recomendação orientadas a ranking.

Implementa as quatro métricas exigidas pelo Tech Challenge:

- **Precision@K**: fração dos itens recomendados que são relevantes.
- **Recall@K**: fração dos itens relevantes que foram recomendados.
- **NDCG@K**: Normalized Discounted Cumulative Gain (posição no ranking).
- **MAP@K**: Mean Average Precision (precisão acumulada média).

Todas operam sobre arrays NumPy e seguem a abordagem: para cada usuário,
as recomendações (item_idx) são comparadas com os itens relevantes
(ground truth do test set). O resultado final é a média sobre os usuários.
"""

import numpy as np
import pandas as pd


def _precision_at_k(user_recs: np.ndarray, user_positives: set[int], k: int) -> float:
    """Precision@K para um único usuário.

    Args:
        user_recs: Itens recomendados (ordenados do mais relevante).
        user_positives: Itens realmente relevantes (comprados).
        k: Número de recomendações a considerar.

    Returns:
        Fração dos top-K itens que são relevantes, em [0, 1].
    """
    if k == 0:
        return 0.0
    top_k = user_recs[:k]
    hits = sum(1 for item in top_k if item in user_positives)
    return hits / k


def _recall_at_k(user_recs: np.ndarray, user_positives: set[int], k: int) -> float:
    """Recall@K para um único usuário.

    Args:
        user_recs: Itens recomendados.
        user_positives: Itens relevantes.
        k: Número de recomendações a considerar.

    Returns:
        Fração dos itens relevantes que estão nos top-K, em [0, 1].
    """
    if len(user_positives) == 0:
        return 0.0
    top_k = set(user_recs[:k].tolist())
    hits = len(top_k & user_positives)
    return hits / len(user_positives)


def _ndcg_at_k(user_recs: np.ndarray, user_positives: set[int], k: int) -> float:
    """NDCG@K para um único usuário.

    Args:
        user_recs: Itens recomendados (ordenados).
        user_positives: Itens relevantes.
        k: Número de recomendações a considerar.

    Returns:
        NDCG@K em [0, 1]. 1 = ranking perfeito.
    """
    if k == 0 or len(user_positives) == 0:
        return 0.0

    dcg = 0.0
    for i, item in enumerate(user_recs[:k]):
        if item in user_positives:
            dcg += 1.0 / np.log2(i + 2)  # i+2 pois i começa em 0

    ideal_hits = min(len(user_positives), k)
    idcg = sum(1.0 / np.log2(i + 2) for i in range(ideal_hits))

    return dcg / idcg if idcg > 0 else 0.0


def _average_precision_at_k(
    user_recs: np.ndarray, user_positives: set[int], k: int
) -> float:
    """AP@K (Average Precision) para um único usuário.

    Args:
        user_recs: Itens recomendados (ordenados).
        user_positives: Itens relevantes.
        k: Número de recomendações a considerar.

    Returns:
        AP@K em [0, 1].
    """
    if k == 0 or len(user_positives) == 0:
        return 0.0

    hits = 0
    ap = 0.0
    for i, item in enumerate(user_recs[:k]):
        if item in user_positives:
            hits += 1
            ap += hits / (i + 1)
    return ap / min(len(user_positives), k)


def evaluate_recommendations(
    recommendations: np.ndarray,
    test_pairs: list[tuple[int, set[int]]],
    k: int,
) -> dict[str, float]:
    """Calcula as 4 métricas de ranking sobre um conjunto de usuários.

    Args:
        recommendations: Matriz ``(n_users, top_k)`` com item_idx recomendados.
        test_pairs: Lista de ``(user_idx, set_itens_relevantes)`` para cada
            usuário do test set.
        k: Top-K a considerar.

    Returns:
        Dicionário com ``precision_at_k``, ``recall_at_k``, ``ndcg_at_k``,
        ``map_at_k`` (médias sobre os usuários).
    """
    metrics: dict[str, list[float]] = {
        "precision_at_k": [],
        "recall_at_k": [],
        "ndcg_at_k": [],
        "map_at_k": [],
    }

    for i, (user_idx, positives) in enumerate(test_pairs):
        recs = recommendations[i]
        metrics["precision_at_k"].append(_precision_at_k(recs, positives, k))
        metrics["recall_at_k"].append(_recall_at_k(recs, positives, k))
        metrics["ndcg_at_k"].append(_ndcg_at_k(recs, positives, k))
        metrics["map_at_k"].append(_average_precision_at_k(recs, positives, k))

    return {name: float(np.mean(values)) for name, values in metrics.items()}


def build_test_pairs(test_df: pd.DataFrame) -> list[tuple[int, set[int]]]:
    """Constrói pares de avaliação a partir do DataFrame de teste.

    Agrupa os itens positivos (comprados) por usuário.

    Args:
        test_df: DataFrame com ``user_idx``, ``item_idx``, ``label``.

    Returns:
        Lista de ``(user_idx, set_itens_relevantes)`` ordenada por user_idx.
    """
    positives = test_df[test_df["label"] == 1]
    grouped = positives.groupby("user_idx")["item_idx"].apply(set)
    return [(int(uid), set(items)) for uid, items in grouped.items()]
