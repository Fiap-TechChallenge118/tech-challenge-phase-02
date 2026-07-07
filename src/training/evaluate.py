"""Avaliação comparativa: MLP vs baselines (≥4 métricas).

Módulo executável que:

1. Carrega o checkpoint ``models/model.pt`` e o test set
2. Re-processa os dados brutos para obter os pares de treino (mesmo seed)
3. Instancia os baselines (Popularity, Sklearn) e os treina no train set
4. Gera recomendações top-K para cada modelo
5. Calcula Precision@K, Recall@K, NDCG@K, MAP@K
6. Salva a tabela comparativa em ``metrics/evaluation.json``

Uso::

    uv run python -m src.training.evaluate
    uv run python -m src.training.evaluate --max-users 5000 --frac 0.2
"""

from __future__ import annotations

import argparse
import json
import logging
import math
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from src.features.preprocessing import (
    AggregatedFeaturesStrategy,
    InstacartFrames,
    InteractionPairsStrategy,
)
from src.models.baselines import PopularityBaseline, SklearnBaseline
from src.models.mlp import RecommendationMLP
from src.settings import get_settings
from src.training.metrics import build_test_pairs, evaluate_recommendations

logger = logging.getLogger(__name__)

# Chunks ajustados para 8 GB VRAM. Cada forward passa ~262k linhas
# (64 users × 4096 itens) que ocupam ≈ 500 MB nos tensores.
_MLP_USER_CHUNK = 64
_MLP_ITEM_CHUNK = 4096


def _load_mlp(path: Path, device: torch.device) -> tuple[RecommendationMLP, dict]:
    """Carrega o modelo MLP treinado e seu checkpoint.

    Args:
        path: Caminho para ``models/model.pt``.
        device: Dispositivo para carregar o modelo.

    Returns:
        Tupla ``(modelo, checkpoint)``.
    """
    ckpt = torch.load(path, map_location=device, weights_only=False)
    model = RecommendationMLP(
        n_users=ckpt["n_users"],
        n_items=ckpt["n_items"],
        embedding_dim=ckpt["embedding_dim"],
        hidden_dims=ckpt["hidden_dims"],
    )
    model.load_state_dict(ckpt["model_state_dict"])
    model.to(device)
    model.eval()
    logger.info(
        "MLP carregado: %d params, best_epoch=%d",
        sum(p.numel() for p in model.parameters()),
        ckpt.get("best_epoch", -1),
    )
    return model, ckpt


def _mlp_recommendations(
    model: RecommendationMLP,
    all_items: np.ndarray,
    test_users: np.ndarray,
    top_k: int,
    device: torch.device,
) -> np.ndarray:
    """Gera recomendações da MLP com pré-compute de embeddings.

    Calcula embeddings de usuário e item uma única vez e só repassa
    o MLP nos cross-joins, evitando re-embedding a cada chunk.

    Args:
        model: MLP treinada.
        all_items: Array com todos os item_idx do catálogo.
        test_users: Array de user_idx para avaliar.
        top_k: Número de recomendações por usuário.
        device: Dispositivo.

    Returns:
        Matriz ``(n_users, top_k)`` com item_idx ordenados por score.
    """
    n_items = len(all_items)
    n_users = len(test_users)

    # Pré-computa embeddings de todos os itens (~50k × 64 × 4B = 13 MB)
    item_ids_tensor = torch.as_tensor(all_items, device=device)
    with torch.no_grad():
        item_embs_all = model.item_embedding(item_ids_tensor)

    item_chunks = math.ceil(n_items / _MLP_ITEM_CHUNK)
    recs: list[np.ndarray] = []

    for u_start in range(0, n_users, _MLP_USER_CHUNK):
        u_end = min(u_start + _MLP_USER_CHUNK, n_users)
        batch_users = test_users[u_start:u_end]
        n_batch = len(batch_users)

        user_tensor = torch.as_tensor(batch_users, device=device)
        with torch.no_grad():
            user_embs = model.user_embedding(user_tensor)

        all_scores: list[torch.Tensor] = []
        for c in range(item_chunks):
            i_start = c * _MLP_ITEM_CHUNK
            i_end = min(i_start + _MLP_ITEM_CHUNK, n_items)
            item_embs = item_embs_all[i_start:i_end]
            chunk_size = i_end - i_start

            # Cross-join: user × item do chunk
            user_expanded = user_embs.repeat_interleave(chunk_size, dim=0)
            item_expanded = item_embs.repeat(n_batch, 1)
            combined = torch.cat([user_expanded, item_expanded], dim=-1)

            with torch.no_grad():
                chunk_scores = model.mlp(combined).view(n_batch, chunk_size)
            all_scores.append(chunk_scores)

        scores = torch.cat(all_scores, dim=1)
        top_idx = scores.argsort(dim=-1, descending=True)[:, :top_k]
        for j in range(n_batch):
            recs.append(all_items[top_idx[j].cpu().numpy()])

    return np.array(recs)


def _load_frames(data_dir: Path) -> InstacartFrames:
    """Carrega CSVs brutos do Instacart."""
    orders = pd.read_csv(data_dir / "orders.csv")
    prior = pd.read_csv(data_dir / "order_products__prior.csv")
    train_orders = pd.read_csv(data_dir / "order_products__train.csv")
    products = pd.read_csv(data_dir / "products.csv")
    aisles = pd.read_csv(data_dir / "aisles.csv")
    departments = pd.read_csv(data_dir / "departments.csv")
    return InstacartFrames(
        orders=orders,
        order_products=pd.concat([prior, train_orders], ignore_index=True),
        products=products,
        aisles=aisles,
        departments=departments,
    )


def _split_train_val_test(
    n: int, val_frac: float, test_frac: float, seed: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Divide índices em treino, validação e teste deterministicamente."""
    rng = np.random.default_rng(seed)
    indices = rng.permutation(n)
    test_end = int(n * test_frac)
    val_end = test_end + int(n * val_frac)
    return indices[val_end:], indices[test_end:val_end], indices[:test_end]


def _parse_args() -> argparse.Namespace:
    """CLI com opção de limitar usuários e fração dos dados."""
    parser = argparse.ArgumentParser(
        description="Avaliação comparativa MLP vs baselines"
    )
    parser.add_argument(
        "--max-users",
        type=int,
        default=5000,
        help="Número máximo de usuários do test set (default: 5000)",
    )
    parser.add_argument(
        "--frac",
        type=float,
        default=0.2,
        help="Fração dos dados usada no treino (default: 0.2)",
    )
    return parser.parse_args()


def main() -> None:
    """Ponto de entrada do script de avaliação."""
    args = _parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(levelname)-8s %(message)s",
    )

    settings = get_settings()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    top_k = settings.evaluation_k
    model_path = Path("models/model.pt")
    test_path = Path(settings.data_processed_dir) / "test_pairs.parquet"

    if not model_path.exists():
        logger.error("Checkpoint não encontrado: %s", model_path)
        raise SystemExit(1)

    # ---- Carregar MLP ----------------------------------------------------
    mlp, ckpt = _load_mlp(model_path, device)

    # ---- Re-processar dados para baselines --------------------------------
    # Replica o pipeline de treino com o mesmo seed para obter os pares
    # de treino SEM vazamento do test set.
    logger.info("Carregando dados brutos e gerando pares de interação...")
    frames = _load_frames(Path(settings.data_raw_dir))
    strategy = InteractionPairsStrategy(seed=settings.random_seed)
    all_pairs = strategy.fit_transform(frames)
    logger.info(
        "Pares gerados: %d (usuários=%d, itens=%d)",
        len(all_pairs),
        len(strategy.user_to_idx),
        len(strategy.item_to_idx),
    )

    if args.frac < 1.0:
        all_pairs = all_pairs.sample(
            frac=args.frac, random_state=settings.random_seed
        )
        logger.info(
            "Usando %.0f%% dos dados = %d pares",
            args.frac * 100,
            len(all_pairs),
        )

    # Split triplo determinístico (mesmo do treino)
    train_idx, val_idx, test_idx = _split_train_val_test(
        len(all_pairs),
        settings.validation_split,
        settings.test_split,
        settings.random_seed,
    )
    train_pairs = all_pairs.iloc[train_idx].copy()

    logger.info(
        "Split: %d treino / %d validação / %d teste",
        len(train_idx), len(val_idx), len(test_idx),
    )

    # ---- Carregar test set (holdout original) ----------------------------
    if test_path.exists():
        test_pairs = pd.read_parquet(test_path)
        logger.info("Test set carregado de %s (%d pares)", test_path, len(test_pairs))
    else:
        test_pairs = all_pairs.iloc[test_idx].copy()
        logger.warning("Test set não encontrado, usando split atual.")

    all_test_users = np.sort(test_pairs["user_idx"].unique())

    # Amostra usuários para avaliação rápida
    rng = np.random.default_rng(settings.random_seed)
    if args.max_users and args.max_users < len(all_test_users):
        test_users = rng.choice(
            all_test_users, size=args.max_users, replace=False
        )
        test_users.sort()
    else:
        test_users = all_test_users

    user_mask = test_pairs["user_idx"].isin(test_users)
    filtered_pairs = test_pairs[user_mask]
    test_pairs_list = build_test_pairs(filtered_pairs)

    logger.info(
        "Avaliando %d usuários (%d com itens positivos)",
        len(test_users),
        len(test_pairs_list),
    )

    # Catálogo completo de itens (do checkpoint, que cobre todos os itens)
    all_items = np.array(sorted(ckpt["item_to_idx"].values()), dtype=np.int64)
    logger.info("Catálogo: %d itens", len(all_items))

    # ---- MLP -------------------------------------------------------------
    logger.info("Gerando recomendações da MLP (pré-compute de embeddings)...")
    mlp_recs = _mlp_recommendations(mlp, all_items, test_users, top_k, device)
    mlp_metrics = evaluate_recommendations(mlp_recs, test_pairs_list, top_k)
    logger.info(
        "MLP  — P@%d: %.4f | R@%d: %.4f | NDCG@%d: %.4f | MAP@%d: %.4f",
        top_k,
        mlp_metrics["precision_at_k"],
        top_k,
        mlp_metrics["recall_at_k"],
        top_k,
        mlp_metrics["ndcg_at_k"],
        top_k,
        mlp_metrics["map_at_k"],
    )

    # ---- Popularity Baseline (treinado no TRAIN set) ---------------------
    logger.info("Treinando PopularityBaseline no train set...")
    pop = PopularityBaseline().fit(train_pairs)
    pop_recs = pop.predict(test_users, top_k=top_k)
    pop_metrics = evaluate_recommendations(pop_recs, test_pairs_list, top_k)
    logger.info(
        "Pop  — P@%d: %.4f | R@%d: %.4f | NDCG@%d: %.4f | MAP@%d: %.4f",
        top_k,
        pop_metrics["precision_at_k"],
        top_k,
        pop_metrics["recall_at_k"],
        top_k,
        pop_metrics["ndcg_at_k"],
        top_k,
        pop_metrics["map_at_k"],
    )

    # ---- Sklearn Baseline (treinado no TRAIN set) ------------------------
    logger.info("Treinando SklearnBaseline no train set...")
    agg = AggregatedFeaturesStrategy(scale=True).fit_transform(frames)
    agg = agg.rename(columns={"user_id": "user_idx"})
    sk = SklearnBaseline(seed=settings.random_seed).fit(agg, train_pairs)
    sk_recs = sk.predict(
        agg[agg["user_idx"].isin(test_users)],
        top_k=top_k,
    )
    sk_metrics = evaluate_recommendations(sk_recs, test_pairs_list, top_k)
    logger.info(
        "SKL  — P@%d: %.4f | R@%d: %.4f | NDCG@%d: %.4f | MAP@%d: %.4f",
        top_k,
        sk_metrics["precision_at_k"],
        top_k,
        sk_metrics["recall_at_k"],
        top_k,
        sk_metrics["ndcg_at_k"],
        top_k,
        sk_metrics["map_at_k"],
    )

    # ---- Resultados ------------------------------------------------------
    results = {
        "top_k": top_k,
        "test_users": int(len(test_pairs_list)),
        "sampled_from": int(len(all_test_users)),
        "models": {
            "MLP (PyTorch)": mlp_metrics,
            "Popularity": pop_metrics,
            "Sklearn (LogReg)": sk_metrics,
        },
    }

    metrics_dir = Path("metrics")
    metrics_dir.mkdir(exist_ok=True)
    out = metrics_dir / "evaluation.json"
    out.write_text(
        json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    logger.info("Métricas salvas em %s", out)

    # ---- Tabela ----------------------------------------------------------
    logger.info("=" * 70)
    logger.info("TABELA COMPARATIVA — Métricas@%d", top_k)
    logger.info(
        "%-25s %12s %12s %12s %12s",
        "", "Precision", "Recall", "NDCG", "MAP",
    )
    for name, m in results["models"].items():
        logger.info(
            "%-25s %12.4f %12.4f %12.4f %12.4f",
            name,
            m["precision_at_k"],
            m["recall_at_k"],
            m["ndcg_at_k"],
            m["map_at_k"],
        )
    logger.info("=" * 70)


if __name__ == "__main__":
    main()
