"""Carregamento do modelo e geração de recomendações top-K.

Encapsula o checkpoint ``models/model.pt`` (pesos + vocabulários user/item) e
expõe uma única operação de negócio: dado um ``user_id`` original do dataset,
retornar os ``k`` produtos com maior score previsto pela MLP.

O checkpoint é autossuficiente — carrega ``user_to_idx`` e ``item_to_idx`` junto
com os pesos, dispensando os artefatos do pipeline em runtime.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

import torch

from src.models.mlp import RecommendationMLP

logger = logging.getLogger(__name__)

# Itens pontuados por forward pass. Mantém o uso de memória previsível em
# instâncias pequenas (App Runner roda com 2 GB).
_ITEM_CHUNK = 4096


@dataclass(frozen=True)
class Recommendation:
    """Um produto recomendado.

    Args:
        product_id: ``product_id`` original do dataset Instacart.
        score: Probabilidade prevista de o usuário comprar o produto.
    """

    product_id: int
    score: float


class ModelService:
    """Serve recomendações a partir do checkpoint treinado.

    Args:
        checkpoint_path: Caminho para ``models/model.pt``.
        device: Dispositivo do PyTorch. Default: CPU (o container não tem GPU).
    """

    def __init__(
        self,
        checkpoint_path: Path,
        device: torch.device | None = None,
    ) -> None:
        self._device = device or torch.device("cpu")
        self._model, self._user_to_idx, self._item_to_idx = self._load(checkpoint_path)
        self._idx_to_item = {idx: item for item, idx in self._item_to_idx.items()}

    def _load(
        self, path: Path
    ) -> tuple[RecommendationMLP, dict[str, int], dict[str, int]]:
        """Carrega pesos e vocabulários do checkpoint.

        Args:
            path: Caminho para o arquivo ``.pt``.

        Returns:
            Tupla ``(modelo em modo eval, user_to_idx, item_to_idx)``.

        Raises:
            FileNotFoundError: Se o checkpoint não existir.
        """
        if not path.exists():
            raise FileNotFoundError(f"Checkpoint não encontrado: {path}")

        ckpt = torch.load(path, map_location=self._device, weights_only=False)
        model = RecommendationMLP(
            n_users=ckpt["n_users"],
            n_items=ckpt["n_items"],
            embedding_dim=ckpt["embedding_dim"],
            hidden_dims=ckpt["hidden_dims"],
        )
        model.load_state_dict(ckpt["model_state_dict"])
        model.to(self._device)
        model.eval()
        logger.info(
            "Modelo carregado: %d usuários, %d itens, best_epoch=%s",
            ckpt["n_users"],
            ckpt["n_items"],
            ckpt.get("best_epoch", "?"),
        )
        return model, ckpt["user_to_idx"], ckpt["item_to_idx"]

    @property
    def n_users(self) -> int:
        """Número de usuários no vocabulário do modelo."""
        return len(self._user_to_idx)

    @property
    def n_items(self) -> int:
        """Número de itens no vocabulário do modelo."""
        return len(self._item_to_idx)

    def knows_user(self, user_id: int) -> bool:
        """Informa se o usuário existe no vocabulário (evita cold start).

        Args:
            user_id: ``user_id`` original do dataset.

        Returns:
            ``True`` se o usuário foi visto no treino.
        """
        return str(user_id) in self._user_to_idx or user_id in self._user_to_idx

    def _user_index(self, user_id: int) -> int:
        """Traduz ``user_id`` original para o índice interno do embedding."""
        # As chaves viram string ao passar por JSON; o checkpoint pode ter ambos.
        if str(user_id) in self._user_to_idx:
            return int(self._user_to_idx[str(user_id)])
        return int(self._user_to_idx[user_id])

    @torch.inference_mode()
    def recommend(self, user_id: int, k: int = 10) -> list[Recommendation]:
        """Retorna os ``k`` produtos com maior score para o usuário.

        Pontua o catálogo inteiro em blocos e seleciona o top-K.

        Args:
            user_id: ``user_id`` original do dataset Instacart.
            k: Quantidade de recomendações.

        Returns:
            Lista de ``Recommendation`` ordenada por score decrescente.

        Raises:
            KeyError: Se o usuário não existir no vocabulário do modelo.
        """
        if not self.knows_user(user_id):
            raise KeyError(f"Usuário {user_id} não existe no vocabulário do modelo.")

        user_idx = self._user_index(user_id)
        scores = self._score_catalog(user_idx)
        top_scores, top_indices = torch.topk(scores, k=min(k, scores.numel()))

        return [
            Recommendation(
                product_id=int(self._idx_to_item[int(idx)]),
                score=float(score),
            )
            for score, idx in zip(top_scores, top_indices, strict=True)
        ]

    def _score_catalog(self, user_idx: int) -> torch.Tensor:
        """Pontua todos os itens do catálogo para um usuário.

        Args:
            user_idx: Índice interno do usuário.

        Returns:
            Tensor 1-D com um score (probabilidade) por item do catálogo.
        """
        n_items = self._model.item_embedding.num_embeddings
        all_scores: list[torch.Tensor] = []

        for start in range(0, n_items, _ITEM_CHUNK):
            items = torch.arange(
                start, min(start + _ITEM_CHUNK, n_items), device=self._device
            )
            users = torch.full_like(items, user_idx)
            logits = self._model(users, items).squeeze(-1)
            all_scores.append(torch.sigmoid(logits))

        return torch.cat(all_scores)
