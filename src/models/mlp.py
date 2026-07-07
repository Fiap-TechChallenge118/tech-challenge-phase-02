"""MLP com embeddings de usuário e item para recomendação.

Arquitetura embedding-based: cada ``user_idx`` e ``item_idx`` é projetado em um
vetor denso de dimensão ``embedding_dim``. Os dois vetores são concatenados e
passam por uma MLP com camadas ``Linear → BatchNorm → ReLU → Dropout`` (se
ativadas) até um score final via sigmoid.

A MLP é construída dinamicamente a partir de ``configs/config.yaml`` (ou dos
argumentos do construtor), portanto o número de camadas, dropout, ativação e
BatchNorm são configuráveis sem alterar código.
"""

from typing import Literal

import torch
import torch.nn as nn


class RecommendationMLP(nn.Module):
    """Rede neural de recomendação com embeddings + MLP.

    Args:
        n_users: Número de usuários distintos no vocabulário (tamanho da
            camada de embedding de usuário).
        n_items: Número de itens distintos no vocabulário (tamanho da
            camada de embedding de item).
        embedding_dim: Dimensão dos vetores de embedding de usuário e item.
        hidden_dims: Lista com as unidades de cada camada oculta da MLP.
        dropout: Taxa de dropout aplicada após cada camada oculta.
        activation: Função de ativação: ``"relu"``, ``"gelu"`` ou ``"tanh"``.
        batch_norm: Se ``True``, insere ``BatchNorm1d`` após cada camada linear.
    """

    def __init__(
        self,
        n_users: int,
        n_items: int,
        embedding_dim: int = 64,
        hidden_dims: list[int] | None = None,
        dropout: float = 0.3,
        activation: Literal["relu", "gelu", "tanh"] = "relu",
        batch_norm: bool = True,
    ) -> None:
        super().__init__()
        if hidden_dims is None:
            hidden_dims = [256, 128, 64]

        self.n_users = n_users
        self.n_items = n_items
        self.embedding_dim = embedding_dim

        self.user_embedding = nn.Embedding(n_users, embedding_dim)
        self.item_embedding = nn.Embedding(n_items, embedding_dim)

        self.mlp = self._build_mlp(
            embedding_dim, hidden_dims, dropout, activation, batch_norm
        )

        self._init_weights()

    @staticmethod
    def _build_mlp(
        embedding_dim: int,
        hidden_dims: list[int],
        dropout: float,
        activation: str,
        batch_norm: bool,
    ) -> nn.Sequential:
        """Constrói as camadas da MLP dinamicamente.

        Args:
            embedding_dim: Dimensão da entrada (2 × embedding_dim).
            hidden_dims: Unidades por camada oculta.
            dropout: Taxa de dropout.
            activation: Nome da função de ativação.
            batch_norm: Se ``True``, adiciona BatchNorm após cada Linear.

        Returns:
            MLP sequencial com a arquitetura especificada.
        """
        activations: dict[str, type[nn.Module]] = {
            "relu": nn.ReLU,
            "gelu": nn.GELU,
            "tanh": nn.Tanh,
        }
        activation_fn = activations[activation]

        layers: list[nn.Module] = []
        input_dim = 2 * embedding_dim
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(input_dim, hidden_dim))
            if batch_norm:
                layers.append(nn.BatchNorm1d(hidden_dim))
            layers.append(activation_fn())
            layers.append(nn.Dropout(dropout))
            input_dim = hidden_dim
        layers.append(nn.Linear(input_dim, 1))
        return nn.Sequential(*layers)

    def _init_weights(self) -> None:
        """Inicializa embeddings (normal) e camadas lineares (Xavier)."""
        nn.init.normal_(self.user_embedding.weight, mean=0.0, std=0.01)
        nn.init.normal_(self.item_embedding.weight, mean=0.0, std=0.01)
        for module in self.mlp:
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def forward(self, user_idx: torch.Tensor, item_idx: torch.Tensor) -> torch.Tensor:
        """Propaga os índices de usuário e item pela rede.

        Args:
            user_idx: Tensor de índices de usuário com shape ``(batch_size,)``.
            item_idx: Tensor de índices de item com shape ``(batch_size,)``.

        Returns:
            Logits (score bruto) com shape ``(batch_size, 1)``.
            Use ``torch.sigmoid`` para converter em probabilidade.
        """
        user_emb = self.user_embedding(user_idx)
        item_emb = self.item_embedding(item_idx)
        combined = torch.cat([user_emb, item_emb], dim=-1)
        return self.mlp(combined)
