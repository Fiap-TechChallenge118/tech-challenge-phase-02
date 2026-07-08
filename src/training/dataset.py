"""Dataset PyTorch para pares de interação user-item.

Converte o ``DataFrame`` produzido pelo ``InteractionPairsStrategy`` (colunas
``user_idx``, ``item_idx``, ``label``) em tensores prontos para o ``DataLoader``.
"""

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset


class InteractionDataset(Dataset[tuple[torch.Tensor, torch.Tensor, torch.Tensor]]):
    """Dataset que encapsula pares ``(user_idx, item_idx, label)``.

    Args:
        pairs: DataFrame com colunas ``user_idx``, ``item_idx`` e ``label``,
            como produzido por ``InteractionPairsStrategy.fit_transform``.
    """

    def __init__(self, pairs: pd.DataFrame) -> None:
        self.users = torch.as_tensor(pairs["user_idx"].to_numpy(dtype=np.int64))
        self.items = torch.as_tensor(pairs["item_idx"].to_numpy(dtype=np.int64))
        self.labels = torch.as_tensor(
            pairs["label"].to_numpy(dtype=np.float32)
        ).unsqueeze(1)

    def __len__(self) -> int:
        """Retorna o número de pares no dataset."""
        return len(self.users)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Retorna a tripla ``(user_idx, item_idx, label)`` na posição ``idx``."""
        return self.users[idx], self.items[idx], self.labels[idx]
