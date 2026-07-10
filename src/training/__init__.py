"""Loop de treino, dataset, avaliação e I/O do modelo de recomendação.

Exporta o ``InteractionDataset`` (Dataset PyTorch) e as funções de I/O
compartilhadas entre os scripts de treino e avaliação.
"""

from src.training.dataset import InteractionDataset
from src.training.io import load_config, load_mappings, load_pairs, split_train_val_test

__all__ = [
    "InteractionDataset",
    "load_config",
    "load_mappings",
    "load_pairs",
    "split_train_val_test",
]
