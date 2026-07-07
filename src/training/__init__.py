"""Loop de treino, dataset e avaliação do modelo de recomendação.

Exporta o ``InteractionDataset`` (Dataset PyTorch) e o script de treino
executável ``trainer.py``.
"""

from src.training.dataset import InteractionDataset

__all__ = [
    "InteractionDataset",
]
