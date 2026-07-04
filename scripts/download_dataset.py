"""Download do dataset Instacart via kagglehub e cópia para ``data/raw/``.

Uso::

    uv run python scripts/download_dataset.py
"""

import logging
import shutil
from pathlib import Path

import kagglehub

logger = logging.getLogger(__name__)


def main() -> None:
    """Baixa o dataset Instacart e move os CSVs para ``data/raw/``."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(levelname)-8s %(message)s",
    )

    raw_dir = Path("data/raw")
    raw_dir.mkdir(parents=True, exist_ok=True)

    logger.info("Baixando dataset Instacart via kagglehub...")
    kaggle_path = Path(
        kagglehub.dataset_download(
            "yasserh/instacart-online-grocery-basket-analysis-dataset"
        )
    )
    logger.info("Dataset baixado em: %s", kaggle_path)

    # Copia todos os CSVs do diretório de download para data/raw/
    csv_files = list(kaggle_path.glob("*.csv"))
    if not csv_files:
        logger.warning("Nenhum CSV encontrado em %s", kaggle_path)
        return

    for csv_file in csv_files:
        dest = raw_dir / csv_file.name
        shutil.copy2(csv_file, dest)
        logger.info("  copied %s → data/raw/%s", csv_file.name, csv_file.name)

    logger.info(
        "Download concluído! %d arquivos copiados para %s",
        len(csv_files),
        raw_dir.absolute(),
    )


if __name__ == "__main__":
    main()
