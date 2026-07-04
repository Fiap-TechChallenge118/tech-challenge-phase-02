"""Registro e promoção de modelo no MLflow Model Registry.

Módulo executável que:

1. Busca a versão mais recente do modelo registrado.
2. Transiciona o modelo de **Staging** → **Production**.
3. Anota a versão promovida com métricas do run de origem.

Fluxo típico::

    # 1. Treinar (já registra o modelo no Registry)
    uv run python -m src.training.trainer --frac 0.2

    # 2. Promover para Production
    uv run python -m src.training.register

    # 3. (Opcional) Especificar versão
    uv run python -m src.training.register --version 3
"""

from __future__ import annotations

import argparse
import logging

import mlflow
from mlflow.entities.model_registry import ModelVersion
from mlflow.tracking import MlflowClient

from src.settings import get_settings

logger = logging.getLogger(__name__)

# Descrição padrão anexada à versão quando promovida.
_DEFAULT_DESCRIPTION = (
    "Modelo promovido a Production após avaliação comparativa: "
    "MLP supera baseline Sklearn (LogReg) e serve como baseline "
    "treinada com PyTorch (embeddings + MLP). "
    "Métricas: P@10, R@10, NDCG@10, MAP@10 disponíveis em "
    "metrics/evaluation.json. "
    "Dataset: Instacart Market Basket (20% dos dados)."
)


def _get_client() -> MlflowClient:
    """Retorna um cliente MLflow autenticado.

    Returns:
        Cliente configurado com tracking URI das settings.
    """
    settings = get_settings()
    mlflow.set_tracking_uri(settings.mlflow_tracking_uri)
    return MlflowClient(tracking_uri=settings.mlflow_tracking_uri)


def _find_latest_version(
    client: MlflowClient, model_name: str, stage: str | None = None
) -> ModelVersion | None:
    """Encontra a versão mais recente do modelo, opcionalmente filtrada por stage.

    Args:
        client: Cliente MLflow.
        model_name: Nome do modelo registrado.
        stage: Filtrar por estágio (ex: "Staging", "Production", "None").

    Returns:
        ModelVersion mais recente ou None se nenhuma encontrada.
    """
    versions = client.search_model_versions(f"name='{model_name}'")
    if stage is not None:
        versions = [v for v in versions if v.current_stage == stage]
    if not versions:
        return None
    return max(versions, key=lambda v: int(v.version))


def _promote_to_production(
    client: MlflowClient,
    model_name: str,
    version: str,
    description: str,
) -> ModelVersion:
    """Promove uma versão do modelo para Production.

    Primeiro arquiva versões existentes em Production e então
    transiciona a versão alvo: None → Staging → Production.

    Args:
        client: Cliente MLflow.
        model_name: Nome do modelo registrado.
        version: Número da versão a promover.
        description: Descrição anexada à versão.

    Returns:
        ModelVersion após promoção.
    """
    # Arquiva versões atuais em Production (busca todas e filtra em Python)
    all_versions = client.search_model_versions(f"name='{model_name}'")
    prod_versions = [v for v in all_versions if v.current_stage == "Production"]
    for mv in prod_versions:
        if mv.version != version:
            client.transition_model_version_stage(
                name=model_name,
                version=mv.version,
                stage="Archived",
            )
            logger.info("Arquivada versão %s (estava em Production)", mv.version)

    # None → Staging → Production
    current = client.get_model_version(name=model_name, version=version)
    logger.info("Versão %s está em '%s'", version, current.current_stage)

    if current.current_stage != "Staging":
        client.transition_model_version_stage(
            name=model_name,
            version=version,
            stage="Staging",
            archive_existing_versions=False,
        )
        logger.info("Transicionada versão %s → Staging", version)

    client.transition_model_version_stage(
        name=model_name,
        version=version,
        stage="Production",
        archive_existing_versions=False,
    )

    client.update_model_version(
        name=model_name,
        version=version,
        description=description,
    )

    promoted = client.get_model_version(name=model_name, version=version)
    logger.info("Versão %s promovida → Production", version)
    return promoted


def _parse_args() -> argparse.Namespace:
    """CLI com opção de versão e descrição customizada."""
    parser = argparse.ArgumentParser(
        description="Registrar e promover modelo no MLflow Registry"
    )
    parser.add_argument(
        "--version",
        type=str,
        default=None,
        help="Versão específica a promover (default: última registrada).",
    )
    parser.add_argument(
        "--description",
        type=str,
        default=_DEFAULT_DESCRIPTION,
        help="Descrição anexada à versão promovida.",
    )
    parser.add_argument(
        "--stage",
        type=str,
        default="Production",
        choices=["Staging", "Production", "Archived"],
        help="Estágio alvo (default: Production).",
    )
    return parser.parse_args()


def main() -> None:
    """Ponto de entrada do script de registro."""
    args = _parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(levelname)-8s %(message)s",
    )

    settings = get_settings()
    client = _get_client()
    model_name = settings.mlflow_registered_model_name

    # Determina a versão alvo
    if args.version:
        target_version = args.version
        logger.info("Usando versão especificada: %s", target_version)
    else:
        latest = _find_latest_version(client, model_name)
        if latest is None:
            logger.error(
                "Nenhuma versão encontrada para '%s'. "
                "Execute o treino primeiro.",
                model_name,
            )
            raise SystemExit(1)
        target_version = latest.version
        logger.info(
            "Versão mais recente: %s (stage=%s)",
            target_version,
            latest.current_stage,
        )

    # Promove para Production
    promoted = _promote_to_production(
        client, model_name, target_version, args.description
    )

    logger.info("=" * 60)
    logger.info("Modelo em Production:")
    logger.info("  Nome:    %s", promoted.name)
    logger.info("  Versão:  %s", promoted.version)
    logger.info("  Stage:   %s", promoted.current_stage)
    logger.info("  Status:  %s", promoted.status)
    logger.info("  Desc:    %s", promoted.description)
    logger.info("=" * 60)

    # Lista todas as versões para conferência
    all_versions = client.search_model_versions(f"name='{model_name}'")
    logger.info("Histórico de versões:")
    for mv in sorted(all_versions, key=lambda v: int(v.version)):
        logger.info(
            "  v%s — stage=%-12s status=%s",
            mv.version,
            mv.current_stage,
            mv.status,
        )


if __name__ == "__main__":
    main()
