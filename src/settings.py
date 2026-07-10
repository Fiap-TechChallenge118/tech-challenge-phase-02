"""Configurações centralizadas da aplicação via Pydantic BaseSettings.

Todas as configurações são lidas de variáveis de ambiente e/ou de um arquivo .env.
Hiperparâmetros do pipeline de ML vivem em configs/config.yaml;
apenas valores de runtime e infraestrutura ficam aqui.

Uso::

    from src.settings import get_settings

    settings = get_settings()
    print(settings.mlflow_tracking_uri)
"""

import json
from functools import lru_cache
from pathlib import Path
from typing import Any, ClassVar, Literal

from pydantic import Field, field_validator
from pydantic_settings import (
    BaseSettings,
    DotEnvSettingsSource,
    EnvSettingsSource,
    SettingsConfigDict,
)

from src.features.preprocessing.enums import PreprocessorStrategyName

# * ---------------------------------------------------------------------------
# * Utilitário público — também usado diretamente nos testes unitários
# * ---------------------------------------------------------------------------


def _coerce_int_list(value: object) -> list[int]:
    """Converte string separada por vírgulas, array JSON, lista ou tupla em list[int].

    Aceita tanto o formato legível ('256,128,64') quanto o formato JSON estrito
    do pydantic-settings ('[256,128,64]').

    Args:
        value: Valor bruto da variável de ambiente, arquivo .env ou código Python.

    Returns:
        Lista de inteiros.

    Raises:
        ValueError: Se algum elemento não puder ser convertido para int.
        TypeError: Se o tipo não puder ser convertido.
    """
    if isinstance(value, str):
        stripped = value.strip()
        # * Tenta interpretar como array JSON primeiro (ex: '[256,128,64]')
        if stripped.startswith("["):
            return [int(v) for v in json.loads(stripped)]
        # * Fallback para formato separado por vírgulas (ex: '256,128,64')
        parts = [p.strip() for p in stripped.split(",") if p.strip()]
        return [int(p) for p in parts]
    if isinstance(value, (list, tuple)):
        return [int(v) for v in value]
    msg = f"Não é possível converter {type(value).__name__!r} para list[int]."
    raise TypeError(msg)


# * ---------------------------------------------------------------------------
# * Mixin compartilhado para parsing flexível de campos de lista
# * ---------------------------------------------------------------------------


class _CommaListMixin:
    """Mixin que normaliza campos de lista separados por vírgula antes do decode JSON.

    O pydantic-settings chama json.loads em campos do tipo list[] lidos de env vars.
    Este mixin intercepta a string bruta e converte valores separados por vírgula
    como '256,128,64' para um array JSON válido '[256,128,64]' antes que a
    classe pai tente fazer o parse.
    """

    # ! Campos que aceitam strings separadas por vírgula além do formato JSON
    _COMMA_LIST_FIELDS: ClassVar[frozenset[str]] = frozenset({"hidden_layers"})

    def prepare_field_value(  # type: ignore[override]
        self,
        field_name: str,
        field: Any,
        value: Any,
        value_is_complex: bool,  # noqa: FBT001
    ) -> Any:
        """Normaliza valores de lista separados por vírgula para JSON antes do parse.

        Args:
            field_name: Nome do campo de settings.
            field: Informações do campo Pydantic.
            value: Valor bruto da string da source.
            value_is_complex: Se o pydantic-settings trata o campo como complexo.

        Returns:
            Valor normalizado, convertido para array JSON ou mantido sem alteração.
        """
        if (
            field_name in self._COMMA_LIST_FIELDS
            and isinstance(value, str)
            and not value.strip().startswith("[")
        ):
            # * Converte '256,128,64' → '[256,128,64]'
            # * para que o json.loads do pydantic-settings consiga interpretar
            parts = [p.strip() for p in value.split(",") if p.strip()]
            value = json.dumps([int(p) for p in parts])
        return super().prepare_field_value(  # type: ignore[misc]
            field_name, field, value, value_is_complex
        )


class _FlexibleEnvSource(_CommaListMixin, EnvSettingsSource):
    """Source de variáveis de processo com parsing flexível de listas."""


class _FlexibleDotEnvSource(_CommaListMixin, DotEnvSettingsSource):
    """Source de arquivo .env com parsing flexível de listas."""


# * ---------------------------------------------------------------------------
# * Settings
# * ---------------------------------------------------------------------------


class Settings(BaseSettings):
    """Configurações de runtime carregadas de variáveis de ambiente / arquivo .env."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    @classmethod
    def settings_customise_sources(  # type: ignore[override]
        cls,
        settings_cls: type[BaseSettings],
        init_settings: Any,
        env_settings: Any,
        dotenv_settings: Any,
        file_secret_settings: Any,
    ) -> tuple[Any, ...]:
        """Substitui as sources padrão de env e dotenv pelas versões flexíveis.

        Args:
            settings_cls: A classe Settings sendo construída.
            init_settings: Source para valores passados no init.
            env_settings: Source padrão de variáveis de processo (substituída).
            dotenv_settings: Source para valores do arquivo .env (substituída).
            file_secret_settings: Source para diretório de secrets.

        Returns:
            Tupla de sources de settings em ordem de prioridade.
        """
        # * Substitui as sources padrão pelas versões com suporte a CSV em listas
        return (
            init_settings,
            _FlexibleEnvSource(settings_cls),
            _FlexibleDotEnvSource(settings_cls, env_file=".env"),
            file_secret_settings,
        )

    # * -----------------------------------------------------------------------
    # * Projeto
    # * -----------------------------------------------------------------------

    project_name: str = Field(
        default="tech-challenge-02",
        description="Nome do projeto usado em logs e no MLflow.",
    )

    env: Literal["development", "staging", "production"] = Field(
        default="development",
        description="Ambiente de execução.",
    )

    random_seed: int = Field(
        default=42,
        ge=0,
        description="Semente global para PyTorch, NumPy e random do Python.",
    )

    # * -----------------------------------------------------------------------
    # * MLflow
    # * -----------------------------------------------------------------------

    mlflow_tracking_uri: str = Field(
        default="http://localhost:5000",
        description="URI do servidor de tracking do MLflow "
        "(sqlite:/// suporta Registry sem servidor).",
    )

    mlflow_experiment_name: str = Field(
        default="recommendation-system",
        description="Nome do experimento no MLflow.",
    )

    mlflow_registered_model_name: str = Field(
        default="recommender-mlp",
        description="Nome do modelo registrado no MLflow Model Registry.",
    )

    mlflow_tracking_username: str | None = Field(
        default=None,
        description="Usuário opcional para autenticação básica do MLflow.",
    )

    mlflow_tracking_password: str | None = Field(
        default=None,
        description="Senha opcional para autenticação básica do MLflow.",
    )

    # * -----------------------------------------------------------------------
    # * DVC
    # * -----------------------------------------------------------------------

    dvc_remote_type: Literal["local", "s3", "gs", "azure"] = Field(
        default="local",
        description="Tipo de armazenamento remoto do DVC.",
    )

    dvc_remote_url: str = Field(
        default="/tmp/dvc-remote",
        description="Caminho ou URI do remote DVC.",
    )

    # * -----------------------------------------------------------------------
    # * Caminhos de dados
    # * -----------------------------------------------------------------------

    data_dir: str = Field(
        default="data",
        description="Diretório raiz dos dados.",
    )

    data_raw_dir: str = Field(
        default="data/raw",
        description="Diretório para arquivos de entrada brutos.",
    )

    data_processed_dir: str = Field(
        default="data/processed",
        description="Diretório para outputs processados do pipeline.",
    )

    dataset_name: Literal["instacart"] = Field(
        default="instacart",
        description="Identificador do dataset.",
    )

    # * -----------------------------------------------------------------------
    # * Pré-processamento
    # * -----------------------------------------------------------------------

    preprocessing_strategy: PreprocessorStrategyName = Field(
        default=PreprocessorStrategyName.INTERACTION_PAIRS,
        description=(
            "Estratégia de pré-processamento via PreprocessorFactory. "
            "Definida em configs/config.yaml (data.strategy)."
        ),
    )

    n_negatives: int = Field(
        default=1,
        ge=1,
        description=(
            "Negativos amostrados por par positivo (interaction_pairs). "
            "Definido em configs/config.yaml (data.n_negatives)."
        ),
    )

    # * -----------------------------------------------------------------------
    # * Hiperparâmetros do modelo (sobrescrevem os valores do config.yaml)
    # * -----------------------------------------------------------------------

    embedding_dim: int = Field(
        default=64,
        ge=1,
        description="Dimensão dos embeddings de usuários e itens.",
    )

    hidden_layers: list[int] = Field(
        default=[256, 128, 64],
        description="Tamanhos das camadas ocultas da MLP. Aceita JSON ou CSV no env.",
    )

    dropout_rate: float = Field(
        default=0.3,
        ge=0.0,
        lt=1.0,
        description="Taxa de dropout aplicada após cada camada oculta.",
    )

    learning_rate: float = Field(
        default=0.001,
        gt=0.0,
        description="Taxa de aprendizado inicial para o optimizer Adam.",
    )

    max_epochs: int = Field(
        default=50,
        ge=1,
        description="Número máximo de épocas de treinamento.",
    )

    batch_size: int = Field(
        default=1024,
        ge=1,
        description="Tamanho do batch de treinamento.",
    )

    early_stopping_patience: int = Field(
        default=5,
        ge=1,
        description="Épocas sem melhora na val_loss antes de parar o treinamento.",
    )

    # * -----------------------------------------------------------------------
    # * Avaliação
    # * -----------------------------------------------------------------------

    evaluation_k: int = Field(
        default=10,
        ge=1,
        description="K para Precision@K, Recall@K, NDCG@K e MAP@K.",
    )

    validation_split: float = Field(
        default=0.1,
        gt=0.0,
        lt=1.0,
        description="Fração dos dados reservada para validação.",
    )

    test_split: float = Field(
        default=0.1,
        gt=0.0,
        lt=1.0,
        description="Fração dos dados reservada para teste.",
    )

    # * -----------------------------------------------------------------------
    # * Logging
    # * -----------------------------------------------------------------------

    log_level: Literal["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"] = Field(
        default="INFO",
        description="Nível do logger raiz.",
    )

    log_format: Literal["json", "text"] = Field(
        default="json",
        description="Formato de saída dos logs.",
    )

    # * -----------------------------------------------------------------------
    # * Validadores
    # * -----------------------------------------------------------------------

    @field_validator("validation_split", "test_split", mode="after")
    @classmethod
    def _splits_below_half(cls, value: float) -> float:
        """Garante que as frações de split são inferiores a 0.5.

        Args:
            value: Fração de split.

        Returns:
            A fração de split validada.

        Raises:
            ValueError: Se a fração for >= 0.5.
        """
        if value >= 0.5:
            msg = "As frações de split devem ser < 0.5 para sobrar dados para treino."
            raise ValueError(msg)
        return value


# * ---------------------------------------------------------------------------
# * Fábrica singleton
# * ---------------------------------------------------------------------------


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Retorna uma instância cacheada de Settings.

    Carrega ``configs/config.yaml`` e injeta os valores da seção ``data``
    como overrides, permitindo que o config seja a fonte da verdade para
    parâmetros do pipeline enquanto o ``.env`` continua responsável pelos
    valores de runtime e infraestrutura.

    Returns:
        Objeto Settings singleton.
    """
    import yaml  # ? import local — yaml só é necessário aqui

    config_path = Path("configs/config.yaml")
    overrides: dict = {}

    if config_path.exists():
        with config_path.open(encoding="utf-8") as fh:
            cfg = yaml.safe_load(fh) or {}
        data_cfg = cfg.get("data", {})
        if "strategy" in data_cfg:
            overrides["preprocessing_strategy"] = data_cfg["strategy"]
        if "n_negatives" in data_cfg:
            overrides["n_negatives"] = data_cfg["n_negatives"]

    return Settings(**overrides)
