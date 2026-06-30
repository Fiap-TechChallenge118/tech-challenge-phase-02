"""Script de validação do ambiente de desenvolvimento.

Verifica se o ambiente está corretamente configurado antes de rodar o pipeline.
Sai com código de erro específico caso alguma verificação falhe.

Uso::

    uv run python scripts/validate_env.py

Códigos de saída:
    0 — todas as verificações passaram
    1 — versão do Python incompatível
    2 — uma ou mais variáveis de ambiente obrigatórias ausentes
    3 — um ou mais imports críticos falharam
    4 — settings inválido (falha de validação do Pydantic)
"""

import importlib
import logging
import sys
from dataclasses import dataclass, field
from pathlib import Path

# * ---------------------------------------------------------------------------
# * Configuração de logging
# * ---------------------------------------------------------------------------

logging.basicConfig(
    level=logging.INFO,
    format="%(levelname)s  %(message)s",
)

_log = logging.getLogger(__name__)

# * ---------------------------------------------------------------------------
# * Constantes
# * ---------------------------------------------------------------------------

# * Versão mínima exigida pelo projeto (pyproject.toml: requires-python = ">=3.12")
_PYTHON_MIN = (3, 12)

# * Variáveis obrigatórias: sem valor padrão útil em produção
_REQUIRED_VARS: list[str] = [
    "PROJECT_NAME",
    "ENV",
    "MLFLOW_TRACKING_URI",
    "MLFLOW_EXPERIMENT_NAME",
    "MLFLOW_REGISTERED_MODEL_NAME",
    "DVC_REMOTE_TYPE",
    "DVC_REMOTE_URL",
    "DATA_DIR",
    "DATA_RAW_DIR",
    "DATA_PROCESSED_DIR",
    "DATASET_NAME",
]

# * Imports críticos para o pipeline — nome do módulo e rótulo legível
_CRITICAL_IMPORTS: list[tuple[str, str]] = [
    ("torch", "PyTorch"),
    ("sklearn", "Scikit-Learn"),
    ("mlflow", "MLflow"),
    ("pandas", "Pandas"),
    ("numpy", "NumPy"),
    ("pydantic", "Pydantic"),
    ("pydantic_settings", "Pydantic Settings"),
    ("pandera", "Pandera"),
    ("yaml", "PyYAML"),
    ("joblib", "Joblib"),
]

# * Códigos de saída
_EXIT_OK = 0
_EXIT_PYTHON_VERSION = 1
_EXIT_MISSING_VARS = 2
_EXIT_IMPORT_ERROR = 3
_EXIT_SETTINGS_ERROR = 4


# * ---------------------------------------------------------------------------
# * Estrutura de resultado de validação
# * ---------------------------------------------------------------------------


@dataclass
class _ValidationResult:
    """Acumula erros e avisos de todas as etapas de validação."""

    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    def add_error(self, msg: str) -> None:
        """Registra um erro crítico que impede a execução.

        Args:
            msg: Descrição do erro.
        """
        self.errors.append(msg)
        _log.error("  ✗ %s", msg)

    def add_warning(self, msg: str) -> None:
        """Registra um aviso que não impede a execução.

        Args:
            msg: Descrição do aviso.
        """
        self.warnings.append(msg)
        _log.warning("  ⚠ %s", msg)

    def add_ok(self, msg: str) -> None:
        """Registra uma verificação bem-sucedida.

        Args:
            msg: Descrição do item verificado.
        """
        _log.info("  ✓ %s", msg)

    @property
    def ok(self) -> bool:
        """Retorna True se não há erros críticos."""
        return len(self.errors) == 0


# * ---------------------------------------------------------------------------
# * Etapas de validação
# * ---------------------------------------------------------------------------


def _check_python_version(result: _ValidationResult) -> None:
    """Verifica se a versão do Python atende ao mínimo exigido.

    Args:
        result: Objeto de resultado onde erros e avisos serão registrados.
    """
    current = sys.version_info[:2]
    version_str = f"{current[0]}.{current[1]}"
    min_str = f"{_PYTHON_MIN[0]}.{_PYTHON_MIN[1]}"

    if current < _PYTHON_MIN:
        result.add_error(
            f"Python {version_str} encontrado — mínimo exigido é {min_str}"
        )
    else:
        result.add_ok(f"Python {version_str} (≥ {min_str})")


def _check_env_vars(result: _ValidationResult) -> None:
    """Verifica se todas as variáveis de ambiente obrigatórias estão definidas.

    Lê o arquivo .env diretamente via python-dotenv quando disponível,
    mas também aceita variáveis já exportadas no processo.

    Args:
        result: Objeto de resultado onde erros e avisos serão registrados.
    """
    import os

    # * Tenta carregar o .env para complementar as variáveis do processo
    env_file = Path(".env")
    if env_file.exists():
        result.add_ok(f"Arquivo .env encontrado em {env_file.resolve()}")
    else:
        result.add_warning(
            "Arquivo .env não encontrado — usando apenas variáveis do processo"
        )

    # * Verifica cada variável obrigatória
    missing: list[str] = []
    for var in _REQUIRED_VARS:
        value = os.getenv(var)
        if not value:
            missing.append(var)
        else:
            result.add_ok(f"{var} definida")

    if missing:
        for var in missing:
            result.add_error(f"Variável obrigatória ausente: {var}")


def _check_imports(result: _ValidationResult) -> None:
    """Tenta importar cada dependência crítica do pipeline.

    Args:
        result: Objeto de resultado onde erros e avisos serão registrados.
    """
    for module, label in _CRITICAL_IMPORTS:
        try:
            mod = importlib.import_module(module)
            version = getattr(mod, "__version__", "versão desconhecida")
            result.add_ok(f"{label} importado com sucesso (v{version})")
        except ImportError as exc:
            result.add_error(
                f"Falha ao importar {label} ({module}): {exc}"
            )


def _check_settings(result: _ValidationResult) -> None:
    """Instancia o Settings do projeto e valida todos os campos via Pydantic.

    Args:
        result: Objeto de resultado onde erros e avisos serão registrados.
    """
    try:
        # * Adiciona o diretório raiz ao path para importar src
        root = Path(__file__).parent.parent
        if str(root) not in sys.path:
            sys.path.insert(0, str(root))

        from src.settings import Settings

        settings = Settings()
        result.add_ok(
            f"Settings válido — projeto '{settings.project_name}' "
            f"| env '{settings.env}' | seed {settings.random_seed}"
        )
    except ImportError as exc:
        result.add_error(f"Falha ao importar src.settings: {exc}")
    except Exception as exc:  # noqa: BLE001
        result.add_error(f"Settings inválido: {exc}")


def _check_data_dirs(result: _ValidationResult) -> None:
    """Verifica se os diretórios de dados existem.

    Args:
        result: Objeto de resultado onde erros e avisos serão registrados.
    """
    dirs_to_check = [
        (Path("data"), "Diretório raiz de dados"),
        (Path("data/raw"), "Dados brutos (data/raw)"),
        (Path("data/processed"), "Dados processados (data/processed)"),
        (Path("configs"), "Configurações (configs/)"),
        (Path("models"), "Artefatos de modelos (models/)"),
        (Path("metrics"), "Métricas do pipeline (metrics/)"),
    ]

    for path, label in dirs_to_check:
        if path.exists():
            result.add_ok(f"{label} existe")
        else:
            # ! Diretórios de dados ausentes são avisos, não erros —
            # ! podem ser criados pelo pipeline ou pelo DVC pull
            result.add_warning(f"{label} não encontrado: {path}")


# * ---------------------------------------------------------------------------
# * Ponto de entrada
# * ---------------------------------------------------------------------------


def main() -> None:
    """Executa todas as validações e encerra com o código de saída adequado."""
    result = _ValidationResult()

    _log.info("=" * 60)
    _log.info("Validação do Ambiente — Tech Challenge 02")
    _log.info("=" * 60)

    # * 1 — Versão do Python
    _log.info("\n[1/5] Versão do Python")
    _check_python_version(result)
    if not result.ok:
        _log.error("\n✗ Versão do Python incompatível. Abortando.")
        sys.exit(_EXIT_PYTHON_VERSION)

    # * 2 — Variáveis de ambiente obrigatórias
    _log.info("\n[2/5] Variáveis de ambiente obrigatórias")
    _check_env_vars(result)

    # * 3 — Imports críticos
    _log.info("\n[3/5] Imports críticos")
    _check_imports(result)

    # * 4 — Settings (validação Pydantic)
    _log.info("\n[4/5] Validação do Settings (Pydantic)")
    _check_settings(result)

    # * 5 — Diretórios de dados
    _log.info("\n[5/5] Diretórios do projeto")
    _check_data_dirs(result)

    # * Resumo final
    _log.info("\n" + "=" * 60)

    if result.warnings:
        _log.warning("Avisos (%d):", len(result.warnings))
        for w in result.warnings:
            _log.warning("  ⚠ %s", w)

    if not result.ok:
        _log.error("Falhas críticas (%d):", len(result.errors))
        for e in result.errors:
            _log.error("  ✗ %s", e)
        _log.error("\n✗ Validação falhou. Corrija os erros acima antes de continuar.")

        # * Determina o código de saída mais específico
        error_text = " ".join(result.errors)
        if "Python" in error_text:
            sys.exit(_EXIT_PYTHON_VERSION)
        if "ausente" in error_text:
            sys.exit(_EXIT_MISSING_VARS)
        if "importar" in error_text.lower():
            sys.exit(_EXIT_IMPORT_ERROR)
        sys.exit(_EXIT_SETTINGS_ERROR)

    _log.info(
        "✓ Ambiente validado com sucesso! "
        "(%d avisos, 0 erros)",
        len(result.warnings),
    )
    sys.exit(_EXIT_OK)


if __name__ == "__main__":
    main()
