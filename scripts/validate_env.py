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
import sys
from dataclasses import dataclass, field
from pathlib import Path

# * ---------------------------------------------------------------------------
# * Cores ANSI — funcionam em qualquer terminal moderno
# * ---------------------------------------------------------------------------

_RESET = "\033[0m"
_BOLD = "\033[1m"
_DIM = "\033[2m"

_GREEN = "\033[32m"
_YELLOW = "\033[33m"
_RED = "\033[31m"
_CYAN = "\033[36m"
_WHITE = "\033[37m"
_BLUE = "\033[34m"
_MAGENTA = "\033[35m"

# * Ícones e prefixos visuais
_OK = f"{_GREEN}✓{_RESET}"
_WARN = f"{_YELLOW}⚠{_RESET}"
_FAIL = f"{_RED}✗{_RESET}"
_ARROW = f"{_DIM}→{_RESET}"


def _c(text: str, color: str, bold: bool = False) -> str:
    """Aplica cor e opcionalmente negrito a um texto para o terminal.

    Args:
        text: Texto a ser colorido.
        color: Código ANSI de cor.
        bold: Se True, aplica negrito além da cor.

    Returns:
        Texto com as sequências ANSI aplicadas.
    """
    prefix = _BOLD if bold else ""
    return f"{prefix}{color}{text}{_RESET}"


def _print_header(title: str) -> None:
    """Imprime um cabeçalho de seção com separadores visuais.

    Args:
        title: Título da seção.
    """
    line = "─" * 60
    print(f"\n{_c(line, _CYAN)}")
    print(f"  {_c(title, _WHITE, bold=True)}")
    print(f"{_c(line, _CYAN)}")


def _print_step(number: int, total: int, description: str) -> None:
    """Imprime o cabeçalho de uma etapa de validação.

    Args:
        number: Número da etapa atual.
        total: Total de etapas.
        description: Descrição da etapa.
    """
    step = _c(f"[{number}/{total}]", _CYAN, bold=True)
    desc = _c(description, _WHITE, bold=True)
    print(f"\n  {step} {desc}")


def _print_ok(message: str) -> None:
    """Imprime uma linha de verificação bem-sucedida.

    Args:
        message: Descrição do item verificado com sucesso.
    """
    print(f"      {_OK}  {message}")


def _print_warn(message: str) -> None:
    """Imprime um aviso que não bloqueia a execução.

    Args:
        message: Descrição do aviso.
    """
    print(f"      {_WARN}  {_c(message, _YELLOW)}")


def _print_fail(message: str) -> None:
    """Imprime um erro crítico que impede a execução.

    Args:
        message: Descrição do erro.
    """
    print(f"      {_FAIL}  {_c(message, _RED)}")


# * ---------------------------------------------------------------------------
# * Constantes
# * ---------------------------------------------------------------------------

# * Versão mínima exigida pelo projeto (pyproject.toml: requires-python = ">=3.12")
_PYTHON_MIN = (3, 12)

# * Variáveis obrigatórias sem valor padrão útil em produção
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

# * Imports críticos para o pipeline — (módulo, rótulo legível)
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
        """Registra um erro crítico e imprime no terminal.

        Args:
            msg: Descrição do erro.
        """
        self.errors.append(msg)
        _print_fail(msg)

    def add_warning(self, msg: str) -> None:
        """Registra um aviso e imprime no terminal.

        Args:
            msg: Descrição do aviso.
        """
        self.warnings.append(msg)
        _print_warn(msg)

    def add_ok(self, msg: str) -> None:
        """Registra uma verificação bem-sucedida e imprime no terminal.

        Args:
            msg: Descrição do item verificado.
        """
        _print_ok(msg)

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
            f"Python {version_str} encontrado {_ARROW} mínimo exigido é {min_str}"
        )
    else:
        result.add_ok(
            f"Python {_c(version_str, _GREEN, bold=True)} "
            f"{_c(f'(≥ {min_str})', _DIM)}"
        )


def _check_env_vars(result: _ValidationResult) -> None:
    """Verifica se todas as variáveis de ambiente obrigatórias estão definidas.

    Args:
        result: Objeto de resultado onde erros e avisos serão registrados.
    """
    import os

    env_file = Path(".env")
    if env_file.exists():
        result.add_ok(
            f"Arquivo {_c('.env', _CYAN)} encontrado em "
            f"{_c(str(env_file.resolve()), _DIM)}"
        )
    else:
        result.add_warning(
            f"Arquivo {_c('.env', _CYAN)} não encontrado "
            f"{_ARROW} usando apenas variáveis do processo"
        )

    missing: list[str] = []
    for var in _REQUIRED_VARS:
        if os.getenv(var):
            result.add_ok(
                f"{_c(var, _CYAN)} {_c('definida', _DIM)}"
            )
        else:
            missing.append(var)

    for var in missing:
        result.add_error(
            f"{_c(var, _CYAN)} {_ARROW} variável obrigatória ausente"
        )


def _check_imports(result: _ValidationResult) -> None:
    """Tenta importar cada dependência crítica do pipeline.

    Args:
        result: Objeto de resultado onde erros e avisos serão registrados.
    """
    for module, label in _CRITICAL_IMPORTS:
        try:
            mod = importlib.import_module(module)
            version = getattr(mod, "__version__", "?")
            result.add_ok(
                f"{_c(label, _WHITE)} "
                f"{_c(f'v{version}', _DIM)}"
            )
        except ImportError as exc:
            result.add_error(
                f"{_c(label, _WHITE)} {_ARROW} {exc}"
            )


def _check_settings(result: _ValidationResult) -> None:
    """Instancia o Settings do projeto e valida todos os campos via Pydantic.

    Args:
        result: Objeto de resultado onde erros e avisos serão registrados.
    """
    try:
        root = Path(__file__).parent.parent
        if str(root) not in sys.path:
            sys.path.insert(0, str(root))

        from src.settings import Settings

        s = Settings()
        result.add_ok(
            f"Settings válido {_ARROW} "
            f"projeto {_c(repr(s.project_name), _CYAN)}  "
            f"env {_c(repr(s.env), _CYAN)}  "
            f"seed {_c(str(s.random_seed), _CYAN)}"
        )
    except ImportError as exc:
        result.add_error(f"Falha ao importar src.settings {_ARROW} {exc}")
    except Exception as exc:  # noqa: BLE001
        result.add_error(f"Settings inválido {_ARROW} {exc}")


def _check_data_dirs(result: _ValidationResult) -> None:
    """Verifica se os diretórios do projeto existem.

    Args:
        result: Objeto de resultado onde erros e avisos serão registrados.
    """
    dirs_to_check: list[tuple[Path, str]] = [
        (Path("data"),           "data/"),
        (Path("data/raw"),       "data/raw/"),
        (Path("data/processed"), "data/processed/"),
        (Path("configs"),        "configs/"),
        (Path("models"),         "models/"),
        (Path("metrics"),        "metrics/"),
    ]

    for path, label in dirs_to_check:
        if path.exists():
            result.add_ok(f"{_c(label, _CYAN)} existe")
        else:
            # ! Ausência de data/ é esperada antes do dvc pull — não é erro crítico
            result.add_warning(
                f"{_c(label, _CYAN)} não encontrado "
                f"{_c('(rode dvc pull)', _DIM)}"
            )


# * ---------------------------------------------------------------------------
# * Ponto de entrada
# * ---------------------------------------------------------------------------


def main() -> None:
    """Executa todas as validações e encerra com o código de saída adequado."""
    result = _ValidationResult()

    # * Cabeçalho principal
    print(f"\n{_c('═' * 62, _CYAN)}")
    print(
        f"  {_c('🔍 Validação do Ambiente', _WHITE, bold=True)}"
        f"  {_c('Tech Challenge 02', _DIM)}"
    )
    print(f"{_c('═' * 62, _CYAN)}")

    total_steps = 5

    # * 1 — Versão do Python
    _print_step(1, total_steps, "Versão do Python")
    _check_python_version(result)
    if not result.ok:
        _print_summary(result)
        sys.exit(_EXIT_PYTHON_VERSION)

    # * 2 — Variáveis de ambiente
    _print_step(2, total_steps, "Variáveis de ambiente obrigatórias")
    _check_env_vars(result)

    # * 3 — Imports críticos
    _print_step(3, total_steps, "Imports críticos")
    _check_imports(result)

    # * 4 — Settings Pydantic
    _print_step(4, total_steps, "Validação do Settings (Pydantic)")
    _check_settings(result)

    # * 5 — Diretórios do projeto
    _print_step(5, total_steps, "Diretórios do projeto")
    _check_data_dirs(result)

    # * Rodapé com resumo
    _print_summary(result)

    if not result.ok:
        error_text = " ".join(result.errors)
        if "Python" in error_text:
            sys.exit(_EXIT_PYTHON_VERSION)
        if "ausente" in error_text:
            sys.exit(_EXIT_MISSING_VARS)
        if "importar" in error_text.lower() or "import" in error_text.lower():
            sys.exit(_EXIT_IMPORT_ERROR)
        sys.exit(_EXIT_SETTINGS_ERROR)

    sys.exit(_EXIT_OK)


def _print_summary(result: _ValidationResult) -> None:
    """Imprime o resumo final com contagem de erros e avisos.

    Args:
        result: Resultado acumulado de todas as etapas.
    """
    print(f"\n{_c('─' * 62, _CYAN)}")

    if result.warnings:
        label = _c(f"⚠  {len(result.warnings)} aviso(s)", _YELLOW, bold=True)
        print(f"\n  {label}")
        for w in result.warnings:
            print(f"     {_WARN}  {_c(w, _YELLOW)}")

    if not result.ok:
        label = _c(f"✗  {len(result.errors)} erro(s) crítico(s)", _RED, bold=True)
        print(f"\n  {label}")
        for e in result.errors:
            print(f"     {_FAIL}  {_c(e, _RED)}")
        print(
            f"\n  {_c('Corrija os erros acima antes de continuar.', _RED)}\n"
        )
    else:
        n_warn = len(result.warnings)
        warn_label = (
            f"{_c(str(n_warn), _YELLOW)} aviso(s)" if n_warn else
            f"{_c('0', _GREEN)} avisos"
        )
        print(
            f"\n  {_c('✓  Ambiente validado com sucesso!', _GREEN, bold=True)}"
            f"  {_c(f'({warn_label}, 0 erros)', _DIM)}"
        )

    print(f"{_c('─' * 62, _CYAN)}\n")


if __name__ == "__main__":
    main()
