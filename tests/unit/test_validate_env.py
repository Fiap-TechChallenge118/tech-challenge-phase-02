"""Testes unitários para scripts/validate_env.py.

Coberturas:
- Verificação de versão do Python (compatível e incompatível)
- Verificação de variáveis de ambiente (presentes e ausentes)
- Verificação de imports (sucesso e falha)
- Verificação de diretórios (existentes e ausentes)
- _ValidationResult acumula erros e avisos corretamente
- main() sai com os códigos de erro corretos
"""

import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

# * Adiciona a raiz do projeto ao path para importar scripts/
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from scripts.validate_env import (
    _EXIT_MISSING_VARS,
    _EXIT_OK,
    _EXIT_PYTHON_VERSION,
    _PYTHON_MIN,
    _REQUIRED_VARS,
    _check_data_dirs,
    _check_env_vars,
    _check_imports,
    _check_python_version,
    _check_settings,
    _ValidationResult,
)

# * ---------------------------------------------------------------------------
# * _ValidationResult
# * ---------------------------------------------------------------------------


class TestValidationResult:
    """_ValidationResult deve acumular erros e avisos corretamente."""

    def test_inicia_sem_erros_nem_avisos(self) -> None:
        """Instância nova não deve ter erros nem avisos."""
        result = _ValidationResult()
        assert result.errors == []
        assert result.warnings == []

    def test_ok_sem_erros(self) -> None:
        """ok deve ser True quando não há erros."""
        result = _ValidationResult()
        assert result.ok is True

    def test_ok_falso_com_erros(self) -> None:
        """ok deve ser False quando há pelo menos um erro."""
        result = _ValidationResult()
        result.add_error("erro qualquer")
        assert result.ok is False

    def test_add_error_acumula(self) -> None:
        """add_error deve acumular todos os erros."""
        result = _ValidationResult()
        result.add_error("erro 1")
        result.add_error("erro 2")
        assert len(result.errors) == 2

    def test_add_warning_acumula(self) -> None:
        """add_warning deve acumular todos os avisos."""
        result = _ValidationResult()
        result.add_warning("aviso 1")
        result.add_warning("aviso 2")
        assert len(result.warnings) == 2

    def test_warning_nao_afeta_ok(self) -> None:
        """Avisos não devem tornar ok False."""
        result = _ValidationResult()
        result.add_warning("apenas um aviso")
        assert result.ok is True


# * ---------------------------------------------------------------------------
# * _check_python_version
# * ---------------------------------------------------------------------------


class TestCheckPythonVersion:
    """_check_python_version deve validar a versão mínima exigida."""

    def test_versao_compativel(self) -> None:
        """Versão igual ou superior ao mínimo não deve gerar erros."""
        result = _ValidationResult()
        versao_ok = (_PYTHON_MIN[0], _PYTHON_MIN[1] + 1)
        with patch.object(sys, "version_info", versao_ok):
            _check_python_version(result)
        assert result.ok

    def test_versao_minima_exata(self) -> None:
        """Versão igual ao mínimo exato não deve gerar erros."""
        result = _ValidationResult()
        with patch.object(sys, "version_info", _PYTHON_MIN):
            _check_python_version(result)
        assert result.ok

    def test_versao_incompativel(self) -> None:
        """Versão abaixo do mínimo deve gerar exatamente um erro."""
        result = _ValidationResult()
        versao_antiga = (_PYTHON_MIN[0], _PYTHON_MIN[1] - 1)
        with patch.object(sys, "version_info", versao_antiga):
            _check_python_version(result)
        assert not result.ok
        assert len(result.errors) == 1
        assert "mínimo exigido" in result.errors[0]


# * ---------------------------------------------------------------------------
# * _check_env_vars
# * ---------------------------------------------------------------------------


class TestCheckEnvVars:
    """_check_env_vars deve detectar variáveis ausentes ou presentes."""

    def test_todas_variaveis_presentes(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        """Sem variáveis ausentes não deve gerar erros."""
        # * Define todas as variáveis obrigatórias no ambiente
        for var in _REQUIRED_VARS:
            monkeypatch.setenv(var, "valor-de-teste")

        result = _ValidationResult()
        # ! Aponta para tmp_path para evitar dependência do .env real
        with patch("scripts.validate_env.Path") as mock_path:
            mock_path.return_value.exists.return_value = False
            mock_path.return_value.__str__ = lambda s: ".env"
            _check_env_vars(result)

        assert result.ok

    def test_variavel_ausente_gera_erro(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Cada variável ausente deve gerar um erro distinto."""
        # * Remove todas as variáveis obrigatórias do ambiente
        for var in _REQUIRED_VARS:
            monkeypatch.delenv(var, raising=False)

        result = _ValidationResult()
        with patch("scripts.validate_env.Path") as mock_path:
            mock_path.return_value.exists.return_value = False
            mock_path.return_value.__str__ = lambda s: ".env"
            _check_env_vars(result)

        assert not result.ok
        assert len(result.errors) == len(_REQUIRED_VARS)

    def test_mensagem_erro_cita_variavel(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A mensagem de erro deve citar o nome da variável ausente."""
        for var in _REQUIRED_VARS:
            monkeypatch.delenv(var, raising=False)

        result = _ValidationResult()
        with patch("scripts.validate_env.Path") as mock_path:
            mock_path.return_value.exists.return_value = False
            _check_env_vars(result)

        nomes_com_erro = " ".join(result.errors)
        for var in _REQUIRED_VARS:
            assert var in nomes_com_erro

    def test_env_file_ausente_gera_aviso(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Ausência do arquivo .env deve gerar aviso, não erro."""
        for var in _REQUIRED_VARS:
            monkeypatch.setenv(var, "qualquer")

        result = _ValidationResult()
        with patch("scripts.validate_env.Path") as mock_path:
            mock_path.return_value.exists.return_value = False
            _check_env_vars(result)

        assert any(".env" in w for w in result.warnings)


# * ---------------------------------------------------------------------------
# * _check_imports
# * ---------------------------------------------------------------------------


class TestCheckImports:
    """_check_imports deve detectar imports com falha."""

    def test_imports_disponiveis(self) -> None:
        """Todos os módulos instalados devem ser importados sem erro."""
        result = _ValidationResult()
        _check_imports(result)
        # * Permite warnings mas não erros de import em ambiente configurado
        import_errors = [e for e in result.errors if "importar" in e.lower()]
        assert import_errors == [], f"Imports falharam: {import_errors}"

    def test_import_ausente_gera_erro(self) -> None:
        """Módulo inexistente deve gerar exatamente um erro."""
        from scripts.validate_env import _CRITICAL_IMPORTS

        imports_com_fake = [("modulo_inexistente_xyz", "Fake")] + list(
            _CRITICAL_IMPORTS
        )
        result = _ValidationResult()
        with patch("scripts.validate_env._CRITICAL_IMPORTS", imports_com_fake):
            _check_imports(result)

        erros_fake = [e for e in result.errors if "Fake" in e]
        assert len(erros_fake) == 1
        assert "modulo_inexistente_xyz" in erros_fake[0]


# * ---------------------------------------------------------------------------
# * _check_data_dirs
# * ---------------------------------------------------------------------------


class TestCheckDataDirs:
    """_check_data_dirs deve avisar sobre diretórios ausentes."""

    def test_diretorios_existentes_sem_aviso(self, tmp_path: Path) -> None:
        """Diretórios presentes não devem gerar avisos."""
        # * Cria todos os diretórios esperados dentro de tmp_path
        (tmp_path / "data").mkdir()
        (tmp_path / "data" / "raw").mkdir()
        (tmp_path / "data" / "processed").mkdir()
        (tmp_path / "configs").mkdir()
        (tmp_path / "models").mkdir()
        (tmp_path / "metrics").mkdir()

        result = _ValidationResult()

        # * Substitui Path para resolver caminhos relativos em tmp_path
        original_path = Path

        def patched_path(*args: str) -> Path:
            """Resolve caminhos relativos dentro de tmp_path."""
            p = original_path(*args)
            if not p.is_absolute():
                return tmp_path / p
            return p

        with patch("scripts.validate_env.Path", side_effect=patched_path):
            _check_data_dirs(result)

        assert result.warnings == []

    def test_diretorios_ausentes_geram_avisos(self, tmp_path: Path) -> None:
        """Diretórios ausentes devem gerar avisos, não erros."""
        result = _ValidationResult()

        # * Usa tmp_path vazio — nenhum diretório existe
        original_path = Path

        def patched_path(*args: str) -> Path:
            """Resolve caminhos relativos dentro de tmp_path vazio."""
            p = original_path(*args)
            if not p.is_absolute():
                return tmp_path / p
            return p

        with patch("scripts.validate_env.Path", side_effect=patched_path):
            _check_data_dirs(result)

        assert len(result.warnings) > 0
        # ! Diretórios ausentes devem ser avisos, nunca erros críticos
        assert result.ok


# * ---------------------------------------------------------------------------
# * _check_settings
# * ---------------------------------------------------------------------------


class TestCheckSettings:
    """_check_settings deve validar o Settings via Pydantic."""

    def test_settings_valido(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Settings configurado corretamente não deve gerar erros."""
        monkeypatch.delenv("HIDDEN_LAYERS", raising=False)
        result = _ValidationResult()
        _check_settings(result)
        assert result.ok

    def test_settings_invalido_gera_erro(self) -> None:
        """Settings com campo inválido deve gerar um erro."""
        result = _ValidationResult()

        mock_settings_cls = MagicMock(
            side_effect=Exception("campo inválido simulado")
        )

        with patch("scripts.validate_env.Path"):
            with patch.dict(
                sys.modules, {"src.settings": MagicMock(Settings=mock_settings_cls)}
            ):
                _check_settings(result)

        assert not result.ok
        assert len(result.errors) == 1


# * ---------------------------------------------------------------------------
# * main() — códigos de saída
# * ---------------------------------------------------------------------------


class TestMainExitCodes:
    """main() deve encerrar com o código de saída correto para cada cenário."""

    def test_saida_ok_ambiente_valido(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Ambiente completamente válido deve sair com código 0."""
        # * Valores válidos para os campos Literal do Settings
        valores_validos: dict[str, str] = {
            "PROJECT_NAME": "tech-challenge-02",
            "ENV": "development",
            "MLFLOW_TRACKING_URI": "http://localhost:5000",
            "MLFLOW_EXPERIMENT_NAME": "recommendation-system",
            "MLFLOW_REGISTERED_MODEL_NAME": "recommender-mlp",
            "DVC_REMOTE_TYPE": "local",
            "DVC_REMOTE_URL": "/tmp/dvc-remote",
            "DATA_DIR": "data",
            "DATA_RAW_DIR": "data/raw",
            "DATA_PROCESSED_DIR": "data/processed",
            "DATASET_NAME": "instacart",
        }
        for var, val in valores_validos.items():
            monkeypatch.setenv(var, val)
        monkeypatch.delenv("HIDDEN_LAYERS", raising=False)

        from scripts.validate_env import main

        with pytest.raises(SystemExit) as exc:
            with patch("scripts.validate_env.Path") as mock_path:
                mock_path.return_value.exists.return_value = True
                mock_path.return_value.resolve.return_value = Path(".env")
                main()

        assert exc.value.code == _EXIT_OK

    def test_saida_versao_python(self) -> None:
        """Versão do Python incompatível deve sair com código 1."""
        from scripts.validate_env import main

        versao_antiga = (2, 7)
        with pytest.raises(SystemExit) as exc:
            with patch.object(sys, "version_info", versao_antiga):
                main()

        assert exc.value.code == _EXIT_PYTHON_VERSION

    def test_saida_variaveis_ausentes(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Variáveis ausentes devem sair com código 2."""
        for var in _REQUIRED_VARS:
            monkeypatch.delenv(var, raising=False)
        monkeypatch.delenv("HIDDEN_LAYERS", raising=False)

        from scripts.validate_env import main

        with pytest.raises(SystemExit) as exc:
            with patch("scripts.validate_env.Path") as mock_path:
                mock_path.return_value.exists.return_value = False
                main()

        assert exc.value.code == _EXIT_MISSING_VARS
