"""Testes unitários para src/settings.py.

Coberturas:
- Valores padrão estão corretos
- Sobrescrita por variáveis de ambiente funciona
- Parsing de hidden_layers separado por vírgulas via _coerce_int_list
- Valores inválidos levantam ValidationError
- get_settings() retorna um singleton cacheado
"""

import pytest
from pydantic import ValidationError

from src.settings import Settings, _coerce_int_list, get_settings

# * ---------------------------------------------------------------------------
# * Utilitários de teste
# * ---------------------------------------------------------------------------


def _make(env_vars: dict, monkeypatch: pytest.MonkeyPatch) -> Settings:
    """Instancia Settings com as env vars fornecidas, ignorando o .env real.

    Args:
        env_vars: Mapeamento de nome da variável → valor string a injetar.
        monkeypatch: Fixture do pytest para manipulação do ambiente.

    Returns:
        Nova instância de Settings construída a partir das variáveis fornecidas.
    """
    # ! Remove HIDDEN_LAYERS do processo para que o valor real do .env
    # ! não contamine testes que não definem essa variável explicitamente.
    monkeypatch.delenv("HIDDEN_LAYERS", raising=False)
    for key, value in env_vars.items():
        monkeypatch.setenv(key, value)
    return Settings(_env_file=None)  # type: ignore[call-arg]


# * ---------------------------------------------------------------------------
# * Valores padrão
# * ---------------------------------------------------------------------------


class TestSettingsDefaults:
    """Settings instanciado sem env vars deve usar os valores padrão corretos."""

    def test_project_name_default(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Nome do projeto padrão deve coincidir com o nome no pyproject.toml."""
        settings = _make({}, monkeypatch)
        assert settings.project_name == "tech-challenge-02"

    def test_env_default(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Ambiente padrão deve ser development."""
        settings = _make({}, monkeypatch)
        assert settings.env == "development"

    def test_random_seed_default(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Semente padrão deve ser 42."""
        settings = _make({}, monkeypatch)
        assert settings.random_seed == 42

    def test_mlflow_tracking_uri_default(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """URI padrão do MLflow deve usar SQLite (Registry sem servidor)."""
        settings = _make({}, monkeypatch)
        assert settings.mlflow_tracking_uri == "sqlite:///mlflow.db"

    def test_hidden_layers_default(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Camadas ocultas padrão devem ser [256, 128, 64]."""
        settings = _make({}, monkeypatch)
        assert settings.hidden_layers == [256, 128, 64]

    def test_evaluation_k_default(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """K padrão para métricas de ranking deve ser 10."""
        settings = _make({}, monkeypatch)
        assert settings.evaluation_k == 10

    def test_log_level_default(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Nível de log padrão deve ser INFO."""
        settings = _make({}, monkeypatch)
        assert settings.log_level == "INFO"

    def test_log_format_default(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Formato de log padrão deve ser json."""
        settings = _make({}, monkeypatch)
        assert settings.log_format == "json"


# * ---------------------------------------------------------------------------
# * Sobrescrita por variáveis de ambiente
# * ---------------------------------------------------------------------------


class TestSettingsEnvOverrides:
    """Valores definidos como env vars devem sobrescrever os padrões."""

    def test_env_override(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Variável ENV deve sobrescrever o literal de ambiente."""
        settings = _make({"ENV": "production"}, monkeypatch)
        assert settings.env == "production"

    def test_random_seed_override(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """RANDOM_SEED deve ser lido como inteiro."""
        settings = _make({"RANDOM_SEED": "123"}, monkeypatch)
        assert settings.random_seed == 123

    def test_mlflow_experiment_name_override(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """MLFLOW_EXPERIMENT_NAME deve ser repassado corretamente."""
        settings = _make({"MLFLOW_EXPERIMENT_NAME": "my-exp"}, monkeypatch)
        assert settings.mlflow_experiment_name == "my-exp"

    def test_evaluation_k_override(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """EVALUATION_K deve ser lido como inteiro."""
        settings = _make({"EVALUATION_K": "20"}, monkeypatch)
        assert settings.evaluation_k == 20

    def test_log_level_override(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """LOG_LEVEL deve aceitar um literal válido."""
        settings = _make({"LOG_LEVEL": "DEBUG"}, monkeypatch)
        assert settings.log_level == "DEBUG"

    def test_optional_mlflow_credentials(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Credenciais opcionais do MLflow devem ser carregadas quando presentes."""
        settings = _make(
            {
                "MLFLOW_TRACKING_USERNAME": "admin",
                "MLFLOW_TRACKING_PASSWORD": "secret",
            },
            monkeypatch,
        )
        assert settings.mlflow_tracking_username == "admin"
        assert settings.mlflow_tracking_password == "secret"


# * ---------------------------------------------------------------------------
# * Parsing de hidden_layers — testado via _coerce_int_list diretamente
# * ---------------------------------------------------------------------------


class TestHiddenLayersParsing:
    """O utilitário _coerce_int_list deve tratar todas as formas de entrada válidas."""

    def test_comma_separated_string(self) -> None:
        """String separada por vírgulas deve ser convertida para list[int]."""
        assert _coerce_int_list("512,256,128") == [512, 256, 128]

    def test_single_element_string(self) -> None:
        """String com valor único deve produzir uma lista com um elemento."""
        assert _coerce_int_list("128") == [128]

    def test_list_of_ints_passthrough(self) -> None:
        """Lista de inteiros deve ser retornada sem alteração."""
        assert _coerce_int_list([64, 32]) == [64, 32]

    def test_list_of_strings_is_coerced(self) -> None:
        """Lista de strings numéricas deve ser convertida para list[int]."""
        assert _coerce_int_list(["64", "32"]) == [64, 32]

    def test_hidden_layers_env_override(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """HIDDEN_LAYERS em formato JSON deve ser parseado corretamente."""
        settings = _make({"HIDDEN_LAYERS": "[512,256,128]"}, monkeypatch)
        assert settings.hidden_layers == [512, 256, 128]


# * ---------------------------------------------------------------------------
# * Erros de validação
# * ---------------------------------------------------------------------------


class TestSettingsValidation:
    """Valores inválidos devem levantar pydantic.ValidationError."""

    def test_invalid_env_literal(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Valor desconhecido em ENV deve levantar ValidationError."""
        monkeypatch.setenv("ENV", "unknown")
        monkeypatch.delenv("HIDDEN_LAYERS", raising=False)
        with pytest.raises(ValidationError):
            Settings(_env_file=None)  # type: ignore[call-arg]

    def test_negative_random_seed(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """RANDOM_SEED negativo deve levantar ValidationError (restrição ge=0)."""
        with pytest.raises(ValidationError):
            _make({"RANDOM_SEED": "-1"}, monkeypatch)

    def test_zero_batch_size(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """BATCH_SIZE igual a 0 deve levantar ValidationError (restrição ge=1)."""
        with pytest.raises(ValidationError):
            _make({"BATCH_SIZE": "0"}, monkeypatch)

    def test_dropout_rate_out_of_range(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """DROPOUT_RATE >= 1.0 deve levantar ValidationError (restrição lt=1.0)."""
        with pytest.raises(ValidationError):
            _make({"DROPOUT_RATE": "1.0"}, monkeypatch)

    def test_invalid_log_level(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """LOG_LEVEL desconhecido deve levantar ValidationError."""
        with pytest.raises(ValidationError):
            _make({"LOG_LEVEL": "VERBOSE"}, monkeypatch)

    def test_validation_split_too_large(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """VALIDATION_SPLIT >= 0.5 deve levantar ValidationError."""
        with pytest.raises(ValidationError):
            _make({"VALIDATION_SPLIT": "0.6"}, monkeypatch)

    def test_test_split_too_large(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """TEST_SPLIT >= 0.5 deve levantar ValidationError."""
        with pytest.raises(ValidationError):
            _make({"TEST_SPLIT": "0.5"}, monkeypatch)

    def test_learning_rate_zero(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """LEARNING_RATE igual a 0.0 deve levantar ValidationError (gt=0.0)."""
        with pytest.raises(ValidationError):
            _make({"LEARNING_RATE": "0.0"}, monkeypatch)

    def test_invalid_log_format(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """LOG_FORMAT desconhecido deve levantar ValidationError."""
        with pytest.raises(ValidationError):
            _make({"LOG_FORMAT": "xml"}, monkeypatch)


# * ---------------------------------------------------------------------------
# * Comportamento de singleton
# * ---------------------------------------------------------------------------


class TestGetSettings:
    """get_settings() deve retornar o mesmo singleton em múltiplas chamadas."""

    def setup_method(self) -> None:
        """Limpa o lru_cache antes de cada teste para garantir isolamento."""
        # * Limpa o cache antes de cada teste para que o singleton seja recriado
        get_settings.cache_clear()

    def test_returns_settings_instance(self) -> None:
        """get_settings() deve retornar um objeto Settings."""
        result = get_settings()
        assert isinstance(result, Settings)

    def test_singleton_identity(self) -> None:
        """Duas chamadas a get_settings() devem retornar o mesmo objeto (lru_cache)."""
        first = get_settings()
        second = get_settings()
        assert first is second
