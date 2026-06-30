# * Raiz do pacote — expõe o singleton de configurações para conveniência
from src.settings import Settings, get_settings

__all__ = ["Settings", "get_settings"]
