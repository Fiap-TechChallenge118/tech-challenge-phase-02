"""Schemas pandera para validação dos DataFrames de entrada.

Validação leve, focada na presença das colunas exigidas pelas estratégias. Não
força dtypes estritos para tolerar variações de leitura (ex.: ``NaN`` em
``days_since_prior_order``). ``strict=False`` permite colunas extras.
"""

import pandera.pandas as pa

from src.features.preprocessing.base import InstacartFrames

# * Colunas mínimas exigidas em cada frame pelas estratégias de preprocessing.
_ORDERS_SCHEMA = pa.DataFrameSchema(
    {"order_id": pa.Column(), "user_id": pa.Column()},
    strict=False,
    name="orders",
)

_ORDER_PRODUCTS_SCHEMA = pa.DataFrameSchema(
    {"order_id": pa.Column(), "product_id": pa.Column()},
    strict=False,
    name="order_products",
)


def validate_frames(data: InstacartFrames) -> None:
    """Valida a presença das colunas essenciais nos frames de entrada.

    Args:
        data: DataFrames brutos do Instacart.

    Raises:
        pandera.errors.SchemaError: Se alguma coluna obrigatória estiver ausente.
    """
    _ORDERS_SCHEMA.validate(data.orders)
    _ORDER_PRODUCTS_SCHEMA.validate(data.order_products)
