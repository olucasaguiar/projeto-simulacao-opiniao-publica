"""
Pacote utilizado como serviço de integração com o serviço de dados agregados do IBGE.
Documentaçao: https://servicodados.ibge.gov.br/api/docs/agregados?versao=3
"""

from .client import SidraClient
from .query_builder import SidraQueryBuilder
from .models import (
    SidraMetadata,
    SidraVariableResponse,
    SidraResultado,
    SidraSerie,
    SidraClassificacao,
    SidraLocalidade,
    SidraNivel,
)

__all__ = [
    "SidraClient",
    "SidraQueryBuilder",
    "SidraMetadata",
    "SidraVariableResponse",
    "SidraResultado",
    "SidraSerie",
    "SidraClassificacao",
    "SidraLocalidade",
    "SidraNivel",
]
