from pydantic import BaseModel
from typing import List, Dict, Any, Optional


class SidraMetadata(BaseModel):
    id: int
    nome: str
    assunto: str
    periodicidade: Dict[str, Any]
    nivelTerritorial: Dict[str, Any]
    variaveis: List[Dict[str, Any]]
    classificacoes: List[Dict[str, Any]]


class SidraNivel(BaseModel):
    id: str
    nome: str


class SidraLocalidade(BaseModel):
    id: str
    nome: str
    nivel: SidraNivel


class SidraSerie(BaseModel):
    localidade: SidraLocalidade
    serie: Dict[str, Optional[str]]


class SidraClassificacao(BaseModel):
    id: str
    nome: str
    categoria: Dict[str, str]


class SidraResultado(BaseModel):
    classificacoes: List[SidraClassificacao]
    series: List[SidraSerie]


class SidraVariableResponse(BaseModel):
    id: str
    variavel: str
    unidade: str
    resultados: List[SidraResultado]
