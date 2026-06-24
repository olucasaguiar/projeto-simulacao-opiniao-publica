from typing import Dict, List
from pydantic import BaseModel
from features.generate_persona.models import Persona
from infrastructure.llm.models import ModelAnswer


class SurveyQuestion(BaseModel):
    """Uma pergunta do questionário."""

    id: str
    topic: str
    text: str
    options: Dict[str, str]  # {"a": "Concordo totalmente", "b": "Discordo", ...}


class FormResults(BaseModel):
    """Distribuição das respostas para uma pergunta.
    Key: id da opção, Value: proporção (0.0 a 1.0)."""

    distribution: Dict[str, float]  # {"a": 0.4, "b": 0.4, "c": 0.2}


class FormResponse(BaseModel):
    """Resultado de uma persona respondendo uma pergunta N vezes."""

    question: str
    options: Dict[str, str]
    answers: List[ModelAnswer]
    result: FormResults


class PersonaSimulationResult(BaseModel):
    """Resultado completo de uma persona respondendo o questionário."""

    persona: Persona
    responses: List[FormResponse]


class Survey(BaseModel):
    """Uma pesquisa completa com múltiplas perguntas."""

    id: str
    title: str
    questions: List[SurveyQuestion]


class SimulationConfig(BaseModel):
    """Configuração do cenário de simulação, carregada do YAML."""

    personas: int
    repetitions: int  # reproduções
    models: List[str]
    results_path: str
    surveys: List[Survey]
