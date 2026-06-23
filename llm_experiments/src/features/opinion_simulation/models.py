from typing import List
from pydantic import BaseModel
from features.generate_persona.models import Persona


class OpinionQuestion(BaseModel):
    id: str
    topic: str
    question_text: str
    options: List[str]


class ModelOpinionResponse(BaseModel):
    model_id: str
    question_id: str
    persona_id: str
    chosen_option: str
    explanation: str


class PersonaSimulationResult(BaseModel):
    persona: Persona
    responses: List[ModelOpinionResponse]


class SimulationComparison(BaseModel):
    question: OpinionQuestion
    persona_results: List[PersonaSimulationResult]
