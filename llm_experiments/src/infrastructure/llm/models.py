from typing import Tuple, Literal
from pydantic import BaseModel

# Literal type com todos os modelos suportados pelo factory
ModelID = Literal[
    "sabia-4",
    "sabiazinho-4",
    "tucano2-qwen-3.7b-think",
    "jurema-7b",
    "llama-3.2-1b",
    "llama-3.2-3b",
]


class ModelAnswer(BaseModel):
    answer: Tuple[str, str]  # (key, value) — ex: ("a", "Concordo totalmente")
    explanation: str  # justificativa do modelo
