from abc import ABC, abstractmethod
from typing import Optional, Dict, List, Any
from .models import ModelAnswer


class BaseLLMClient(ABC):
    def __init__(self, model_id: str):
        self.model_id = model_id

    @abstractmethod
    def generate(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        json_schema: Optional[Dict[str, Any]] = None,
    ) -> str:
        """
        Generate a response from the LLM.

        Args:
            prompt: The user query/prompt.
            system_prompt: Optional system prompt to guide the persona.
            json_schema: Optional JSON schema to force structured output.

        Returns:
            The string response from the LLM.
        """
        pass

    @abstractmethod
    def question_answer(
        self,
        system: str,
        question: str,
        options: Dict[str, str],
        messages: List[Dict[str, str]] = None,
        **kwargs,
    ) -> ModelAnswer:
        """
        Recebe uma pergunta e opções, prepara o modelo para responder
        exclusivamente com a opção escolhida e justificativa.

        Args:
            system: system prompt (persona)
            question: texto da pergunta
            options: dict de opções {key: value} ex: {"a": "Concordo", "b": "Discordo"}
            messages: histórico de mensagens anteriores do chat
                      [{"role": "assistant", "content": "..."}, {"role": "user", "content": "..."}]
            **kwargs: parâmetros extras específicos do adapter

        Returns:
            ModelAnswer com answer=(key, value) e explanation
        """
        pass

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Default cleanup — subclasses with GPU resources should override."""
        return False
