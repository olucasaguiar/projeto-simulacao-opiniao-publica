from abc import ABC, abstractmethod
from typing import Optional, Dict, Any


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
