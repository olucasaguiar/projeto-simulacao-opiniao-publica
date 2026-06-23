import os
import httpx
from typing import Optional, Dict, Any
from .client_base import BaseLLMClient
from settings import settings


class MaritacaAdapter(BaseLLMClient):
    def __init__(self, model_id: str):
        super().__init__(model_id)
        self.api_key = os.environ.get("MARITACA_API_KEY")
        if not self.api_key:
            raise ValueError("MARITACA_API_KEY environment variable is required.")

        self.base_url = settings.llm.maritaca.base_url
        self.timeout = settings.llm.maritaca.timeout

    def generate(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        json_schema: Optional[Dict[str, Any]] = None,
    ) -> str:

        headers = {
            "Authorization": f"Key {self.api_key}",
            "Content-Type": "application/json",
        }

        # Maritaca requires an array of messages
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})

        messages.append({"role": "user", "content": prompt})

        payload = {
            "model": self.model_id,
            "messages": messages,
            "max_tokens": 1024,
            "temperature": 0.7,
        }

        if json_schema:
            # Maritaca structure for JSON schema extraction
            payload["response_format"] = {
                "type": "json_schema",
                "json_schema": {
                    "name": "response_schema",
                    "schema": json_schema,
                    "strict": True,
                },
            }

        # Optional: We could use the official /chat/completions endpoint
        # to ensure compatibility with standard schemas.
        endpoint = f"{self.base_url}/chat/completions"

        with httpx.Client(timeout=self.timeout) as client:
            response = client.post(endpoint, headers=headers, json=payload)
            response.raise_for_status()

            data = response.json()
            return data["choices"][0]["message"]["content"]
