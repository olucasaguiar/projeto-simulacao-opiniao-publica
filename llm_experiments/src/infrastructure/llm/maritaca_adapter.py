import os
import json
import time
import httpx
from typing import Optional, Dict, Any, List
from .client_base import BaseLLMClient
from .models import ModelAnswer
from settings import settings


class MaritacaAdapter(BaseLLMClient):
    def __init__(self, model_id: str):
        super().__init__(model_id)
        self.api_key = os.environ.get("MARITACA_API_KEY")
        if not self.api_key:
            raise ValueError("MARITACA_API_KEY environment variable is required.")

        self.base_url = settings.llm.maritaca.base_url
        self.timeout = settings.llm.maritaca.timeout

    def _post_with_backoff(self, endpoint: str, headers: dict, payload: dict) -> dict:
        max_retries = 5
        base_wait_time = 2.0

        with httpx.Client(timeout=self.timeout) as client:
            for attempt in range(max_retries):
                response = client.post(endpoint, headers=headers, json=payload)
                if response.status_code == 429:
                    if attempt == max_retries - 1:
                        response.raise_for_status()
                    wait_time = base_wait_time * (2**attempt)
                    time.sleep(wait_time)
                    continue
                response.raise_for_status()
                return response.json()
        raise Exception("Max retries exceeded")

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

        data = self._post_with_backoff(endpoint, headers, payload)
        return data["choices"][0]["message"]["content"]

    def question_answer(
        self,
        system: str,
        question: str,
        options: Dict[str, str],
        messages: List[Dict[str, str]] = None,
        **kwargs,
    ) -> ModelAnswer:
        headers = {
            "Authorization": f"Key {self.api_key}",
            "Content-Type": "application/json",
        }

        api_messages = []
        if system:
            api_messages.append({"role": "system", "content": system})

        if messages:
            api_messages.extend(messages)

        prompt_lines = [question, ""]
        for k, v in options.items():
            prompt_lines.append(f"{k}) {v}")
        prompt = "\n".join(prompt_lines)

        api_messages.append({"role": "user", "content": prompt})

        schema = {
            "type": "object",
            "properties": {
                "opcao": {
                    "type": "string",
                    "description": "A letra da opção escolhida",
                },
                "justificativa": {
                    "type": "string",
                    "description": "Justificativa da escolha",
                },
            },
            "required": ["opcao", "justificativa"],
        }

        payload = {
            "model": self.model_id,
            "messages": api_messages,
            "max_tokens": kwargs.get("max_tokens", 1024),
            "temperature": kwargs.get("temperature", 0.7),
            "response_format": {
                "type": "json_schema",
                "json_schema": {
                    "name": "response_schema",
                    "schema": schema,
                    "strict": True,
                },
            },
        }

        endpoint = f"{self.base_url}/chat/completions"
        data = self._post_with_backoff(endpoint, headers, payload)
        content = data["choices"][0]["message"]["content"]

        try:
            parsed = json.loads(content)
            opcao = str(parsed.get("opcao", "")).lower().strip()
            justificativa = str(parsed.get("justificativa", "")).strip()

            if len(opcao) > 0 and opcao[0] in options:
                key = opcao[0]
                return ModelAnswer(
                    answer=(key, options[key]), explanation=justificativa
                )

            for k in options:
                if k in opcao:
                    return ModelAnswer(
                        answer=(k, options[k]), explanation=justificativa
                    )
        except Exception:
            pass

        first_key = list(options.keys())[0]
        return ModelAnswer(
            answer=(first_key, options[first_key]),
            explanation=f"Parse error. Raw: {content}",
        )
