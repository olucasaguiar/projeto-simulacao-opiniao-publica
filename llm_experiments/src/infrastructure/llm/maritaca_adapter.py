import os
import json
import re
import time
import httpx
import logging
from typing import Dict, List, Any

from .client_base import BaseLLMClient
from .models import ModelAnswer
from settings import settings

logger = logging.getLogger(__name__)


class MaritacaAdapter(BaseLLMClient):
    def __init__(
        self, model_id: str, maritaca_api_key: str = os.getenv("MARITACA_API_KEY", None)
    ):
        super().__init__(model_id)
        self.maritaca_api_key = maritaca_api_key
        if not self.maritaca_api_key:
            raise ValueError("MARITACA_API_KEY environment variable is required.")

        self.base_url = settings.llm.maritaca.base_url
        self.timeout = settings.llm.maritaca.timeout

    def _post_with_backoff(self, endpoint: str, headers: dict, payload: dict) -> dict:
        max_retries = 5
        base_wait_time = 2.0

        with httpx.Client(timeout=self.timeout) as client:
            for attempt in range(max_retries):
                try:
                    logger.info(
                        f"Sending request to {endpoint} (Attempt {attempt + 1}/{max_retries})"
                    )
                    response = client.post(endpoint, headers=headers, json=payload)
                    if response.status_code == 429:
                        if attempt == max_retries - 1:
                            logger.error(
                                "Max retries exceeded due to rate limiting (429)."
                            )
                            response.raise_for_status()
                        wait_time = base_wait_time * (2**attempt)
                        logger.warning(
                            f"Rate limited (429). Waiting {wait_time}s before retry."
                        )
                        time.sleep(wait_time)
                        continue

                    response.raise_for_status()
                    logger.info("Request successful.")
                    return response.json()
                except httpx.HTTPError as e:
                    logger.error(f"HTTP error occurred: {e}")
                    if attempt == max_retries - 1:
                        raise

        raise Exception("Max retries exceeded")

    def _build_chat_messages(
        self,
        question: str,
        options: Dict[str, str],
        historical_messages: List[Dict[str, str]],
    ) -> List[Dict[str, str]]:
        chat_messages = []

        if historical_messages:
            chat_messages.extend(historical_messages)

        options_text = "\n".join([f"{k}: {v}" for k, v in options.items()])
        user_prompt = (
            f"{question}\n\n"
            f"Opções:\n{options_text}\n\n"
            f"O JSON deve ter as chaves 'answer' (apenas a letra da opção escolhida) e 'explanation' (resumo do porquê)."
        )
        chat_messages.append({"role": "user", "content": user_prompt})
        return chat_messages

    def _build_payload(
        self, system: str, chat_messages: List[Dict[str, str]], kwargs: Dict[str, Any]
    ) -> dict:
        schema = {
            "type": "object",
            "properties": {
                "answer": {
                    "type": "string",
                    "description": "A letra da alternativa escolhida para a sua resposta",
                },
                "explanation": {
                    "type": "string",
                    "description": "Justificativa em relação a alternativa escolhida",
                },
            },
            "required": ["answer", "explanation"],
        }

        payload = {
            "model": self.model_id,
            "instructions": system,
            "input": chat_messages,
            "text": {
                "format": {
                    "type": "json_schema",
                    "name": "response_schema",
                    "schema": schema,
                    "strict": True,
                }
            },
            "top_p": kwargs.get("top_p", 1.0),
            "temperature": kwargs.get("temperature", 0.1),
            "max_output_tokens": kwargs.get("max_output_tokens", 150),
        }

        return payload

    def _parse_model_response(self, text: str, options: Dict[str, str]) -> ModelAnswer:
        try:
            match = re.search(r"\{.*\}", text.strip(), re.DOTALL)
            if match:
                data = json.loads(match.group(0))
            else:
                data = json.loads(text.strip())

            ans_key = str(data.get("answer", "")).lower().strip()
            explanation = str(data.get("explanation", "")).strip()

            if ans_key in options:
                return ModelAnswer(
                    answer=(ans_key, options[ans_key]), explanation=explanation
                )

            for key, value in options.items():
                if re.search(rf"\b{re.escape(key.lower())}\b", ans_key):
                    return ModelAnswer(answer=(key, value), explanation=explanation)

            return ModelAnswer(
                answer=(ans_key, "Opção inválida"), explanation=explanation
            )
        except Exception as error:
            logger.error(f"Failed to parse JSON from response: {text}")
            logger.debug(f"Exception details: {error}")
            return ModelAnswer(
                answer=("error", str(error)),
                explanation=f"Falha ao extrair JSON da resposta original: {text}",
            )

    def question_answer(
        self,
        system: str,
        question: str,
        options: Dict[str, str],
        messages: List[Dict[str, str]] = None,
        **kwargs,
    ) -> ModelAnswer:
        headers = {
            "Authorization": f"Key {self.maritaca_api_key}",
            "Content-Type": "application/json",
        }

        chat_messages = self._build_chat_messages(question, options, messages)
        payload = self._build_payload(system, chat_messages, kwargs)

        endpoint = f"{self.base_url}/v1/responses"
        data = self._post_with_backoff(endpoint, headers, payload)

        content = data["output"][0]["content"][0]["text"]
        return self._parse_model_response(content, options)
