import gc
import json
import logging
import os
import re
from typing import Dict, List, Optional
import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    GenerationConfig,
)

from settings import settings
from .client_base import BaseLLMClient
from .models import ModelAnswer

logger = logging.getLogger(__name__)


class TucanoAdapter(BaseLLMClient):
    def __init__(
        self,
        model_id: str,
        huggingface_id: str,
        huggingface_token: Optional[str] = os.getenv("HF_TOKEN", None),
    ):
        super().__init__(model_id)
        self.huggingface_id = huggingface_id
        self.huggingface_token = huggingface_token
        self.device = settings.llm.local.device
        self.quantization = settings.llm.local.quantization

        # Lazy loading to avoid blocking when initializing other models
        self.model = None
        self.tokenizer = None

    def __enter__(self):
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
        )
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.huggingface_id, token=self.huggingface_token
        )
        self.model = AutoModelForCausalLM.from_pretrained(
            self.huggingface_id,
            token=self.huggingface_token,
            device_map="auto",
            dtype=torch.bfloat16,
            quantization_config=quantization_config,
        )

        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.model is not None:
            del self.model
            self.model = None
        if self.tokenizer is not None:
            del self.tokenizer
            self.tokenizer = None

        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        return False

    def _build_chat_messages(
        self,
        system: str,
        question: str,
        options: Dict[str, str],
        historical_messages: List[Dict[str, str]],
    ) -> List[Dict[str, str]]:
        chat_messages = []

        if system:
            chat_messages.append({"role": "system", "content": system})

        if historical_messages:
            chat_messages.extend(historical_messages)

        options_text = "\n".join([f"{k}: {v}" for k, v in options.items()])
        user_prompt = (
            f"{question}\n\n"
            f"Opções:\n{options_text}\n\n"
            f"Responda fornecendo APENAS um JSON contendo as chaves 'answer' (a letra da alternativa escolhida para a sua resposta) e 'explanation' (justificativa do porquê, contendo no máximo 85 palavras)."
        )
        chat_messages.append({"role": "user", "content": user_prompt})
        return chat_messages

    def _build_generation_config(self, **kwargs) -> GenerationConfig:
        return GenerationConfig(
            do_sample=kwargs.get("do_sample", True),
            temperature=kwargs.get("temperature", 0.7),
            top_k=kwargs.get("top_k", 50),
            top_p=kwargs.get("top_p", 1.0),
            repetition_penalty=kwargs.get("repetition_penalty", 1.2),
            max_new_tokens=kwargs.get("max_output_tokens", 1000),
            pad_token_id=self.tokenizer.eos_token_id,
        )

    def _parse_model_response(self, text: str, options: Dict[str, str]) -> ModelAnswer:
        try:
            _, response_text = self._split_think_content(text)
            match = re.search(r"\{.*\}", response_text.strip(), re.DOTALL)
            if match:
                data = json.loads(match.group(0))
            else:
                data = json.loads(response_text.strip())

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

    def _split_think_content(self, text: str) -> tuple[str, str]:
        match = re.search(r"<think>(.*?)</think>(.*?)", text, flags=re.DOTALL)
        if match:
            return match[0], match[1]
        else:
            return "", text

    def question_answer(
        self,
        system: str,
        question: str,
        options: Dict[str, str],
        messages: List[Dict[str, str]] = None,
        **kwargs,
    ) -> ModelAnswer:
        if not self.model or not self.tokenizer:
            raise RuntimeError("Hugging Face model or tokenizer not initialized.")

        prompt_messages = self._build_chat_messages(system, question, options, messages)
        prompt_text = self.tokenizer.apply_chat_template(
            prompt_messages, tokenize=False, add_generation_prompt=True
        )

        inputs = self.tokenizer(prompt_text, return_tensors="pt").to(self.model.device)

        generation_config = self._build_generation_config(**kwargs)

        with torch.no_grad():
            outputs = self.model.generate(**inputs, generation_config=generation_config)

        raw_response = self.tokenizer.decode(
            outputs[0][len(inputs.input_ids[0]) :], skip_special_tokens=True
        ).strip()

        return self._parse_model_response(raw_response, options)
