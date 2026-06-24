import json
import gc
import re
from typing import Dict, List, Optional, Any
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

from settings import settings
from .client_base import BaseLLMClient
from .models import ModelAnswer


class JuremaAdapter(BaseLLMClient):
    def __init__(self, model_id: str, huggingface_id: str):
        super().__init__(model_id)
        self.huggingface_id = huggingface_id
        self.device = settings.llm.local.device
        self.quantization = settings.llm.local.quantization

        self.model = None
        self.tokenizer = None

    def _initialize(self):
        if self.model is not None:
            return

        print(f"Loading {self.huggingface_id} to {self.device}...")
        self.tokenizer = AutoTokenizer.from_pretrained(self.huggingface_id)

        model_kwargs = {"device_map": self.device}
        if self.quantization == "4bit":
            model_kwargs["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=True, bnb_4bit_compute_dtype=torch.float16
            )
        elif self.quantization == "8bit":
            model_kwargs["quantization_config"] = BitsAndBytesConfig(load_in_8bit=True)

        self.model = AutoModelForCausalLM.from_pretrained(
            self.huggingface_id, **model_kwargs
        )

    def generate(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        json_schema: Optional[Dict[str, Any]] = None,
    ) -> str:
        self._initialize()

        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})

        if json_schema:
            schema_instructions = f"\n\nResponda APENAS com um objeto JSON válido que obedeça ao seguinte schema:\n{json.dumps(json_schema, indent=2)}\nNão adicione nenhum texto antes ou depois do JSON."
            messages[-1]["content"] += schema_instructions

        text_prompt = self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )

        inputs = self.tokenizer(text_prompt, return_tensors="pt").to(self.device)

        outputs = self.model.generate(
            **inputs,
            max_new_tokens=1024,
            temperature=0.5,
            do_sample=True,
            pad_token_id=self.tokenizer.eos_token_id,
        )

        generated_ids = outputs[0][inputs.input_ids.shape[-1] :]
        return self.tokenizer.decode(generated_ids, skip_special_tokens=True)

    def question_answer(
        self,
        system: str,
        question: str,
        options: Dict[str, str],
        messages: List[Dict[str, str]] = None,
        **kwargs,
    ) -> ModelAnswer:
        self._initialize()

        if messages is None:
            messages = []

        # Prepare prompt
        prompt = system + "\n\n" + question + "\n\nOpções:\n"
        for k, v in options.items():
            prompt += f"{k}: {v}\n"

        prompt += "\nResponda APENAS com um objeto JSON válido no formato:\n"
        prompt += '{"answer": "letra_da_opcao", "explanation": "sua justificativa"}'

        msgs = [{"role": "user", "content": prompt}]

        # Use ChatML via apply_chat_template
        text_prompt = self.tokenizer.apply_chat_template(
            msgs, tokenize=False, add_generation_prompt=True
        )

        inputs = self.tokenizer(text_prompt, return_tensors="pt").to(self.device)

        outputs = self.model.generate(
            **inputs,
            max_new_tokens=1024,
            temperature=0.3,
            do_sample=True,
            pad_token_id=self.tokenizer.eos_token_id,
        )

        generated_ids = outputs[0][inputs.input_ids.shape[-1] :]
        response_text = self.tokenizer.decode(generated_ids, skip_special_tokens=True)

        return self._extract_json(response_text, options)

    def _extract_json(self, text: str, options: Dict[str, str]) -> ModelAnswer:
        # Fallback robusto para extração de JSON
        try:
            # Tenta extrair qualquer coisa entre chaves {}
            match = re.search(r"\{.*\}", text.strip(), re.DOTALL)
            if match:
                json_str = match.group(0)
                data = json.loads(json_str)
            else:
                data = json.loads(text.strip())

            ans_key = str(data.get("answer", "")).lower().strip()
            explanation = str(data.get("explanation", ""))

            # Se a resposta for a chave pura, ex: 'a'
            if ans_key in options:
                return ModelAnswer(
                    answer=(ans_key, options[ans_key]), explanation=explanation
                )

            # Verifica se alguma chave está contida na resposta
            for k, v in options.items():
                if k.lower() in ans_key:
                    return ModelAnswer(answer=(k, v), explanation=explanation)

            return ModelAnswer(
                answer=(ans_key, "Opção inválida"), explanation=explanation
            )
        except Exception as e:
            # Fallback final se falhar o parsing do JSON
            return ModelAnswer(
                answer=("error", str(e)),
                explanation=f"Falha ao extrair JSON da resposta original: {text}",
            )

    def __exit__(self, exc_type, exc_val, exc_tb):
        # O contexto manager __exit__ deve liberar explicitamente GPU
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
