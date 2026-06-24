import gc
import json
import re
from typing import Optional, Dict, Any, List
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

from settings import settings
from .client_base import BaseLLMClient
from .models import ModelAnswer


class TucanoAdapter(BaseLLMClient):
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

    def free_memory(self):
        """Releases GPU memory manually if needed."""
        self.__exit__(None, None, None)

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
            temperature=0.7,
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

        full_messages = [{"role": "system", "content": system}]
        full_messages.extend(messages)

        options_text = "\n".join([f"{k}: {v}" for k, v in options.items()])
        prompt = (
            f"{question}\n\n"
            f"Opções:\n{options_text}\n\n"
            f"Por favor, use <think> para raciocinar sobre a pergunta e então responda fornecendo um JSON imediatamente após a tag </think>.\n"
            f"O JSON deve ter as chaves 'answer' (apenas a letra da opção escolhida) e 'explanation' (resumo do porquê)."
        )
        full_messages.append({"role": "user", "content": prompt})

        text_prompt = self.tokenizer.apply_chat_template(
            full_messages, tokenize=False, add_generation_prompt=True
        )

        inputs = self.tokenizer(text_prompt, return_tensors="pt").to(self.device)

        outputs = self.model.generate(
            **inputs,
            max_new_tokens=2048,
            temperature=0.7,
            do_sample=True,
            pad_token_id=self.tokenizer.eos_token_id,
        )

        generated_ids = outputs[0][inputs.input_ids.shape[-1] :]
        response_text = self.tokenizer.decode(generated_ids, skip_special_tokens=True)

        # Parse <think>...</think> and extract the rest as JSON
        think_match = re.search(r"<think>(.*?)</think>", response_text, flags=re.DOTALL)
        explanation_from_think = ""
        if think_match:
            explanation_from_think = think_match.group(1).strip()
            rest_of_text = response_text[think_match.end() :].strip()
        else:
            rest_of_text = response_text.strip()

        json_match = re.search(r"\{.*\}", rest_of_text, flags=re.DOTALL)
        if json_match:
            json_str = json_match.group(0)
        else:
            json_str = rest_of_text

        ans_key = ""
        explanation_from_json = ""
        try:
            parsed_json = json.loads(json_str)
            ans_key = parsed_json.get("answer", "").strip()
            explanation_from_json = parsed_json.get("explanation", "").strip()
        except json.JSONDecodeError:
            pass

        if ans_key not in options:
            # Fallback by sliding window / regex search for the key
            for k in options.keys():
                # checks for something like "answer": "A"
                if re.search(
                    rf'["\']answer["\']\s*:\s*["\']{k}["\']',
                    rest_of_text,
                    re.IGNORECASE,
                ):
                    ans_key = k
                    break
            if ans_key not in options:
                for k in options.keys():
                    if k in rest_of_text:
                        ans_key = k
                        break

        # Use explanation from <think> if available, otherwise from JSON
        explanation = (
            explanation_from_json if explanation_from_json else explanation_from_think
        )

        return ModelAnswer(
            answer=(ans_key, options.get(ans_key, "")), explanation=explanation
        )
