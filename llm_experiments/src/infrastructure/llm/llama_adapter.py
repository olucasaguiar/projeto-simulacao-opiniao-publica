import os
import json
from typing import Optional, Dict, List, Any
from .client_base import BaseLLMClient
from .models import ModelAnswer


class LlamaAdapter(BaseLLMClient):
    def __init__(self, model_id: str, huggingface_id: str):
        super().__init__(model_id)
        self.huggingface_id = huggingface_id

        # Lazy loading to avoid blocking when initializing other models
        self.model = None
        self.tokenizer = None
        self.device = "cpu"

    def _initialize(self):
        if self.model is not None:
            return

        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        hf_token = os.getenv("HF_TOKEN")
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        print(f"Loading {self.huggingface_id} to {self.device}...")
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.huggingface_id, token=hf_token
        )
        self.model = AutoModelForCausalLM.from_pretrained(
            self.huggingface_id,
            device_map="auto",
            torch_dtype=torch.float16,
            token=hf_token,
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

        inputs = self.tokenizer(text_prompt, return_tensors="pt").to(self.model.device)

        outputs = self.model.generate(
            **inputs,
            max_new_tokens=1024,
            temperature=0.7,
            do_sample=True,
            pad_token_id=self.tokenizer.eos_token_id,
        )

        generated_ids = outputs[0][inputs.input_ids.shape[-1] :]
        response_text = self.tokenizer.decode(generated_ids, skip_special_tokens=True)

        return response_text

    def question_answer(
        self,
        system: str,
        question: str,
        options: Dict[str, str],
        messages: List[Dict[str, str]] = None,
        **kwargs,
    ) -> ModelAnswer:
        self._initialize()

        # Build prompt for QA
        prompt_lines = [question, ""]
        for k, v in options.items():
            prompt_lines.append(f"{k}) {v}")
        prompt = "\n".join(prompt_lines)

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

        schema_instructions = f"\n\nResponda APENAS com um objeto JSON válido que obedeça ao seguinte schema:\n{json.dumps(schema, indent=2)}\nNão adicione nenhum texto antes ou depois do JSON."

        chat_messages = []
        if system:
            chat_messages.append({"role": "system", "content": system})

        if messages:
            chat_messages.extend(messages)

        chat_messages.append(
            {"role": "user", "content": f"{prompt}{schema_instructions}"}
        )

        text_prompt = self.tokenizer.apply_chat_template(
            chat_messages, tokenize=False, add_generation_prompt=True
        )

        inputs = self.tokenizer(text_prompt, return_tensors="pt").to(self.model.device)

        outputs = self.model.generate(
            **inputs,
            max_new_tokens=kwargs.get("max_tokens", 1024),
            temperature=kwargs.get("temperature", 0.7),
            do_sample=True,
            pad_token_id=self.tokenizer.eos_token_id,
        )

        generated_ids = outputs[0][inputs.input_ids.shape[-1] :]
        response_text = self.tokenizer.decode(generated_ids, skip_special_tokens=True)

        try:
            start = response_text.find("{")
            end = response_text.rfind("}") + 1
            if start != -1 and end != 0:
                json_str = response_text[start:end]
                data = json.loads(json_str)
                opcao = str(data.get("opcao", "")).lower().strip()
                justificativa = str(data.get("justificativa", "")).strip()

                if len(opcao) > 0 and opcao[0] in options:
                    key = opcao[0]
                    return ModelAnswer(
                        answer=(key, options[key]), explanation=justificativa
                    )

                # fallback iterando nas chaves
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
            explanation=f"Parse error. Raw: {response_text}",
        )

    def free_memory(self):
        import gc
        import torch

        if self.model is not None:
            del self.model
            self.model = None
        if self.tokenizer is not None:
            del self.tokenizer
            self.tokenizer = None

        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.free_memory()
        return False
