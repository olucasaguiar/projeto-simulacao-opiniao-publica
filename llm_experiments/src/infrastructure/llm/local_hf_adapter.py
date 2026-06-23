import json
from typing import Optional, Dict, Any
from .client_base import BaseLLMClient
from settings import settings


class LocalHFAdapter(BaseLLMClient):
    def __init__(self, model_id: str, huggingface_id: str):
        super().__init__(model_id)
        self.huggingface_id = huggingface_id
        self.device = settings.llm.local.device
        self.quantization = settings.llm.local.quantization

        # Lazy loading to avoid blocking when initializing other models
        self.model = None
        self.tokenizer = None

    def _initialize(self):
        if self.model is not None:
            return

        from transformers import AutoModelForCausalLM, AutoTokenizer
        import torch

        print(f"Loading {self.huggingface_id} to {self.device}...")
        self.tokenizer = AutoTokenizer.from_pretrained(self.huggingface_id)

        model_kwargs = {"device_map": self.device}

        if self.quantization == "4bit":
            from transformers import BitsAndBytesConfig

            model_kwargs["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=True, bnb_4bit_compute_dtype=torch.float16
            )
        elif self.quantization == "8bit":
            from transformers import BitsAndBytesConfig

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

        # Adding instructions for JSON if schema provided
        if json_schema:
            schema_instructions = f"\n\nResponda APENAS com um objeto JSON válido que obedeça ao seguinte schema:\n{json.dumps(json_schema, indent=2)}\nNão adicione nenhum texto antes ou depois do JSON."
            messages[-1]["content"] += schema_instructions

        try:
            text_prompt = self.tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
        except Exception:
            # Manual fallback formatting for models without native chat templates
            text_prompt = ""
            for msg in messages:
                role = msg["role"]
                content = msg["content"]
                text_prompt += f"<|im_start|>{role}\n{content}<|im_end|>\n"
            text_prompt += "<|im_start|>assistant\n"

        inputs = self.tokenizer(text_prompt, return_tensors="pt").to(self.device)

        # Simple generation
        outputs = self.model.generate(
            **inputs,
            max_new_tokens=1024,
            temperature=0.7,
            do_sample=True,
            pad_token_id=self.tokenizer.eos_token_id,
        )

        # Extract the response
        generated_ids = outputs[0][inputs.input_ids.shape[-1] :]
        response_text = self.tokenizer.decode(generated_ids, skip_special_tokens=True)

        return response_text

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
