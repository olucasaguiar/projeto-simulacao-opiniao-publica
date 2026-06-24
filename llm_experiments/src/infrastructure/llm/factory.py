from typing import Callable, Dict, Optional
from .client_base import BaseLLMClient
from .maritaca_adapter import MaritacaAdapter
from .tucano_adapter import TucanoAdapter
from .jurema_adapter import JuremaAdapter
from .llama_adapter import LlamaAdapter

from settings import settings


class LLMFactory:
    """Factory responsável por gerenciar instâncias de adapters LLM."""

    def __init__(self):
        self._registry: Dict[str, Callable[[], BaseLLMClient]] = {}
        for m in settings.llm.active_models:
            if m.adapter == "maritaca":
                self._registry[m.id] = lambda m_id=m.id: MaritacaAdapter(m_id)
            elif m.adapter == "tucano":
                self._registry[m.id] = lambda m_id=m.id, hf_id=m.huggingface_id: (
                    TucanoAdapter(m_id, hf_id)
                )
            elif m.adapter == "jurema":
                self._registry[m.id] = lambda m_id=m.id, hf_id=m.huggingface_id: (
                    JuremaAdapter(m_id, hf_id)
                )
            elif m.adapter == "llama":
                self._registry[m.id] = lambda m_id=m.id, hf_id=m.huggingface_id: (
                    LlamaAdapter(m_id, hf_id)
                )

    def provide(self, model: str) -> Optional[BaseLLMClient]:
        """
        Retorna uma instância do adapter correspondente ao modelo.
        Retorna None se o modelo não existir no registry.
        """
        builder = self._registry.get(model)
        if builder:
            return builder()
        return None
