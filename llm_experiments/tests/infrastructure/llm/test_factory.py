from infrastructure.llm import LLMFactory
from infrastructure.llm.maritaca_adapter import MaritacaAdapter
from infrastructure.llm.tucano_adapter import TucanoAdapter
from infrastructure.llm.jurema_adapter import JuremaAdapter
from infrastructure.llm.llama_adapter import LlamaAdapter


def test_llm_factory_provide_maritaca(monkeypatch):
    monkeypatch.setenv("MARITACA_API_KEY", "fake-key")
    factory = LLMFactory()
    client = factory.provide("sabia-4")
    assert isinstance(client, MaritacaAdapter)
    assert client.model_id == "sabia-4"


def test_llm_factory_provide_tucano():
    factory = LLMFactory()
    client = factory.provide("tucano2-qwen-3.7b-think")
    assert isinstance(client, TucanoAdapter)
    assert client.huggingface_id == "Polygl0t/Tucano2-qwen-3.7B-Think"


def test_llm_factory_provide_jurema():
    factory = LLMFactory()
    client = factory.provide("jurema-7b")
    assert isinstance(client, JuremaAdapter)
    assert client.huggingface_id == "Jurema-br/Jurema-7B"


def test_llm_factory_provide_llama():
    factory = LLMFactory()
    client = factory.provide("llama-3.2-1b")
    assert isinstance(client, LlamaAdapter)
    assert client.huggingface_id == "meta-llama/Llama-3.2-1B-Instruct"


def test_llm_factory_provide_invalid():
    factory = LLMFactory()
    client = factory.provide("invalid-model")
    assert client is None
