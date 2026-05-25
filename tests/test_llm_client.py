import pytest

from app.core.config import settings
from app.services import llm_client
from app.services.llm_client import DeepSeekLLMClient, get_llm_client


def test_get_llm_client_selects_deepseek(monkeypatch):
    monkeypatch.setattr(settings, "llm_provider", "deepseek")
    monkeypatch.setattr(settings, "llm_api_key", "test-key")
    monkeypatch.setattr(settings, "llm_base_url", "https://example.com")
    monkeypatch.setattr(settings, "llm_model", "test-model")

    class DummyOpenAI:
        def __init__(self, api_key, base_url):
            self.api_key = api_key
            self.base_url = base_url

    monkeypatch.setattr(llm_client, "OpenAI", DummyOpenAI)

    client = get_llm_client()

    assert isinstance(client, DeepSeekLLMClient)
    assert client.provider_name == "deepseek"
    assert client.model_name == "test-model"


def test_get_llm_client_rejects_unknown_provider(monkeypatch):
    monkeypatch.setattr(settings, "llm_provider", "unknown")

    with pytest.raises(ValueError, match="Unsupported LLM provider"):
        get_llm_client()
