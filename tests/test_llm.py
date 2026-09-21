"""Tests for the provider-agnostic chat model factory."""

from __future__ import annotations

import pytest
from langchain_openai import ChatOpenAI

from app.llm.model import (
    NIM_BASE_URL,
    OPENAI_COMPATIBLE_PROVIDERS,
    get_chat_model,
)


class TestOpenAICompatibleNIM:
    def test_nim_points_at_hosted_base_url(self, monkeypatch):
        monkeypatch.setenv("NVIDIA_API_KEY", "nvapi-test")
        model = get_chat_model(provider="nim")
        assert isinstance(model, ChatOpenAI)
        assert model.openai_api_base == NIM_BASE_URL
        assert model.model_name == "meta/llama-3.3-70b-instruct"
        assert model.openai_api_key.get_secret_value() == "nvapi-test"

    @pytest.mark.parametrize("alias", sorted(OPENAI_COMPATIBLE_PROVIDERS))
    def test_aliases_resolve_to_nim(self, monkeypatch, alias):
        monkeypatch.setenv("NVIDIA_API_KEY", "nvapi-test")
        model = get_chat_model(provider=alias)
        assert model.openai_api_base == NIM_BASE_URL

    def test_llm_api_key_wins_over_nvidia_key(self, monkeypatch):
        monkeypatch.setenv("NVIDIA_API_KEY", "nvapi-default")
        monkeypatch.setenv("LLM_API_KEY", "llm-override")
        model = get_chat_model(provider="nim")
        assert model.openai_api_key.get_secret_value() == "llm-override"

    def test_explicit_base_url_override(self, monkeypatch):
        monkeypatch.setenv("NVIDIA_API_KEY", "nvapi-test")
        model = get_chat_model(
            provider="nim", base_url="http://localhost:8000/v1", model="my/local-model"
        )
        assert model.openai_api_base == "http://localhost:8000/v1"
        assert model.model_name == "my/local-model"

    def test_custom_base_url_via_env(self, monkeypatch):
        monkeypatch.setenv("NVIDIA_API_KEY", "nvapi-test")
        monkeypatch.setenv("LLM_BASE_URL", "http://10.0.0.5:8000/v1")
        monkeypatch.delenv("LLM_PROVIDER", raising=False)
        monkeypatch.setenv("LLM_PROVIDER", "nim")
        model = get_chat_model()
        assert model.openai_api_base == "http://10.0.0.5:8000/v1"

    def test_missing_key_raises_clear_error(self, monkeypatch):
        monkeypatch.delenv("NVIDIA_API_KEY", raising=False)
        monkeypatch.delenv("LLM_API_KEY", raising=False)
        with pytest.raises(ValueError, match="NVIDIA_API_KEY"):
            get_chat_model(provider="nim")


class TestOpenAI:
    def test_defaults_to_openai(self, monkeypatch):
        monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
        model = get_chat_model()
        assert isinstance(model, ChatOpenAI)
        assert model.model_name == "gpt-4o-mini"

    def test_openai_with_custom_base_url(self, monkeypatch):
        monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
        model = get_chat_model(provider="openai", base_url="http://localhost:8000/v1")
        assert model.openai_api_base == "http://localhost:8000/v1"

    def test_missing_key_raises(self, monkeypatch):
        monkeypatch.delenv("OPENAI_API_KEY", raising=False)
        with pytest.raises(OpenAIError := __import__("openai", fromlist=["OpenAIError"]).OpenAIError):
            get_chat_model(provider="openai")