"""Provider-agnostic chat model factory.

The research agent needs a general-purpose LLM for understanding, reasoning,
planning and generation. This module builds a LangChain chat model from
environment variables so the implementation stays provider-agnostic.

Supported ``LLM_PROVIDER`` values:

- ``openai``              — OpenAI (default). Set ``OPENAI_API_KEY``.
- ``nim`` / ``nvidia`` / ``openai_compatible`` — any OpenAI-compatible HTTP
  endpoint, including NVIDIA NIM. Defaults to the hosted NIM endpoint
  ``https://integrate.api.nvidia.com/v1`` using ``NVIDIA_API_KEY``; override
  with ``LLM_BASE_URL`` to point at a self-hosted NIM container, an OpenAI
  proxy/VLM, or a local vLLM server.
- ``anthropic``, ``google_genai``, ... — other LangChain providers
  (requires the matching ``langchain-<provider>`` package).

Environment variables:

- ``LLM_PROVIDER``      provider to use (default ``openai``)
- ``LLM_MODEL``         model name used by the provider
- ``LLM_BASE_URL``      base URL for OpenAI-compatible providers
- ``LLM_TEMPERATURE``   sampling temperature
- ``LLM_API_KEY``       optional key that wins over the provider-specific key
- ``OPENAI_API_KEY``    key for ``openai``
- ``NVIDIA_API_KEY``    key for ``nim`` (default)
"""

from __future__ import annotations

import os

from langchain.chat_models import init_chat_model
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_openai import ChatOpenAI

#: Hosted NVIDIA NIM endpoint, OpenAI-compatible chat completions API.
NIM_BASE_URL = "https://integrate.api.nvidia.com/v1"

#: Default model per provider when LLM_MODEL is unset.
DEFAULT_MODELS = {
    "openai": "gpt-4o-mini",
    "nim": "meta/llama-3.3-70b-instruct",
}

#: Aliases that map to the OpenAI-compatible ChatOpenAI client.
OPENAI_COMPATIBLE_PROVIDERS = {"nim", "nvidia", "openai_compatible", "openai-compatible"}

#: API key env var per provider when neither LLM_API_KEY nor the key arg is set.
PROVIDER_KEY_ENV = {
    "openai": "OPENAI_API_KEY",
    "nim": "NVIDIA_API_KEY",
}


def get_chat_model(
    *,
    provider: str | None = None,
    model: str | None = None,
    temperature: float | None = None,
    base_url: str | None = None,
    api_key: str | None = None,
) -> BaseChatModel:
    """Build a chat model from explicit arguments or environment variables."""
    provider = (provider or os.getenv("LLM_PROVIDER", "openai")).strip().lower()
    model = model or os.getenv("LLM_MODEL") or DEFAULT_MODELS.get(provider, "gpt-4o-mini")
    temperature = temperature if temperature is not None else _float_env("LLM_TEMPERATURE", 0.0)
    base_url = base_url if base_url is not None else os.getenv("LLM_BASE_URL") or None
    api_key = api_key if api_key is not None else (os.getenv("LLM_API_KEY") or None)

    if provider in OPENAI_COMPATIBLE_PROVIDERS:
        return _openai_compatible_model(
            model=model,
            temperature=temperature,
            base_url=base_url or NIM_BASE_URL,
            api_key=api_key or _provider_key("nim"),
        )

    if provider == "openai":
        kwargs: dict = {"temperature": temperature}
        if api_key:
            kwargs["openai_api_key"] = api_key
        if base_url:
            kwargs["openai_api_base"] = base_url
        return ChatOpenAI(model_name=model, **kwargs)

    kwargs = {"temperature": temperature}
    if provider == "anthropic" and api_key:
        kwargs["anthropic_api_key"] = api_key
    elif provider == "google_genai" and api_key:
        kwargs["google_api_key"] = api_key

    # Generic fallback for any other installed LangChain provider.
    return init_chat_model(model, model_provider=provider, **kwargs)


def _openai_compatible_model(
    *,
    model: str,
    temperature: float,
    base_url: str,
    api_key: str | None,
) -> ChatOpenAI:
    """ChatOpenAI pointing at any OpenAI-compatible endpoint (NIM, vLLM, ...)."""
    if not api_key:
        raise ValueError(
            "No API key configured for the OpenAI-compatible provider. "
            "Set NVIDIA_API_KEY (or LLM_API_KEY). For keyless local servers "
            "set LLM_API_KEY to any placeholder, e.g. 'EMPTY'."
        )
    return ChatOpenAI(
        model_name=model,
        temperature=temperature,
        openai_api_base=base_url,
        openai_api_key=api_key,
    )


def _provider_key(provider: str) -> str | None:
    env_name = PROVIDER_KEY_ENV.get(provider)
    return os.getenv(env_name) if env_name else None


def _float_env(name: str, default: float) -> float:
    raw = os.getenv(name)
    if raw is None or raw.strip() == "":
        return default
    try:
        return float(raw)
    except ValueError:
        return default