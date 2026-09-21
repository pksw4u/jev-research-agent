"""Tests for search tool provider selection and error normalization."""

from __future__ import annotations

import pytest

from app.tools.news import get_news_provider
from app.tools.search import (
    DuckDuckGoProvider,
    SearchError,
    TavilyProvider,
    get_search_provider,
    resolve_provider_name,
    search_web,
)


@pytest.fixture(autouse=True)
def _clean_env(monkeypatch):
    for var in ("SEARCH_PROVIDER", "TAVILY_API_KEY"):
        monkeypatch.delenv(var, raising=False)


class TestProviderResolution:
    def test_defaults_to_duckduckgo_without_tavily_key(self):
        assert resolve_provider_name() == "duckduckgo"
        assert isinstance(get_search_provider(), DuckDuckGoProvider)

    def test_tavily_key_auto_selects_tavily(self, monkeypatch):
        monkeypatch.setenv("TAVILY_API_KEY", "tvly-test")
        assert resolve_provider_name() == "tavily"
        assert isinstance(get_search_provider(), TavilyProvider)

    def test_explicit_env_wins_over_key_detection(self, monkeypatch):
        monkeypatch.setenv("TAVILY_API_KEY", "tvly-test")
        monkeypatch.setenv("SEARCH_PROVIDER", "duckduckgo")
        assert resolve_provider_name() == "duckduckgo"
        assert isinstance(get_search_provider(), DuckDuckGoProvider)

    def test_explicit_argument_wins_over_env(self, monkeypatch):
        monkeypatch.setenv("SEARCH_PROVIDER", "duckduckgo")
        assert resolve_provider_name("tavily") == "tavily"
        assert isinstance(get_search_provider("tavily"), TavilyProvider)

    def test_news_provider_follows_same_resolution(self, monkeypatch):
        monkeypatch.setenv("TAVILY_API_KEY", "tvly-test")
        provider = get_news_provider()
        assert provider.name == "tavily"


class TestErrorNormalization:
    def test_tavily_failure_becomes_search_error(self, monkeypatch):
        monkeypatch.setenv("TAVILY_API_KEY", "tvly-test")

        class Boom:
            def __init__(self, *a, **k):
                pass

            def search(self, *a, **k):
                raise RuntimeError("rate limited")

        monkeypatch.setattr("tavily.TavilyClient", Boom)
        with pytest.raises(SearchError, match="tavily web search failed"):
            search_web("anything", provider=TavilyProvider("tvly-test"))