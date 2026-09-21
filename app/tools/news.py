"""News search tool with pluggable providers.

Same interface as :mod:`app.tools.search` but scoped to recent news, which is
useful for time-sensitive research (e.g. "Company X recently raised funding").
"""

from __future__ import annotations

import os

from app.tools.search import (
    SearchError,
    SearchProvider,
    SearchResult,
    _load_ddgs,
    resolve_provider_name,
)


class DuckDuckGoNewsProvider:
    """Search recent news via DuckDuckGo (no API key)."""

    name = "duckduckgo"

    def results(self, query: str, max_results: int) -> list[SearchResult]:
        DDGS = _load_ddgs()
        try:
            with DDGS() as ddgs:
                raw_items = list(ddgs.news(query, max_results=max_results))
        except Exception as exc:  # noqa: BLE001 - normalize provider errors
            raise SearchError(f"duckduckgo news search failed: {exc}") from exc
        return [
            SearchResult(
                title=item.get("title", ""),
                url=item.get("url") or item.get("href", ""),
                snippet=item.get("body", ""),
                source=item.get("source", "duckduckgo"),
                published=item.get("date") or None,
                query=query,
            )
            for item in raw_items
        ]


class TavilyNewsProvider:
    """Search recent news via Tavily (requires TAVILY_API_KEY)."""

    name = "tavily"

    def __init__(self, api_key: str | None = None) -> None:
        self.api_key = api_key or os.getenv("TAVILY_API_KEY")

    def results(self, query: str, max_results: int) -> list[SearchResult]:
        from tavily import TavilyClient

        try:
            client = TavilyClient(api_key=self.api_key)
            response = client.search(
                query=query,
                max_results=max_results,
                topic="news",
            )
        except Exception as exc:  # noqa: BLE001 - normalize provider errors
            raise SearchError(f"tavily news search failed: {exc}") from exc
        return [
            SearchResult(
                title=result.get("title", ""),
                url=result.get("url", ""),
                snippet=result.get("content", ""),
                source=result.get("source", "tavily"),
                published=result.get("published_date") or None,
                query=query,
            )
            for result in response.get("results", [])
        ]


def get_news_provider(provider_name: str | None = None) -> SearchProvider:
    name = resolve_provider_name(provider_name)
    if name == "tavily":
        return TavilyNewsProvider()
    return DuckDuckGoNewsProvider()


def search_news(
    query: str,
    *,
    max_results: int | None = None,
    provider: SearchProvider | None = None,
) -> list[SearchResult]:
    """Search recent news and return normalized results.

    Raises :class:`SearchError` on backend failure.
    """
    provider = provider or get_news_provider()
    limit = max_results or _int_env("SEARCH_RESULTS", 5)
    return list(provider.results(query, limit))


def _int_env(name: str, default: int) -> int:
    raw = os.getenv(name)
    if raw is None or raw.strip() == "":
        return default
    try:
        return int(raw)
    except ValueError:
        return default


__all__ = ["get_news_provider", "search_news", "SearchResult", "SearchProvider", "SearchError"]