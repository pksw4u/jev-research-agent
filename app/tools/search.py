"""Web search tools with pluggable providers.

The harness executes tools: it never asks Jev to decide tool arguments, and it
never lets the LLM call the network directly. ``search_web`` / ``search_news``
are plain Python functions the graph nodes call.

Providers:

- ``tavily``      (requires ``TAVILY_API_KEY``) — selected automatically when
  that key is present and ``SEARCH_PROVIDER`` is unset.
- ``duckduckgo``  (default when no Tavily key; no API key required)
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Iterable, Protocol


class SearchError(RuntimeError):
    """Raised when a search backend fails (rate limit, auth, network, ...).

    The harness catches this so one failed search does not abort the run.
    """


@dataclass
class SearchResult:
    """A single normalized search result."""

    title: str
    url: str
    snippet: str
    source: str = ""
    published: str | None = None
    query: str = ""

    def to_text(self, index: int) -> str:
        published = f" (published: {self.published})" if self.published else ""
        return (
            f"[{index}] {self.title}{published}\n"
            f"    url: {self.url}\n"
            f"    source: {self.source}\n"
            f"    snippet: {self.snippet}"
        )


class SearchProvider(Protocol):
    """Common interface implemented by each search backend."""

    name: str

    def results(self, query: str, max_results: int) -> Iterable[SearchResult]:
        ...


def _load_ddgs():
    """Import the DuckDuckGo client, preferring the renamed ``ddgs`` package."""
    try:
        from ddgs import DDGS  # type: ignore[import-not-found]

        return DDGS
    except ImportError:
        from duckduckgo_search import DDGS

        return DDGS


class DuckDuckGoProvider:
    """Search the general web via DuckDuckGo (no API key)."""

    name = "duckduckgo"

    def results(self, query: str, max_results: int) -> list[SearchResult]:
        DDGS = _load_ddgs()
        try:
            with DDGS() as ddgs:
                raw_items = list(ddgs.text(query, max_results=max_results))
        except Exception as exc:  # noqa: BLE001 - normalize provider errors
            raise SearchError(f"duckduckgo web search failed: {exc}") from exc
        return [
            SearchResult(
                title=item.get("title", ""),
                url=item.get("href") or item.get("url", ""),
                snippet=item.get("body", ""),
                source="duckduckgo",
                query=query,
            )
            for item in raw_items
        ]


class TavilyProvider:
    """Search the general web via Tavily (requires TAVILY_API_KEY)."""

    name = "tavily"

    def __init__(self, api_key: str | None = None) -> None:
        self.api_key = api_key or os.getenv("TAVILY_API_KEY")

    def results(self, query: str, max_results: int) -> list[SearchResult]:
        from tavily import TavilyClient

        try:
            client = TavilyClient(api_key=self.api_key)
            response = client.search(query=query, max_results=max_results)
        except Exception as exc:  # noqa: BLE001 - normalize provider errors
            raise SearchError(f"tavily web search failed: {exc}") from exc
        return [
            SearchResult(
                title=result.get("title", ""),
                url=result.get("url", ""),
                snippet=result.get("content", ""),
                source=result.get("source", "tavily"),
                query=query,
            )
            for result in response.get("results", [])
        ]


def resolve_provider_name(provider_name: str | None = None) -> str:
    """Pick the provider: explicit arg > SEARCH_PROVIDER > key detection.

    When ``SEARCH_PROVIDER`` is unset, a configured ``TAVILY_API_KEY`` selects
    Tavily; otherwise DuckDuckGo is used.
    """
    if provider_name:
        return provider_name.strip().lower()
    configured = (os.getenv("SEARCH_PROVIDER") or "").strip().lower()
    if configured:
        return configured
    if os.getenv("TAVILY_API_KEY"):
        return "tavily"
    return "duckduckgo"


def get_search_provider(provider_name: str | None = None) -> SearchProvider:
    """Build the configured search provider from env or an explicit name."""
    name = resolve_provider_name(provider_name)
    if name == "tavily":
        return TavilyProvider()
    return DuckDuckGoProvider()


def search_web(
    query: str,
    *,
    max_results: int | None = None,
    provider: SearchProvider | None = None,
) -> list[SearchResult]:
    """Search the general web and return normalized results.

    Raises :class:`SearchError` on backend failure.
    """
    provider = provider or get_search_provider()
    limit = max_results or _int_env("SEARCH_RESULTS", 5)
    return list(provider.results(query, limit))


def render_results(results: list[SearchResult]) -> str:
    """Render accumulated results as the observation block for prompts/state."""
    if not results:
        return "No observations gathered yet."
    return "\n\n".join(r.to_text(i) for i, r in enumerate(results, start=1))


def _int_env(name: str, default: int) -> int:
    raw = os.getenv(name)
    if raw is None or raw.strip() == "":
        return default
    try:
        return int(raw)
    except ValueError:
        return default