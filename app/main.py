"""CLI entry point for the Jev decision-driven research agent.

Usage:
    python -m app.main "Find out whether Acme raised funding recently"
    python -m app.main --steps 6 "What is the latest on ..."
    echo "..." | python -m app.main

Runs the LangGraph, printing the trace and the final answer. Requires a
``TYPESAFE_API_KEY`` for Jev decisions and an LLM API key; without the former
the agent falls back to a deterministic policy so you can still exercise the
harness.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys

from dotenv import load_dotenv

from app.decision.jev import create_decision_layer, describe_decision_layer
from app.graph.graph import build_graph


def _load_env() -> None:
    load_dotenv()  # no-op when .env does not exist


def main(argv: list[str] | None = None) -> int:
    _load_env()

    parser = argparse.ArgumentParser(description="Jev decision-driven research agent")
    parser.add_argument("query", nargs="?", help="The research request.")
    parser.add_argument("--steps", type=int, default=None, help="Override MAX_RESEARCH_STEPS.")
    parser.add_argument("--json", action="store_true", help="Emit the full trace as JSON.")
    parser.add_argument("--verbose", action="store_true", help="Enable INFO logging.")
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=logging.INFO if (args.verbose or os.getenv("VERBOSE")) else logging.WARNING,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    query = args.query or (sys.stdin.read().strip() if not sys.stdin.isatty() else None)
    if not query:
        parser.print_usage()
        return 2

    decision_layer = create_decision_layer()
    layer_label = describe_decision_layer(decision_layer)
    logging.getLogger("app").info("Decision layer: %s", layer_label)

    graph = build_graph(decision_layer=decision_layer, max_steps=args.steps)
    result = graph.invoke({"user_query": query})

    if args.json:
        trace = _trace(result)
        trace["decision_layer"] = layer_label
        print(json.dumps(trace, default=str, indent=2))
    else:
        print("\n=== FINAL ANSWER ===")
        print(result.get("final_answer", "(no answer produced)"))
        print("\n=== RESEARCH LOG ===")
        print(f"  - decision layer: {layer_label}")
        for used in result.get("queries_used", []):
            print(f"  - query: {used}")
        print(f"  - steps: {result.get('steps', 0)}")
        print(f"  - observations: {len(result.get('observations', []))}")
        print(f"  - action probabilities: {result.get('action_probabilities')}")
    return 0


def _trace(result: dict) -> dict:
    keep = {
        "user_query",
        "understanding",
        "plan",
        "web_queries",
        "news_queries",
        "queries_used",
        "steps",
        "attempted",
        "action",
        "action_probabilities",
        "action_confidence",
        "enough_information",
        "enough_probability",
        "stop_research",
        "final_answer",
    }
    trace = {k: v for k, v in result.items() if k in keep}
    trace["observations"] = [
        {
            "index": i,
            "title": obs.title,
            "url": obs.url,
            "snippet": obs.snippet,
            "source": obs.source,
            "published": obs.published,
            "query": obs.query,
        }
        for i, obs in enumerate(result.get("observations", []), start=1)
    ]
    return trace


if __name__ == "__main__":
    raise SystemExit(main())