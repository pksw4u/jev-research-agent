"""Graph nodes.

Every node is built by a factory that takes its dependencies explicitly, so
tests can inject fakes and live runs can wire real components. This keeps the
graph topology in ``graph.py`` clean and the payroll honest: Jev decides, the
harness executes, the LLM reasons.
"""

from __future__ import annotations

import logging
import os
from typing import Callable, List

from langchain_core.language_models.chat_models import BaseChatModel

from app.decision.policy import (
    DecisionLayer,
    StopPolicy,
    create_stop_policy,
)
from app.decision.schemas import (
    ActionSelection,
    DecisionContext,
    ResearchAction,
    SufficiencyDecision,
)
from app.graph.state import ResearchState
from app.prompts.final_answer import build_final_answer_messages
from app.prompts.task_understanding import build_tasks, parse_task_output
from app.tools.news import search_news as default_news_search
from app.tools.search import (
    SearchResult,
    render_results,
    search_web as default_web_search,
)

logger = logging.getLogger(__name__)


# --------------------------------------------------------------------------- #
# Shared helpers
# --------------------------------------------------------------------------- #

def _int_env(name: str, default: int) -> int:
    raw = os.getenv(name)
    if raw is None or raw.strip() == "":
        return default
    try:
        return int(raw)
    except ValueError:
        return default


def recent_observations(state: ResearchState, limit: int | None = None) -> list[SearchResult]:
    limit = limit if limit is not None else _int_env("RECENT_OBSERVATIONS", 4)
    return list(state.get("observations", []))[-limit:]


def build_decision_context(
    state: ResearchState,
    *,
    max_steps: int | None = None,
    observations_limit: int | None = None,
) -> DecisionContext:
    return DecisionContext(
        user_query=state.get("user_query", ""),
        understanding=state.get("understanding", ""),
        steps_taken=int(state.get("steps", 0)),
        max_steps=max_steps if max_steps is not None else _int_env("MAX_RESEARCH_STEPS", 4),
        recent_observations=render_results(recent_observations(state, observations_limit)),
        attempted=list(state.get("attempted", [])),
    )


# --------------------------------------------------------------------------- #
# LLM node: task understanding, plan, candidate queries
# --------------------------------------------------------------------------- #

def make_understand_task_node(
    chat_model: BaseChatModel,
) -> Callable[[ResearchState], dict]:
    """Turn the user request into understanding + plan + candidate queries."""

    def understand_task(state: ResearchState) -> dict:
        messages = build_tasks(state["user_query"])
        response = chat_model.invoke(messages)
        parsed = parse_task_output(
            response.content if isinstance(response.content, str) else str(response.content)
        )
        return {
            "understanding": parsed["understanding"],
            "plan": parsed["plan"],
            "web_queries": parsed["web_queries"],
            "news_queries": parsed["news_queries"],
            "steps": 0,
            "attempted": [],
            "queries_used": [],
            "observations": [],
        }

    return understand_task


# --------------------------------------------------------------------------- #
# Decision layer
# --------------------------------------------------------------------------- #

def make_select_action_node(
    decision_layer: DecisionLayer,
    *,
    max_steps: int | None = None,
) -> Callable[[ResearchState], dict]:
    """Let Jev (or the policy fallback) select the next action."""

    def select_action(state: ResearchState) -> dict:
        context = build_decision_context(state, max_steps=max_steps)
        selection: ActionSelection = decision_layer.select_action(context)
        return {
            "action": selection.decision,
            "action_probabilities": selection.probabilities,
            "action_confidence": selection.confidence,
        }

    return select_action


def make_decide_continue_node(
    decision_layer: DecisionLayer,
    stop_policy: StopPolicy | None = None,
    *,
    max_steps: int | None = None,
) -> Callable[[ResearchState], dict]:
    """Let Jev (or the policy fallback) decide whether research is complete."""

    def decide_continue(state: ResearchState) -> dict:
        policy = stop_policy or create_stop_policy()
        context = build_decision_context(state, max_steps=max_steps)
        decision: SufficiencyDecision = decision_layer.decide_sufficiency(context)
        should_stop = policy.should_stop(decision, context)
        return {
            "enough_information": decision.enough,
            "enough_probability": decision.probability,
            "stop_research": should_stop,
        }

    return decide_continue


# --------------------------------------------------------------------------- #
# Harness: execute the selected action
# --------------------------------------------------------------------------- #

def make_execute_tool_node(
    web_search: Callable = default_web_search,
    news_search: Callable = default_news_search,
) -> Callable[[ResearchState], dict]:
    """Execute the action Jev selected. Jev never executes tools itself."""

    def execute_tool(state: ResearchState) -> dict:
        action = state["action"]
        if action is ResearchAction.FINISH:
            return {}

        query, remaining = _consume_query(state, action)
        logger.info("Executing %s with query: %s", action.value, query)

        try:
            if action is ResearchAction.SEARCH_WEB:
                results: list[SearchResult] = list(web_search(query))
            else:
                results = list(news_search(query))
        except Exception as exc:  # noqa: BLE001 - a failed tool must not abort the run
            logger.warning("Tool %s failed for query %r: %s", action.value, query, exc)
            results = [
                SearchResult(
                    title=f"{action.value} failed",
                    url="",
                    snippet=str(exc),
                    source="harness-error",
                    query=query,
                )
            ]

        return {
            "observations": results,
            "queries_used": [query],
            "attempted": list(state.get("attempted", [])) + [action.value],
            "steps": int(state.get("steps", 0)) + 1,
            "web_queries": remaining if action is ResearchAction.SEARCH_WEB else state.get("web_queries", []),
            "news_queries": remaining if action is ResearchAction.SEARCH_NEWS else state.get("news_queries", []),
        }

    return execute_tool


def _consume_query(state: ResearchState, action: ResearchAction) -> tuple[str, List[str]]:
    """Pop the next planned query for ``action``, falling back to the request."""
    key = "web_queries" if action is ResearchAction.SEARCH_WEB else "news_queries"
    queue: List[str] = list(state.get(key, []))
    if queue:
        return queue.pop(0), queue
    return state.get("user_query", ""), queue


# --------------------------------------------------------------------------- #
# LLM node: final answer
# --------------------------------------------------------------------------- #

def make_generate_answer_node(
    chat_model: BaseChatModel,
) -> Callable[[ResearchState], dict]:
    """Synthesize accumulated observations into the user-facing answer."""

    def generate_answer(state: ResearchState) -> dict:
        findings = render_results(list(state.get("observations", [])))
        messages = build_final_answer_messages(state["user_query"], findings)
        response = chat_model.invoke(messages)
        content = response.content if isinstance(response.content, str) else str(response.content)
        return {"final_answer": content}

    return generate_answer