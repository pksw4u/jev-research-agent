"""End-to-end tests for the research graph with faked components.

No network: the chat model, Jev, and search backends are all injected fakes,
exercising the full loop (plan -> decide -> execute -> decide -> answer).
"""

from __future__ import annotations

import pytest
from langchain_core.messages import AIMessage

from app.decision.policy import PolicyDecisionLayer
from app.decision.schemas import (
    ActionSelection,
    ResearchAction,
    SufficiencyDecision,
)
from app.graph.graph import (
    build_graph,
    route_after_action,
    route_after_continue,
)
from app.tools.search import SearchResult

PLAN_JSON = (
    '{"understanding": "Check whether Acme raised funding", '
    '"plan": "Search web then news", '
    '"web_queries": ["Acme funding round recent"], '
    '"news_queries": ["Acme Corp Series B"]}'
)


class FakeChatModel:
    """Returns scripted replies; records the message lists it received."""

    def __init__(self, responses):
        self.responses = list(responses)
        self.calls = []

    def invoke(self, messages):
        self.calls.append(messages)
        if not self.responses:
            raise AssertionError("no scripted chat response left")
        return AIMessage(content=self.responses.pop(0))


class ScriptedDecisionLayer:
    """Pops scripted action selections and sufficiency probabilities."""

    def __init__(self, actions, sufficiency):
        self.actions = list(actions)
        self.sufficiency = list(sufficiency)

    def select_action(self, context) -> ActionSelection:
        action = self.actions.pop(0) if self.actions else ResearchAction.FINISH
        return ActionSelection(
            decision=action,
            confidence=0.99,
            probabilities={action.value: 0.99},
        )

    def decide_sufficiency(self, context) -> SufficiencyDecision:
        p = self.sufficiency.pop(0) if self.sufficiency else 1.0
        return SufficiencyDecision(enough=p >= 0.5, probability=p)


def stub_web(query):
    return [
        SearchResult(title=f"web result for {query}", url=f"https://w/{i}", snippet="body", source="stub-web", query=query)
        for i in range(2)
    ]


def stub_news(query):
    return [
        SearchResult(title=f"news result for {query}", url="https://n/1", snippet="body", source="stub-news", published="2026-09-01", query=query)
    ]


def make_graph(chat, layer, **kwargs):
    kwargs.setdefault("web_search", stub_web)
    kwargs.setdefault("news_search", stub_news)
    return build_graph(
        chat_model=chat,
        decision_layer=layer,
        **kwargs,
    )


class TestRouting:
    def test_route_after_action(self):
        assert route_after_action({"action": ResearchAction.FINISH}) == "generate_answer"
        state = {"action": ResearchAction.SEARCH_WEB}
        assert route_after_action(state) == "execute_tool"

    def test_route_after_continue(self):
        assert route_after_continue({"stop_research": True}) == "generate_answer"
        assert route_after_continue({"stop_research": False}) == "select_action"


class TestGraphLoops:
    def test_finishes_immediately_when_action_is_finish(self):
        chat = FakeChatModel([PLAN_JSON, "No research was needed."])
        layer = ScriptedDecisionLayer(actions=[ResearchAction.FINISH], sufficiency=[])
        graph = make_graph(chat, layer)

        result = graph.invoke({"user_query": "Is 2+2 4?"})

        assert result["steps"] == 0
        assert result["observations"] == []
        assert result["final_answer"] == "No research was needed."
        assert result["action"] is ResearchAction.FINISH

    def test_runs_two_research_loops_and_synthesizes(self):
        chat = FakeChatModel([PLAN_JSON, "Acme raised a Series B in September 2026."])
        layer = ScriptedDecisionLayer(
            actions=[ResearchAction.SEARCH_WEB, ResearchAction.SEARCH_NEWS],
            sufficiency=[0.4, 0.9],
        )
        graph = make_graph(chat, layer)

        result = graph.invoke({"user_query": "Did Acme raise funding recently?"})

        assert result["steps"] == 2
        assert [o.title for o in result["observations"]] == [
            "web result for Acme funding round recent",
            "web result for Acme funding round recent",
            "news result for Acme Corp Series B",
        ]
        assert result["attempted"] == ["search_web", "search_news"]
        assert result["queries_used"] == ["Acme funding round recent", "Acme Corp Series B"]
        assert result["enough_probability"] == 0.9
        assert result["stop_research"] is True
        assert result["final_answer"] == "Acme raised a Series B in September 2026."

    def test_forced_finish_at_max_steps(self):
        chat = FakeChatModel([PLAN_JSON, "Stopped by step limit."])
        layer = ScriptedDecisionLayer(
            actions=[ResearchAction.SEARCH_WEB, ResearchAction.SEARCH_WEB],
            sufficiency=[0.1, 0.1, 0.1],
        )
        graph = make_graph(chat, layer, max_steps=2, stop_policy=None)

        result = graph.invoke({"user_query": "keep searching forever?"})

        assert result["steps"] == 2
        assert result["stop_research"] is True
        assert result["final_answer"] == "Stopped by step limit."

    def test_no_queries_left_falls_back_to_user_query(self):
        chat = FakeChatModel(
            [
                '{"understanding":"u","plan":"p","web_queries":[],"news_queries":[]}',
                "Answer without useful findings.",
            ]
        )
        layer = ScriptedDecisionLayer(actions=[ResearchAction.SEARCH_WEB], sufficiency=[0.8])
        graph = make_graph(chat, layer)

        result = graph.invoke({"user_query": "fallback query here"})

        assert result["queries_used"] == ["fallback query here"]
        assert result["observations"][0].query == "fallback query here"

    def test_tool_failure_is_recorded_and_run_continues(self):
        from app.tools.search import SearchError

        def failing_web(query):
            raise SearchError("tavily web search failed: 429 rate limited")

        chat = FakeChatModel([PLAN_JSON, "Answer despite the failed search."])
        layer = ScriptedDecisionLayer(actions=[ResearchAction.SEARCH_WEB], sufficiency=[0.9])
        graph = make_graph(chat, layer, web_search=failing_web)

        result = graph.invoke({"user_query": "Did Acme raise funding?"})

        assert result["steps"] == 1
        assert result["stop_research"] is True
        error_obs = result["observations"][0]
        assert error_obs.source == "harness-error"
        assert "rate limited" in error_obs.snippet
        assert result["final_answer"] == "Answer despite the failed search."


class TestGraphWiring:
    def test_policy_only_decision_layer_and_fake_model(self):
        chat = FakeChatModel([PLAN_JSON, "final"])
        layer = PolicyDecisionLayer()
        graph = make_graph(chat, layer, max_steps=1)
        names = {n for n in graph.get_graph().nodes}
        assert {"understand_task", "select_action", "execute_tool", "decide_continue", "generate_answer"} <= names

    def test_build_with_defaults_compiles(self, monkeypatch):
        monkeypatch.setenv("OPENAI_API_KEY", "dummy")
        monkeypatch.delenv("TYPESAFE_API_KEY", raising=False)
        graph = build_graph()  # real ChatOpenAI object, policy fallback layer
        names = {n for n in graph.get_graph().nodes}
        assert {"understand_task", "select_action", "execute_tool", "decide_continue", "generate_answer"} <= names