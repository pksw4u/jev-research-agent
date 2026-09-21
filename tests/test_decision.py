"""Tests for the Jev decision layer: parsing SDK responses and fallback.

Jev is faked by injecting an in-memory client — nothing in this file calls the
network.
"""

from __future__ import annotations

import pytest
from typesafe_sdk import ChoiceAnswer, NoulAnswer

from app.decision.jev import (
    JevDecisionLayer,
    JevUnavailableError,
    create_decision_layer,
    describe_decision_layer,
)
from app.decision.policy import ActionPolicy, PolicyDecisionLayer
from app.decision.schemas import (
    ActionSelection,
    DecisionContext,
    ResearchAction,
    SufficiencyDecision,
)


class FakeResponse:
    def __init__(self, answers):
        self.answers = answers


class FakeClient:
    """Scripted TypeSafeClient double: each call pops the next scenario."""

    def __init__(self, script):
        self.script = list(script)
        self.calls = []

    def system_one(self, state, questions):
        self.calls.append({"state": state, "questions": questions})
        if not self.script:
            raise AssertionError("no scripted response left")
        return self.script.pop(0)


def _action_answer(decision: str, confidence: float, probabilities=None):
    return ChoiceAnswer(
        choice=decision,
        confidence=confidence,
        probabilities=probabilities or {decision: confidence, "other": 1 - confidence},
    )


def _noul_answer(p: float):
    return NoulAnswer(noul=p)


class TestActionSelection:
    def _layer(self, script):
        return JevDecisionLayer(
            client=FakeClient(script),
            api_key="sk-test",
            policy=None,
        )

    def test_parses_choice_into_selection(self):
        layer = self._layer(
            [
                FakeResponse(
                    {
                        "action": _action_answer(
                            "search_news", 0.78, {"search_web": 0.15, "search_news": 0.78, "finish": 0.07}
                        )
                    }
                )
            ]
        )
        context = DecisionContext(user_query="did Acme raise funds?")
        selection = layer.select_action(context)

        assert selection.decision is ResearchAction.SEARCH_NEWS
        assert selection.confidence == 0.78
        assert selection.probabilities == {
            "search_web": 0.15,
            "search_news": 0.78,
            "finish": 0.07,
        }

    def test_parses_finish(self):
        layer = self._layer(
            [FakeResponse({"action": _action_answer("finish", 0.9, {"finish": 0.9})})]
        )
        selection = layer.select_action(DecisionContext(user_query="q"))
        assert selection.decision is ResearchAction.FINISH
        assert selection.is_finish

    def test_confidence_gate_forces_fallback_action(self):
        # Low confidence pick -> policy resolves to the unattempted read-only action.
        layer = JevDecisionLayer(
            client=FakeClient([FakeResponse({"action": _action_answer("search_news", 0.2)})]),
            api_key="sk-test",
            policy=ActionPolicy(confidence_threshold=0.6, max_steps=4),
        )
        selection = layer.select_action(
            DecisionContext(user_query="q", steps_taken=0, attempted=[])
        )
        assert selection.decision is ResearchAction.SEARCH_WEB

    def test_sends_full_state_to_jev(self):
        layer = self._layer(
            [FakeResponse({"action": _action_answer("search_web", 0.9)})]
        )
        layer.select_action(
            DecisionContext(user_query="query", understanding="understand", steps_taken=2)
        )
        sent = layer._client.calls[0]["state"]
        assert sent["query"] == "query"
        assert sent["understanding"] == "understand"
        assert sent["steps_taken"] == 2


class TestSufficiency:
    def _layer(self, script):
        return JevDecisionLayer(client=FakeClient(script), api_key="sk-test")

    def test_yes_when_noul_high(self):
        layer = self._layer([FakeResponse({"enough_information": _noul_answer(0.92)})])
        decision = layer.decide_sufficiency(DecisionContext(user_query="q"))
        assert decision.enough is True
        assert decision.probability == 0.92

    def test_no_when_noul_low(self):
        layer = self._layer([FakeResponse({"enough_information": _noul_answer(0.3)})])
        decision = layer.decide_sufficiency(DecisionContext(user_query="q"))
        assert decision.enough is False
        assert decision.probability == 0.3


class TestDescribeDecisionLayer:
    def test_describes_jev_layer(self):
        layer = JevDecisionLayer(client=object(), api_key="sk-test", model="jev-1.13.0")
        label = describe_decision_layer(layer)
        assert "Jev" in label
        assert "jev-1.13.0" in label

    def test_describes_fallback_layer(self, monkeypatch):
        monkeypatch.delenv("TYPESAFE_API_KEY", raising=False)
        layer = create_decision_layer()
        assert isinstance(layer, PolicyDecisionLayer)
        assert "fallback" in describe_decision_layer(layer)

    def test_create_decision_layer_uses_jev_when_key_present(self, monkeypatch):
        monkeypatch.setenv("TYPESAFE_API_KEY", "sk-test")
        assert isinstance(create_decision_layer(), JevDecisionLayer)

    def test_create_decision_layer_falls_back_without_key(self, monkeypatch):
        monkeypatch.delenv("TYPESAFE_API_KEY", raising=False)
        assert isinstance(create_decision_layer(), PolicyDecisionLayer)


class TestUnavailability:
    def test_missing_key_raises(self, monkeypatch):
        monkeypatch.delenv("TYPESAFE_API_KEY", raising=False)
        with pytest.raises(JevUnavailableError, match="TYPESAFE_API_KEY"):
            JevDecisionLayer(api_key=None)

    def test_missing_key_explicit_none_string_is_a_key(self, monkeypatch):
        # An empty-string key is *not* treated as configured.
        monkeypatch.delenv("TYPESAFE_API_KEY", raising=False)
        with pytest.raises(JevUnavailableError, match="TYPESAFE_API_KEY"):
            JevDecisionLayer(api_key="")

    def test_from_env_without_key_returns_fallback(self, monkeypatch):
        monkeypatch.delenv("TYPESAFE_API_KEY", raising=False)
        layer = JevDecisionLayer.from_env()
        assert isinstance(layer, PolicyDecisionLayer)

    def test_exception_falls_back_to_policy(self):
        class Boom:
            def system_one(self, state, questions):
                raise TimeoutError("jev timed out")

        layer = JevDecisionLayer(client=Boom(), api_key="sk-test")
        selection = layer.select_action(DecisionContext(user_query="q", steps_taken=0))
        assert selection is not None
        assert isinstance(selection, ActionSelection)

        decision = layer.decide_sufficiency(DecisionContext(user_query="q", steps_taken=0))
        assert isinstance(decision, SufficiencyDecision)