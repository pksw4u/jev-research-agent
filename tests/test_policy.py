"""Tests for the policy layer: thresholds, routing, and the policy fallback."""

from __future__ import annotations

from app.decision.policy import (
    ActionPolicy,
    PolicyDecisionLayer,
    StopPolicy,
)
from app.decision.schemas import (
    ActionSelection,
    DecisionContext,
    ResearchAction,
    SufficiencyDecision,
)


def ctx(**kwargs) -> DecisionContext:
    base = dict(user_query="q", max_steps=4)
    base.update(kwargs)
    return DecisionContext(**base)


class TestActionPolicy:
    def test_finish_when_max_steps_reached(self):
        policy = ActionPolicy(confidence_threshold=0.6, max_steps=3)
        selection = ActionSelection(decision=ResearchAction.SEARCH_WEB, confidence=0.9)
        outcome = policy.resolve(selection, ctx(steps_taken=3))
        assert outcome is ResearchAction.FINISH

    def test_honors_finish_decision(self):
        policy = ActionPolicy(confidence_threshold=0.6, max_steps=3)
        selection = ActionSelection(decision=ResearchAction.FINISH, confidence=0.3)
        assert policy.resolve(selection, ctx(steps_taken=0)) is ResearchAction.FINISH

    def test_uses_decision_above_threshold(self):
        policy = ActionPolicy(confidence_threshold=0.6, max_steps=3)
        selection = ActionSelection(decision=ResearchAction.SEARCH_NEWS, confidence=0.8)
        assert policy.resolve(selection, ctx(steps_taken=0)) is ResearchAction.SEARCH_NEWS

    def test_finish_is_honored_even_at_low_confidence(self):
        # An explicit finish pick is a safe exit: the answer node handles
        # sparse findings, so we do not re-route it.
        policy = ActionPolicy(confidence_threshold=0.6, max_steps=3)
        selection = ActionSelection(decision=ResearchAction.FINISH, confidence=0.1)
        assert policy.resolve(selection, ctx(steps_taken=0, attempted=[])) is ResearchAction.FINISH

    def test_low_confidence_skips_exhausted_action(self):
        policy = ActionPolicy(confidence_threshold=0.6, max_steps=4)
        selection = ActionSelection(decision=ResearchAction.SEARCH_WEB, confidence=0.1)
        outcome = policy.resolve(
            selection, ctx(steps_taken=1, attempted=["search_web"])
        )
        assert outcome is ResearchAction.SEARCH_NEWS

    def test_low_confidence_finishes_when_all_searches_exhausted(self):
        policy = ActionPolicy(confidence_threshold=0.6, max_steps=4)
        selection = ActionSelection(decision=ResearchAction.SEARCH_NEWS, confidence=0.1)
        outcome = policy.resolve(
            selection, ctx(steps_taken=2, attempted=["search_web", "search_news"])
        )
        assert outcome is ResearchAction.FINISH


class TestStopPolicy:
    def test_stops_when_high_confidence(self):
        policy = StopPolicy(stop_threshold=0.7, min_searches=1)
        decision = SufficiencyDecision(enough=True, probability=0.9)
        assert policy.should_stop(decision, ctx(steps_taken=1, attempted=["search_web"])) is True

    def test_keeps_going_when_only_slightly_sure(self):
        policy = StopPolicy(stop_threshold=0.7, min_searches=1)
        decision = SufficiencyDecision(enough=True, probability=0.6)
        assert policy.should_stop(decision, ctx(steps_taken=1, attempted=["search_web"])) is False

    def test_does_not_stop_before_any_search_unless_almost_certain(self):
        policy = StopPolicy(stop_threshold=0.7, min_searches=1)
        decision = SufficiencyDecision(enough=True, probability=0.9)
        # 0.9 >= threshold is not enough: we must search at least once first.
        assert policy.should_stop(decision, ctx(steps_taken=0, attempted=[])) is False
        decision_certain = SufficiencyDecision(enough=True, probability=0.98)
        assert policy.should_stop(decision_certain, ctx(steps_taken=0, attempted=[])) is True

    def test_forced_stop_at_max_steps(self):
        policy = StopPolicy(stop_threshold=0.7, min_searches=1)
        decision = SufficiencyDecision(enough=False, probability=0.1)
        assert policy.should_stop(decision, ctx(steps_taken=4, attempted=["search_web"])) is True


class TestPolicyDecisionLayer:
    def test_cycles_through_actions_then_finishes(self):
        layer = PolicyDecisionLayer()
        context = ctx(steps_taken=0, attempted=[])
        assert layer.select_action(context).decision is ResearchAction.SEARCH_WEB

        context = ctx(steps_taken=1, attempted=["search_web"])
        assert layer.select_action(context).decision is ResearchAction.SEARCH_NEWS

        context = ctx(steps_taken=2, attempted=["search_web", "search_news"])
        assert layer.select_action(context).decision is ResearchAction.FINISH

    def test_finishes_at_max_steps(self):
        layer = PolicyDecisionLayer()
        context = ctx(steps_taken=4, attempted=["search_web", "search_news"])
        assert layer.select_action(context).decision is ResearchAction.FINISH

    def test_sufficiency_tracks_max_steps(self):
        layer = PolicyDecisionLayer()
        first = layer.decide_sufficiency(ctx(steps_taken=1))
        assert first.enough is False
        last = layer.decide_sufficiency(ctx(steps_taken=4))
        assert last.enough is True
        assert last.probability == 1.0