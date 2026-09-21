"""Policy layer: risk tolerance lives in code, not in the model.

Jev returns probabilities, but thresholds are a policy decision. This module
turns a Jev decision into the next graph route, and provides a deterministic
fallback decision layer used when Jev is unavailable (no key, or the call
fails).

- ``ActionPolicy``  applies a confidence gate to action selection.
- ``StopPolicy``    decides when the agent must stop gathering information.
- ``PolicyDecisionLayer`` is a pure-code fallback so the agent degrades
  gracefully instead of crashing when Jev cannot be reached.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Protocol

from app.decision.schemas import (
    ActionSelection,
    DecisionContext,
    ResearchAction,
    SufficiencyDecision,
)


@dataclass(frozen=True)
class ActionPolicy:
    """Turn a Jev action selection into an actual route for the graph."""

    confidence_threshold: float = 0.6
    max_steps: int = 4

    def resolve(self, selection: ActionSelection, context: DecisionContext) -> ResearchAction:
        """Apply the confidence gate and step limit, then return the action."""
        if context.steps_taken >= context.max_steps or context.steps_taken >= self.max_steps:
            return ResearchAction.FINISH

        action = selection.decision
        if action is ResearchAction.FINISH:
            return action

        # Below the gate we do not trust the pick. Fall back to a read-only
        # search action we have not exhausted rather than guessing.
        if selection.confidence < self.confidence_threshold:
            return self._fallback_action(context)

        return action

    def _fallback_action(self, context: DecisionContext) -> ResearchAction:
        attempted = set(context.attempted)
        for candidate in (ResearchAction.SEARCH_WEB, ResearchAction.SEARCH_NEWS):
            if candidate.value not in attempted:
                return candidate
        return ResearchAction.FINISH


@dataclass(frozen=True)
class StopPolicy:
    """Decide when the agent has enough information to stop."""

    stop_threshold: float = 0.7
    min_searches: int = 1

    def should_stop(self, decision: SufficiencyDecision, context: DecisionContext) -> bool:
        if context.steps_taken >= context.max_steps:
            return True
        # Avoid stopping before any search has produced an observation unless
        # Jev is very confident the query is already answerable.
        searches = len(context.attempted)
        if searches < self.min_searches and decision.probability < 0.95:
            return False
        return decision.enough and decision.probability >= self.stop_threshold


class DecisionLayer(Protocol):
    """Anything that can make the agent's two decisions."""

    def select_action(self, context: DecisionContext) -> ActionSelection: ...

    def decide_sufficiency(self, context: DecisionContext) -> SufficiencyDecision: ...


class PolicyDecisionLayer:
    """Deterministic, policy-only decision layer (no model calls).

    Used as the fallback when Jev is unavailable. It works through the action
    set in order and stops once every action has been tried or the step limit
    is reached.
    """

    def __init__(self, policy: ActionPolicy | None = None) -> None:
        self.policy = policy or create_action_policy()

    def select_action(self, context: DecisionContext) -> ActionSelection:
        attempted = set(context.attempted)
        if context.steps_taken >= context.max_steps:
            return ActionSelection(decision=ResearchAction.FINISH, confidence=1.0)
        for candidate in (ResearchAction.SEARCH_WEB, ResearchAction.SEARCH_NEWS):
            if candidate.value not in attempted:
                return ActionSelection(decision=candidate, confidence=1.0)
        return ActionSelection(decision=ResearchAction.FINISH, confidence=1.0)

    def decide_sufficiency(self, context: DecisionContext) -> SufficiencyDecision:
        return SufficiencyDecision(
            enough=context.steps_taken >= context.max_steps,
            probability=1.0 if context.steps_taken >= context.max_steps else 0.0,
        )


def create_action_policy(
    confidence_threshold: float | None = None,
    max_steps: int | None = None,
) -> ActionPolicy:
    return ActionPolicy(
        confidence_threshold=confidence_threshold
        if confidence_threshold is not None
        else _float_env("ACTION_CONFIDENCE_THRESHOLD", 0.6),
        max_steps=max_steps if max_steps is not None else _int_env("MAX_RESEARCH_STEPS", 4),
    )


def create_stop_policy(
    stop_threshold: float | None = None,
    max_steps: int | None = None,
) -> StopPolicy:
    return StopPolicy(
        stop_threshold=stop_threshold
        if stop_threshold is not None
        else _float_env("STOP_CONFIDENCE_THRESHOLD", 0.7),
        min_searches=_int_env("MIN_SEARCHES_BEFORE_STOP", 1),
    )


def _float_env(name: str, default: float) -> float:
    raw = os.getenv(name)
    if raw is None or raw.strip() == "":
        return default
    try:
        return float(raw)
    except ValueError:
        return default


def _int_env(name: str, default: int) -> int:
    raw = os.getenv(name)
    if raw is None or raw.strip() == "":
        return default
    try:
        return int(raw)
    except ValueError:
        return default


__all__ = [
    "ActionPolicy",
    "StopPolicy",
    "DecisionLayer",
    "PolicyDecisionLayer",
    "create_action_policy",
    "create_stop_policy",
]