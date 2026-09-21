"""Jev decision layer.

Jev (TypeSafe) is the agent's decision maker: it selects the next action and
decides when research is complete. It never executes tools and never writes
the final answer — it returns typed, probabilistic decisions that the harness
turns into graph routes.

The layer degrades gracefully: if the ``TYPESAFE_API_KEY`` is missing or the
API call fails, callers can fall back to :class:`PolicyDecisionLayer`.
"""

from __future__ import annotations

import logging
import os
from typing import Dict

from app.decision.policy import (
    DecisionLayer,
    PolicyDecisionLayer,
    create_action_policy,
)
from app.decision.schemas import (
    ActionSelection,
    DecisionContext,
    ResearchAction,
    SufficiencyDecision,
)

logger = logging.getLogger(__name__)


class JevUnavailableError(RuntimeError):
    """Raised when Jev cannot be reached; callers should fall back to policy."""


ACTION_CRITERIA: Dict[str, str] = {
    action.value: action.label for action in ResearchAction
}

ACTION_INSTRUCTIONS = (
    "Given the research request and the findings gathered so far, which action "
    "should the agent take next?"
)

SUFFICIENCY_INSTRUCTIONS = (
    "The agent has gathered enough relevant, authoritative information to "
    "answer the user's research question and can stop researching. Return true "
    "only if the findings genuinely support a well-grounded final answer."
)


class JevDecisionLayer:
    """Make decisions with TypeSafe Jev, one system_one call per decision."""

    def __init__(
        self,
        client=None,
        *,
        model: str | None = None,
        api_key: str | None = None,
        fallback: DecisionLayer | None = None,
        policy=None,
    ) -> None:
        """
        Args:
            client: optional pre-built ``TypeSafeClient`` (injecting a fake in tests).
            model: Jev model id/alias (default ``TYPESAFE_MODEL`` env or ``jev-latest``).
            api_key: optional key override (default ``TYPESAFE_API_KEY`` env).
            fallback: decision layer used when Jev is unreachable.
            policy: optional ActionPolicy for gating (falls back to env defaults).
        """
        self.model = model or os.getenv("TYPESAFE_MODEL", "jev-latest")
        api_key = api_key if api_key is not None else os.getenv("TYPESAFE_API_KEY")
        self._policy = policy or create_action_policy()
        self._fallback = fallback or PolicyDecisionLayer(self._policy)
        if client is not None:
            self._client = client
        elif api_key:
            from typesafe_sdk import TypeSafeClient

            self._client = TypeSafeClient(model=self.model, api_key=api_key)
        else:
            raise JevUnavailableError(
                "TYPESAFE_API_KEY is not set; falling back to the policy decision layer."
            )

    @classmethod
    def from_env(cls, *args, **kwargs) -> DecisionLayer:
        """Build a Jev layer, or the policy fallback when no API key is set."""
        try:
            return cls(*args, **kwargs)
        except JevUnavailableError as exc:
            logger.warning("%s", exc)
            return PolicyDecisionLayer(kwargs.get("policy") or create_action_policy())

    def select_action(self, context: DecisionContext) -> ActionSelection:
        try:
            result = self._call_actions(context)
        except Exception as exc:  # network, rate limit, auth, parse ...
            logger.warning("Jev select_action failed: %s", exc)
            return self._fallback.select_action(context)
        answer = result.answers["action"]
        selection = ActionSelection(
            decision=ResearchAction(answer.choice),
            probabilities=dict(answer.probabilities),
            confidence=float(answer.confidence),
        )
        resolved = self._policy.resolve(selection, context)
        logger.info(
            "Jev action=%s confidence=%.3f probabilities=%s",
            resolved.value,
            selection.confidence,
            selection.probabilities,
        )
        return selection.model_copy(update={"decision": resolved})

    def decide_sufficiency(self, context: DecisionContext) -> SufficiencyDecision:
        try:
            result = self._call_sufficiency(context)
        except Exception as exc:
            logger.warning("Jev decide_sufficiency failed: %s", exc)
            return self._fallback.decide_sufficiency(context)
        answer = result.answers["enough_information"]
        decision = SufficiencyDecision(
            enough=bool(answer.noul >= 0.5), probability=float(answer.noul)
        )
        logger.info(
            "Jev sufficiency enough=%s p=%.3f", decision.enough, decision.probability
        )
        return decision

    def _call_actions(self, context: DecisionContext):
        from typesafe_sdk import Choice

        return self._client.system_one(
            state=context.to_dict(),
            questions={
                "action": Choice(
                    instructions=ACTION_INSTRUCTIONS,
                    criteria=ACTION_CRITERIA,
                )
            },
        )

    def _call_sufficiency(self, context: DecisionContext):
        from typesafe_sdk import Noul

        return self._client.system_one(
            state=context.to_dict(),
            questions={
                "enough_information": Noul(instructions=SUFFICIENCY_INSTRUCTIONS)
            },
        )


def create_decision_layer(policy=None) -> DecisionLayer:
    """Build the Jev decision layer, or the policy fallback when no key is set."""
    return JevDecisionLayer.from_env(policy=policy)


def describe_decision_layer(layer: DecisionLayer) -> str:
    """Human-readable label for logging which brain is deciding."""
    if isinstance(layer, JevDecisionLayer):
        return f"Jev (TypeSafe, model={layer.model})"
    return f"{type(layer).__name__} (TYPESAFE_API_KEY not set — deterministic fallback)"


__all__ = [
    "JevDecisionLayer",
    "JevUnavailableError",
    "create_decision_layer",
    "describe_decision_layer",
    "ACTION_CRITERIA",
    "ACTION_INSTRUCTIONS",
    "SUFFICIENCY_INSTRUCTIONS",
]