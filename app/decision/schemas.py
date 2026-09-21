"""Typed schema for the decisions Jev can make.

Jev answers typed, probabilistic questions. These Pydantic models are the
contract between the Jev decision layer, the policy layer and the graph, so
downstream code never has to touch the raw SDK response.
"""

from __future__ import annotations

from datetime import date
from enum import Enum
from typing import Dict

from pydantic import BaseModel, Field


class ResearchAction(str, Enum):
    """Actions the agent may take on each research loop.

    Jev selects among these; the harness executes; the LLM never owns the loop.
    """

    SEARCH_WEB = "search_web"
    SEARCH_NEWS = "search_news"
    FINISH = "finish"

    @property
    def label(self) -> str:
        return {
            "search_web": "Search the general web for broad, current information.",
            "search_news": "Search recent news articles for time-sensitive developments.",
            "finish": "Enough information has been gathered; stop and write the answer.",
        }[self.value]


class ActionSelection(BaseModel):
    """Jev's choice of the next action, with the full probability distribution."""

    decision: ResearchAction
    probabilities: Dict[str, float] = Field(
        default_factory=dict, description="Probability per action option."
    )
    confidence: float = Field(0.0, ge=0.0, le=1.0)

    @property
    def is_finish(self) -> bool:
        return self.decision is ResearchAction.FINISH


class SufficiencyDecision(BaseModel):
    """Jev's judgment of whether enough information has been gathered."""

    enough: bool = Field(..., description="True when research may stop.")
    probability: float = Field(0.0, ge=0.0, le=1.0, description="P(enough).")


class DecisionContext(BaseModel):
    """Everything Jev needs to make a decision about the current state."""

    user_query: str
    understanding: str = ""
    steps_taken: int = 0
    max_steps: int = 4
    recent_observations: str = ""
    attempted: list[str] = Field(default_factory=list)
    today: str = ""

    def to_dict(self) -> Dict[str, object]:
        return {
            "query": self.user_query,
            "today": self.today or date.today().isoformat(),
            "understanding": self.understanding,
            "steps_taken": self.steps_taken,
            "max_steps": self.max_steps,
            "attempted": self.attempted,
            "observations": self.recent_observations,
        }

    def to_prompt(self) -> str:
        attempted = ", ".join(self.attempted) if self.attempted else "none yet"
        return (
            f"Today's date: {self.today or date.today().isoformat()}\n"
            f"User request: {self.user_query}\n"
            f"Task understanding: {self.understanding or 'not yet established'}\n"
            f"Research steps taken so far: {self.steps_taken} / {self.max_steps}\n"
            f"Actions attempted: {attempted}\n"
            "Findings so far:\n"
            f"{self.recent_observations or 'none'}"
        )


__all__ = ["ResearchAction", "ActionSelection", "SufficiencyDecision", "DecisionContext"]