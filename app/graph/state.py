"""LangGraph state for the research agent.

``ResearchState`` is the only thing that flows between nodes. Jev's decisions
and the LLM's generated text are written back here, and node logic reads it to
decide the next step.
"""

from __future__ import annotations

import operator
from typing import Annotated, List, Optional, TypedDict

from app.decision.schemas import ResearchAction
from app.tools.search import SearchResult


class ResearchState(TypedDict, total=False):
    """Mutable state flowing through the research graph."""

    user_query: str

    # Planned before any decision: produced once by the LLM planner.
    understanding: str
    plan: str
    web_queries: List[str]
    news_queries: List[str]

    # Execution bookkeeping.
    queries_used: Annotated[List[str], operator.add]
    observations: Annotated[List[SearchResult], operator.add]
    attempted: List[str]
    steps: int

    # Latest decision (written by the decision layer / policy).
    action: ResearchAction
    action_probabilities: dict
    action_confidence: float
    enough_information: bool
    enough_probability: float
    stop_research: bool

    # Output.
    final_answer: str