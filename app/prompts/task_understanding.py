"""Task-understanding prompt.

The general LLM reads the user's research request and produces a lightweight
task understanding, a research plan, and candidate search queries. None of this
controls the execution loop — Jev picks the actual action afterwards.

The current date is injected so time-sensitive queries use the real present
instead of a year the model remembers from training.
"""

from __future__ import annotations

import json
import re
from datetime import date

from app.prompts.planning import QUERY_GUIDANCE, PLAN_GUIDANCE

SYSTEM_TEMPLATE = (
    "You are the reasoning and planning component of a research agent. You "
    "understand tasks and plan research, but you never decide which tool runs "
    "and you never write the final answer by yourself. A separate decision "
    "system chooses the actions; the harness executes them.\n\n"
    "Today's date is {today} (year {year}). Treat this as the present; your "
    "internal knowledge may be older than it.\n\n"
    "Your job is to produce, in JSON only:\n"
    "1. `understanding` — a concise statement of what the user is asking and "
    "what information is needed to answer it well.\n"
    "2. `plan` — a lightweight research plan.\n"
    "3. `web_queries` — 1-3 search queries for the general web.\n"
    "4. `news_queries` — 1-3 search queries for recent news (may be empty if "
    "the question is not time-sensitive).\n\n"
    f"{QUERY_GUIDANCE}\n\n{PLAN_GUIDANCE}\n\n"
    "Respond with a single JSON object only. No markdown, no prose outside the JSON."
)

USER_TEMPLATE = "Research request:\n{query}"


def build_system_prompt(today: date | None = None) -> str:
    today = today or date.today()
    return SYSTEM_TEMPLATE.format(
        today=today.isoformat(),
        year=today.year,
    )


def build_tasks(user_query: str, today: date | None = None) -> list[tuple[str, str]]:
    """Return LangChain-compatible prompt messages for the understanding node."""
    return [
        ("system", build_system_prompt(today)),
        ("user", USER_TEMPLATE.format(query=user_query)),
    ]


def parse_task_output(text: str) -> dict:
    """Parse the planner's JSON response into a dict with safe defaults."""
    cleaned = re.sub(r"^```(?:json)?\s*|\s*```$", "", text.strip(), flags=re.MULTILINE)
    try:
        data = json.loads(cleaned)
    except json.JSONDecodeError:
        # Try to salvage the JSON object embedded in otherwise-noisy text.
        start, end = cleaned.find("{"), cleaned.rfind("}")
        if start != -1 and end != -1:
            data = json.loads(cleaned[start : end + 1])
        else:
            data = {}
    if not isinstance(data, dict):
        data = {}
    return {
        "understanding": str(data.get("understanding", "")).strip(),
        "plan": str(data.get("plan", "")).strip(),
        "web_queries": _string_list(data.get("web_queries", [])),
        "news_queries": _string_list(data.get("news_queries", [])),
    }


def _string_list(value) -> list[str]:
    if isinstance(value, str):
        return [value.strip()]
    if isinstance(value, list):
        return [str(item).strip() for item in value if str(item).strip()]
    return []