"""Final-answer prompt.

The general LLM synthesizes the accumulated observations into a user-facing
answer. This is the only place text is produced for the user; Jev decided when
the agent had enough information to get here.
"""

from __future__ import annotations

from datetime import date

SYSTEM_PROMPT = (
    "You are the answer-writing component of a research agent. You have been "
    "given the user's research request and the raw findings gathered during "
    "research. Write a concise, well-organized final answer.\n\n"
    "Ground every claim in the provided findings. If a finding is missing or "
    "contradictory, say so explicitly instead of inventing details. Cite the "
    "source title or URL for key claims. Prefer short paragraphs and a clear "
    "summary at the top. Use the current date given in the request to judge "
    "whether findings are recent; never assert a stale year as the present.\n\n"
    "If the findings are empty or irrelevant, state that the research did not "
    "produce usable results and suggest what to try next."
)

USER_TEMPLATE = (
    "Current date: {today}\n\n"
    "Research request:\n{query}\n\n"
    "Findings gathered:\n{findings}\n\n"
    "Write the final answer."
)


def build_final_answer_messages(
    user_query: str, findings: str, today: date | None = None
) -> list[tuple[str, str]]:
    today = today or date.today()
    return [
        ("system", SYSTEM_PROMPT),
        (
            "user",
            USER_TEMPLATE.format(
                today=today.isoformat(), query=user_query, findings=findings
            ),
        ),
    ]