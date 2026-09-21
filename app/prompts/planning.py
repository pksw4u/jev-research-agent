"""Planning guidance.

Shared text used by the task-understanding node: how candidate queries and the
research plan should be phrased. Keeping it modular lets the planning step be
re-prompted on its own later without touching the understanding prompt.
"""

PLAN_GUIDANCE = (
    "Plan the research in 2-4 steps. Prefer two overlapping searches over a "
    "single broad one: first locate the subject, then verify details. Remember "
    "that the plan is guidance, not an execution order — actions are chosen "
    "independently by the decision system at each step."
)

QUERY_GUIDANCE = (
    "Queries must be concrete, self-contained search strings (5-10 words). For "
    "news queries, prefer the subject plus a verb, e.g. \"Acme Corp Series B "
    "funding round\". Do not invent facts that the query would assert — phrase "
    "queries as lookups, not claims.\n"
    "Date rules: use the current date given above for anything time-sensitive. "
    "If the user did not name a year, do NOT append a year that you inferred; "
    "use words like \"recent\" or \"latest\", or the current year only when the "
    "user asked about the present. Never assume an earlier year (e.g. 2024) "
    "just because it is what you remember."
)