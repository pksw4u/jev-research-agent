"""Tests for prompt construction, especially current-date injection."""

from __future__ import annotations

from datetime import date

from app.decision.schemas import DecisionContext
from app.prompts.final_answer import build_final_answer_messages
from app.prompts.task_understanding import build_system_prompt, build_tasks


class TestTaskUnderstandingDate:
    def test_system_prompt_contains_current_year(self):
        prompt = build_system_prompt(today=date(2026, 9, 21))
        assert "2026-09-21" in prompt
        assert "year 2026" in prompt

    def test_system_prompt_forbids_stale_year(self):
        prompt = build_system_prompt(today=date(2026, 9, 21))
        assert "2024" in prompt  # used only as the counter-example to avoid
        assert "do NOT append a year" in prompt

    def test_build_tasks_embeds_date(self):
        messages = build_tasks("latest funding news", today=date(2026, 1, 2))
        system = messages[0][1]
        assert "2026-01-02" in system
        assert messages[1][1].endswith("latest funding news")


class TestFinalAnswerDate:
    def test_messages_include_current_date(self):
        messages = build_final_answer_messages(
            "did Acme raise?", "findings...", today=date(2026, 9, 21)
        )
        assert "2026-09-21" in messages[1][1]


class TestDecisionContextDate:
    def test_to_dict_includes_today(self):
        ctx = DecisionContext(user_query="q")
        assert ctx.to_dict()["today"] == date.today().isoformat()

    def test_explicit_today_is_used(self):
        ctx = DecisionContext(user_query="q", today="2026-09-21")
        assert ctx.to_dict()["today"] == "2026-09-21"
        assert "2026-09-21" in ctx.to_prompt()