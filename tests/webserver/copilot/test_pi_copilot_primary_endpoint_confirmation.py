"""A primary outcome answered in endpoint wording confirms the outcome slot.

Protocols call the primary outcome the primary endpoint.  The confirmation
gate recognized only outcome wording, so a user who answered the review's
endpoint question in endpoint words, then clicked the Copilot's own
affirmative choice, was asked the same question again after each answer.
A mention inside a research question still records candidate intent only.
"""

from __future__ import annotations

from typing import Any

import pytest

from easyicu.webserver.pi_copilot import tools as tool_module
from easyicu.webserver.pi_copilot.contracts import (
    AuthorityBinding,
    PiSessionRecord,
    ToolExecutionContext,
)

_PROPOSED = "90 天全因死亡（自入 ICU 起算）"


@pytest.fixture
def update(monkeypatch: pytest.MonkeyPatch):
    current = {
        "id": "study-endpoint-wording",
        "revision": 1,
        "question": "",
        "active_job_id": None,
        "cohort": {},
        "outcome": "",
        "primary_exposure": "",
    }
    writes: list[dict[str, Any]] = []
    monkeypatch.setattr(tool_module, "_bound_context", lambda binding: dict(current))
    monkeypatch.setattr(
        tool_module.study_contexts,
        "upsert_context",
        lambda raw, **_kwargs: writes.append(dict(raw)) or {**raw, "revision": 2},
    )
    session = PiSessionRecord(
        session_id="pi-endpoint-wording",
        binding=AuthorityBinding(
            study_context_id=current["id"], study_revision=current["revision"],
        ),
    )

    def run(user_message: str, outcome: str = _PROPOSED) -> tuple[dict[str, Any], list]:
        writes.clear()
        result = tool_module.execute_tool(
            "easyicu_update_study_context",
            {"outcome": outcome},
            ToolExecutionContext(
                session=session, user_message=user_message, allowed_actions={"configure"},
            ),
        )
        return result, list(writes)

    return run


@pytest.mark.parametrize(
    "user_message",
    [
        "主要终点：自入 ICU 起 90 天内全因死亡；ICU 住院天数作为次要描述。",
        "主要结局：90 天全因死亡",
        "主要终点采用 90 天全因死亡。",
        "请把主要终点改为 90 天全因死亡。",
        "是，确认该主要终点和次要描述结局",
        "确认该主要终点",
        "再次确认：主要终点为 90 天全因死亡；次要描述结局为 ICU 住院天数。",
        "The primary endpoint is 90-day all-cause mortality.",
        "Primary endpoint: 90-day all-cause mortality from ICU admission.",
    ],
)
def test_an_explicit_endpoint_answer_saves_the_primary_outcome(update, user_message):
    result, writes = update(user_message)

    assert result["code"] == "study_context_updated"
    assert writes[-1]["outcome"] == _PROPOSED


@pytest.mark.parametrize(
    "user_message",
    [
        "研究脓毒症患者的器官功能轨迹与 90 天死亡的关系。",
        "主要终点是什么？",
        "是否确认该主要终点？",
        "确认该主要终点？",
        "不确认该主要终点",
        "主要终点：待定",
        "主要终点：？",
        "次要终点：ICU 住院天数",
        "What is the primary endpoint?",
        "Primary endpoint: TBD",
    ],
)
def test_a_mention_or_open_question_still_needs_a_choice(update, user_message):
    result, writes = update(user_message)

    assert result["code"] == "study_primary_outcome_confirmation_required"
    assert writes == []


def test_the_refusal_says_how_the_outcome_is_confirmed(update):
    result, _ = update("研究脓毒症患者的器官功能轨迹与 90 天死亡的关系。")

    summary = result["summary"]
    assert "exact label" in summary
    assert "主要结局/主要终点：X" in summary
    assert "offer that exact label as a choice" in summary


def test_the_users_own_label_confirms_itself(update):
    result, writes = update(
        "我想看 90 天全因死亡（自入 ICU 起算），院内死亡只作描述。",
    )

    assert result["code"] == "study_context_updated"
    assert writes[-1]["outcome"] == _PROPOSED
