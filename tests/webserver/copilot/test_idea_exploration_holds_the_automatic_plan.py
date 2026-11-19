"""Mined ideas awaiting a choice hold the conversation's automatic first plan.

The browser holds the plan while ``idea_exploration_open`` is true
(tests/js/guided_pi_idea_exploration_hold.test.js); this owner decides it from
the conversation's own recorded tool results and the study's outcome.
"""

from __future__ import annotations

from easyicu.webserver.pi_copilot.idea_exploration import exploration_open


def _turn(*codes: str, status: str = "ok") -> dict:
    return {"job_id": "job", "events": [
        {"type": "tool_end", "code": code, "tool_name": "synthetic", "status": status} for code in codes
    ]}


def test_mined_ideas_hold_until_a_handoff_is_accepted_or_an_outcome_is_named() -> None:
    mined = [_turn("easyicu_research_workflow_projected"), _turn("easyicu_idea_mined", "easyicu_literature_search_completed")]
    assert exploration_open(mined, {"question": "Mine ideas", "outcome": ""}) is True
    assert exploration_open([*mined, _turn("easyicu_idea_handoff_accepted")], {"outcome": ""}) is False
    assert exploration_open(mined, {"outcome": "28-day mortality"}) is False
    # Mining again after an accepted handoff reopens the exploration.
    assert exploration_open([*mined, _turn("easyicu_idea_handoff_accepted"), _turn("easyicu_idea_mined")], {}) is True


def test_a_conversation_without_mined_ideas_holds_nothing() -> None:
    assert exploration_open([], None) is False
    assert exploration_open([_turn("easyicu_data_sources_listed")], {"outcome": ""}) is False
    assert exploration_open([_turn("easyicu_idea_mined", status="blocked")], {"outcome": ""}) is False
    assert exploration_open([{"events": "not-a-list"}, "not-a-turn"], {}) is False
