"""Mined ideas awaiting a choice hold the conversation's automatic first plan.

The browser holds the plan while ``idea_exploration_open`` is true
(tests/js/guided_pi_idea_exploration_hold.test.js); this owner decides it from
the conversation's own recorded tool results; only an accepted handoff ends it.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from easyicu.webserver.pi_copilot.contracts import PiSessionRecord
from easyicu.webserver.pi_copilot.service import PiCopilotService
from easyicu.webserver.pi_copilot.idea_exploration import exploration_open


def _turn(*codes: str, status: str = "ok") -> dict:
    return {"job_id": "job", "events": [
        {"type": "tool_end", "code": code, "tool_name": "synthetic", "status": status} for code in codes
    ]}


def test_mined_ideas_hold_until_a_handoff_is_accepted() -> None:
    mined = [_turn("easyicu_research_workflow_projected"), _turn("easyicu_idea_mined", "easyicu_literature_search_completed")]
    assert exploration_open(mined) is True
    assert exploration_open([*mined, _turn("easyicu_idea_handoff_accepted")]) is False
    # Mining again after an accepted handoff reopens the exploration.
    assert exploration_open([*mined, _turn("easyicu_idea_handoff_accepted"), _turn("easyicu_idea_mined")]) is True


def test_an_outcome_the_agent_saved_is_not_a_choice() -> None:
    """S1442: the UX walk's agent wrote an outcome while assessing a direction; the hold stays."""
    mined = [_turn("easyicu_idea_mined"), _turn("easyicu_study_setup_saved")]
    assert exploration_open(mined) is True


def test_a_conversation_without_mined_ideas_holds_nothing() -> None:
    # S1442: a session that never entered exploration is unaffected by the hold rule.
    assert exploration_open([]) is False
    assert exploration_open([_turn("easyicu_data_sources_listed"), _turn("easyicu_study_setup_saved")]) is False
    assert exploration_open([_turn("easyicu_idea_mined", status="blocked")]) is False
    assert exploration_open([{"events": "not-a-list"}, "not-a-turn"]) is False


class _Gateway:
    environ = {"EASYICU_PI_PROVIDER": "easyicu-local", "EASYICU_PI_API_KEY": "test-only"}

    def __init__(self, root: Path) -> None:
        self.declared_cwd = root / "workspace"
        self.declared_cwd.mkdir()

    def request(self, method: str, params: dict, **_kwargs) -> dict:
        return {"session_id": params.get("session_id"), "transcript": []}

    def close(self) -> None:
        return None


@pytest.mark.parametrize("agent_mode", ["workspace", "research"])
def test_a_session_without_a_bound_study_loads_and_holds_nothing(tmp_path: Path, agent_mode: str) -> None:
    # A workspace session, or a research session before its study exists,
    # must still load its replay; the flag never asks the study store for "".
    service = PiCopilotService(store_path=tmp_path / "sessions.json", gateway=_Gateway(tmp_path))
    record = PiSessionRecord(session_id="pi-synthetic", project_id="project-synthetic", agent_mode=agent_mode)
    store = service.replay_store
    store.start_turn(session_id="pi-synthetic", project_id="project-synthetic", job_id="message-1", allowed_actions=[])
    store.append_event(session_id="pi-synthetic", project_id="project-synthetic", job_id="message-1", event={
        "type": "tool_end", "at": "2026-11-19T00:00:00Z", "tool_call_id": "call-1",
        "tool_name": "easyicu_mine_ideas", "status": "ok", "code": "easyicu_idea_mined",
    })
    store.finish_turn(session_id="pi-synthetic", project_id="project-synthetic", job_id="message-1", status="done")

    public = service._public_session(record, gateway_state={"transcript": []})

    assert public["conversation_replay"]["turns"]
    assert public["idea_exploration_open"] is False
