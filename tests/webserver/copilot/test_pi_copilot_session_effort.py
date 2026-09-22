"""Per-session effort level and the session store's effort migration.

Moved out of test_pi_copilot_contract.py, which is past its large-module
baseline.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from easyicu.webserver.pi_copilot import service as service_module
from easyicu.webserver.pi_copilot.contracts import PiCopilotError
from easyicu.webserver.pi_copilot.service import PiCopilotService
from tests.webserver.copilot.pi_copilot_contract_fixtures import (
    FakeGateway,
    study_state as study_state,
)


def test_session_store_reads_pre_effort_menu_sessions_at_the_default_level(
    tmp_path: Path,
) -> None:
    """Conversations from before the effort menu continue at the default.

    Every session was created at the forced "off" level before the menu
    existed, so "off" in a /1 store records the absence of a choice: it is
    read (and reopened) at the menu default instead of surfacing as an
    "unspecified" state the researcher has to switch by hand. The first
    write persists the result under the current store schema, after which
    "off" is a real state (a model clamp or an explicit request) and stays.
    """
    store_path = tmp_path / "sessions.json"
    store_path.write_text(
        json.dumps(
            {
                "schema_version": "easyicu.pi-copilot-store/1",
                "sessions": [
                    {
                        "session_id": "pi-before-menu",
                        "project_id": "project-a",
                        "title": "Created at the forced level",
                        "thinking_level": "off",
                    },
                    {
                        "session_id": "pi-before-level-field",
                        "project_id": "project-a",
                        "title": "Written before the field existed",
                    },
                    {
                        "session_id": "pi-chosen-low",
                        "project_id": "project-a",
                        "title": "Chosen through the menu before the schema bump",
                        "thinking_level": "low",
                    },
                ],
            }
        ),
        encoding="utf-8",
    )
    gateway = FakeGateway()
    service = PiCopilotService(store_path=store_path, gateway=gateway)

    records = service._read_records()
    assert {row.session_id: row.thinking_level for row in records} == {
        "pi-before-menu": "medium",
        "pi-before-level-field": "medium",
        "pi-chosen-low": "low",
    }

    # A reopen after the sidecar forgot a pre-menu session carries medium, so
    # its next reply is requested at the default rather than with no level.
    original_request = gateway.request

    def forgetful(method, params, **kwargs):
        if method == "session.state":
            raise PiCopilotError("pi_session_not_open", "gone", status_code=409)
        return original_request(method, params, **kwargs)

    gateway.request = forgetful  # type: ignore[assignment]
    service._ensure_open(records[0])
    gateway.request = original_request  # type: ignore[assignment]
    reopened = [p for m, p, _ in gateway.calls if m == "session.create"]
    assert reopened and reopened[-1]["session_id"] == "pi-before-menu"
    assert reopened[-1]["thinking_level"] == "medium"

    # The first write persists the migrated levels under the current schema.
    service._save_record(records[2])
    raw = json.loads(store_path.read_text(encoding="utf-8"))
    assert raw["schema_version"] == service_module.STORE_SCHEMA_VERSION
    assert raw["schema_version"] != "easyicu.pi-copilot-store/1"
    assert {row["session_id"]: row["thinking_level"] for row in raw["sessions"]} == {
        "pi-before-menu": "medium",
        "pi-before-level-field": "medium",
        "pi-chosen-low": "low",
    }

    # In a current-schema store "off" is an explicit state and is preserved.
    raw["sessions"] = [
        {**row, "thinking_level": "off"} if row["session_id"] == "pi-chosen-low" else row
        for row in raw["sessions"]
    ]
    store_path.write_text(json.dumps(raw), encoding="utf-8")
    assert service._get_record("pi-chosen-low").thinking_level == "off"


def test_effort_level_is_a_per_session_choice_that_defaults_to_medium(
    tmp_path: Path,
    study_state: dict[str, Any],
) -> None:
    """The thinking level is requested per session and changed between turns.

    It used to be forced to "off" on both the service and the bridge because
    raw provider reasoning must not cross; the bridge now projects only a
    bounded, sanitized reasoning summary, so the level is the researcher's
    choice: medium by default, any accepted level on request, reopened with
    the session's own level, and switchable through the bridge with the
    effective (model-clamped) level written back to the record.
    """
    gateway = FakeGateway()
    service = PiCopilotService(store_path=tmp_path / "sessions.json", gateway=gateway)

    defaulted = service.create_session(
        project_id="project-effort", external_llm_opt_in=True
    )["session"]
    assert defaulted["thinking_level"] == "medium"
    created_params = next(p for m, p, _ in gateway.calls if m == "session.create")
    assert created_params["thinking_level"] == "medium"

    requested = service.create_session(
        project_id="project-effort", external_llm_opt_in=True, thinking_level="high"
    )["session"]
    assert requested["thinking_level"] == "high"
    unknown = service.create_session(
        project_id="project-effort", external_llm_opt_in=True, thinking_level="max"
    )["session"]
    assert unknown["thinking_level"] == "medium"

    changed = service.set_thinking_level(
        requested["session_id"], project_id="project-effort", thinking_level="low"
    )
    assert changed["ok"] is True
    assert changed["thinking_level"] == "low"
    assert changed["session"]["thinking_level"] == "low"
    set_call = next(
        (m, p) for m, p, _ in gateway.calls if m == "session.set_thinking_level"
    )
    assert set_call[1] == {"session_id": requested["session_id"], "thinking_level": "low"}
    assert service._get_record(requested["session_id"]).thinking_level == "low"

    # A reopen after the sidecar forgot the session carries the record's level.
    record = service._get_record(requested["session_id"])
    original_request = gateway.request

    def forgetful(method, params, **kwargs):
        if method == "session.state":
            raise PiCopilotError("pi_session_not_open", "gone", status_code=409)
        return original_request(method, params, **kwargs)

    gateway.request = forgetful  # type: ignore[assignment]
    service._ensure_open(record)
    reopened = [p for m, p, _ in gateway.calls if m == "session.create" and p.get("session_file")]
    assert reopened and reopened[-1]["thinking_level"] == "low"
    gateway.request = original_request  # type: ignore[assignment]

    with pytest.raises(PiCopilotError) as invalid:
        service.set_thinking_level(
            requested["session_id"], project_id="project-effort", thinking_level="max"
        )
    assert invalid.value.code == "pi_thinking_level_invalid"
