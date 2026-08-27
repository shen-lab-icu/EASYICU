"""Focused persistence and pagination tests for long-lived Pi demonstrations."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import pytest

from easyicu.webserver import study_contexts
from easyicu.webserver.data_package_review import DataPackageReviewSnapshotStore
from easyicu.webserver.pi_copilot.contracts import PiCopilotError, PiSessionRecord
from easyicu.webserver.pi_copilot.projections import project_job, project_pi_replay_event
from easyicu.webserver.pi_copilot.replay_store import PiConversationReplayStore
from easyicu.webserver.pi_copilot.service import PiCopilotService


class _Gateway:
    environ = {
        "EASYICU_PI_PROVIDER": "easyicu-local",
        "EASYICU_PI_API_KEY": "test-only",
    }
    declared_cwd: Path

    def __init__(self, root: Path) -> None:
        self.declared_cwd = root / "workspace"
        self.declared_cwd.mkdir()
        self.calls: list[tuple[str, dict[str, Any]]] = []

    def request(self, method: str, params: dict[str, Any], **_kwargs: Any) -> dict:
        self.calls.append((method, dict(params)))
        return {
            "session_id": params.get("session_id"),
            "transcript": [],
            "transcript_page": {
                "items": [],
                "start": 0,
                "end": 0,
                "total": 0,
                "has_more": False,
                "next_cursor": None,
            },
        }

    def close(self) -> None:
        return None


def _turn(store: PiConversationReplayStore, index: int) -> None:
    job_id = f"message-{index:03d}"
    store.start_turn(
        session_id="pi-demo",
        project_id="project-demo",
        job_id=job_id,
        allowed_actions=["run"] if index % 2 else ["configure"],
    )
    store.append_event(
        session_id="pi-demo",
        project_id="project-demo",
        job_id=job_id,
        event={
            "type": "tool_end",
            "at": f"2026-08-13T00:{index % 60:02d}:00Z",
            "tool_call_id": f"call-{index}",
            "tool_name": "easyicu_inspect_plan",
            "status": "ok",
            "code": "easyicu_plan_projected",
        },
    )
    store.finish_turn(
        session_id="pi-demo",
        project_id="project-demo",
        job_id=job_id,
        status="done",
    )


def _review_snapshot(*, study_id: str, revision: int) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "schema_version": "easyicu.data-package-review/1",
        "study_context_id": study_id,
        "study_context_revision": revision,
        "status": "ready_for_plan",
        "privacy": {"patient_rows_returned": False, "path_values_returned": False},
    }
    payload["review_sha256"] = hashlib.sha256(
        json.dumps(
            payload,
            ensure_ascii=False,
            separators=(",", ":"),
            sort_keys=True,
        ).encode("utf-8")
    ).hexdigest()
    return payload


def test_replay_store_pages_from_latest_without_dropping_earlier_turns(
    tmp_path: Path,
) -> None:
    store = PiConversationReplayStore(tmp_path / "replay")
    for index in range(120):
        _turn(store, index)

    latest = store.snapshot(
        session_id="pi-demo",
        project_id="project-demo",
        limit=10,
    )
    assert [row["job_id"] for row in latest["turns"]] == [
        f"message-{index:03d}" for index in range(110, 120)
    ]
    assert latest["turn_page"]["total"] == 120
    assert latest["turn_page"]["next_cursor"] == "110"

    previous = store.snapshot(
        session_id="pi-demo",
        project_id="project-demo",
        cursor=latest["turn_page"]["next_cursor"],
        limit=10,
    )
    assert [row["job_id"] for row in previous["turns"]] == [
        f"message-{index:03d}" for index in range(100, 110)
    ]


def test_replay_store_hides_superseded_branch_without_deleting_audit_rows(
    tmp_path: Path,
) -> None:
    store = PiConversationReplayStore(tmp_path / "replay")
    for index in range(4):
        _turn(store, index)

    store.supersede_from_turn_index(
        session_id="pi-demo",
        project_id="project-demo",
        turn_index=1,
    )
    _turn(store, 9)

    current = store.snapshot(
        session_id="pi-demo",
        project_id="project-demo",
        limit=10,
    )
    assert [row["job_id"] for row in current["turns"]] == [
        "message-000",
        "message-009",
    ]
    persisted = json.loads(store._path("pi-demo").read_text(encoding="utf-8"))
    assert [row["job_id"] for row in persisted["turns"]] == [
        "message-000",
        "message-001",
        "message-002",
        "message-003",
        "message-009",
    ]
    assert all(row.get("superseded") for row in persisted["turns"][1:4])
    with pytest.raises(PiCopilotError, match="cursor") as invalid:
        store.snapshot(
            session_id="pi-demo",
            project_id="project-demo",
            cursor="121",
        )
    assert invalid.value.code == "pi_replay_cursor_invalid"


def test_replay_event_keeps_only_reopenable_resource_identity_and_digest() -> None:
    digest = "a" * 64
    projected = project_pi_replay_event(
        {
            "type": "tool_end",
            "at": "2026-08-13T00:00:00Z",
            "tool_call_id": "call-1",
            "tool_name": "easyicu_inspect_plan",
            "status": "ok",
            "code": "easyicu_plan_projected",
            "resource": {
                "kind": "research_artifact",
                "run_id": "run_demo",
                "artifact": "agent_plan.json",
                "label": "Plan",
                "media_type": "application/json",
                "sha256": digest,
                "path": "/private/must-not-persist",
            },
        }
    )
    assert projected is not None
    assert projected["resource"] == {
        "kind": "research_artifact",
        "run_id": "run_demo",
        "artifact": "agent_plan.json",
        "label": "Plan",
        "media_type": "application/json",
        "sha256": digest,
    }
    assert "/private" not in json.dumps(projected)


def test_replay_keeps_system_validation_document_as_a_distinct_authority_kind() -> None:
    digest = "c" * 64
    projected = project_pi_replay_event(
        {
            "type": "tool_end",
            "at": "2026-08-15T00:00:00Z",
            "tool_call_id": "call-system-validation",
            "tool_name": "easyicu_inspect_run",
            "status": "ok",
            "code": "easyicu_run_projected",
            "resource": {
                "kind": "system_validation_document",
                "run_id": "run_demo",
                "artifact": "system_validation_report.html",
                "label": "System validation dossier",
                "media_type": "text/html",
                "sha256": digest,
                "path": "/private/must-not-persist",
            },
        }
    )

    assert projected is not None
    assert projected["resource"] == {
        "kind": "system_validation_document",
        "run_id": "run_demo",
        "artifact": "system_validation_report.html",
        "label": "System validation dossier",
        "media_type": "text/html",
        "sha256": digest,
    }


def test_replay_drops_research_documents_without_a_digest() -> None:
    projected = project_pi_replay_event(
        {
            "type": "tool_end",
            "at": "2026-08-15T00:00:00Z",
            "tool_call_id": "call-unbound-document",
            "tool_name": "easyicu_inspect_run",
            "status": "ok",
            "code": "easyicu_run_projected",
            "resource": {
                "kind": "system_validation_document",
                "run_id": "run_demo",
                "artifact": "system_validation_report.html",
                "label": "Unbound dossier",
                "media_type": "text/html",
            },
        }
    )

    assert projected is not None
    assert "resource" not in projected


def test_replay_webpage_resource_keeps_checked_digest() -> None:
    digest = "b" * 64
    projected = project_pi_replay_event(
        {
            "type": "tool_end",
            "at": "2026-08-15T00:00:00Z",
            "tool_call_id": "call-preview",
            "tool_name": "easyicu_preview_project_file",
            "status": "ok",
            "code": "pi_workspace_preview_ready",
            "resource": {
                "kind": "webpage",
                "file": "prototype/index.html",
                "label": "Prototype",
                "media_type": "text/html",
                "checked_sha256": digest,
                "path": "/private/must-not-persist",
            },
        }
    )

    assert projected is not None
    assert projected["resource"] == {
        "kind": "webpage",
        "file": "prototype/index.html",
        "label": "Prototype",
        "media_type": "text/html",
        "checked_sha256": digest,
    }
    assert "/private" not in json.dumps(projected)


@pytest.mark.parametrize("checked_sha256", [None, "", "not-a-digest"])
def test_replay_webpage_resource_without_checked_digest_is_not_reopenable(
    checked_sha256: str | None,
) -> None:
    projected = project_pi_replay_event(
        {
            "type": "tool_end",
            "at": "2026-08-15T00:00:00Z",
            "tool_call_id": "call-preview",
            "tool_name": "easyicu_preview_project_file",
            "status": "ok",
            "code": "pi_workspace_preview_ready",
            "resource": {
                "kind": "webpage",
                "file": "prototype/index.html",
                "label": "Prototype",
                "media_type": "text/html",
                "checked_sha256": checked_sha256,
            },
        }
    )

    assert projected is not None
    assert "resource" not in projected


def test_replay_store_itself_drops_text_deltas_and_private_event_fields(
    tmp_path: Path,
) -> None:
    store = PiConversationReplayStore(tmp_path / "replay")
    store.start_turn(
        session_id="pi-demo",
        project_id="project-demo",
        job_id="message-safe",
        allowed_actions=["run"],
    )
    store.append_event(
        session_id="pi-demo",
        project_id="project-demo",
        job_id="message-safe",
        event={
            "type": "text_delta",
            "at": "2026-08-13T00:00:00Z",
            "delta": "private chain-of-thought",
        },
    )
    store.append_event(
        session_id="pi-demo",
        project_id="project-demo",
        job_id="message-safe",
        event={
            "type": "tool_end",
            "at": "2026-08-13T00:00:01Z",
            "tool_call_id": "call-safe",
            "tool_name": "easyicu_inspect_plan",
            "status": "ok",
            "code": "easyicu_plan_projected",
            "summary": "must not persist free-form output",
            "path": "/private/must-not-persist",
        },
    )

    replay = store.snapshot(session_id="pi-demo", project_id="project-demo")

    assert replay["turns"][0]["events"] == [
        {
            "type": "tool_end",
            "at": "2026-08-13T00:00:01Z",
            "tool_call_id": "call-safe",
            "tool_name": "easyicu_inspect_plan",
            "status": "ok",
            "code": "easyicu_plan_projected",
        }
    ]
    assert "private" not in json.dumps(replay)


def test_archived_child_job_carries_artifact_refs_not_private_result(tmp_path: Path) -> None:
    digest = "b" * 64
    projected = project_job(
        {
            "id": "child-job",
            "kind": "research-agent-pipeline",
            "status": "done",
            "events": [],
            "result": {
                "run_id": "run_demo",
                "project_dir": "/private/run",
                "artifacts": [
                    {
                        "name": "quality_gate.json",
                        "sha256": digest,
                        "size": 42,
                        "path": "/private/run/quality_gate.json",
                    }
                ],
                "gate": {"status": "analysis_only", "reportable": False},
            },
        }
    )
    assert projected["run_id"] == "run_demo"
    assert projected["artifact_refs"][0]["sha256"] == digest
    assert "result" not in projected
    assert "/private" not in json.dumps(projected)


def test_service_forwards_transcript_cursor_and_pages_replay(tmp_path: Path) -> None:
    gateway = _Gateway(tmp_path)
    service = PiCopilotService(
        store_path=tmp_path / "sessions.json",
        gateway=gateway,
    )
    record = PiSessionRecord(
        session_id="pi-demo",
        project_id="project-demo",
    )
    for index in range(3):
        _turn(service.replay_store, index)

    state = service._ensure_open(
        record,
        transcript_cursor="40",
        transcript_limit=25,
    )
    assert state["transcript_page"]["total"] == 0
    assert gateway.calls[-1] == (
        "session.state",
        {
            "session_id": "pi-demo",
            "transcript_limit": 25,
            "transcript_cursor": "40",
        },
    )
    public = service._public_session(
        record,
        gateway_state=state,
        replay_cursor="2",
        replay_limit=1,
    )
    assert [row["job_id"] for row in public["conversation_replay"]["turns"]] == [
        "message-001"
    ]
    assert public["conversation_replay"]["turn_page"]["next_cursor"] == "1"


def test_historical_data_package_link_reads_exact_snapshot_after_study_advances(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    snapshots = DataPackageReviewSnapshotStore(tmp_path / "reviews")
    historical = _review_snapshot(study_id="study-demo", revision=5)
    snapshots.persist(historical)
    service = PiCopilotService(
        store_path=tmp_path / "sessions.json",
        gateway=_Gateway(tmp_path),
        review_snapshot_store=snapshots,
    )
    monkeypatch.setattr(
        service,
        "_assert_project_initialized",
        lambda project_id: str(project_id),
    )
    monkeypatch.setattr(service.project_store, "resolve", lambda _project_id: "study-demo")
    monkeypatch.setattr(
        study_contexts,
        "get_context",
        lambda _study_id: {"id": "study-demo", "revision": 9},
    )

    opened = service.get_data_package_review(
        project_id="project-demo",
        study_revision=5,
        review_sha256=historical["review_sha256"],
    )

    assert opened["payload"]["study_context_revision"] == 5
    assert opened["payload"]["review_sha256"] == historical["review_sha256"]


def test_historical_data_package_link_never_rebuilds_against_new_revision(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    service = PiCopilotService(
        store_path=tmp_path / "sessions.json",
        gateway=_Gateway(tmp_path),
        review_snapshot_store=DataPackageReviewSnapshotStore(tmp_path / "reviews"),
    )
    monkeypatch.setattr(
        service,
        "_assert_project_initialized",
        lambda project_id: str(project_id),
    )
    monkeypatch.setattr(service.project_store, "resolve", lambda _project_id: "study-demo")
    monkeypatch.setattr(
        study_contexts,
        "get_context",
        lambda _study_id: {"id": "study-demo", "revision": 9},
    )

    with pytest.raises(PiCopilotError) as missing:
        service.get_data_package_review(
            project_id="project-demo",
            study_revision=5,
            review_sha256="c" * 64,
        )

    assert missing.value.code == "pi_data_package_review_snapshot_missing"
    assert missing.value.status_code == 404


def test_node_and_browser_owners_use_cursor_pages_without_a_last_100_slice() -> None:
    root = Path(__file__).resolve().parents[1]
    node = (
        root
        / "src/easyicu/webserver/pi_copilot/node_app/src/main.mjs"
    ).read_text(encoding="utf-8")
    replay = (
        root / "src/easyicu/webserver/static/js/screens-guided-pi-replay.js"
    ).read_text(encoding="utf-8")
    index = (root / "src/easyicu/webserver/static/index.html").read_text(
        encoding="utf-8"
    )
    assert "function transcriptPage(messages, cursor, limit = 100, manager)" in node
    assert "session.messages.slice(-100)" not in node
    assert 'new Set(["session_id", "transcript_cursor", "transcript_limit"])' in node
    assert "next_cursor" in replay
    assert "loadPiCopilotSession(sessionId, project" in replay
    assert "screens-guided-pi-replay.js" in index
    guided = (
        root / "src/easyicu/webserver/static/js/screens-guided-pi.js"
    ).read_text(encoding="utf-8")
    assert "resources: Array.isArray(job.artifact_refs) ? job.artifact_refs : []" in guided
