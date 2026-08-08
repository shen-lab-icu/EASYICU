"""Focused owner and authority tests for the Pi Copilot integration."""

from __future__ import annotations

import json
import threading
import time
from pathlib import Path
from typing import Any

import pytest
from pydantic import ValidationError

from easyicu.webserver import guided_sessions, settings
from easyicu.webserver.pi_copilot.contracts import (
    AuthorityBinding,
    HostTurnGrant,
    PiCopilotError,
    PiSessionRecord,
    ToolExecutionContext,
)
from easyicu.webserver.pi_copilot.projections import (
    ensure_safe_projection,
    project_job,
    project_study_context,
    reject_sensitive_message,
)
from easyicu.webserver.pi_copilot.provider_config import PiProviderConfig
from easyicu.webserver.pi_copilot.service import PiCopilotService
from easyicu.webserver.pi_copilot import service as service_module
from easyicu.webserver.pi_copilot import tools as tool_module


class FakeGateway:
    def __init__(self, session_dir: Path | None = None) -> None:
        self.environ = {
            "EASYICU_PI_PROVIDER": "easyicu-local",
            "EASYICU_PI_API_KEY": "test-only-placeholder",
        }
        self.calls: list[tuple[str, dict[str, Any], Any]] = []
        self.tool_contexts: list[ToolExecutionContext] = []
        self.session_dir = session_dir
        self.applied_config: PiProviderConfig | None = None

    def installation_status(self) -> dict[str, Any]:
        return {
            "node_available": True,
            "node_version": "24.11.0",
            "node_version_supported": True,
            "entrypoint_available": True,
            "dependency_installed": True,
            "lockfile_present": True,
            "runtime_integrity_verified": True,
            "api_key_configured": True,
            "provider_connection_verified": True,
            "base_url_configured": True,
            "provider": "easyicu-local",
            "model": "gpt5.6 luna",
            "api_transport": "openai-completions",
        }

    def request(
        self,
        method: str,
        params: dict[str, Any],
        **kwargs: Any,
    ) -> dict[str, Any]:
        self.calls.append((method, dict(params), kwargs.get("tool_context")))
        if kwargs.get("tool_context") is not None:
            self.tool_contexts.append(kwargs["tool_context"])
        session_id = str(params.get("session_id") or "")
        return {
            "session_id": session_id,
            "pi_session_id": "pi-internal-test",
            "session_file": "/private/test-session.jsonl",
            "model": {"provider": "easyicu-local", "id": "gpt5.6 luna"},
            "thinking_level": params.get("thinking_level") or "medium",
            "message_count": 1 if method == "session.prompt" else 0,
            "streaming": False,
            "enabled_tools": ["easyicu_inspect_context"],
            "transcript": [],
            "aborted": method == "session.abort",
        }

    def close(self) -> None:
        return None

    def apply_provider_config(self, config: PiProviderConfig) -> None:
        self.applied_config = config
        self.environ.update(config.as_environment())


@pytest.fixture
def study_state(monkeypatch: pytest.MonkeyPatch) -> dict[str, Any]:
    current = {
        "id": "study-test",
        "revision": 3,
        "title": "Aggregate ICU study",
        "question": "Is aggregate lactate associated with mortality?",
        "data_source": {"database": "mimiciv", "path": "/private/export"},
        "cohort": {"cohort_size": 140},
        "modules": ["lactate"],
        "outcome": "mortality",
        "time_window": {"hours": 24},
        "confirmations": {"cohort": True},
        "active_job_id": None,
    }
    monkeypatch.setattr(
        settings, "load_settings", lambda: {"ai_enabled": True, "language": "en"}
    )
    monkeypatch.setattr(
        service_module.study_contexts, "get_active_context", lambda: dict(current)
    )
    monkeypatch.setattr(
        service_module.study_contexts,
        "get_context",
        lambda context_id: dict(current) if context_id == current["id"] else None,
    )
    monkeypatch.setattr(
        service_module.study_contexts,
        "upsert_context",
        lambda raw, **kwargs: dict(current),
    )
    monkeypatch.setattr(
        service_module.agent_runs,
        "list_run_history",
        lambda **kwargs: {"runs": []},
    )
    return current


def test_runtime_status_has_local_defaults_without_secret_values(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(settings, "load_settings", lambda: {"ai_enabled": True})
    gateway = FakeGateway()
    service = PiCopilotService(store_path=tmp_path / "sessions.json", gateway=gateway)

    payload = service.runtime_status()

    runtime = payload["runtime"]
    assert runtime["status"] == "ready"
    assert runtime["provider"] == "easyicu-local"
    assert runtime["model"] == "gpt5.6 luna"
    assert runtime["built_in_tools_enabled"] == []
    assert runtime["credential_values_exposed"] is False
    assert "test-only-placeholder" not in json.dumps(payload)


def test_development_metadata_alias_migrates_message_job_field() -> None:
    record = PiSessionRecord.model_validate(
        {
            "session_id": "pi-test",
            "last_job_id": "old-message-job",
        }
    )
    assert record.last_message_job_id == "old-message-job"
    dumped = record.model_dump(mode="json")
    assert dumped["last_message_job_id"] == "old-message-job"
    assert "last_job_id" not in dumped


def test_runtime_status_fails_closed_when_credential_is_missing(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(settings, "load_settings", lambda: {"ai_enabled": True})
    gateway = FakeGateway()
    original = gateway.installation_status
    gateway.installation_status = lambda: {**original(), "api_key_configured": False}
    payload = PiCopilotService(
        store_path=tmp_path / "sessions.json",
        gateway=gateway,
    ).runtime_status()
    assert payload["runtime"]["status"] == "setup_required"
    assert payload["runtime"]["blockers"] == ["api_key_configured"]


def test_runtime_status_fails_closed_until_provider_is_verified(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(settings, "load_settings", lambda: {"ai_enabled": True})
    gateway = FakeGateway()
    original = gateway.installation_status
    gateway.installation_status = lambda: {
        **original(),
        "provider_connection_verified": False,
    }

    payload = PiCopilotService(
        store_path=tmp_path / "sessions.json",
        gateway=gateway,
    ).runtime_status()

    assert payload["runtime"]["status"] == "setup_required"
    assert payload["runtime"]["blockers"] == ["provider_connection_unverified"]


def test_verified_first_use_config_is_applied_before_chat_unlocks(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    current_settings = {"ai_enabled": False, "language": "en"}
    monkeypatch.setattr(settings, "load_settings", lambda: dict(current_settings))

    def update_settings(patch: dict[str, Any]) -> dict[str, Any]:
        current_settings.update(patch)
        return dict(current_settings)

    monkeypatch.setattr(settings, "update_settings", update_settings)

    class RecordingStore:
        def __init__(self) -> None:
            self.calls: list[dict[str, Any]] = []

        def verify_and_save(self, **kwargs: Any):
            self.calls.append(dict(kwargs))
            config = PiProviderConfig(
                provider=kwargs["provider"],
                api_key=kwargs["api_key"],
                base_url=kwargs["base_url"],
                model=kwargs["model"],
                api_transport=kwargs["api_transport"],
            )
            return config, {
                "credential_present": True,
                "connection_verified": True,
                "secrets_returned": False,
            }

    gateway = FakeGateway()
    store = RecordingStore()
    service = PiCopilotService(
        store_path=tmp_path / "sessions.json",
        gateway=gateway,
        provider_store=store,  # type: ignore[arg-type]
    )

    payload = service.configure_provider(
        provider="easyicu-local",
        api_key="private-setup-key",
        base_url="http://127.0.0.1:8317/v1",
        model="gpt5.6 luna",
        api_transport="openai-completions",
        enable_ai=True,
    )

    assert current_settings["ai_enabled"] is True
    assert gateway.applied_config is not None
    assert gateway.applied_config.api_key == "private-setup-key"
    assert payload["runtime"]["status"] == "ready"
    assert payload["secrets_returned"] is False
    assert "private-setup-key" not in json.dumps(payload)


def test_session_requires_both_global_and_per_session_opt_in(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        settings, "load_settings", lambda: {"ai_enabled": False, "language": "en"}
    )
    service = PiCopilotService(
        store_path=tmp_path / "sessions.json", gateway=FakeGateway()
    )
    with pytest.raises(PiCopilotError) as global_block:
        service.create_session(project_id="project-opt-in", external_llm_opt_in=True)
    assert global_block.value.code == "external_llm_opt_in_required"

    monkeypatch.setattr(
        settings, "load_settings", lambda: {"ai_enabled": True, "language": "en"}
    )
    with pytest.raises(PiCopilotError) as turn_block:
        service.create_session(project_id="project-opt-in", external_llm_opt_in=False)
    assert turn_block.value.code == "external_llm_opt_in_required"


def test_session_binding_stales_then_rebinds_explicitly(
    tmp_path: Path,
    study_state: dict[str, Any],
) -> None:
    service = PiCopilotService(
        store_path=tmp_path / "sessions.json", gateway=FakeGateway()
    )
    created = service.create_session(
        project_id="project-binding", external_llm_opt_in=True
    )
    session_id = created["session"]["session_id"]
    assert created["session"]["binding"]["study_revision"] == 3

    study_state["revision"] = 4
    with pytest.raises(PiCopilotError) as stale:
        service.send_message(
            session_id,
            project_id="project-binding",
            message="Explain the current plan.",
        )
    assert stale.value.code == "pi_session_authority_stale"
    assert stale.value.status_code == 409

    rebound = service.rebind_session(session_id, project_id="project-binding")
    assert rebound["rebound"] is True
    assert rebound["session"]["binding"]["study_revision"] == 4
    assert rebound["session"]["stale"]["stale"] is False


def test_session_binding_tracks_the_current_run(
    tmp_path: Path,
    study_state: dict[str, Any],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    service = PiCopilotService(
        store_path=tmp_path / "sessions.json",
        gateway=FakeGateway(),
    )
    created = service.create_session(project_id="project-run", external_llm_opt_in=True)
    session_id = created["session"]["session_id"]
    monkeypatch.setattr(
        service_module.agent_runs,
        "list_run_history",
        lambda **kwargs: {"runs": [{"run_id": "run-new"}]},
    )

    with pytest.raises(PiCopilotError) as caught:
        service.send_message(
            session_id,
            project_id="project-run",
            message="Inspect the current run",
        )

    assert caught.value.code == "pi_session_authority_stale"
    assert caught.value.details["mismatches"]["run_id"] == {
        "session": None,
        "current": "run-new",
    }


def test_session_retention_disposes_and_unlinks_evicted_jsonl(
    tmp_path: Path,
) -> None:
    session_dir = tmp_path / "pi-sessions"
    session_dir.mkdir()
    gateway = FakeGateway(session_dir=session_dir)
    service = PiCopilotService(
        store_path=tmp_path / "sessions.json",
        gateway=gateway,
    )
    records = []
    for index in range(100):
        session_file = session_dir / f"pi-{index}.jsonl"
        session_file.write_text("{}\n", encoding="utf-8")
        records.append(
            PiSessionRecord(
                session_id=f"pi-{index}",
                pi_session_file=str(session_file),
            )
        )
    service._write_records(records)

    service._save_record(PiSessionRecord(session_id="pi-new"))

    assert not (session_dir / "pi-99.jsonl").exists()
    assert any(
        method == "session.dispose" and params["session_id"] == "pi-99"
        for method, params, _ in gateway.calls
    )


def test_busy_session_retirement_is_deferred_until_message_finishes(
    tmp_path: Path,
) -> None:
    session_dir = tmp_path / "pi-sessions"
    session_dir.mkdir()
    session_file = session_dir / "busy.jsonl"
    session_file.write_text("{}\n", encoding="utf-8")
    gateway = FakeGateway(session_dir=session_dir)
    service = PiCopilotService(
        store_path=tmp_path / "sessions.json",
        gateway=gateway,
    )
    record = PiSessionRecord(
        session_id="pi-busy",
        pi_session_file=str(session_file),
    )
    service._busy_sessions.add(record.session_id)

    service._retire_record(record)

    assert session_file.exists()
    assert record.session_id in service._pending_retirements
    service._busy_sessions.clear()
    service._flush_pending_retirement(record.session_id)
    assert not session_file.exists()
    assert record.session_id not in service._pending_retirements


def test_legacy_unbound_session_get_is_pure_until_explicit_initialization(
    tmp_path: Path,
    study_state: dict[str, Any],
) -> None:
    service = PiCopilotService(
        store_path=tmp_path / "sessions.json", gateway=FakeGateway()
    )
    record = PiSessionRecord(
        session_id="pi-unbound",
        project_id="project-unbound",
        external_llm_opt_in=True,
    )
    service._save_record(record)
    session_bytes = service.store_path.read_bytes()

    with pytest.raises(PiCopilotError) as required:
        service.list_sessions(project_id="project-unbound")
    assert required.value.code == "pi_project_initialization_required"
    assert service.store_path.read_bytes() == session_bytes
    assert service.project_store.resolve("project-unbound") is None

    initialized = service.initialize_project(
        project_id="project-unbound",
        title="Existing project",
        confirm_initialization=True,
    )
    assert initialized["migrated_sessions"] == 1
    assert initialized["migration_receipt"]["schema_version"] == (
        "easyicu.project-studycontext-migration/1"
    )

    listed = service.list_sessions(project_id="project-unbound")
    assert listed["sessions"][0]["binding"]["study_context_id"] == study_state["id"]

    opened = service.get_session(
        "pi-unbound",
        project_id="project-unbound",
    )
    assert opened["session"]["binding"]["study_context_id"] == study_state["id"]
    assert opened["session"]["stale"]["stale"] is False
    assert service.project_store.resolve("project-unbound") == study_state["id"]


def test_get_does_not_backfill_missing_project_store_from_session_binding(
    tmp_path: Path,
) -> None:
    service = PiCopilotService(
        store_path=tmp_path / "sessions.json",
        gateway=FakeGateway(),
    )
    service._save_record(
        PiSessionRecord(
            session_id="pi-legacy-bound",
            project_id="project-legacy-bound",
            binding=AuthorityBinding(study_context_id="study-legacy"),
        )
    )
    before = service.store_path.read_bytes()

    with pytest.raises(PiCopilotError) as required:
        service.list_sessions(project_id="project-legacy-bound")

    assert required.value.code == "pi_project_initialization_required"
    assert service.store_path.read_bytes() == before
    assert not service.project_store.path.exists()


def test_existing_guided_project_migrates_exact_study_coordinates(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config_path = tmp_path / "guided.json"
    projects_root = tmp_path / "projects"
    projects_root.mkdir()
    monkeypatch.setattr(guided_sessions, "_CONFIG_PATH", config_path)
    monkeypatch.setattr(guided_sessions, "_CONFIG_DIR", tmp_path)
    monkeypatch.setattr(guided_sessions, "_PROJECTS_ROOT", projects_root)
    created = guided_sessions.create_guided_draft(
        {
            "title": "Existing lactate study",
            "question": "Is lactate associated with hospital mortality?",
            "cohort_hint": "Adult first ICU stay",
            "parent_dir": str(projects_root),
        }
    )
    draft = created["draft"]
    opened = guided_sessions.open_guided_project(
        {
            "project_dir": draft["project_dir"],
            "draft_id": draft["id"],
            "title": draft["title"],
        }
    )
    saved = guided_sessions.execute_guided_action(
        {
            "action": "update_slots",
            "session_id": opened["session"]["id"],
            "slots": {
                "study_design": {
                    "outcome_label": "Hospital mortality",
                    "window": "First 24 hours",
                    "comparator_label": "Lower lactate",
                    "collected": True,
                },
                "extraction": {
                    "cohort": "Adult first ICU stay",
                    "modules": ["lactate", "demographics", "sofa"],
                    "format": "parquet",
                    "registered": True,
                },
                "active_export": {
                    "label": "MIMIC-IV research export",
                    "database": "mimiciv",
                    "path": str(tmp_path / "aggregate-export"),
                },
            },
        }
    )
    assert saved["ok"] is True
    setup = guided_sessions.read_project_study_setup(draft["id"])
    assert setup is not None
    assert setup.missing_required == []

    captured: dict[str, Any] = {}

    def upsert(raw: dict[str, Any], **kwargs: Any) -> dict[str, Any]:
        captured.update(raw)
        return {**raw, "id": "study-migrated", "revision": 1}

    monkeypatch.setattr(service_module.study_contexts, "upsert_context", upsert)
    monkeypatch.setattr(
        service_module.study_contexts,
        "get_context",
        lambda context_id: {**captured, "id": context_id, "revision": 1},
    )
    monkeypatch.setattr(
        service_module.agent_runs,
        "list_run_history",
        lambda **kwargs: {"runs": []},
    )
    service = PiCopilotService(
        store_path=tmp_path / "sessions.json",
        gateway=FakeGateway(),
    )
    initialized = service.initialize_project(
        project_id=draft["id"],
        title=draft["title"],
    )

    assert initialized["migration_receipt"]["status"] == "migrated"
    assert initialized["migration_receipt"]["source_digest"] == setup.source_digest
    assert captured["question"] == "Is lactate associated with hospital mortality?"
    assert captured["outcome"] == "Hospital mortality"
    assert captured["time_window"] == {
        "preset": "First 24 hours",
        "label": "First 24 hours",
    }
    assert captured["modules"] == ["lactate", "demographics", "sofa"]
    assert captured["cohort"] == {
        "preset": "Adult first ICU stay",
        "label": "Adult first ICU stay",
    }


def test_pi_conversations_are_immutably_scoped_to_one_project(
    tmp_path: Path,
) -> None:
    gateway = FakeGateway()
    service = PiCopilotService(
        store_path=tmp_path / "sessions.json",
        gateway=gateway,
    )
    service.project_store.bind("project-alpha", "study-alpha")
    alpha = PiSessionRecord(
        session_id="pi-alpha",
        project_id="project-alpha",
        binding=AuthorityBinding(study_context_id="study-alpha"),
    )
    beta = PiSessionRecord(session_id="pi-beta", project_id="project-beta")
    legacy = PiSessionRecord(session_id="pi-legacy")
    service._write_records([alpha, beta, legacy])

    listed = service.list_sessions(project_id="project-alpha")
    assert [row["session_id"] for row in listed["sessions"]] == ["pi-alpha"]
    assert listed["sessions"][0]["project_id"] == "project-alpha"

    with pytest.raises(PiCopilotError) as mismatch:
        service.get_session("pi-alpha", project_id="project-beta")
    assert mismatch.value.code == "pi_session_project_mismatch"
    assert not gateway.calls

    with pytest.raises(ValidationError):
        alpha.project_id = "project-beta"


def test_project_authority_resolves_context_without_global_active_fallback(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        settings,
        "load_settings",
        lambda: {"ai_enabled": True, "language": "en"},
    )
    contexts: dict[str, dict[str, Any]] = {
        "study-global": {"id": "study-global", "revision": 9}
    }
    created_count = 0

    def create_context(raw: dict[str, Any], **kwargs: Any) -> dict[str, Any]:
        nonlocal created_count
        created_count += 1
        context = {
            "id": f"study-project-{created_count}",
            "revision": 1,
            **raw,
            "active_job_id": None,
        }
        contexts[context["id"]] = context
        return dict(context)

    monkeypatch.setattr(
        service_module.study_contexts,
        "get_active_context",
        lambda: dict(contexts["study-global"]),
    )
    monkeypatch.setattr(
        service_module.study_contexts,
        "get_context",
        lambda context_id: (
            dict(contexts[context_id]) if context_id in contexts else None
        ),
    )
    monkeypatch.setattr(
        service_module.study_contexts,
        "upsert_context",
        create_context,
    )
    monkeypatch.setattr(
        service_module.agent_runs,
        "list_run_history",
        lambda **kwargs: {"runs": []},
    )
    gateway = FakeGateway()
    service = PiCopilotService(
        store_path=tmp_path / "sessions.json",
        gateway=gateway,
    )

    first = service.create_session(
        project_id="project-alpha",
        title="Alpha",
        external_llm_opt_in=True,
    )["session"]
    second = service.create_session(
        project_id="project-alpha",
        title="Alpha second chat",
        external_llm_opt_in=True,
    )["session"]

    assert first["binding"]["study_context_id"] == "study-project-1"
    assert second["binding"]["study_context_id"] == "study-project-1"
    assert first["binding"]["study_context_id"] != "study-global"
    assert created_count == 1

    with pytest.raises(PiCopilotError) as cross_project:
        service.send_message(
            first["session_id"],
            project_id="project-beta",
            message="Inspect context",
        )
    assert cross_project.value.code == "pi_session_project_mismatch"
    assert not any(method == "session.prompt" for method, _, _ in gateway.calls)


def test_new_pi_conversation_requires_a_project_binding(tmp_path: Path) -> None:
    service = PiCopilotService(
        store_path=tmp_path / "sessions.json",
        gateway=FakeGateway(),
    )

    with pytest.raises(PiCopilotError) as missing:
        service.create_session(project_id="", external_llm_opt_in=True)

    assert missing.value.code == "pi_project_binding_required"


def test_message_grants_are_host_held_and_message_job_is_not_scientific(
    tmp_path: Path,
    study_state: dict[str, Any],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    gateway = FakeGateway()
    service = PiCopilotService(store_path=tmp_path / "sessions.json", gateway=gateway)
    session_id = service.create_session(
        project_id="project-grants", external_llm_opt_in=True
    )["session"]["session_id"]

    submitted = service.send_message(
        session_id,
        project_id="project-grants",
        message="Inspect the study status.",
        allowed_actions=["run"],
    )
    deadline = time.monotonic() + 3
    job = None
    while time.monotonic() < deadline:
        job = service_module.jobs.MANAGER.get(submitted["job_id"])
        if job and job.status != "running":
            break
        time.sleep(0.01)
    assert job is not None and job.status == "done"
    assert gateway.tool_contexts[-1].allowed_actions == frozenset({"run"})

    study_state["revision"] = 4
    with pytest.raises(PiCopilotError) as stale_tool:
        tool_module.execute_tool(
            "easyicu_inspect_context",
            {},
            gateway.tool_contexts[-1],
        )
    assert stale_tool.value.code == "pi_session_authority_stale"
    study_state["revision"] = 3

    record = service._get_record(session_id)
    assert record.last_message_job_id == submitted["job_id"]
    assert record.binding.active_job_id is None

    class RecordingManager:
        def __init__(self) -> None:
            self.lookups: list[str] = []

        def get(self, job_id: str) -> None:
            self.lookups.append(job_id)
            return None

    manager = RecordingManager()
    monkeypatch.setattr(tool_module.jobs, "MANAGER", manager)
    monkeypatch.setattr(
        tool_module.agent_runs, "list_run_history", lambda **kwargs: {"runs": []}
    )
    result = tool_module.execute_tool(
        "easyicu_inspect_run",
        {},
        ToolExecutionContext(session=record),
    )
    assert result["code"] == "easyicu_run_not_found"
    assert manager.lookups == []

    with pytest.raises(PiCopilotError) as unrelated_abort:
        service.abort_session(
            session_id,
            project_id="project-grants",
            message_job_id="some-other-job",
        )
    assert unrelated_abort.value.code == "pi_message_job_mismatch"


def test_session_rejects_overlapping_prompts(
    tmp_path: Path,
    study_state: dict[str, Any],
) -> None:
    entered = threading.Event()
    release = threading.Event()

    class BlockingGateway(FakeGateway):
        def request(
            self, method: str, params: dict[str, Any], **kwargs: Any
        ) -> dict[str, Any]:
            if method == "session.prompt":
                entered.set()
                assert release.wait(timeout=3)
            return super().request(method, params, **kwargs)

    service = PiCopilotService(
        store_path=tmp_path / "sessions.json",
        gateway=BlockingGateway(),
    )
    session_id = service.create_session(
        project_id="project-overlap", external_llm_opt_in=True
    )["session"]["session_id"]
    first = service.send_message(
        session_id,
        project_id="project-overlap",
        message="First aggregate question",
    )
    assert entered.wait(timeout=3)
    try:
        with pytest.raises(PiCopilotError) as busy:
            service.send_message(
                session_id,
                project_id="project-overlap",
                message="Second aggregate question",
            )
        assert busy.value.code == "pi_session_busy"
        assert busy.value.status_code == 409
    finally:
        release.set()

    deadline = time.monotonic() + 3
    while time.monotonic() < deadline:
        job = service_module.jobs.MANAGER.get(first["job_id"])
        if job and job.status != "running":
            break
        time.sleep(0.01)
    assert job is not None and job.status == "done"


def test_control_tools_fail_closed_without_owner_contracts() -> None:
    session = PiSessionRecord(
        session_id="pi-test",
        binding=AuthorityBinding(run_id="run-1"),
    )
    no_grant = ToolExecutionContext(session=session)
    run_block = tool_module.execute_tool(
        "easyicu_run", {"run_type": "preflight"}, no_grant
    )
    assert run_block["code"] == "pi_action_authorization_required"

    run_grant = ToolExecutionContext(
        session=session, allowed_actions=frozenset({"run"})
    )
    full_block = tool_module.execute_tool(
        "easyicu_run", {"run_type": "full"}, run_grant
    )
    assert full_block["code"] == "pi_full_run_requires_dedicated_confirmation"

    resume_block = tool_module.execute_tool("easyicu_resume", {}, no_grant)
    assert resume_block["code"] == "scientific_resume_not_supported"
    replan_block = tool_module.execute_tool(
        "easyicu_request_replan",
        {"reason": "The aggregate outcome changed."},
        no_grant,
    )
    assert replan_block["code"] == "scientific_replan_not_supported"


def test_tool_surface_has_no_generic_or_scientific_authority_mutators() -> None:
    assert tool_module.ALLOWED_TOOLS == {
        "easyicu_workspace_status",
        "easyicu_inspect_context",
        "easyicu_inspect_plan",
        "easyicu_inspect_capability",
        "easyicu_inspect_run",
        "easyicu_inspect_step",
        "easyicu_inspect_validation",
        "easyicu_list_artifacts",
        "easyicu_inspect_evidence",
        "easyicu_explain_blocker",
        "easyicu_update_study_context",
        "easyicu_run",
        "easyicu_resume",
        "easyicu_cancel",
        "easyicu_request_replan",
    }
    forbidden = {
        "read",
        "write",
        "edit",
        "bash",
        "easyicu_mutate_plan",
        "easyicu_write_evidence",
        "easyicu_authorize_paper",
    }
    assert forbidden.isdisjoint(tool_module.ALLOWED_TOOLS)


def test_study_setup_requires_one_turn_grant_and_uses_typed_owner(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    current = {
        "id": "study-1",
        "revision": 5,
        "question": "Old aggregate question",
        "active_job_id": None,
    }
    session = PiSessionRecord(
        session_id="pi-test",
        binding=AuthorityBinding(study_context_id="study-1", study_revision=5),
    )
    monkeypatch.setattr(tool_module, "_bound_context", lambda binding: dict(current))
    writes = []

    def upsert(raw, **kwargs):
        writes.append((dict(raw), dict(kwargs)))
        return {
            **current,
            **raw,
            "revision": 6,
            "data_source": None,
            "cohort": raw.get("cohort") or {},
            "modules": raw.get("modules") or [],
            "time_window": raw.get("time_window") or {},
            "confirmations": raw.get("confirmations") or {},
        }

    monkeypatch.setattr(tool_module.study_contexts, "upsert_context", upsert)
    proposal = {
        "question": "Is aggregate lactate associated with mortality?",
        "purpose": "Observational ICU research",
        "cohort": {"age_min": 18, "exclude_readmissions": True},
        "modules": ["lactate", "demographics"],
        "outcome": "hospital mortality",
        "time_window": {"hours": 24, "anchor": "ICU admission"},
        "export_format": "parquet",
        "analysis_goal": "Adjusted association",
        "confirmations": {"guided_configuration_collected": True},
    }

    blocked = tool_module.execute_tool(
        "easyicu_update_study_context",
        proposal,
        ToolExecutionContext(session=session),
    )
    assert blocked["code"] == "pi_action_authorization_required"
    assert writes == []

    saved = tool_module.execute_tool(
        "easyicu_update_study_context",
        proposal,
        ToolExecutionContext(
            session=session,
            allowed_actions=frozenset({"configure"}),
        ),
    )
    assert saved["code"] == "study_context_updated"
    assert saved["details"]["rebind_required"] is True
    assert writes[0][0]["id"] == "study-1"
    assert writes[0][1] == {
        "active": True,
        "expected_revision": 5,
        "require_revision": True,
        "lifecycle_write": False,
    }

    one_grant = ToolExecutionContext(
        session=session,
        allowed_actions=frozenset({"configure"}),
    )
    first = tool_module.execute_tool(
        "easyicu_update_study_context", proposal, one_grant
    )
    assert first["code"] == "study_context_updated"
    with pytest.raises(PiCopilotError) as invalidated:
        tool_module.execute_tool("easyicu_inspect_context", {}, one_grant)
    assert invalidated.value.code == "pi_session_authority_stale"

    grant = HostTurnGrant.from_actions(["configure"])
    assert grant.consume("configure") == "granted"
    assert grant.consume("configure") == "consumed"
    assert grant.consume("run") == "missing"


def test_preflight_delegates_to_the_existing_agent_submission_owner(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    submitted_bodies = []

    def submit(body):
        submitted_bodies.append(dict(body))
        return {
            "job_id": "scientific-job-1",
            "kind": "agent-run",
            "status": "running",
            "study_context_id": "study-1",
            "study_context_revision": 8,
        }

    from easyicu.webserver.routes import agent as agent_route

    monkeypatch.setattr(agent_route, "jobs_agent_run", submit)
    monkeypatch.setattr(
        tool_module,
        "_bound_context",
        lambda binding: {
            "id": "study-1",
            "revision": 7,
            "question": "Aggregate association question",
        },
    )
    context = ToolExecutionContext(
        session=PiSessionRecord(
            session_id="pi-test",
            binding=AuthorityBinding(study_context_id="study-1", study_revision=7),
        ),
        allowed_actions=frozenset({"run"}),
    )

    result = tool_module.execute_tool(
        "easyicu_run",
        {"run_type": "preflight"},
        context,
    )

    assert result["code"] == "easyicu_run_submitted"
    assert submitted_bodies == [
        {
            "study_context_id": "study-1",
            "question": "Aggregate association question",
            "run_type": "preflight",
            "llm_provider": "mock",
            "external_llm_opt_in": False,
        }
    ]


def test_phi_and_projection_boundaries_reject_rows_identifiers_and_paths() -> None:
    with pytest.raises(PiCopilotError, match="row-level"):
        reject_sensitive_message("Please inspect patient_id=12345")
    with pytest.raises(PiCopilotError) as raw_rows:
        ensure_safe_projection({"rows": [{"value": 1}]})
    assert raw_rows.value.code == "pi_projection_blocked"
    with pytest.raises(PiCopilotError) as raw_path:
        ensure_safe_projection({"path": "/private/export"})
    assert raw_path.value.code == "pi_projection_blocked"
    for unsafe_value in (
        "failed reading /Users/researcher/patient_12345/raw.csv",
        "Authorization: Bearer secret-token-value",
        'row fragment: {"subject_id": 12345, "value": 7.1}',
    ):
        with pytest.raises(PiCopilotError) as unsafe_string:
            ensure_safe_projection({"reason": unsafe_value})
        assert unsafe_string.value.code == "pi_projection_blocked"

    projected = project_study_context(
        {
            "id": "study-safe",
            "revision": 1,
            "question": "Aggregate lactate analysis",
            "data_source": {"database": "mimiciv", "path": "/private/export"},
            "cohort": {"cohort_size": 140},
        }
    )
    assert "/private/export" not in json.dumps(projected)
    assert len(projected["data_source"]["path_digest"]) == 32

    projected_job = project_job(
        {
            "id": "job-safe",
            "status": "failed",
            "cancel_reason": "/Users/reviewer/private.csv",
            "events": [
                {
                    "seq": 1,
                    "type": "progress",
                    "label": "patient_id=123",
                    "reason": "/private/raw.csv",
                }
            ],
        }
    )
    encoded_job = json.dumps(projected_job)
    assert "private.csv" not in encoded_job
    assert "patient_id" not in encoded_job
    assert projected_job["progress"][0]["reason_code"] is None


def test_validation_projection_is_owner_specific_and_value_safe(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    context = ToolExecutionContext(session=PiSessionRecord(session_id="pi-safe"))
    monkeypatch.setattr(
        tool_module,
        "_select_run",
        lambda context, requested_run_id=None: {
            "run_id": "run-safe",
            "project_dir": "/private/not-projected",
        },
    )
    monkeypatch.setattr(
        tool_module,
        "_run_review",
        lambda row: {
            "gate": {
                "status": "blocked",
                "reason": "/Users/reviewer/patient_123.csv",
                "checks": [
                    {"id": "numeric_evidence_missing", "passed": False},
                    {"id": "unsafe /private/path", "passed": False},
                ],
                "nested": {"raw": "patient_id=123"},
            },
            "readiness": {
                "status": "blocked",
                "reason": "Bearer hidden-secret",
                "non_human_failures": [
                    "evidence_not_ready",
                    "/private/source.csv",
                ],
            },
        },
    )

    result = tool_module.execute_tool(
        "easyicu_inspect_validation",
        {"run_id": "run-safe"},
        context,
    )

    encoded = json.dumps(result)
    assert "/Users" not in encoded
    assert "/private" not in encoded
    assert "patient_id" not in encoded
    assert "Bearer" not in encoded
    assert result["details"]["gate"]["failed_requirement_codes"] == [
        "numeric_evidence_missing"
    ]


def test_complete_tool_result_sanitizes_summary_and_authority_values(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    context = ToolExecutionContext(session=PiSessionRecord(session_id="pi-safe"))
    monkeypatch.setattr(
        tool_module,
        "_select_run",
        lambda context, requested_run_id=None: {
            "run_id": "/Users/reviewer/private-run",
        },
    )

    with pytest.raises(PiCopilotError) as caught:
        tool_module.execute_tool("easyicu_inspect_run", {}, context)

    assert caught.value.code == "pi_projection_blocked"


def test_unknown_tool_arguments_and_missing_plan_keep_owner_codes(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    context = ToolExecutionContext(session=PiSessionRecord(session_id="pi-test"))
    with pytest.raises(PiCopilotError) as unknown:
        tool_module.execute_tool("easyicu_inspect_context", {"raw": True}, context)
    assert unknown.value.code == "pi_tool_unknown_arguments"

    monkeypatch.setattr(
        tool_module.agent_runs, "list_run_history", lambda **kwargs: {"runs": []}
    )
    missing = tool_module.execute_tool(
        "easyicu_inspect_step",
        {"step_id": "analysis"},
        context,
    )
    assert missing["code"] == "easyicu_plan_not_found"
