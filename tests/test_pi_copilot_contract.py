"""Focused owner and authority tests for the Pi Copilot integration."""

from __future__ import annotations

import hashlib
import json
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from easyicu.research_agent.acquisition.patient_grouping import (
    PatientGroupingBinding,
)
from pydantic import ValidationError

from easyicu.webserver import agent_pipeline_runs, guided_sessions, settings
from easyicu.webserver.pi_copilot.contracts import (
    AuthorityBinding,
    HostTurnGrant,
    LEGACY_SESSION_SCHEMA_VERSION,
    PiCopilotError,
    PiProjectBindingHandoffReceipt,
    ResearchProviderBinding,
    SESSION_SCHEMA_VERSION,
    PiSessionDataSourceAuthorization,
    PiSessionRecord,
    ToolExecutionContext,
    WorkspaceMutationLimitError,
)
from easyicu.webserver.pi_copilot.projections import (
    ensure_safe_projection,
    project_job,
    project_pi_replay_event,
    project_run_row,
    project_study_context,
    reject_sensitive_message,
)
from easyicu.webserver.pi_copilot.provider_config import PiProviderConfig
from easyicu.webserver.pi_copilot.gateway import PiGatewayClient
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


def _allow_unrelated_message_test(
    service: PiCopilotService,
    session_id: str,
) -> None:
    """Keep non-consent tests focused on their original owner contract."""

    record = service._get_record(session_id)
    record.data_source_authorization = PiSessionDataSourceAuthorization(
        status="legacy_confirmed",
        confirmation_mode="legacy_session",
    )
    service._save_record(record)


def test_research_provider_binding_is_coherent_and_browser_projection_is_safe() -> None:
    with pytest.raises(ValidationError, match="codex research provider binding"):
        ResearchProviderBinding(
            provider="codex",
            credential_source="codex_user_auth",
            authentication_mode="chatgpt_account",
            model="gpt-5.6-luna",
        )

    binding = ResearchProviderBinding(
        provider="codex",
        credential_source="codex_user_auth",
        authentication_mode="chatgpt_account",
        model="gpt-5.6-luna",
        account_session_sha256="a" * 64,
    )

    assert binding.public_projection() == {
        "schema_version": "easyicu.research-provider-binding/1",
        "provider": "codex",
        "credential_source": "codex_user_auth",
        "authentication_mode": "chatgpt_account",
        "model": "gpt-5.6-luna",
    }
    assert "account_session_sha256" not in binding.public_projection()


def test_session_schema_v2_unifies_the_model_connection_without_migrating_v1() -> None:
    legacy = PiSessionRecord(
        schema_version=LEGACY_SESSION_SCHEMA_VERSION,
        session_id="pi-legacy-models",
    )
    current = PiSessionRecord(session_id="pi-one-model")

    assert legacy.uses_unified_model_connection is False
    assert current.schema_version == SESSION_SCHEMA_VERSION
    assert current.uses_unified_model_connection is True


def test_service_rejects_symlinked_gateway_workspace_root(tmp_path: Path) -> None:
    outside = tmp_path / "outside"
    outside.mkdir()
    declared_workspace = tmp_path / "workspace"
    declared_workspace.symlink_to(outside, target_is_directory=True)
    gateway = PiGatewayClient(
        app_dir=tmp_path / "pi-app",
        session_dir=tmp_path / "sessions",
        cwd=declared_workspace,
        environ={},
    )

    with pytest.raises(PiCopilotError) as caught:
        PiCopilotService(
            store_path=tmp_path / "pi-sessions.json",
            gateway=gateway,
        )

    assert caught.value.code == "pi_workspace_base_root_symlink_blocked"


def test_service_workspace_seal_survives_tool_context_composition(
    tmp_path: Path,
) -> None:
    first = tmp_path / "first"
    second = tmp_path / "second"
    first.mkdir()
    second.mkdir()
    alias = tmp_path / "easyicu-home"
    alias.symlink_to(first, target_is_directory=True)
    gateway = PiGatewayClient(
        app_dir=tmp_path / "pi-app",
        session_dir=alias / "sessions",
        environ={},
    )
    service = PiCopilotService(
        store_path=tmp_path / "pi-sessions.json",
        gateway=gateway,
    )
    context = ToolExecutionContext(
        session=PiSessionRecord(
            session_id="pi-workspace-seal",
            project_id="project-a",
            agent_mode="workspace",
        ),
        workspace=service.workspace,
    )

    alias.unlink()
    alias.symlink_to(second, target_is_directory=True)

    with pytest.raises(PiCopilotError) as caught:
        tool_module.execute_tool("easyicu_list_project_files", {}, context)

    assert caught.value.code == "pi_workspace_base_root_changed"
    assert not (second / "workspace" / "projects").exists()


def test_session_store_migrates_only_retired_specialization_fields(
    tmp_path: Path,
) -> None:
    store_path = tmp_path / "sessions.json"
    store_path.write_text(
        json.dumps(
            {
                "schema_version": "easyicu.pi-copilot-store/1",
                "sessions": [
                    {
                        "session_id": "pi-retired-specialization",
                        "project_id": "project-a",
                        "title": "Retired development-only conversation",
                        "canonical_task_id": "retired-development-task",
                        "canonical_input_sha256": "a" * 64,
                        "canonical_job_id": None,
                    },
                    {
                        "session_id": "pi-ordinary-after-migration",
                        "project_id": "project-a",
                        "title": "Ordinary research conversation",
                        "canonical_task_id": None,
                        "canonical_input_sha256": None,
                        "canonical_job_id": None,
                    },
                ],
            }
        ),
        encoding="utf-8",
    )
    service = PiCopilotService(
        store_path=store_path,
        gateway=FakeGateway(),
    )

    records = service._read_records()

    assert [record.session_id for record in records] == ["pi-ordinary-after-migration"]
    assert records[0].title == "Ordinary research conversation"
    assert set(records[0].model_dump()).isdisjoint(
        {"canonical_task_id", "canonical_input_sha256", "canonical_job_id"}
    )

    raw = json.loads(store_path.read_text(encoding="utf-8"))
    raw["sessions"][1]["unexpected_new_field"] = True
    store_path.write_text(json.dumps(raw), encoding="utf-8")
    with pytest.raises(PiCopilotError) as caught:
        service._read_records()
    assert caught.value.code == "pi_session_store_invalid"


def test_project_workflow_projects_active_job_for_session_timeline(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    service = PiCopilotService(
        store_path=tmp_path / "sessions.json",
        gateway=FakeGateway(),
    )
    service.project_store.bind("project-a", "study-a")
    study = {
        "id": "study-a",
        "revision": 9,
        "question": "A bounded ICU research question",
        "purpose": "Review an existing configured project",
        "data_source": {
            "path": "/private/already-bound-export",
            "label": "Existing export",
            "database": "mimiciv",
        },
        "cohort": {"preset": "adult_icu", "cohort_size": 140},
        "modules": ["vitals", "outcome"],
        "outcome": "In-hospital mortality",
        "time_window": {"hours": 24, "anchor": "ICU admission"},
        "export_format": "parquet",
        "analysis_goal": "Descriptive prognostic association",
        "confirmations": {"study_design_reviewed": True},
        "active_job_id": "job-demo",
    }
    baseline = json.loads(json.dumps(study))
    monkeypatch.setattr(
        service_module.study_contexts,
        "get_context",
        lambda study_id: study,
    )
    monkeypatch.setattr(
        service_module.sources,
        "load_registry",
        lambda: {"active_path": None},
    )
    monkeypatch.setattr(
        service_module.agent_runs,
        "list_run_history",
        lambda **kwargs: {"runs": []},
    )

    class DemoJob:
        def snapshot(self) -> dict[str, Any]:
            return {
                "id": "job-demo",
                "kind": "agent-run",
                "status": "running",
                "events": [
                    {
                        "seq": 1,
                        "type": "progress",
                        "step": "materialize",
                        "current": 1,
                        "total": 4,
                    }
                ],
            }

    monkeypatch.setattr(
        service_module.jobs.MANAGER,
        "get",
        lambda job_id: DemoJob() if job_id == "job-demo" else None,
    )

    payload = service.get_project_workflow(project_id="project-a")

    assert payload["active_job"]["job_id"] == "job-demo"
    assert payload["active_job"]["status"] == "running"
    assert payload["active_job"]["progress"] == [
        {
            "seq": 1,
            "type": "progress",
            "current": 1,
            "total": 4,
            "step": "materialize",
            "reason_code": None,
        }
    ]
    receipt = payload["workflow"]["study_setup_receipt"]
    assert receipt["study_context_id"] == "study-a"
    assert receipt["revision"] == 9
    assert receipt["configuration"]["cohort"] == study["cohort"]
    assert receipt["configuration"]["modules"] == study["modules"]
    assert receipt["configuration"]["confirmations"] == study["confirmations"]
    assert receipt["configuration"]["data_source"]["label"] == "Existing export"
    assert "path" not in receipt["configuration"]["data_source"]
    assert "/private/already-bound-export" not in json.dumps(payload)

    repeated = service.get_project_workflow(project_id="project-a")
    assert repeated["workflow"]["study_setup_receipt"] == receipt
    assert study == baseline


def test_project_workflow_prefers_newer_project_pipeline_run(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    service = PiCopilotService(
        store_path=tmp_path / "sessions.json",
        gateway=FakeGateway(),
    )
    service.project_store.bind("project-a", "study-a")
    study = {
        "id": "study-a",
        "revision": 4,
        "question": "Describe the sealed adult ICU cohort.",
        "data_source": {"path": "/private/export", "database": "mimiciv"},
        "cohort": {"preset": "adult_icu"},
        "modules": ["vitals", "outcome"],
        "outcome": "In-hospital mortality",
        "time_window": {"hours": 24, "anchor": "ICU admission"},
        "export_format": "parquet",
        "analysis_goal": "Descriptive epidemiology",
        "confirmations": {"study_design_reviewed": True},
    }
    monkeypatch.setattr(service_module.study_contexts, "get_context", lambda _: study)
    monkeypatch.setattr(
        service_module.sources,
        "load_registry",
        lambda: {"active_path": None},
    )

    def history(**kwargs: Any) -> dict[str, Any]:
        if kwargs.get("project_root"):
            return {
                "runs": [
                    {
                        "run_id": "run-pipeline-plan",
                        "updated_at_epoch": 20.0,
                        "artifact_names": ["agent_plan.json"],
                    }
                ]
            }
        return {
            "runs": [
                {
                    "run_id": "run-old-preflight",
                    "updated_at_epoch": 10.0,
                    "artifact_names": ["evidence_ledger.json"],
                }
            ]
        }

    selected: list[str] = []
    monkeypatch.setattr(service_module.agent_runs, "list_run_history", history)
    monkeypatch.setattr(
        service_module.agent_pipeline_runs,
        "pending_review",
        lambda run_id: selected.append(str(run_id)) or None,
    )

    payload = service.get_project_workflow(project_id="project-a")

    assert payload["ok"] is True
    assert selected == ["run-pipeline-plan"]


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


def test_new_research_session_allows_planning_but_blocks_data_tools_until_confirmation(
    tmp_path: Path,
    study_state: dict[str, Any],
) -> None:
    gateway = FakeGateway()
    service = PiCopilotService(store_path=tmp_path / "sessions.json", gateway=gateway)
    created = service.create_session(
        project_id="project-data-consent",
        external_llm_opt_in=True,
    )
    session_id = created["session"]["session_id"]
    authorization = created["session"]["data_source_authorization"]

    assert authorization["status"] == "pending"
    assert authorization["reason"] == "project_source_confirmation_required"
    assert authorization["source"]["database"] == "mimiciv"
    assert authorization["source"]["label"] == "MIMIC-IV"
    assert authorization["source"]["reference_release"] == "3.1"
    assert "/private/export" not in json.dumps(authorization)

    submitted = service.send_message(
        session_id,
        project_id="project-data-consent",
        message="Help me refine the research question without reading any data.",
    )
    deadline = time.monotonic() + 3
    job = None
    while time.monotonic() < deadline:
        job = service_module.jobs.MANAGER.get(submitted["job_id"])
        if job and job.status != "running":
            break
        time.sleep(0.01)
    assert job is not None and job.status == "done"
    assert [call[0] for call in gateway.calls] == [
        "session.create",
        "session.state",
        "session.prompt",
    ]
    context = gateway.tool_contexts[-1]
    assert context.session.data_source_authorization.status == "pending"

    listed = tool_module.execute_tool("easyicu_list_data_sources", {}, context)
    assert listed["status"] == "ok"
    blocked = tool_module.execute_tool("easyicu_review_cohort", {}, context)
    assert blocked["status"] == "blocked"
    assert blocked["code"] == "pi_session_data_source_confirmation_required"

    confirmed = service.authorize_data_source(
        session_id,
        project_id="project-data-consent",
        action="reuse_project_source",
    )
    assert confirmed["session"]["data_source_authorization"]["status"] == "confirmed"
    assert (
        confirmed["session"]["data_source_authorization"]["confirmation_mode"]
        == "reuse_project_source"
    )


def test_explicit_prepared_source_choice_confirms_same_session_after_binding(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    study_state: dict[str, Any],
) -> None:
    gateway = FakeGateway()
    source_path = str(study_state["data_source"]["path"])
    monkeypatch.setattr(
        service_module.sources,
        "load_registry",
        lambda: {
            "sources": [
                {
                    "id": "src_prepared",
                    "path": source_path,
                    "database": "mimiciv",
                    "label": "MIMIC-IV v3.1",
                    "ok": True,
                }
            ]
        },
    )
    service = PiCopilotService(store_path=tmp_path / "sessions.json", gateway=gateway)
    created = service.create_session(
        project_id="project-prepared-source-choice",
        external_llm_opt_in=True,
    )

    submitted = service.send_message(
        created["session"]["session_id"],
        project_id="project-prepared-source-choice",
        message="使用 EasyICU 已准备好的完整 **MIMIC-IV v3.1**。",
    )
    deadline = time.monotonic() + 3
    job = None
    while time.monotonic() < deadline:
        job = service_module.jobs.MANAGER.get(submitted["job_id"])
        if job and job.status != "running":
            break
        time.sleep(0.01)

    assert job is not None and job.status == "done"
    session = service.get_session(
        created["session"]["session_id"],
        project_id="project-prepared-source-choice",
    )["session"]
    assert session["data_source_authorization"]["status"] == "confirmed"
    assert session["data_source_authorization"]["confirmation_mode"] == (
        "reuse_project_source"
    )
    assert session["binding"]["study_revision"] == study_state["revision"]


def test_prepared_local_source_choice_unlocks_the_same_provider_turn(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    study_state: dict[str, Any],
) -> None:
    gateway = FakeGateway()
    source_path = str(study_state["data_source"]["path"])
    monkeypatch.setattr(
        service_module.sources,
        "load_registry",
        lambda: {
            "sources": [
                {
                    "id": "src_prepared",
                    "path": source_path,
                    "database": "mimiciv",
                    "label": "MIMIC-IV v3.1",
                    "ok": True,
                }
            ]
        },
    )
    service = PiCopilotService(store_path=tmp_path / "sessions.json", gateway=gateway)
    created = service.create_session(
        project_id="project-prepared-local-same-turn",
        external_llm_opt_in=True,
    )

    submitted = service.send_message(
        created["session"]["session_id"],
        project_id="project-prepared-local-same-turn",
        message="确认使用已准备好的完整本地 MIMIC-IV v3.1，并生成新的分析计划。",
    )
    deadline = time.monotonic() + 3
    job = None
    while time.monotonic() < deadline:
        job = service_module.jobs.MANAGER.get(submitted["job_id"])
        if job and job.status != "running":
            break
        time.sleep(0.01)

    assert job is not None and job.status == "done"
    turn_context = gateway.tool_contexts[-1]
    assert turn_context.session.data_source_authorization.status == "confirmed"
    assert turn_context.session.data_source_authorization.confirmation_mode == (
        "reuse_project_source"
    )


def test_research_question_alone_does_not_confirm_registered_source(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    study_state: dict[str, Any],
) -> None:
    gateway = FakeGateway()
    monkeypatch.setattr(
        service_module.sources,
        "load_registry",
        lambda: {
            "sources": [
                {
                    "id": "src_prepared",
                    "path": study_state["data_source"]["path"],
                    "database": "mimiciv",
                    "ok": True,
                }
            ]
        },
    )
    service = PiCopilotService(store_path=tmp_path / "sessions.json", gateway=gateway)
    created = service.create_session(
        project_id="project-source-mention-only",
        external_llm_opt_in=True,
    )

    submitted = service.send_message(
        created["session"]["session_id"],
        project_id="project-source-mention-only",
        message="我想研究 MIMIC-IV 成人 ICU 人群。",
    )
    deadline = time.monotonic() + 3
    job = None
    while time.monotonic() < deadline:
        job = service_module.jobs.MANAGER.get(submitted["job_id"])
        if job and job.status != "running":
            break
        time.sleep(0.01)

    assert job is not None and job.status == "done"
    session = service.get_session(
        created["session"]["session_id"],
        project_id="project-source-mention-only",
    )["session"]
    assert session["data_source_authorization"]["status"] == "pending"


def test_local_folder_selection_stays_locked_until_study_source_is_saved(
    tmp_path: Path,
    study_state: dict[str, Any],
) -> None:
    study_state["data_source"] = {}
    gateway = FakeGateway()
    service = PiCopilotService(store_path=tmp_path / "sessions.json", gateway=gateway)
    created = service.create_session(
        project_id="project-local-selection",
        external_llm_opt_in=True,
    )
    session_id = created["session"]["session_id"]

    assert created["session"]["data_source_authorization"] == {
        "schema_version": "easyicu.pi-session-data-source-authorization/1",
        "status": "pending",
        "reason": "local_data_selection_required",
        "confirmation_mode": None,
        "source": None,
        "confirmed_at": None,
    }
    with pytest.raises(PiCopilotError) as misplaced_database:
        service.authorize_data_source(
            session_id,
            project_id="project-local-selection",
            action="confirm_selected_source",
            database="miiv",
        )
    assert misplaced_database.value.code == "pi_data_source_database_not_allowed"

    selecting = service.authorize_data_source(
        session_id,
        project_id="project-local-selection",
        action="begin_local_selection",
        database="miiv",
    )
    assert selecting["session"]["data_source_authorization"]["status"] == (
        "selection_in_progress"
    )
    assert selecting["resource"]["kind"] == "native_workspace"
    assert selecting["resource"]["route"] == "extraction"
    assert selecting["resource"]["entry_mode"] == "source_binding"
    assert selecting["resource"]["expected_database"] == "miiv"
    assert selecting["resource"]["label"] == "Data source setup"
    assert "path" not in selecting["resource"]

    with pytest.raises(PiCopilotError) as no_source:
        service.authorize_data_source(
            session_id,
            project_id="project-local-selection",
            action="confirm_selected_source",
        )
    assert no_source.value.code == "pi_session_data_source_unavailable"

    study_state["revision"] = 4
    study_state["data_source"] = {
        "source_id": "src_0123456789ab",
        "label": "MIMIC-IV v3.1",
        "database": "miiv",
        "path": "/private/new-local-source",
    }
    confirmed = service.authorize_data_source(
        session_id,
        project_id="project-local-selection",
        action="confirm_selected_source",
    )
    session = confirmed["session"]
    assert session["binding"]["study_revision"] == 4
    assert session["data_source_authorization"]["status"] == "confirmed"
    assert session["data_source_authorization"]["confirmation_mode"] == (
        "select_local_source"
    )
    assert "/private/new-local-source" not in json.dumps(session)


def test_workspace_session_does_not_require_research_data_confirmation(
    tmp_path: Path,
    study_state: dict[str, Any],
) -> None:
    service = PiCopilotService(
        store_path=tmp_path / "sessions.json",
        gateway=FakeGateway(),
    )
    created = service.create_session(
        project_id="project-workspace-data-gate",
        agent_mode="workspace",
        external_llm_opt_in=True,
    )

    assert created["session"]["data_source_authorization"]["status"] == (
        "not_required"
    )


def test_blank_guided_project_persists_no_fixture_question_or_source(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    projects_root = tmp_path / "projects"
    projects_root.mkdir()
    monkeypatch.setattr(guided_sessions, "_CONFIG_PATH", tmp_path / "guided.json")
    monkeypatch.setattr(guided_sessions, "_CONFIG_DIR", tmp_path)
    monkeypatch.setattr(guided_sessions, "_PROJECTS_ROOT", projects_root)

    created = guided_sessions.create_guided_draft(
        {
            "title": "Truly blank project",
            "data_mode": "unbound",
            "question": "",
            "source": None,
            "parent_dir": str(projects_root),
        }
    )

    assert created["ok"] is True
    assert created["draft"]["data_mode"] == "unbound"
    assert created["draft"]["question"] == ""
    assert created["draft"]["source"] is None
    stored = json.loads(
        (Path(created["draft"]["project_dir"]) / "guided_draft.json").read_text(
            encoding="utf-8"
        )
    )
    assert stored["data_mode"] == "unbound"
    assert stored["question"] == ""
    assert stored["source"] is None


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


def test_new_codex_session_uses_one_account_model_for_chat_and_analysis(
    tmp_path: Path,
    study_state: dict[str, Any],
) -> None:
    base_gateway = FakeGateway()
    account_gateway = FakeGateway()

    class RecordingCodexPool:
        def __init__(self) -> None:
            self.calls: list[tuple[ResearchProviderBinding, bool]] = []

        def gateway_for(
            self,
            binding: ResearchProviderBinding,
            *,
            refresh_account: bool = False,
        ) -> FakeGateway:
            self.calls.append((binding, refresh_account))
            return account_gateway

        def close(self) -> None:
            return None

    pool = RecordingCodexPool()
    service = PiCopilotService(
        store_path=tmp_path / "sessions.json",
        gateway=base_gateway,
        codex_gateway_pool=pool,  # type: ignore[arg-type]
    )
    model_connection = ResearchProviderBinding(
        provider="codex",
        credential_source="codex_user_auth",
        authentication_mode="chatgpt_account",
        model="gpt-5.6-luna",
        account_session_sha256="a" * 64,
    )

    created = service.create_session(
        project_id="project-one-model",
        language="zh",
        external_llm_opt_in=True,
        research_provider=model_connection,
    )

    assert not base_gateway.calls
    assert account_gateway.calls[0][0] == "session.create"
    assert account_gateway.calls[0][1]["language"] == "zh"
    assert pool.calls == [(model_connection, True)]
    assert created["session"]["model_connection"] == (
        model_connection.public_projection()
    )
    assert created["session"]["research_provider"] == (
        model_connection.public_projection()
    )

    record = service._get_record(created["session"]["session_id"])
    legacy = record.model_copy(
        update={"schema_version": LEGACY_SESSION_SCHEMA_VERSION}
    )
    assert service._conversation_gateway(record) is account_gateway
    assert service._conversation_gateway(legacy) is base_gateway


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


def test_session_binding_prefers_newer_pipeline_run_in_project_workspace(
    tmp_path: Path,
    study_state: dict[str, Any],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[str | None] = []

    def history(
        *,
        study_id: str,
        project_root: str | None = None,
        limit: int,
    ) -> dict[str, Any]:
        calls.append(project_root)
        if project_root is not None:
            return {
                "runs": [
                    {
                        "run_id": "run-pipeline-new",
                        "updated_at_epoch": 200.0,
                    }
                ]
            }
        return {
            "runs": [
                {
                    "run_id": "run-native-old",
                    "updated_at_epoch": 100.0,
                }
            ]
        }

    monkeypatch.setattr(service_module.agent_runs, "list_run_history", history)
    service = PiCopilotService(
        store_path=tmp_path / "sessions.json",
        gateway=FakeGateway(),
    )

    created = service.create_session(
        project_id="project-pipeline-run",
        external_llm_opt_in=True,
    )

    assert created["session"]["binding"]["run_id"] == "run-pipeline-new"
    assert calls[0] is None
    assert any(call is not None for call in calls)


def test_implicit_run_inspection_uses_latest_history_not_stale_session_binding(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    context = ToolExecutionContext(
        session=PiSessionRecord(
            session_id="pi-latest-run",
            binding=AuthorityBinding(
                study_context_id="study-latest-run",
                run_id="run-older",
            ),
        )
    )
    monkeypatch.setattr(
        tool_module,
        "_run_rows",
        lambda _context: [
            {"run_id": "run-newest"},
            {"run_id": "run-older"},
        ],
    )

    assert tool_module._select_run(context) == {"run_id": "run-newest"}
    assert tool_module._select_run(context, "run-older") == {"run_id": "run-older"}


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
    exact_export_path = tmp_path / "ICU  研究 (A)" / "aggregate-export"
    saved = guided_sessions.execute_guided_action(
        {
            "action": "update_slots",
            "session_id": opened["session"]["id"],
            "slots": {
                "study_design": {
                    "outcome_label": "Hospital mortality",
                    "window": {"hours": 24, "anchor": "ICU admission"},
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
                    "path": str(exact_export_path),
                },
            },
        }
    )
    assert saved["ok"] is True
    setup = guided_sessions.read_project_study_setup(draft["id"])
    assert setup is not None
    assert setup.missing_required == []
    assert setup.data_source == {
        "path": str(exact_export_path),
        "label": "MIMIC-IV research export",
        "database": "mimiciv",
    }

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
    assert captured["time_window"] == {"hours": 24, "anchor": "ICU admission"}
    assert captured["modules"] == ["lactate", "demographics", "sofa"]
    assert captured["cohort"] == {
        "preset": "Adult first ICU stay",
        "label": "Adult first ICU stay",
    }
    assert captured["data_source"]["path"] == str(exact_export_path)


def test_guided_project_rejects_overlong_exact_path_without_persisting(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config_path = tmp_path / "guided.json"
    monkeypatch.setattr(guided_sessions, "_CONFIG_PATH", config_path)
    monkeypatch.setattr(guided_sessions, "_CONFIG_DIR", tmp_path)
    config_path.write_text(
        json.dumps(
            {
                "schema_version": 2,
                "drafts": [],
                "sessions": [
                    {
                        "id": "guided-overlong-write",
                        "memory_scope": "project_folder",
                        "slots": {},
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    before = config_path.read_bytes()

    result = guided_sessions.execute_guided_action(
        {
            "action": "update_slots",
            "session_id": "guided-overlong-write",
            "slots": {"active_export": {"path": "/" + "x" * 4096}},
        }
    )

    assert result["ok"] is False
    assert result["blocked"] is True
    assert result["error"] == "guided_project_path_too_long"
    assert result["details"] == {
        "field": "active_export.path",
        "max_length": 4096,
    }
    assert result["persisted"] is False
    assert config_path.read_bytes() == before


def test_overlong_guided_path_cannot_create_study_context_or_authority_binding(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config_path = tmp_path / "guided.json"
    monkeypatch.setattr(guided_sessions, "_CONFIG_PATH", config_path)
    monkeypatch.setattr(guided_sessions, "_CONFIG_DIR", tmp_path)
    config_path.write_text(
        json.dumps(
            {
                "schema_version": 2,
                "drafts": [],
                "sessions": [
                    {
                        "id": "guided-overlong-migration",
                        "draft_id": "project-overlong-migration",
                        "project_title": "Overlong path project",
                        "slots": {
                            "question_hint": "Does lactate predict mortality?",
                            "outcome_hint": "Hospital mortality",
                            "time_window_hint": "First 24 hours",
                            "extraction": {
                                "cohort": "Adult first ICU stay",
                                "modules": ["lactate"],
                            },
                            "active_export": {"path": "/" + "x" * 4096},
                        },
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    upserts: list[dict[str, Any]] = []
    monkeypatch.setattr(
        service_module.study_contexts,
        "upsert_context",
        lambda raw, **kwargs: upserts.append(raw),
    )
    service = PiCopilotService(
        store_path=tmp_path / "sessions.json",
        gateway=FakeGateway(),
    )

    with pytest.raises(PiCopilotError) as blocked:
        service.initialize_project(
            project_id="project-overlong-migration",
            title="Overlong path project",
        )

    assert blocked.value.code == "guided_project_path_too_long"
    assert blocked.value.status_code == 409
    assert blocked.value.details == {
        "field": "data_source.path",
        "max_length": 4096,
    }
    assert upserts == []
    assert not service.project_store.path.exists()


@pytest.mark.parametrize(
    ("window_slots", "expected"),
    [
        (
            {"time_window_hint": "First 24 hours"},
            {"preset": "First 24 hours", "label": "First 24 hours"},
        ),
        (
            {"study_params": {"window": {"hours": 24, "anchor": "ICU admission"}}},
            {"hours": 24, "anchor": "ICU admission"},
        ),
    ],
    ids=["guided-v1-text", "guided-v2-typed"],
)
def test_guided_legacy_time_window_schemas_migrate_without_stringifying(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    window_slots: dict[str, Any],
    expected: dict[str, Any],
) -> None:
    config_path = tmp_path / "guided.json"
    monkeypatch.setattr(guided_sessions, "_CONFIG_PATH", config_path)
    monkeypatch.setattr(guided_sessions, "_CONFIG_DIR", tmp_path)
    slots = {
        "question_hint": "Does lactate vary in the first ICU day?",
        "outcome_hint": "Lactate distribution",
        "extraction": {
            "cohort": "Adult first ICU stay",
            "modules": ["lactate"],
        },
        **window_slots,
    }
    config_path.write_text(
        json.dumps(
            {
                "schema_version": 2,
                "drafts": [],
                "sessions": [
                    {
                        "id": "guided-legacy-window",
                        "draft_id": "project-legacy-window",
                        "project_title": "Legacy window project",
                        "slots": slots,
                    }
                ],
            }
        ),
        encoding="utf-8",
    )

    setup = guided_sessions.read_project_study_setup("project-legacy-window")

    assert setup is not None
    assert setup.time_window == expected
    assert setup.missing_required == []


def test_concurrent_project_initialization_creates_one_study_context(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    service = PiCopilotService(
        store_path=tmp_path / "sessions.json",
        gateway=FakeGateway(),
    )
    created: dict[str, dict[str, Any]] = {}
    created_lock = threading.Lock()

    def upsert(raw: dict[str, Any], **kwargs: Any) -> dict[str, Any]:
        with created_lock:
            context_id = f"study-concurrent-{len(created) + 1}"
            context = {**raw, "id": context_id, "revision": 1}
            created[context_id] = context
        time.sleep(0.05)
        return context

    monkeypatch.setattr(service_module.study_contexts, "upsert_context", upsert)
    monkeypatch.setattr(
        service_module.study_contexts,
        "get_context",
        lambda context_id: created.get(context_id),
    )
    monkeypatch.setattr(
        service_module.agent_runs,
        "list_run_history",
        lambda **kwargs: {"runs": []},
    )

    def initialize() -> dict[str, Any]:
        return service.initialize_project(
            project_id="project-concurrent",
            title="Concurrent project",
            confirm_initialization=True,
        )

    with ThreadPoolExecutor(max_workers=2) as executor:
        results = list(executor.map(lambda _: initialize(), range(2)))

    assert len(created) == 1
    assert {row["study_context_id"] for row in results} == {"study-concurrent-1"}
    assert service.project_store.resolve("project-concurrent") == "study-concurrent-1"


def test_agent_handoff_binds_existing_study_context_without_creating_a_new_one(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    service = PiCopilotService(
        store_path=tmp_path / "sessions.json",
        gateway=FakeGateway(),
    )
    study = {
        "id": "study-agent-handoff",
        "revision": 7,
        "title": "Cross-database mortality plan",
    }
    monkeypatch.setattr(
        service_module.study_contexts,
        "get_context",
        lambda context_id: dict(study) if context_id == study["id"] else None,
    )
    monkeypatch.setattr(
        service_module.study_contexts,
        "upsert_context",
        lambda *_args, **_kwargs: pytest.fail(
            "an Agent handoff must not create a second StudyContext"
        ),
    )
    monkeypatch.setattr(
        service_module.agent_runs,
        "list_run_history",
        lambda **_kwargs: {"runs": []},
    )
    receipt = PiProjectBindingHandoffReceipt(
        project_id=study["id"],
        project_title=study["title"],
        study_context_id=study["id"],
        study_context_revision=study["revision"],
    )

    initialized = service.initialize_project(
        project_id=study["id"],
        title=study["title"],
        binding_receipt=receipt,
    )

    assert initialized["study_context_id"] == study["id"]
    assert initialized["study_context_revision"] == 7
    assert initialized["binding_receipt"] == receipt.model_dump(mode="json")
    assert service.project_store.resolve(study["id"]) == study["id"]


def test_agent_handoff_rejects_a_stale_revision_before_binding(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    service = PiCopilotService(
        store_path=tmp_path / "sessions.json",
        gateway=FakeGateway(),
    )
    monkeypatch.setattr(
        service_module.study_contexts,
        "get_context",
        lambda _context_id: {"id": "study-stale", "revision": 8},
    )
    receipt = PiProjectBindingHandoffReceipt(
        project_id="study-stale",
        project_title="Stale handoff",
        study_context_id="study-stale",
        study_context_revision=7,
    )

    with pytest.raises(PiCopilotError) as caught:
        service.initialize_project(
            project_id="study-stale",
            title="Stale handoff",
            binding_receipt=receipt,
        )

    assert caught.value.code == "pi_project_handoff_revision_conflict"
    assert service.project_store.resolve("study-stale") is None


def test_agent_handoff_revision_race_does_not_publish_a_project_binding(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    service = PiCopilotService(
        store_path=tmp_path / "sessions.json",
        gateway=FakeGateway(),
    )
    revisions = iter((7, 8))
    monkeypatch.setattr(
        service_module.study_contexts,
        "get_context",
        lambda _context_id: {"id": "study-race", "revision": next(revisions)},
    )
    receipt = PiProjectBindingHandoffReceipt(
        project_id="study-race",
        project_title="Racing handoff",
        study_context_id="study-race",
        study_context_revision=7,
    )

    with pytest.raises(PiCopilotError) as caught:
        service.initialize_project(
            project_id="study-race",
            title="Racing handoff",
            binding_receipt=receipt,
        )

    assert caught.value.code == "pi_project_handoff_revision_conflict"
    assert service.project_store.resolve("study-race") is None


def test_agent_handoff_cannot_replace_an_existing_project_binding(
    tmp_path: Path,
) -> None:
    service = PiCopilotService(
        store_path=tmp_path / "sessions.json",
        gateway=FakeGateway(),
    )
    service.project_store.bind("project-bound", "study-original")
    receipt = PiProjectBindingHandoffReceipt(
        project_id="project-bound",
        project_title="Bound project",
        study_context_id="study-forged",
        study_context_revision=1,
    )

    with pytest.raises(PiCopilotError) as caught:
        service.initialize_project(
            project_id="project-bound",
            title="Bound project",
            binding_receipt=receipt,
        )

    assert caught.value.code == "pi_project_study_context_mismatch"
    assert service.project_store.resolve("project-bound") == "study-original"


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


def test_pi_conversation_mode_filter_runs_before_the_list_limit(tmp_path: Path) -> None:
    service = PiCopilotService(
        store_path=tmp_path / "sessions.json",
        gateway=FakeGateway(),
    )
    service.project_store.bind("project-alpha", "study-alpha")
    research = [
        PiSessionRecord(
            session_id=f"pi-research-{index}",
            project_id="project-alpha",
            agent_mode="research",
            binding=AuthorityBinding(study_context_id="study-alpha"),
        )
        for index in range(35)
    ]
    workspace = PiSessionRecord(
        session_id="pi-workspace-existing",
        project_id="project-alpha",
        agent_mode="workspace",
        binding=AuthorityBinding(study_context_id="study-alpha"),
    )
    service._write_records([*research, workspace])

    listed = service.list_sessions(
        project_id="project-alpha",
        limit=1,
        agent_mode="workspace",
    )

    assert [row["session_id"] for row in listed["sessions"]] == [
        "pi-workspace-existing"
    ]


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
    _allow_unrelated_message_test(service, session_id)

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
    assert record.last_turn_status == "done"
    assert record.last_turn_allowed_actions == ["run"]
    public = service.get_session(session_id, project_id="project-grants")["session"]
    assert public["active_message_job_id"] is None
    assert public["last_turn_allowed_actions"] == ["run"]

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


def test_current_user_explicit_extraction_confirmation_is_host_granted(
    tmp_path: Path,
) -> None:
    gateway = FakeGateway()
    service = PiCopilotService(store_path=tmp_path / "sessions.json", gateway=gateway)
    session_id = service.create_session(
        project_id="project-explicit-extract", external_llm_opt_in=True
    )["session"]["session_id"]
    _allow_unrelated_message_test(service, session_id)

    submitted = service.send_message(
        session_id,
        project_id="project-explicit-extract",
        message="授权下载并准备官方 MIMIC-IV demo。",
    )
    deadline = time.monotonic() + 3
    job = None
    while time.monotonic() < deadline:
        job = service_module.jobs.MANAGER.get(submitted["job_id"])
        if job and job.status != "running":
            break
        time.sleep(0.01)

    assert job is not None and job.status == "done"
    assert gateway.tool_contexts[-1].allowed_actions == frozenset({"extract"})
    record = service._get_record(session_id)
    assert record.last_turn_allowed_actions == ["extract"]


def test_provider_error_marks_message_job_failed_without_raw_network_detail(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        settings,
        "load_settings",
        lambda: {"ai_enabled": True, "language": "en"},
    )

    class ProviderErrorGateway(FakeGateway):
        def request(
            self, method: str, params: dict[str, Any], **kwargs: Any
        ) -> dict[str, Any]:
            state = super().request(method, params, **kwargs)
            if method == "session.prompt":
                state["transcript"] = [
                    {
                        "role": "assistant",
                        "content": [],
                        "stop_reason": "error",
                        "error_code": "pi_model_provider_unavailable",
                    }
                ]
            return state

    service = PiCopilotService(
        store_path=tmp_path / "sessions.json",
        gateway=ProviderErrorGateway(),
    )
    session_id = service.create_session(
        project_id="project-provider-error", external_llm_opt_in=True
    )["session"]["session_id"]
    _allow_unrelated_message_test(service, session_id)

    submitted = service.send_message(
        session_id,
        project_id="project-provider-error",
        message="Retry the confirmed action.",
    )
    deadline = time.monotonic() + 3
    job = None
    while time.monotonic() < deadline:
        job = service_module.jobs.MANAGER.get(submitted["job_id"])
        if job and job.status != "running":
            break
        time.sleep(0.01)

    assert job is not None and job.status == "failed"
    assert job.error == "pi_model_provider_unavailable"
    record = service._get_record(session_id)
    assert record.last_turn_status == "failed"
    assert record.active_message_job_id is None


def test_conversation_prompt_uses_bounded_host_deadline(tmp_path: Path) -> None:
    prompt_timeouts: list[float | None] = []

    class DeadlineGateway(FakeGateway):
        def request(
            self, method: str, params: dict[str, Any], **kwargs: Any
        ) -> dict[str, Any]:
            if method == "session.prompt":
                prompt_timeouts.append(kwargs.get("timeout"))
            return super().request(method, params, **kwargs)

    service = PiCopilotService(
        store_path=tmp_path / "sessions.json",
        gateway=DeadlineGateway(),
    )
    session_id = service.create_session(
        project_id="project-bounded-prompt", external_llm_opt_in=True
    )["session"]["session_id"]
    _allow_unrelated_message_test(service, session_id)

    submitted = service.send_message(
        session_id,
        project_id="project-bounded-prompt",
        message="Confirm the selected source.",
    )
    deadline = time.monotonic() + 3
    while time.monotonic() < deadline:
        job = service_module.jobs.MANAGER.get(submitted["job_id"])
        if job and job.status != "running":
            break
        time.sleep(0.01)

    assert job is not None and job.status == "done"
    assert prompt_timeouts == [service_module.COPILOT_MESSAGE_TIMEOUT_SECONDS]
    assert prompt_timeouts[0] < 2 * 60


def test_orphaned_replay_execution_is_marked_interrupted(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(settings, "load_settings", lambda: {"ai_enabled": True})
    service = PiCopilotService(
        store_path=tmp_path / "sessions.json",
        gateway=FakeGateway(),
    )
    session_id = service.create_session(
        project_id="project-replay", external_llm_opt_in=True
    )["session"]["session_id"]
    record = service._get_record(session_id)
    record.active_message_job_id = "process-local-job-that-no-longer-exists"
    record.last_turn_status = "running"
    service._save_record(record)

    public = service.get_session(session_id, project_id="project-replay")["session"]

    assert public["active_message_job_id"] is None
    assert public["last_turn_status"] == "interrupted"


def test_pi_replay_projection_excludes_text_deltas_and_sensitive_shapes() -> None:
    assert (
        project_pi_replay_event(
            {"type": "text_delta", "at": "2026-08-13T00:00:00Z", "delta": "private"}
        )
        is None
    )
    projected = project_pi_replay_event(
        {
            "type": "tool_end",
            "at": "2026-08-13T00:00:01Z",
            "tool_call_id": "call_1",
            "tool_name": "easyicu_inspect_data_package",
            "status": "ok",
            "code": "easyicu_data_package_review_ready",
            "owner": "easyicu.webserver.data_package_review",
            "job_id": "job_1",
            "summary": "must not persist free-form output",
            "resource": {"path": "/private/source"},
        }
    )
    assert projected == {
        "type": "tool_end",
        "at": "2026-08-13T00:00:01Z",
        "tool_call_id": "call_1",
        "tool_name": "easyicu_inspect_data_package",
        "status": "ok",
        "code": "easyicu_data_package_review_ready",
        "owner": "easyicu.webserver.data_package_review",
        "job_id": "job_1",
    }


def test_pi_replay_survives_service_restart_and_archives_only_safe_child_job(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(settings, "load_settings", lambda: {"ai_enabled": True})
    store_path = tmp_path / "sessions.json"
    service = PiCopilotService(store_path=store_path, gateway=FakeGateway())
    session_id = service.create_session(
        project_id="project-replay-archive",
        external_llm_opt_in=True,
    )["session"]["session_id"]
    service.replay_store.start_turn(
        session_id=session_id,
        project_id="project-replay-archive",
        job_id="message-job-1",
        allowed_actions=["run", "provider_run"],
    )
    service.replay_store.append_event(
        session_id=session_id,
        project_id="project-replay-archive",
        job_id="message-job-1",
        event={
            "type": "tool_end",
            "at": "2026-08-13T01:00:00Z",
            "tool_call_id": "call-1",
            "tool_name": "easyicu_run",
            "status": "ok",
            "code": "easyicu_full_run_submitted",
            "owner": "easyicu.webserver.agent_pipeline_runs",
            "job_id": "child-job-1",
        },
    )
    service.replay_store.finish_turn(
        session_id=session_id,
        project_id="project-replay-archive",
        job_id="message-job-1",
        status="done",
    )

    child = service_module.jobs.Job("child-job-1", "research-agent-pipeline")
    child.emit(
        {
            "type": "progress",
            "step": "plan",
            "label": "Evidence-bound plan prepared",
            "path": "/private/must-not-persist.json",
        }
    )
    child.finish(
        "done",
        result={"project_dir": "/private/result", "patient_id": "hidden"},
    )

    class ChildManager:
        def get(self, job_id: str) -> Any:
            return child if job_id == child.id else None

    monkeypatch.setattr(service_module.jobs, "MANAGER", ChildManager())
    archived = service.archive_child_job(
        session_id,
        project_id="project-replay-archive",
        job_id=child.id,
    )
    assert archived["job"]["status"] == "done"
    assert "result" not in archived["job"]
    assert "/private" not in json.dumps(archived)

    restarted = PiCopilotService(store_path=store_path, gateway=FakeGateway())
    public = restarted.get_session(
        session_id,
        project_id="project-replay-archive",
    )["session"]
    assert public["conversation_replay"]["turns"][0]["allowed_actions"] == [
        "provider_run",
        "run",
    ]
    assert public["archived_child_jobs"][0]["job_id"] == child.id
    assert public["archived_child_jobs"][0]["created_at_epoch"] == child.created
    assert public["archived_child_jobs"][0]["finished_at_epoch"] == child.finished
    assert len(public["conversation_replay"]["replay_sha256"]) == 64


def test_presentation_pin_protects_conversation_from_retention_eviction(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(service_module, "MAX_SESSIONS", 2)
    service = PiCopilotService(
        store_path=tmp_path / "sessions.json",
        gateway=FakeGateway(),
    )
    pinned = PiSessionRecord(
        session_id="pi-pinned",
        project_id="project-a",
        pinned_for_presentation=True,
    )
    service._save_record(pinned)
    service._save_record(PiSessionRecord(session_id="pi-old", project_id="project-a"))
    service._save_record(PiSessionRecord(session_id="pi-new", project_id="project-a"))

    records = service._read_records()
    assert [row.session_id for row in records] == ["pi-new", "pi-pinned"]
    assert records[1].pinned_for_presentation is True


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
    _allow_unrelated_message_test(service, session_id)
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


def test_control_tools_fail_closed_without_owner_contracts(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(tool_module, "_bound_context", lambda _binding: None)
    monkeypatch.setattr(tool_module, "_run_rows", lambda _context: [])
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
        session=session, allowed_actions=frozenset({"provider_run"})
    )
    full_block = tool_module.execute_tool(
        "easyicu_run", {"run_type": "full"}, run_grant
    )
    assert full_block["code"] == "external_llm_opt_in_required"

    resume_block = tool_module.execute_tool("easyicu_resume", {}, no_grant)
    assert resume_block["code"] == "scientific_resume_not_supported"
    replan_block = tool_module.execute_tool(
        "easyicu_request_replan",
        {"reason": "The aggregate outcome changed."},
        no_grant,
    )
    assert replan_block["code"] == "scientific_replan_not_supported"


def test_superseded_plan_replan_starts_fresh_pipeline_run(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from easyicu.webserver import study_contexts as study_owner
    from easyicu.webserver.routes import agent as agent_routes

    study = {
        "id": "study-fresh-plan",
        "revision": 3,
        "question": "Is standard Sepsis-3 associated with mortality?",
        "data_source": {"path": "/private/export", "database": "miiv"},
    }
    monkeypatch.setattr(tool_module, "_bound_context", lambda _binding: dict(study))
    monkeypatch.setattr(
        tool_module,
        "_workflow_snapshot",
        lambda _context, *, study_override=None: {
            "next_action_code": "provider_ready_to_generate_plan",
            "missing_setup_fields": [],
        },
    )
    monkeypatch.setattr(
        tool_module,
        "_run_rows",
        lambda _context: [
            {
                "run_id": "run-old-plan",
                "study_id": study["id"],
                "run_status": "human_review_pending",
                "pending_review_reason_codes": ["operator_plan_approval_required"],
                "scientific_configuration_sha256": "a" * 64,
                "artifact_names": ["agent_plan.json"],
            }
        ],
    )
    monkeypatch.setattr(
        tool_module.agent_pipeline_runs,
        "pending_review",
        lambda _run_id: {
            "run_id": "run-old-plan",
            "scientific_configuration_sha256": "a" * 64,
            "resumable_here": True,
        },
    )
    submitted: list[dict[str, Any]] = []
    monkeypatch.setattr(
        agent_routes,
        "submit_agent_run",
        lambda payload, *, account_environment=None: (
            submitted.append(dict(payload))
            or {
                "job_id": "job-fresh-plan",
                "kind": "agent-run",
                "status": "queued",
                "study_context_id": study["id"],
                "study_context_revision": study["revision"],
                "engine": "research_agent_pipeline",
            }
        ),
    )
    assert study_owner.scientific_configuration_sha256(study) != "a" * 64
    context = ToolExecutionContext(
        session=PiSessionRecord(
            session_id="pi-fresh-plan",
            external_llm_opt_in=True,
            binding=AuthorityBinding(
                study_context_id=study["id"],
                study_revision=study["revision"],
                run_id="run-old-plan",
            ),
        ),
        allowed_actions={"provider_run"},
    )

    result = tool_module.execute_tool(
        "easyicu_request_replan",
        {"reason": "The standard Sepsis definition replaced an experimental one."},
        context,
    )

    assert result["code"] == "easyicu_full_run_submitted"
    assert result["details"]["job_id"] == "job-fresh-plan"
    assert submitted[0]["study_context_id"] == study["id"]
    assert submitted[0]["run_type"] == "full"
    assert submitted[0]["engine"] == "research_agent_pipeline"


def test_terminal_blocked_plan_replan_starts_fresh_pipeline_run(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from easyicu.webserver.routes import agent as agent_routes

    study = {
        "id": "study-terminal-plan",
        "revision": 21,
        "question": "What is the prevalence and outcome association?",
        "data_source": {"path": "/private/export", "database": "miiv"},
    }
    monkeypatch.setattr(tool_module, "_bound_context", lambda _binding: dict(study))
    monkeypatch.setattr(
        tool_module,
        "_workflow_snapshot",
        lambda _context, *, study_override=None: {
            "next_action_code": "provider_ready_to_generate_plan",
            "missing_setup_fields": [],
        },
    )
    monkeypatch.setattr(
        tool_module,
        "_run_rows",
        lambda _context: [
            {
                "run_id": "run-terminal-blocked",
                "study_id": study["id"],
                "run_status": "blocked",
                "gate_status": "blocked",
                "pending_review_reason_codes": [],
                "scientific_configuration_sha256": "b" * 64,
                "artifact_names": ["agent_plan.json", "source_run_manifest.json"],
            }
        ],
    )
    submitted: list[dict[str, Any]] = []
    monkeypatch.setattr(
        agent_routes,
        "submit_agent_run",
        lambda payload, *, account_environment=None: (
            submitted.append(dict(payload))
            or {
                "job_id": "job-terminal-retry",
                "kind": "agent-run",
                "status": "queued",
                "study_context_id": study["id"],
                "study_context_revision": study["revision"],
                "engine": "research_agent_pipeline",
            }
        ),
    )
    context = ToolExecutionContext(
        session=PiSessionRecord(
            session_id="pi-terminal-retry",
            external_llm_opt_in=True,
            binding=AuthorityBinding(
                study_context_id=study["id"],
                study_revision=study["revision"],
                run_id="run-terminal-blocked",
            ),
        ),
        allowed_actions={"provider_run"},
    )

    result = tool_module.execute_tool(
        "easyicu_request_replan",
        {"reason": "Retry the unchanged analysis after the old runtime failed."},
        context,
    )

    assert result["code"] == "easyicu_full_run_submitted"
    assert result["details"]["job_id"] == "job-terminal-retry"
    assert submitted == [
        {
            "path": "/private/export",
            "study_context_id": study["id"],
            "question": study["question"],
            "run_type": "full",
            "llm_provider": "openai",
            "external_llm_opt_in": True,
            "engine": "research_agent_pipeline",
            "credential_source": "pi_verified",
        }
    ]


def test_preflight_only_history_replan_starts_fresh_pipeline_run(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A failed pre-projection pipeline must not deadlock fresh planning.

    The nested Research Agent may have produced a candidate Plan before the
    Web review projection failed.  That candidate is not a registered current
    Plan authority and cannot be rejected through the resume endpoint.  The
    durable registered history therefore remains the deterministic preflight.
    A user-authorized replan starts a new run without mutating or reusing the
    unregistered candidate.
    """

    from easyicu.webserver.routes import agent as agent_routes

    study = {
        "id": "study-preflight-only",
        "revision": 11,
        "question": "What is the prevalence and outcome association?",
        "data_source": {"path": "/private/export", "database": "miiv"},
        "active_job_id": None,
    }
    monkeypatch.setattr(tool_module, "_bound_context", lambda _binding: dict(study))
    monkeypatch.setattr(
        tool_module,
        "_workflow_snapshot",
        lambda _context, *, study_override=None: {
            "next_action_code": "provider_ready_to_generate_plan",
            "missing_setup_fields": [],
        },
    )
    monkeypatch.setattr(
        tool_module,
        "_run_rows",
        lambda _context: [
            {
                "run_id": "run-deterministic-preflight",
                "study_id": study["id"],
                "run_status": None,
                "gate_status": "analysis_only",
                "pending_review_reason_codes": [],
                "artifact_names": [
                    "run_context.json",
                    "cohort_summary.json",
                    "quality_gate.json",
                ],
            }
        ],
    )
    submitted: list[dict[str, Any]] = []
    monkeypatch.setattr(
        agent_routes,
        "submit_agent_run",
        lambda payload, *, account_environment=None: (
            submitted.append(dict(payload))
            or {
                "job_id": "job-fresh-after-bridge-failure",
                "kind": "agent-run",
                "status": "queued",
                "study_context_id": study["id"],
                "study_context_revision": study["revision"],
                "engine": "research_agent_pipeline",
            }
        ),
    )
    context = ToolExecutionContext(
        session=PiSessionRecord(
            session_id="pi-preflight-only",
            external_llm_opt_in=True,
            binding=AuthorityBinding(
                study_context_id=study["id"],
                study_revision=study["revision"],
                run_id="run-deterministic-preflight",
            ),
        ),
        allowed_actions={"provider_run"},
    )

    result = tool_module.execute_tool(
        "easyicu_request_replan",
        {"reason": "The earlier pipeline ended before a review was registered."},
        context,
    )

    assert result["code"] == "easyicu_full_run_submitted"
    assert result["details"]["job_id"] == "job-fresh-after-bridge-failure"
    assert submitted[0]["study_context_id"] == study["id"]
    assert submitted[0]["run_type"] == "full"


def test_current_digest_matching_plan_review_cannot_be_restarted_as_replan(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from easyicu.webserver import study_contexts as study_owner
    from easyicu.webserver.routes import agent as agent_routes

    study = {
        "id": "study-current-plan",
        "revision": 4,
        "question": "What is the current association?",
        "data_source": {"path": "/private/export", "database": "miiv"},
    }
    digest = study_owner.scientific_configuration_sha256(study)
    monkeypatch.setattr(tool_module, "_bound_context", lambda _binding: dict(study))
    monkeypatch.setattr(
        tool_module,
        "_run_rows",
        lambda _context: [
            {
                "run_id": "run-current-review",
                "study_id": study["id"],
                "run_status": "human_review_pending",
                "pending_review_reason_codes": ["operator_plan_approval_required"],
                "scientific_configuration_sha256": digest,
                "artifact_names": ["agent_plan.json"],
            }
        ],
    )
    monkeypatch.setattr(
        tool_module.agent_pipeline_runs,
        "pending_review",
        lambda _run_id: {
            "run_id": "run-current-review",
            "scientific_configuration_sha256": digest,
            "resumable_here": True,
        },
    )
    submitted: list[dict[str, Any]] = []
    monkeypatch.setattr(
        agent_routes,
        "submit_agent_run",
        lambda payload, *, account_environment=None: submitted.append(dict(payload)),
    )
    context = ToolExecutionContext(
        session=PiSessionRecord(
            session_id="pi-current-review",
            external_llm_opt_in=True,
            binding=AuthorityBinding(
                study_context_id=study["id"],
                study_revision=study["revision"],
                run_id="run-current-review",
            ),
        ),
        allowed_actions={"provider_run"},
    )

    result = tool_module.execute_tool(
        "easyicu_request_replan",
        {"reason": "Start this exact plan again."},
        context,
    )

    assert result["code"] == "scientific_replan_not_supported"
    assert submitted == []


def test_tool_surface_has_no_generic_or_scientific_authority_mutators() -> None:
    research_tools = {
        "easyicu_workspace_status",
        "easyicu_list_data_sources",
        "easyicu_list_source_concepts",
        "easyicu_inspect_data_package",
        "easyicu_review_cohort",
        "easyicu_open_data_download",
        "easyicu_preview_icd_cohort",
        "easyicu_review_patient_timeline",
        "easyicu_compare_data_sources",
        "easyicu_inspect_workflow",
        "easyicu_inspect_context",
        "easyicu_inspect_plan",
        "easyicu_inspect_literature",
        "easyicu_inspect_capability",
        "easyicu_inspect_run",
        "easyicu_inspect_step",
        "easyicu_inspect_validation",
        "easyicu_list_artifacts",
        "easyicu_inspect_evidence",
        "easyicu_explain_blocker",
        "easyicu_inspect_interpretation",
        "easyicu_inspect_manuscript",
        "easyicu_update_study_context",
        "easyicu_mine_ideas",
        "easyicu_search_literature",
        "easyicu_prepare_idea_handoff",
        "easyicu_accept_idea_handoff",
        "easyicu_prepare_demo_source",
        "easyicu_start_extraction",
        "easyicu_run",
        "easyicu_resume",
        "easyicu_cancel",
        "easyicu_request_replan",
        "easyicu_list_extensions",
        "easyicu_load_skill",
        "easyicu_call_mcp_tool",
    }
    workspace_tools = {
        "easyicu_list_project_files",
        "easyicu_read_project_file",
        "easyicu_write_project_file",
        "easyicu_edit_project_file",
        "easyicu_check_project_file",
        "easyicu_preview_project_file",
    }
    assert tool_module.READ_TOOLS | tool_module.CONTROL_TOOLS == research_tools
    assert tool_module.WORKSPACE_TOOLS == workspace_tools
    assert tool_module.ALLOWED_TOOLS == research_tools | workspace_tools
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


def test_registered_data_source_choices_are_path_free(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        tool_module.sources,
        "load_registry",
        lambda: {
            "active_path": "/private/full-export",
            "sources": [
                {
                    "id": "src_demo",
                    "path": "/private/demo-export",
                    "label": "MIMIC-IV Clinical Database Demo v2.2",
                    "database": "miiv",
                    "generated": "2026-07-28T00:21:10",
                    "ok": True,
                    "modules": ["demographics", "outcome"],
                    "summary": {
                        "stays": 140,
                        "modules": 2,
                        "file_count": 2,
                        "total_rows": 1000,
                    },
                },
                {
                    "id": "src_full",
                    "path": "/private/full-export",
                    "label": "MIMIC-IV full export",
                    "database": "miiv",
                    "ok": True,
                    "modules": ["demographics", "outcome", "vitals"],
                    "summary": {"stays": 94458, "modules": 3},
                },
            ],
        },
    )
    context = ToolExecutionContext(session=PiSessionRecord(session_id="pi-sources"))

    catalog = tool_module.execute_tool("easyicu_list_data_sources", {}, context)

    assert catalog["code"] == "easyicu_data_sources_listed"
    assert catalog["details"]["sources"] == []
    assert catalog["details"]["source_count"] == 0
    assert catalog["details"]["selected_database"] is None
    assert catalog["details"]["source_modes"] == []
    assert catalog["details"]["database_selection_deferred"] is False
    assert [row["database"] for row in catalog["details"]["supported_databases"]] == [
        "miiv",
        "eicu",
        "aumc",
        "hirid",
        "mimic",
        "sic",
    ]

    result = tool_module.execute_tool(
        "easyicu_list_data_sources", {"database": "miiv"}, context
    )

    assert result["code"] == "easyicu_data_sources_listed"
    assert result["details"]["source_count"] == 2
    assert result["details"]["sources"][0]["source_id"] == "src_demo"
    assert result["details"]["sources"][0]["source_scope"] == "official_demo"
    assert result["details"]["sources"][0]["aggregate"]["stays"] == 140
    assert result["details"]["sources"][1]["source_scope"] == "registered_export"
    assert result["details"]["sources"][1]["label"] == "MIMIC-IV v3.1"
    assert (
        result["details"]["sources"][1]["availability"]
        == "available_in_easyicu"
    )
    assert result["details"]["sources"][1]["active"] is True
    assert result["details"]["recommended_source"] == {
        "source_id": "src_full",
        "label": "MIMIC-IV v3.1",
        "database": "miiv",
        "availability": "available_in_easyicu",
        "module_count": 3,
        "aggregate": {"stays": 94458, "modules": 3},
        "selection_reason": "most_complete_local_dataset",
        "auto_select_for_exact_database_request": False,
    }
    supported = {
        row["database"]: row for row in result["details"]["supported_databases"]
    }
    assert supported["miiv"] == {
        "database": "miiv",
        "label": "MIMIC-IV",
        "reference_release": "3.1",
        "display_label": "MIMIC-IV v3.1",
        "selection_required": True,
    }
    assert supported["mimic"]["label"] == "MIMIC-III"
    assert supported["mimic"]["reference_release"] == "1.4"
    assert supported["mimic"]["display_label"] == "MIMIC-III v1.4"
    assert supported["aumc"]["reference_release"] is None
    assert supported["aumc"]["display_label"] == "AmsterdamUMCdb"
    demos = result["details"]["official_demos"]
    assert [(row["source_id"], row["version"]) for row in demos] == [
        ("mimic_iv_demo_v2_2", "2.2"),
    ]
    assert all(row["research_scope"] == "demo_only" for row in demos)
    assert result["details"]["selected_database"] == {
        "database": "miiv",
        "label": "MIMIC-IV",
        "reference_release": "3.1",
    }
    assert result["details"]["source_modes"] == [
        "local_full_database",
        "registered_export",
        "official_demo",
    ]
    assert result["details"]["database_selection_deferred"] is False
    assert result["details"]["selection_policy"]["mimic_database_choices"] == [
        "mimic",
        "miiv",
    ]
    assert result["details"]["selection_policy"][
        "recommended_source_is_already_available_export"
    ] is True
    assert result["details"]["selection_policy"][
        "alternative_local_directory_requires_registration"
    ] is True
    assert result["details"]["selection_policy"][
        "show_returned_safe_aggregates_for_recommendation"
    ] is True
    assert "/private/" not in json.dumps(result)
    assert "full export" not in json.dumps(result)

    ambiguous = tool_module.execute_tool(
        "easyicu_list_data_sources",
        {"database": "miiv"},
        ToolExecutionContext(
            session=PiSessionRecord(session_id="pi-sources-ambiguous"),
            user_message="帮我查一下 MIMIC 数据库里的 A41 队列",
        ),
    )
    assert ambiguous["details"]["selected_database"] is None
    assert ambiguous["details"]["sources"] == []
    assert ambiguous["details"]["source_modes"] == []
    assert ambiguous["details"]["database_selection_deferred"] is True

    explicit = tool_module.execute_tool(
        "easyicu_list_data_sources",
        {"database": "miiv"},
        ToolExecutionContext(
            session=PiSessionRecord(session_id="pi-sources-explicit"),
            user_message="使用 MIMIC-IV 3.1",
        ),
    )
    assert explicit["details"]["selected_database"]["database"] == "miiv"
    assert explicit["details"]["database_selection_deferred"] is False
    assert explicit["details"]["source_modes"] == ["easyicu_available_data"]
    assert explicit["details"]["recommended_source"] == {
        "source_id": "src_full",
        "label": "MIMIC-IV v3.1",
        "database": "miiv",
        "availability": "available_in_easyicu",
        "module_count": 3,
        "aggregate": {"stays": 94458, "modules": 3},
        "selection_reason": "most_complete_local_dataset",
        "auto_select_for_exact_database_request": True,
    }

    corrected_model_argument = tool_module.execute_tool(
        "easyicu_list_data_sources",
        {"database": "mimic"},
        ToolExecutionContext(
            session=PiSessionRecord(session_id="pi-sources-corrected-model-argument"),
            user_message="研究 MIMIC-IV 成人 ICU 人群",
        ),
    )
    assert corrected_model_argument["details"]["selected_database"]["database"] == "miiv"
    assert corrected_model_argument["details"]["sources"][0]["database"] == "miiv"


def test_conversational_setup_requires_explicit_first_stay_selection(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    current = {
        "id": "study-first-stay-authority",
        "revision": 1,
        "question": "Estimate prevalence in adult ICU patients.",
        "active_job_id": None,
        "cohort": {},
    }
    writes: list[dict[str, Any]] = []
    monkeypatch.setattr(tool_module, "_bound_context", lambda binding: dict(current))
    monkeypatch.setattr(
        tool_module.study_contexts,
        "upsert_context",
        lambda raw, **_kwargs: writes.append(dict(raw)) or {**raw, "revision": 2},
    )
    session = PiSessionRecord(
        session_id="pi-first-stay-authority",
        binding=AuthorityBinding(
            study_context_id=current["id"],
            study_revision=current["revision"],
        ),
    )

    blocked = tool_module.execute_tool(
        "easyicu_update_study_context",
        {"cohort": {"preset": "adult_first", "label": "Adult first ICU stay"}},
        ToolExecutionContext(
            session=session,
            user_message="研究成人 ICU 人群的患病率",
            allowed_actions={"configure"},
        ),
    )
    assert blocked["code"] == "study_cohort_first_stay_confirmation_required"
    assert blocked["details"]["field"] == "cohort.preset"
    assert writes == []

    accepted = tool_module.execute_tool(
        "easyicu_update_study_context",
        {"cohort": {"preset": "adult_first", "label": "Adult first ICU stay"}},
        ToolExecutionContext(
            session=session,
            user_message="采用每位患者首次 ICU 入住",
            allowed_actions={"configure"},
        ),
    )
    assert accepted["code"] == "study_context_updated"
    assert writes[0]["cohort"]["preset"] == "adult_first"


def test_conversational_setup_requires_explicit_all_stays_selection(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    current = {
        "id": "study-all-stays-authority",
        "revision": 1,
        "question": "Estimate prevalence in adult ICU patients.",
        "active_job_id": None,
        "cohort": {"preset": "all_icu"},
    }
    writes: list[dict[str, Any]] = []
    monkeypatch.setattr(tool_module, "_bound_context", lambda binding: dict(current))
    monkeypatch.setattr(
        tool_module.study_contexts,
        "upsert_context",
        lambda raw, **_kwargs: writes.append(dict(raw)) or {**raw, "revision": 2},
    )
    session = PiSessionRecord(
        session_id="pi-all-stays-authority",
        binding=AuthorityBinding(
            study_context_id=current["id"],
            study_revision=current["revision"],
        ),
    )

    blocked = tool_module.execute_tool(
        "easyicu_update_study_context",
        {"cohort": {"preset": "adult_all", "label": "All adult ICU stays"}},
        ToolExecutionContext(
            session=session,
            user_message="研究成人 ICU 人群的患病率",
            allowed_actions={"configure"},
        ),
    )
    assert blocked["code"] == "study_cohort_all_stays_confirmation_required"
    assert blocked["details"]["field"] == "cohort.preset"
    assert writes == []

    accepted = tool_module.execute_tool(
        "easyicu_update_study_context",
        {"cohort": {"preset": "adult_all", "label": "All adult ICU stays"}},
        ToolExecutionContext(
            session=session,
            user_message="纳入所有符合条件的成人 ICU stays，包括重复 ICU 入住。",
            allowed_actions={"configure"},
        ),
    )
    assert accepted["code"] == "study_context_updated"
    assert writes[0]["cohort"]["preset"] == "adult_all"


def test_opening_question_saves_without_inventing_an_analysis_unit(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    current = {
        "id": "study-opening-question-authority",
        "revision": 1,
        "question": "",
        "purpose": "",
        "active_job_id": None,
        "cohort": {"preset": "all_icu"},
    }
    writes: list[dict[str, Any]] = []
    monkeypatch.setattr(tool_module, "_bound_context", lambda binding: dict(current))
    monkeypatch.setattr(
        tool_module.study_contexts,
        "upsert_context",
        lambda raw, **_kwargs: writes.append(dict(raw)) or {**raw, "revision": 2},
    )
    monkeypatch.setattr(
        tool_module,
        "_workflow_snapshot",
        lambda _context, **_kwargs: {"current_stage": "study_setup"},
    )
    session = PiSessionRecord(
        session_id="pi-opening-question-authority",
        binding=AuthorityBinding(
            study_context_id=current["id"],
            study_revision=current["revision"],
        ),
    )

    result = tool_module.execute_tool(
        "easyicu_update_study_context",
        {
            "question": "Estimate Sepsis-3 prevalence and its association with ICU mortality.",
            "purpose": "Estimate prevalence and evaluate an observational association.",
            "cohort": {"preset": "adult_all", "label": "Adult ICU population"},
        },
        ToolExecutionContext(
            session=session,
            user_message=(
                "研究 MIMIC-IV 成人 ICU 人群中 Sepsis-3 的患病率，"
                "以及 Sepsis-3 与 ICU 死亡的关系。"
            ),
            allowed_actions={"configure"},
        ),
    )

    assert result["code"] == "study_context_updated"
    assert writes[0]["question"].startswith("Estimate Sepsis-3 prevalence")
    assert writes[0]["cohort"]["preset"] == "all_icu"
    assert result["details"]["omitted_unconfirmed_fields"] == ["cohort.preset"]


def test_conversational_setup_requires_direct_outcome_and_exposure_choices(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    current = {
        "id": "study-candidate-intent-authority",
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
        session_id="pi-candidate-intent-authority",
        binding=AuthorityBinding(
            study_context_id=current["id"],
            study_revision=current["revision"],
        ),
    )

    blocked_outcome = tool_module.execute_tool(
        "easyicu_update_study_context",
        {"outcome": "ICU mortality"},
        ToolExecutionContext(
            session=session,
            user_message="研究 Sepsis-3 与 ICU 死亡的关系。",
            allowed_actions={"configure"},
        ),
    )
    assert blocked_outcome["code"] == "study_primary_outcome_confirmation_required"

    blocked_exposure = tool_module.execute_tool(
        "easyicu_update_study_context",
        {"primary_exposure": "Standard Sepsis-3"},
        ToolExecutionContext(
            session=session,
            user_message="研究 Sepsis-3 与 ICU 死亡的关系。",
            allowed_actions={"configure"},
        ),
    )
    assert blocked_exposure["code"] == "study_primary_exposure_confirmation_required"
    assert writes == []

    accepted_outcome = tool_module.execute_tool(
        "easyicu_update_study_context",
        {"outcome": "ICU mortality"},
        ToolExecutionContext(
            session=session,
            user_message="主要结局使用 ICU stay 期间死亡，不改成住院期间死亡。",
            allowed_actions={"configure"},
        ),
    )
    assert accepted_outcome["code"] == "study_context_updated"

    accepted_outcome_change = tool_module.execute_tool(
        "easyicu_update_study_context",
        {
            "question": "研究 Sepsis-3 与院内死亡的关系。",
            "outcome": "院内死亡（本次住院期间死亡）",
        },
        ToolExecutionContext(
            session=session,
            user_message=(
                "当前导出没有可验证的 ICU 专用死亡结局。"
                "请把主要结局改为院内死亡（本次住院期间死亡）。"
            ),
            allowed_actions={"configure"},
        ),
    )
    assert accepted_outcome_change["code"] == "study_context_updated"
    assert writes[-1]["outcome"] == "院内死亡（本次住院期间死亡）"
    assert writes[-1]["question"] == "研究 Sepsis-3 与院内死亡的关系。"

    accepted_compact_choice = tool_module.execute_tool(
        "easyicu_update_study_context",
        {
            "outcome": "ICU mortality",
            "primary_exposure": "Standard Sepsis-3",
        },
        ToolExecutionContext(
            session=session,
            user_message="ICU 住院期间死亡（推荐）",
            allowed_actions={"configure"},
        ),
    )
    assert accepted_compact_choice["code"] == "study_context_updated"
    assert writes[-1]["outcome"] == "ICU mortality"
    assert "primary_exposure" not in writes[-1]

    accepted_exposure = tool_module.execute_tool(
        "easyicu_update_study_context",
        {"primary_exposure": "Standard Sepsis-3"},
        ToolExecutionContext(
            session=session,
            user_message="主要暴露采用标准 Sepsis-3，不要使用实验性的 SOFA-2。",
            allowed_actions={"configure"},
        ),
    )
    assert accepted_exposure["code"] == "study_context_updated"


def test_conversational_setup_keeps_confirmed_outcome_when_model_bundles_exposure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    current = {
        "id": "study-partial-confirmation-authority",
        "revision": 1,
        "question": "Estimate prevalence and examine mortality association.",
        "active_job_id": None,
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
        session_id="pi-partial-confirmation-authority",
        binding=AuthorityBinding(
            study_context_id=current["id"],
            study_revision=current["revision"],
        ),
    )

    result = tool_module.execute_tool(
        "easyicu_update_study_context",
        {
            "outcome": "ICU mortality",
            "primary_exposure": "Standard Sepsis-3",
        },
        ToolExecutionContext(
            session=session,
            user_message="主要结局使用 ICU stay 期间死亡，不改成住院期间死亡。",
            allowed_actions={"configure"},
        ),
    )

    assert result["code"] == "study_context_updated"
    assert writes[0]["outcome"] == "ICU mortality"
    assert "primary_exposure" not in writes[0]


def test_conversational_setup_does_not_bundle_all_stays_with_clustering(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    current = {
        "id": "study-clustering-authority",
        "revision": 2,
        "question": "Estimate Sepsis-3 prevalence.",
        "active_job_id": None,
        "cohort": {"preset": "all_icu"},
        "analysis_design": {},
    }
    writes: list[dict[str, Any]] = []
    monkeypatch.setattr(tool_module, "_bound_context", lambda binding: dict(current))
    monkeypatch.setattr(
        tool_module.study_contexts,
        "upsert_context",
        lambda raw, **_kwargs: writes.append(dict(raw)) or {**raw, "revision": 3},
    )
    session = PiSessionRecord(
        session_id="pi-clustering-authority",
        binding=AuthorityBinding(
            study_context_id=current["id"],
            study_revision=current["revision"],
        ),
    )
    proposal = {
        "analysis_design": {
            "analysis_unit": "icu_stay",
            "variance_estimator": "cluster_robust",
            "cluster_unit": "patient",
        }
    }

    blocked = tool_module.execute_tool(
        "easyicu_update_study_context",
        proposal,
        ToolExecutionContext(
            session=session,
            user_message="纳入所有符合条件的成人 ICU stays，包括重复 ICU 入住。",
            allowed_actions={"configure"},
        ),
    )
    assert blocked["code"] == "study_patient_clustering_confirmation_required"
    assert writes == []

    capability_checked = tool_module.execute_tool(
        "easyicu_update_study_context",
        proposal,
        ToolExecutionContext(
            session=session,
            user_message="如果能验证患者聚类坐标，请在统计推断中按患者处理相关性。",
            allowed_actions={"configure"},
        ),
    )
    assert capability_checked["code"] == "research_pipeline_cluster_variance_unsupported"
    assert writes == []


def test_conversational_setup_requires_direct_analysis_goal_choice(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    current = {
        "id": "study-analysis-goal-authority",
        "revision": 2,
        "question": "Estimate prevalence and examine a relationship.",
        "active_job_id": None,
        "analysis_goal": "",
    }
    writes: list[dict[str, Any]] = []
    monkeypatch.setattr(tool_module, "_bound_context", lambda binding: dict(current))
    monkeypatch.setattr(
        tool_module.study_contexts,
        "upsert_context",
        lambda raw, **_kwargs: writes.append(dict(raw)) or {**raw, "revision": 3},
    )
    session = PiSessionRecord(
        session_id="pi-analysis-goal-authority",
        binding=AuthorityBinding(
            study_context_id=current["id"],
            study_revision=current["revision"],
        ),
    )

    blocked = tool_module.execute_tool(
        "easyicu_update_study_context",
        {"analysis_goal": "Descriptive prevalence and observational association"},
        ToolExecutionContext(
            session=session,
            user_message="保留重复 ICU stays 并按患者聚类。",
            allowed_actions={"configure"},
        ),
    )
    assert blocked["code"] == "study_analysis_goal_confirmation_required"
    assert writes == []

    accepted = tool_module.execute_tool(
        "easyicu_update_study_context",
        {"analysis_goal": "Descriptive prevalence and observational association"},
        ToolExecutionContext(
            session=session,
            user_message="先报告患病率，再评估观察性关联；不要写成因果效应。",
            allowed_actions={"configure"},
        ),
    )
    assert accepted["code"] == "study_context_updated"
    assert writes[0]["analysis_goal"].startswith("Descriptive prevalence")

    compact_choice = tool_module.execute_tool(
        "easyicu_update_study_context",
        {"analysis_goal": "Adjusted observational association"},
        ToolExecutionContext(
            session=session,
            user_message="描述患病率，并进行协变量调整后的关联分析（推荐）",
            allowed_actions={"configure"},
        ),
    )
    assert compact_choice["code"] == "study_context_updated"
    assert writes[-1]["analysis_goal"] == "Adjusted observational association"

    synchronized_wording = tool_module.execute_tool(
        "easyicu_update_study_context",
        {"analysis_goal": "描述患病率并评估与院内死亡的调整后关联"},
        ToolExecutionContext(
            session=session,
            user_message=(
                "请把分析目标中的 ICU mortality 同步替换为院内死亡；"
                "这只是保持同一研究决定一致。"
            ),
            allowed_actions={"configure"},
        ),
    )
    assert synchronized_wording["code"] == "study_context_updated"
    assert writes[-1]["analysis_goal"].endswith("院内死亡的调整后关联")


def test_analysis_goal_update_preserves_confirmed_patient_clustering(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    current = {
        "id": "study-analysis-design-preservation",
        "revision": 4,
        "question": "Estimate prevalence and observational mortality association.",
        "active_job_id": None,
        "analysis_goal": "",
        "analysis_design": {
            "analysis_unit": "icu_stay",
            "variance_estimator": "cluster_robust",
            "cluster_unit": "patient",
        },
    }
    writes: list[dict[str, Any]] = []
    monkeypatch.setattr(tool_module, "_bound_context", lambda binding: dict(current))
    monkeypatch.setattr(
        tool_module.study_contexts,
        "upsert_context",
        lambda raw, **_kwargs: writes.append(dict(raw)) or {**raw, "revision": 5},
    )
    monkeypatch.setattr(
        tool_module.agent_pipeline_runs,
        "validate_analysis_design_for_execution",
        lambda _study: None,
    )
    session = PiSessionRecord(
        session_id="pi-analysis-design-preservation",
        binding=AuthorityBinding(
            study_context_id=current["id"],
            study_revision=current["revision"],
        ),
    )

    result = tool_module.execute_tool(
        "easyicu_update_study_context",
        {
            "analysis_goal": "Prevalence followed by observational association",
            "analysis_design": {
                "analysis_family": "association_study",
                "analysis_unit": "icu_stay",
                "variance_estimator": "none_counts_only",
            },
        },
        ToolExecutionContext(
            session=session,
            user_message=(
                "先报告患病率，再评估观察性关联；不要写成因果效应。"
            ),
            allowed_actions={"configure"},
        ),
    )

    assert result["code"] == "study_context_updated"
    assert writes[0]["analysis_design"] == {
        "analysis_family": "association_study",
        "analysis_unit": "icu_stay",
        "variance_estimator": "cluster_robust",
        "cluster_unit": "patient",
    }


def test_explicit_age_sex_adjustment_gets_owner_known_baseline_roles(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    current = {
        "id": "study-demographic-adjustment",
        "revision": 8,
        "question": "Estimate an observational association.",
        "active_job_id": None,
        "covariates": [],
        "covariate_selection": "planner_selectable",
        "data_source": {"database": "mimiciv", "path": "/private/export"},
        "modules": ["demographics"],
        "execution_concepts": {
            "outcome": "death",
            "primary_exposure": "sep3_sofa1",
            "covariates": [],
        },
    }
    writes: list[dict[str, Any]] = []
    monkeypatch.setattr(tool_module, "_bound_context", lambda binding: dict(current))
    monkeypatch.setattr(
        tool_module.study_contexts,
        "upsert_context",
        lambda raw, **_kwargs: writes.append(dict(raw)) or {**raw, "revision": 9},
    )
    monkeypatch.setattr(
        "easyicu.research_agent.acquisition.catalog.build_available_catalog",
        lambda _path: SimpleNamespace(
            concepts=[
                SimpleNamespace(concept_id="death", file_name="outcome.parquet"),
                SimpleNamespace(
                    concept_id="sep3_sofa1",
                    file_name="sepsis3_sofa1.parquet",
                ),
                SimpleNamespace(concept_id="age", file_name="demographics.parquet"),
                SimpleNamespace(concept_id="sex", file_name="demographics.parquet"),
            ]
        ),
    )
    session = PiSessionRecord(
        session_id="pi-demographic-adjustment",
        binding=AuthorityBinding(
            study_context_id=current["id"],
            study_revision=current["revision"],
        ),
    )

    result = tool_module.execute_tool(
        "easyicu_update_study_context",
        {
            "covariates": ["age", "sex"],
            "covariate_selection": "planner_selectable",
        },
        ToolExecutionContext(
            session=session,
            user_message=(
                "主要调整年龄和性别；其他候选协变量必须逐项让我确认。"
            ),
            allowed_actions={"configure"},
        ),
    )

    assert result["code"] == "study_context_updated"
    assert writes[0]["covariate_selection"] == "exact"
    assert writes[0]["covariate_temporal_roles"] == {
        "age": "baseline_static",
        "sex": "baseline_static",
    }
    assert set(writes[0]["covariate_rationales"]) == {"age", "sex"}
    assert writes[0]["execution_concepts"]["covariates"] == ["age", "sex"]


def test_explicit_no_adjustment_clears_covariate_contract_atomically(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    current = {
        "id": "study-descriptive-no-adjustment",
        "revision": 9,
        "question": "Estimate prevalence and describe mortality by exposure group.",
        "active_job_id": None,
        "covariates": ["age", "sex"],
        "covariate_selection": "exact",
        "covariate_rationales": {
            "age": "Prior age adjustment decision.",
            "sex": "Prior sex adjustment decision.",
        },
        "covariate_temporal_roles": {
            "age": "baseline_static",
            "sex": "baseline_static",
        },
        "data_source": {"database": "mimiciv", "path": "/private/export"},
        "modules": ["demographics", "outcome", "sepsis3_sofa1"],
        "execution_concepts": {
            "outcome": "death",
            "primary_exposure": "sep3_sofa1",
            "covariates": ["age", "sex"],
        },
        "analysis_design": {
            "analysis_family": "association_study",
            "analysis_unit": "icu_stay",
            "variance_estimator": "cluster_robust",
            "cluster_unit": "patient",
        },
    }
    writes: list[dict[str, Any]] = []
    monkeypatch.setattr(tool_module, "_bound_context", lambda binding: dict(current))
    monkeypatch.setattr(
        tool_module.study_contexts,
        "upsert_context",
        lambda raw, **_kwargs: writes.append(dict(raw)) or {**raw, "revision": 10},
    )
    monkeypatch.setattr(
        "easyicu.research_agent.acquisition.catalog.build_available_catalog",
        lambda _path: SimpleNamespace(
            concepts=[
                SimpleNamespace(concept_id="death", file_name="outcome.parquet"),
                SimpleNamespace(
                    concept_id="sep3_sofa1",
                    file_name="sepsis3_sofa1.parquet",
                ),
            ]
        ),
    )
    session = PiSessionRecord(
        session_id="pi-descriptive-no-adjustment",
        binding=AuthorityBinding(
            study_context_id=current["id"],
            study_revision=current["revision"],
        ),
    )

    result = tool_module.execute_tool(
        "easyicu_update_study_context",
        {
            "covariates": [],
            "covariate_selection": "exact",
            "covariate_rationales": {},
            "covariate_temporal_roles": {},
            "execution_concepts": {
                "outcome": "death",
                "primary_exposure": "sep3_sofa1",
                "covariates": [],
            },
            "analysis_design": {
                "analysis_family": "descriptive_epidemiology",
                "analysis_unit": "icu_stay",
                "variance_estimator": "none_counts_only",
            },
        },
        ToolExecutionContext(
            session=session,
            user_message=(
                "保留 descriptive_epidemiology，并清除旧 age/sex 调整登记；"
                "age/sex 仅作描述；仅报告计数、比例和绝对差，"
                "不进行关联推断。"
            ),
            allowed_actions={"configure"},
        ),
    )

    assert result["code"] == "study_context_updated"
    assert writes[0]["covariates"] == []
    assert writes[0]["covariate_selection"] == "exact"
    assert writes[0]["covariate_rationales"] == {}
    assert writes[0]["covariate_temporal_roles"] == {}
    assert writes[0]["execution_concepts"]["covariates"] == []
    assert writes[0]["analysis_design"] == {
        "analysis_family": "descriptive_epidemiology",
        "analysis_unit": "icu_stay",
        "variance_estimator": "none_counts_only",
    }


def test_empty_covariate_patch_without_user_clear_authority_preserves_metadata() -> None:
    current = {
        "covariates": ["age", "sex"],
        "covariate_rationales": {"age": "age", "sex": "sex"},
        "covariate_temporal_roles": {
            "age": "baseline_static",
            "sex": "baseline_static",
        },
    }

    merged = tool_module._merge_nested_study_patch(
        current,
        {
            "covariates": [],
            "covariate_rationales": {},
            "covariate_temporal_roles": {},
        },
    )

    assert merged["covariates"] == []
    assert merged["covariate_rationales"] == current["covariate_rationales"]
    assert merged["covariate_temporal_roles"] == current[
        "covariate_temporal_roles"
    ]


def test_exact_covariate_roster_rejects_stale_execution_binding() -> None:
    with pytest.raises(agent_pipeline_runs.ResearchPipelineRunError) as caught:
        agent_pipeline_runs._configured_covariates(
            {
                "covariates": ["age", "sex"],
                "covariate_selection": "exact",
                "execution_concepts": {"covariates": []},
            }
        )

    assert caught.value.code == (
        "research_pipeline_covariate_execution_binding_mismatch"
    )
    assert caught.value.details == {"field": "execution_concepts.covariates"}


def test_explicit_export_format_writes_confirmation_receipt(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    current = {
        "id": "study-export-format-confirmation",
        "revision": 9,
        "question": "Prepare a study package.",
        "active_job_id": None,
        "export_format": "parquet",
        "confirmations": {"feature_time_window": True},
    }
    writes: list[dict[str, Any]] = []
    monkeypatch.setattr(tool_module, "_bound_context", lambda binding: dict(current))
    monkeypatch.setattr(
        tool_module.study_contexts,
        "upsert_context",
        lambda raw, **_kwargs: writes.append(dict(raw)) or {**raw, "revision": 10},
    )
    session = PiSessionRecord(
        session_id="pi-export-format-confirmation",
        binding=AuthorityBinding(
            study_context_id=current["id"],
            study_revision=current["revision"],
        ),
    )

    result = tool_module.execute_tool(
        "easyicu_update_study_context",
        {"export_format": "parquet"},
        ToolExecutionContext(
            session=session,
            user_message="研究数据包导出为 Parquet。",
            allowed_actions={"configure"},
        ),
    )

    assert result["code"] == "study_context_updated"
    assert writes[0]["confirmations"]["export_format"] is True
    assert writes[0]["confirmations"]["feature_time_window"] is True

    compact_choice = tool_module.execute_tool(
        "easyicu_update_study_context",
        {"export_format": "parquet"},
        ToolExecutionContext(
            session=session,
            user_message="Parquet（推荐，适合后续分析）",
            allowed_actions={"configure"},
        ),
    )
    assert compact_choice["code"] == "study_context_updated"
    assert writes[-1]["confirmations"]["export_format"] is True
    assert writes[-1]["confirmations"]["feature_time_window"] is True

def test_conversational_setup_rejects_unrequested_explicit_only_feature_module(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    current = {
        "id": "study-module-authority",
        "revision": 1,
        "question": "Estimate standard Sepsis-3 prevalence.",
        "active_job_id": None,
        "modules": [],
    }
    writes: list[dict[str, Any]] = []
    monkeypatch.setattr(tool_module, "_bound_context", lambda binding: dict(current))
    monkeypatch.setattr(
        tool_module.study_contexts,
        "upsert_context",
        lambda raw, **_kwargs: writes.append(dict(raw)) or {**raw, "revision": 2},
    )
    session = PiSessionRecord(
        session_id="pi-module-authority",
        binding=AuthorityBinding(
            study_context_id=current["id"],
            study_revision=current["revision"],
        ),
    )

    blocked = tool_module.execute_tool(
        "easyicu_update_study_context",
        {"modules": ["demographics", "sepsis3_sofa2", "outcome"]},
        ToolExecutionContext(
            session=session,
            user_message="研究成人 ICU 人群的 Sepsis-3 患病率",
            allowed_actions={"configure"},
        ),
    )
    assert blocked["code"] == "concept_explicit_selection_required"
    assert blocked["details"]["field"] == "modules"
    assert blocked["details"]["canonical_alternative_module"] == "sepsis3_sofa1"
    assert writes == []

    accepted = tool_module.execute_tool(
        "easyicu_update_study_context",
        {"modules": ["demographics", "sepsis3_sofa2", "outcome"]},
        ToolExecutionContext(
            session=session,
            user_message="使用实验性 SOFA-2 Sepsis-3 敏感性表型",
            allowed_actions={"configure"},
        ),
    )
    assert accepted["code"] == "study_context_updated"
    assert writes[0]["confirmations"] == {
        "concept_selection_sep3_sofa2_authorized": True,
    }


def test_data_source_recommendation_fails_closed_on_equally_complete_local_data() -> None:
    choices = [
        {
            "source_id": source_id,
            "source_scope": "registered_export",
            "module_count": 19,
            "aggregate": {"stays": 94458},
        }
        for source_id in ("src_a", "src_b")
    ]

    assert tool_module._dominant_local_source(choices) is None


def test_source_concept_choices_are_exact_module_scoped_and_path_free(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from easyicu.research_agent.acquisition import catalog as catalog_module
    from easyicu.research_agent.acquisition.catalog import (
        AvailableCatalog,
        CatalogConcept,
    )

    monkeypatch.setattr(
        tool_module.sources,
        "load_registry",
        lambda: {
            "sources": [
                {
                    "id": "src_demo",
                    "path": "/private/demo-export",
                    "database": "miiv",
                    "ok": True,
                    "modules": [
                        "demographics",
                        "outcome",
                        "sepsis3_sofa1",
                        "sepsis3_sofa2",
                    ],
                }
            ]
        },
    )
    monkeypatch.setattr(
        catalog_module,
        "build_available_catalog",
        lambda _path: AvailableCatalog(
            source="private-demo",
            concepts=[
                CatalogConcept(
                    concept_id="death",
                    description="in hospital mortality",
                    file_name="outcome.parquet",
                    column_role="event_status",
                ),
                CatalogConcept(
                    concept_id="sep3_sofa2",
                    description="Sepsis-3 using SOFA-2",
                    file_name="sepsis3_sofa2.parquet",
                    column_role="event_status",
                    selection_mode="explicit_only",
                    selection_note="Experimental SOFA-2 sensitivity phenotype.",
                    canonical_alternative="sep3_sofa1",
                ),
                CatalogConcept(
                    concept_id="age",
                    description="Age",
                    file_name="demographics.parquet",
                    column_role="value",
                ),
                CatalogConcept(
                    concept_id="sep3_sofa1",
                    description="Canonical Sepsis-3 using traditional SOFA",
                    file_name="sepsis3_sofa1.parquet",
                    column_role="event_status",
                ),
            ],
        ),
    )
    context = ToolExecutionContext(session=PiSessionRecord(session_id="pi-concepts"))

    result = tool_module.execute_tool(
        "easyicu_list_source_concepts",
        {
            "source_id": "src_demo",
            "modules": ["outcome", "sepsis3_sofa2"],
            "query": "death mortality sepsis-3",
            "limit": 20,
        },
        context,
    )

    assert result["code"] == "easyicu_source_concepts_listed"
    assert [row["concept_id"] for row in result["details"]["concepts"]] == [
        "death",
        "sep3_sofa2",
    ]
    assert result["details"]["concepts"][0]["role"] == "event_status"
    sofa2 = result["details"]["concepts"][1]
    assert sofa2["selection_mode"] == "explicit_only"
    assert sofa2["canonical_alternative"] == "sep3_sofa1"
    assert result["details"]["canonical_alternatives"] == [
        {
            "source_concept_id": "sep3_sofa2",
            "concept_id": "sep3_sofa1",
            "module": "sepsis3_sofa1",
            "role": "event_status",
            "description": "Canonical Sepsis-3 using traditional SOFA",
            "selection_mode": "ordinary",
            "selection_note": "",
            "canonical_alternative": "",
        }
    ]
    assert "/private/" not in json.dumps(result)


def test_conversational_setup_binds_verified_execution_concepts(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from easyicu.research_agent.acquisition import catalog as catalog_module
    from easyicu.research_agent.acquisition.catalog import (
        AvailableCatalog,
        CatalogConcept,
    )

    current = {
        "id": "study-execution-bind",
        "revision": 2,
        "question": (
            "Is the explicitly requested SOFA-2 Sepsis sensitivity phenotype "
            "associated with mortality?"
        ),
        "active_job_id": None,
        "data_source": {
            "path": "/private/full-export",
            "database": "miiv",
        },
        # The user authorizes the exact scientific concept. Its owning parquet
        # module is a technical implementation detail and may be added by the
        # StudyContext owner after catalog verification.
        "modules": ["demographics", "outcome"],
    }
    writes: list[dict[str, Any]] = []
    monkeypatch.setattr(tool_module, "_bound_context", lambda binding: dict(current))
    monkeypatch.setattr(
        catalog_module,
        "build_available_catalog",
        lambda _path: AvailableCatalog(
            source="typed-demo",
            concepts=[
                CatalogConcept(concept_id="age", file_name="demographics.parquet"),
                CatalogConcept(concept_id="sex", file_name="demographics.parquet"),
                CatalogConcept(concept_id="death", file_name="outcome.parquet"),
                CatalogConcept(
                    concept_id="sep3_sofa2",
                    file_name="sepsis3_sofa2.parquet",
                ),
            ],
        ),
    )

    def upsert(raw: dict[str, Any], **_kwargs: Any) -> dict[str, Any]:
        writes.append(dict(raw))
        return {**current, **raw, "revision": 3}

    monkeypatch.setattr(tool_module.study_contexts, "upsert_context", upsert)
    context = ToolExecutionContext(
        session=PiSessionRecord(
            session_id="pi-execution-bind",
            binding=AuthorityBinding(
                study_context_id=current["id"],
                study_revision=current["revision"],
            ),
        ),
        allowed_actions={"configure"},
    )

    result = tool_module.execute_tool(
        "easyicu_update_study_context",
        {
            "outcome": "In-hospital mortality",
            "primary_exposure": (
                "Experimental SOFA-2 Sepsis sensitivity phenotype in the first 24 hours"
            ),
            "covariates": ["Age", "Sex"],
            "execution_concepts": {
                "outcome": "death",
                "primary_exposure": "sep3_sofa2",
                "covariates": ["age", "sex"],
            },
        },
        context,
    )

    assert result["code"] == "study_context_updated"
    assert writes[0]["outcome"] == "In-hospital mortality"
    assert writes[0]["execution_concepts"] == {
        "outcome": "death",
        "primary_exposure": "sep3_sofa2",
        "covariates": ["age", "sex"],
    }
    assert writes[0]["modules"] == [
        "demographics",
        "outcome",
        "sepsis3_sofa2",
    ]


def test_conversational_setup_surfaces_repeated_stay_dependence_gate(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        tool_module.study_contexts,
        "_CONFIG_PATH",
        tmp_path / "cfg" / "study-contexts.json",
    )
    current = tool_module.study_contexts.upsert_context(
        {
            "id": "study-repeat-stays",
            "question": "Is ICU sepsis associated with hospital mortality?",
            "cohort": {"age_min": 18, "exclude_readmissions": False},
        },
        active=True,
    )
    monkeypatch.setattr(
        tool_module,
        "_bound_context",
        lambda _binding: tool_module.study_contexts.get_context(current["id"]),
    )
    context = ToolExecutionContext(
        session=PiSessionRecord(
            session_id="pi-repeat-stays",
            binding=AuthorityBinding(
                study_context_id=current["id"],
                study_revision=current["revision"],
            ),
        ),
        allowed_actions={"configure"},
    )

    result = tool_module.execute_tool(
        "easyicu_update_study_context",
        {
            "analysis_design": {
                "analysis_unit": "icu_stay",
                "variance_estimator": "model_based",
            }
        },
        context,
    )

    assert result["status"] == "blocked"
    assert result["code"] == "study_repeated_stay_dependence_unaddressed"
    assert result["details"]["field"] == "analysis_design"
    persisted = tool_module.study_contexts.get_context(current["id"])
    assert persisted is not None
    assert persisted["analysis_design"] == {}
    assert persisted["revision"] == current["revision"]


def test_conversational_setup_merges_partial_nested_scientific_patch(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    current = {
        "id": "study-nested-patch",
        "revision": 6,
        "question": "Is an ICU phenotype associated with mortality?",
        "active_job_id": None,
        "cohort": {
            "preset": "adult_icu_stays",
            "label": "All eligible adult ICU stays",
            "age_min": 18,
            "exclude_readmissions": False,
            "comparison_mode": "exposure_status",
        },
        "confirmations": {
            "adult_age_min_18": True,
            "first_eligible_icu_stay_only": False,
        },
    }
    writes: list[dict[str, Any]] = []
    monkeypatch.setattr(tool_module, "_bound_context", lambda _binding: current)

    def upsert(raw: dict[str, Any], **_kwargs: Any) -> dict[str, Any]:
        writes.append(dict(raw))
        return {**current, **raw, "revision": 7}

    monkeypatch.setattr(tool_module.study_contexts, "upsert_context", upsert)
    result = tool_module.execute_tool(
        "easyicu_update_study_context",
        {
            "cohort": {"exclude_readmissions": True},
            "confirmations": {"first_eligible_icu_stay_only": True},
        },
        ToolExecutionContext(
            session=PiSessionRecord(
                session_id="pi-nested-patch",
                binding=AuthorityBinding(
                    study_context_id=current["id"],
                    study_revision=current["revision"],
                ),
            ),
            allowed_actions={"configure"},
        ),
    )

    assert result["code"] == "study_context_updated"
    assert writes[0]["cohort"] == {
        "preset": "adult_icu_stays",
        "label": "All eligible adult ICU stays",
        "age_min": 18,
        "exclude_readmissions": True,
        "comparison_mode": "exposure_status",
    }
    assert writes[0]["confirmations"] == {
        "adult_age_min_18": True,
        "first_eligible_icu_stay_only": True,
    }


def test_conversational_setup_rejects_prose_extraction_cohort_preset(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    current = {
        "id": "study-prose-cohort-preset",
        "revision": 3,
        "question": "Describe an adult ICD cohort.",
        "active_job_id": None,
        "cohort": {},
    }
    writes: list[dict[str, Any]] = []
    monkeypatch.setattr(tool_module, "_bound_context", lambda _binding: current)
    monkeypatch.setattr(
        tool_module.study_contexts,
        "upsert_context",
        lambda raw, **_kwargs: writes.append(dict(raw)) or raw,
    )

    result = tool_module.execute_tool(
        "easyicu_update_study_context",
        {"cohort": {"preset": "ICU stay; age >= 18; ICD-10 prefix A41"}},
        ToolExecutionContext(
            session=PiSessionRecord(
                session_id="pi-prose-cohort-preset",
                binding=AuthorityBinding(
                    study_context_id=current["id"],
                    study_revision=current["revision"],
                ),
            ),
            allowed_actions={"configure"},
        ),
    )

    assert result["status"] == "blocked"
    assert result["code"] == "unsupported_cohort_preset"
    assert result["owner"] == "easyicu.webserver.dataio"
    assert result["details"]["field"] == "cohort.preset"
    assert writes == []


def test_conversational_setup_rejects_generic_sepsis_sofa2_drift(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from easyicu.research_agent.acquisition import catalog as catalog_module
    from easyicu.research_agent.acquisition.catalog import (
        AvailableCatalog,
        CatalogConcept,
    )

    current = {
        "id": "study-generic-sepsis",
        "revision": 2,
        "question": "Is standard Sepsis-3 associated with mortality?",
        "active_job_id": None,
        "data_source": {"path": "/private/full-export", "database": "miiv"},
        "modules": ["outcome", "sepsis3_sofa2"],
    }
    writes: list[dict[str, Any]] = []
    monkeypatch.setattr(tool_module, "_bound_context", lambda binding: dict(current))
    monkeypatch.setattr(
        catalog_module,
        "build_available_catalog",
        lambda _path: AvailableCatalog(
            source="typed-demo",
            concepts=[
                CatalogConcept(concept_id="death", file_name="outcome.parquet"),
                CatalogConcept(
                    concept_id="sep3_sofa2",
                    file_name="sepsis3_sofa2.parquet",
                ),
            ],
        ),
    )
    monkeypatch.setattr(
        tool_module.study_contexts,
        "upsert_context",
        lambda raw, **_kwargs: writes.append(dict(raw)) or raw,
    )
    context = ToolExecutionContext(
        session=PiSessionRecord(
            session_id="pi-generic-sepsis",
            binding=AuthorityBinding(
                study_context_id=current["id"],
                study_revision=current["revision"],
            ),
        ),
        allowed_actions={"configure"},
    )

    result = tool_module.execute_tool(
        "easyicu_update_study_context",
        {
            "primary_exposure": "Sepsis-3 using experimental SOFA-2",
            "execution_concepts": {
                "outcome": "death",
                "primary_exposure": "sep3_sofa2",
                "covariates": [],
            },
        },
        context,
    )

    assert result["code"] == "concept_explicit_selection_required"
    assert result["details"]["canonical_alternative"] == "sep3_sofa1"
    assert writes == []


def test_conversational_setup_persists_host_verified_explicit_concept_authorization(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from easyicu.research_agent.acquisition import catalog as catalog_module
    from easyicu.research_agent.acquisition.catalog import AvailableCatalog, CatalogConcept

    current = {
        "id": "study-explicit-sepsis",
        "revision": 2,
        "question": "What is Sepsis-3 prevalence and mortality?",
        "active_job_id": None,
        "data_source": {"path": "/private/full-export", "database": "miiv"},
        "modules": ["outcome", "sepsis3_sofa2"],
        "confirmations": {
            "feature_time_window": True,
            "export_format": True,
        },
    }
    writes: list[dict[str, Any]] = []
    monkeypatch.setattr(tool_module, "_bound_context", lambda binding: dict(current))
    monkeypatch.setattr(
        catalog_module,
        "build_available_catalog",
        lambda _path: AvailableCatalog(
            source="typed-demo",
            concepts=[
                CatalogConcept(concept_id="death", file_name="outcome.parquet"),
                CatalogConcept(
                    concept_id="sep3_sofa2",
                    file_name="sepsis3_sofa2.parquet",
                ),
            ],
        ),
    )
    monkeypatch.setattr(
        tool_module.study_contexts,
        "upsert_context",
        lambda raw, **_kwargs: writes.append(dict(raw)) or {**raw, "revision": 3},
    )
    context = ToolExecutionContext(
        session=PiSessionRecord(
            session_id="pi-explicit-sepsis",
            binding=AuthorityBinding(
                study_context_id=current["id"],
                study_revision=current["revision"],
            ),
        ),
        user_message=(
            "I explicitly authorize the experimental sep3_sofa2 variant for this study."
        ),
        allowed_actions={"configure"},
    )

    result = tool_module.execute_tool(
        "easyicu_update_study_context",
        {
            "question": (
                "Do not filter the cohort using Sepsis-3; compare the "
                "sep3_sofa2 groups descriptively."
            ),
            "execution_concepts": {
                "outcome": "death",
                "primary_exposure": "sep3_sofa2",
                "covariates": [],
            },
        },
        context,
    )

    assert result["code"] == "study_context_updated"
    assert writes[0]["confirmations"] == {
        "feature_time_window": True,
        "export_format": True,
        "concept_selection_sep3_sofa2_authorized": True,
    }


def test_conversational_setup_binds_exact_registered_source_id(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    current = {
        "id": "study-source-bind",
        "revision": 2,
        "question": "Does an aggregate ICU feature predict mortality?",
        "active_job_id": None,
        "confirmations": {"feature_time_window": True},
    }
    writes: list[dict[str, Any]] = []
    monkeypatch.setattr(tool_module, "_bound_context", lambda binding: dict(current))
    monkeypatch.setattr(
        tool_module.sources,
        "load_registry",
        lambda: {
            "active_path": "/private/full-export",
            "sources": [
                {
                    "id": "src_demo",
                    "path": "/private/demo-export",
                    "label": "MIMIC-IV Clinical Database Demo v2.2",
                    "database": "miiv",
                    "ok": True,
                }
            ],
        },
    )

    def upsert(raw: dict[str, Any], **kwargs: Any) -> dict[str, Any]:
        writes.append(dict(raw))
        return {**current, **raw, "revision": 3}

    monkeypatch.setattr(tool_module.study_contexts, "upsert_context", upsert)
    context = ToolExecutionContext(
        session=PiSessionRecord(
            session_id="pi-source-bind",
            binding=AuthorityBinding(
                study_context_id="study-source-bind",
                study_revision=2,
            ),
        ),
        allowed_actions={"configure"},
    )

    result = tool_module.execute_tool(
        "easyicu_update_study_context",
        {"bind_source_id": "src_demo", "bind_active_export": True},
        context,
    )

    assert result["code"] == "study_context_updated"
    assert writes[0]["data_source"] == {
        "path": "/private/demo-export",
        "label": "MIMIC-IV Clinical Database Demo v2.2",
        "database": "miiv",
    }
    assert writes[0]["confirmations"] == {
        "feature_time_window": True,
        "extraction_completed": True,
    }
    assert result["details"]["study"]["data_source"]["database"] == "miiv"
    assert len(result["details"]["study"]["data_source"]["path_digest"]) == 32
    assert "/private/" not in json.dumps(result)


def test_confirmed_conversation_source_is_not_silently_rebound(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    current = {
        "id": "study-confirmed-source",
        "revision": 2,
        "question": "",
        "data_source": {
            "path": "/private/user-selected-source",
            "label": "MIMIC-IV",
            "database": "miiv",
        },
        "active_job_id": None,
    }
    writes: list[dict[str, Any]] = []
    monkeypatch.setattr(tool_module, "_bound_context", lambda binding: dict(current))
    monkeypatch.setattr(
        tool_module.sources,
        "load_registry",
        lambda: {
            "active_path": "/private/model-recommended-source",
            "sources": [
                {
                    "id": "src_recommended",
                    "path": "/private/model-recommended-source",
                    "label": "MIMIC-IV full export",
                    "database": "miiv",
                    "ok": True,
                }
            ],
        },
    )

    def upsert(raw: dict[str, Any], **kwargs: Any) -> dict[str, Any]:
        writes.append(dict(raw))
        return {**current, **raw, "revision": 3}

    monkeypatch.setattr(tool_module.study_contexts, "upsert_context", upsert)
    session = PiSessionRecord(
        session_id="pi-confirmed-source",
        binding=AuthorityBinding(
            study_context_id="study-confirmed-source",
            study_revision=2,
        ),
        data_source_authorization=PiSessionDataSourceAuthorization(
            status="confirmed",
            confirmation_mode="select_local_source",
            source={
                "label": "MIMIC-IV",
                "database": "miiv",
                "identity_sha256": "a" * 64,
                "study_revision": 2,
            },
            confirmed_at="2026-08-25T00:00:00Z",
        ),
    )
    context = ToolExecutionContext(
        session=session,
        user_message="我想研究 MIMIC-IV 成人 ICU 人群中的 Sepsis-3 患病率。",
        allowed_actions={"configure"},
    )

    result = tool_module.execute_tool(
        "easyicu_update_study_context",
        {
            "question": "成人 ICU 人群中的 Sepsis-3 患病率是多少？",
            "bind_source_id": "src_recommended",
        },
        context,
    )

    assert result["code"] == "study_context_updated"
    assert writes[0]["question"] == "成人 ICU 人群中的 Sepsis-3 患病率是多少？"
    assert "data_source" not in writes[0]
    assert result["details"]["study"]["data_source"]["database"] == "miiv"


def test_workspace_tools_are_project_scoped_and_reuse_one_turn_write_grant(
    tmp_path: Path,
) -> None:
    research = PiSessionRecord(
        session_id="pi-research",
        project_id="project-a",
        agent_mode="research",
    )
    research_context = ToolExecutionContext(
        session=research,
        workspace_root=tmp_path,
    )
    blocked_mode = tool_module.execute_tool(
        "easyicu_read_project_file",
        {"file": "index.html"},
        research_context,
    )
    assert blocked_mode["code"] == "pi_workspace_mode_required"

    workspace = PiSessionRecord(
        session_id="pi-workspace",
        project_id="project-a",
        agent_mode="workspace",
    )
    no_grant = ToolExecutionContext(session=workspace, workspace_root=tmp_path)
    blocked_write = tool_module.execute_tool(
        "easyicu_write_project_file",
        {"file": "index.html", "content": "<h1>Blocked</h1>"},
        no_grant,
    )
    assert blocked_write["code"] == "pi_workspace_write_authorization_required"

    granted = ToolExecutionContext(
        session=workspace,
        allowed_actions={"workspace_write"},
        workspace_root=tmp_path,
    )
    first = tool_module.execute_tool(
        "easyicu_write_project_file",
        {"file": "index.html", "content": "<h1>Draft</h1>"},
        granted,
    )
    second = tool_module.execute_tool(
        "easyicu_edit_project_file",
        {
            "file": "index.html",
            "old_text": "Draft",
            "new_text": "Ready",
            "expected_sha256": first["details"]["sha256"],
        },
        granted,
    )
    checked = tool_module.execute_tool(
        "easyicu_check_project_file",
        {"file": "index.html"},
        granted,
    )
    preview = tool_module.execute_tool(
        "easyicu_preview_project_file",
        {
            "file": "index.html",
            "checked_sha256": checked["details"]["checked_sha256"],
        },
        granted,
    )

    assert first["code"] == "pi_workspace_file_written"
    assert second["code"] == "pi_workspace_file_edited"
    assert checked["details"]["valid"] is True
    assert first["details"]["mutation_receipt"]["ordinal"] == 1
    assert second["details"]["mutation_receipt"]["ordinal"] == 2
    assert preview["details"]["resource"] == {
        "kind": "webpage",
        "file": "index.html",
        "label": "index.html",
        "media_type": "text/html",
        "checked_sha256": checked["details"]["checked_sha256"],
        "authority_class": "workspace_artifact",
        "scientific_evidence": False,
        "validation_status": "unvalidated",
        "claim_ceiling": "unsupported",
    }
    assert (
        "Ready"
        in tool_module.execute_tool(
            "easyicu_read_project_file", {"file": "index.html"}, granted
        )["details"]["text"]
    )


def test_workspace_tool_rejects_path_escape_before_host_file_access(
    tmp_path: Path,
) -> None:
    session = PiSessionRecord(
        session_id="pi-workspace",
        project_id="project-a",
        agent_mode="workspace",
    )
    context = ToolExecutionContext(
        session=session,
        allowed_actions={"workspace_write"},
        workspace_root=tmp_path,
    )
    with pytest.raises(PiCopilotError) as raised:
        tool_module.execute_tool(
            "easyicu_write_project_file",
            {"file": "../outside.html", "content": "blocked"},
            context,
        )
    assert raised.value.code == "pi_workspace_path_escape"


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
        "primary_exposure": "lactate",
        "covariates": ["age", "sex"],
        "time_window": {"hours": 24, "anchor": "ICU admission"},
        "export_format": "parquet",
        "analysis_goal": "Adjusted association",
        "sensitivity_specs": [
            {
                "spec_id": "landmark_24h",
                "axis": "timing",
                "strategy": "landmark",
                "landmark_hours": 24,
                "require_alive_at_landmark": True,
                "exclude_negative_event_times": True,
            }
        ],
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
    assert "rebind" not in saved["summary"].lower()
    assert "continue" not in saved["summary"].lower()
    assert saved["details"]["workflow"]["current_stage"] == "setup"
    assert "question" not in saved["details"]["workflow"]["missing_setup_fields"]
    assert "data_source" in saved["details"]["workflow"]["missing_setup_fields"]
    assert saved["details"]["rebind_required"] is True
    assert saved["details"]["host_rebind_after_turn"] is True
    assert writes[0][0]["id"] == "study-1"
    assert writes[0][0]["primary_exposure"] == "lactate"
    assert writes[0][0]["covariates"] == ["age", "sex"]
    assert writes[0][0]["sensitivity_specs"][0]["spec_id"] == "landmark_24h"
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
    assert grant.consume_once("configure") == "granted"
    assert grant.consume_once("configure") == "consumed"
    assert grant.consume_once("run") == "missing"
    workspace_grant = HostTurnGrant.from_actions(["workspace_write"])
    assert workspace_grant.has_capability("workspace_write") is True
    assert workspace_grant.has_capability("configure") is False
    assert workspace_grant.consume_once("workspace_write") == "capability"
    receipts = [
        workspace_grant.reserve_workspace_mutation(
            "write" if index % 2 == 0 else "edit"
        )
        for index in range(8)
    ]
    assert receipts[-1].ordinal == 8
    with pytest.raises(WorkspaceMutationLimitError, match="limit"):
        workspace_grant.reserve_workspace_mutation("write")
    literature_grant = HostTurnGrant.from_actions(["literature"])
    assert literature_grant.was_provided("literature") is True
    assert literature_grant.consume_once("literature") == "granted"
    assert literature_grant.was_provided("literature") is True


def test_rejected_sensitivity_does_not_consume_configure_grant(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    current = {
        "id": "study-1",
        "revision": 5,
        "question": "Is sepsis associated with mortality?",
        "active_job_id": None,
    }
    session = PiSessionRecord(
        session_id="pi-test",
        binding=AuthorityBinding(study_context_id="study-1", study_revision=5),
    )
    monkeypatch.setattr(tool_module, "_bound_context", lambda binding: dict(current))
    writes = []

    def upsert(raw, **kwargs):
        writes.append(dict(raw))
        return {**current, **raw, "revision": 6}

    monkeypatch.setattr(tool_module.study_contexts, "upsert_context", upsert)
    execution = ToolExecutionContext(
        session=session,
        allowed_actions=frozenset({"configure"}),
    )

    rejected = tool_module.execute_tool(
        "easyicu_update_study_context",
        {
            "sensitivity_specs": [
                {
                    "spec_id": "first_stay",
                    "axis": "repeated_stays",
                    "strategy": "first_stay",
                    "execution_variables": [],
                }
            ]
        },
        execution,
    )

    assert rejected["code"] == "study_sensitivity_specs_invalid"
    assert (
        "first_stay sensitivity requires execution_variables"
        in rejected["details"]["reason"]
    )
    assert "configure" in execution.allowed_actions
    assert writes == []

    saved = tool_module.execute_tool(
        "easyicu_update_study_context",
        {
            "sensitivity_specs": [
                {
                    "spec_id": "first_stay",
                    "axis": "repeated_stays",
                    "strategy": "first_stay",
                    "execution_variables": ["icu_readmission"],
                }
            ]
        },
        execution,
    )
    assert saved["code"] == "study_context_updated"
    assert writes[0]["sensitivity_specs"][0]["execution_variables"] == [
        "icu_readmission"
    ]


def test_unsupported_runtime_design_does_not_consume_configure_grant(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    current = {
        "id": "study-1",
        "revision": 5,
        "question": "Is sepsis associated with mortality?",
        "outcome": "in-hospital mortality",
        "primary_exposure": "Sepsis-3",
        "data_source": {"path": "/private/export", "database": "miiv"},
        "cohort": {"exclude_readmissions": False},
        "active_job_id": None,
    }
    session = PiSessionRecord(
        session_id="pi-test",
        binding=AuthorityBinding(study_context_id="study-1", study_revision=5),
    )
    monkeypatch.setattr(tool_module, "_bound_context", lambda binding: dict(current))
    writes = []

    def upsert(raw, **kwargs):
        writes.append(dict(raw))
        return {**current, **raw, "revision": 6}

    monkeypatch.setattr(tool_module.study_contexts, "upsert_context", upsert)
    execution = ToolExecutionContext(
        session=session,
        allowed_actions=frozenset({"configure"}),
    )

    blocked = tool_module.execute_tool(
        "easyicu_update_study_context",
        {
            "cohort": {"exclude_readmissions": False},
            "analysis_design": {
                "analysis_unit": "icu_stay",
                "variance_estimator": "cluster_robust",
                "cluster_unit": "patient",
            },
        },
        execution,
    )
    assert blocked["code"] == "research_pipeline_cluster_variance_unsupported"
    assert blocked["details"]["first_stay_restriction_status"] == (
        "unverified_in_selected_export"
    )
    assert "configure" in execution.allowed_actions
    assert writes == []

    monkeypatch.setattr(
        agent_pipeline_runs.source_identity_authority,
        "resolve_patient_grouping_authority",
        lambda **_kwargs: PatientGroupingBinding(
            mapping_path=Path("/private/mapping.parquet"),
            mapping_sha256="a" * 64,
            mapping_stay_column="stay_id",
            mapping_patient_column="patient_key",
            authority_coordinates={"authority_ref": "owner/bridge/v1"},
        ),
    )
    saved = tool_module.execute_tool(
        "easyicu_update_study_context",
        {
            "cohort": {"exclude_readmissions": False},
            "analysis_design": {
                "analysis_unit": "icu_stay",
                "variance_estimator": "cluster_robust",
                "cluster_unit": "patient",
            },
        },
        execution,
    )
    assert saved["code"] == "study_context_updated"
    assert writes[0]["analysis_design"]["variance_estimator"] == "cluster_robust"


def test_preflight_delegates_to_the_existing_agent_submission_owner(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    submitted_bodies = []

    def submit(body, *, account_environment=None):
        assert account_environment is None
        submitted_bodies.append(dict(body))
        return {
            "job_id": "scientific-job-1",
            "kind": "agent-run",
            "status": "running",
            "study_context_id": "study-1",
            "study_context_revision": 8,
        }

    from easyicu.webserver.routes import agent as agent_route

    monkeypatch.setattr(agent_route, "submit_agent_run", submit)
    monkeypatch.setattr(
        tool_module,
        "_bound_context",
        lambda binding: {
            "id": "study-1",
            "revision": 7,
            "question": "Aggregate association question",
            "data_source": {
                "path": "/private/project-export",
                "database": "miiv",
            },
        },
    )
    monkeypatch.setattr(
        tool_module,
        "_workflow_snapshot",
        lambda _context, *, study_override=None: {
            "next_action_code": "provider_ready_to_generate_plan",
            "missing_setup_fields": [],
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
            "path": "/private/project-export",
            "study_context_id": "study-1",
            "question": "Aggregate association question",
            "run_type": "preflight",
            "llm_provider": "mock",
            "external_llm_opt_in": False,
            "engine": "native_summary",
        }
    ]

    full_context = ToolExecutionContext(
        session=PiSessionRecord(
            session_id="pi-full-test",
            external_llm_opt_in=True,
            binding=AuthorityBinding(study_context_id="study-1", study_revision=7),
        ),
        allowed_actions=frozenset({"provider_run"}),
    )
    full_result = tool_module.execute_tool("easyicu_run", {}, full_context)

    assert full_result["code"] == "easyicu_full_run_submitted"
    assert full_result["details"]["run_id_status"] == "pending_pipeline_start"
    assert submitted_bodies[-1] == {
        "path": "/private/project-export",
        "study_context_id": "study-1",
        "question": "Aggregate association question",
        "run_type": "full",
        "llm_provider": "openai",
        "external_llm_opt_in": True,
        "engine": "research_agent_pipeline",
        "credential_source": "pi_verified",
    }

    conservative_model_context = ToolExecutionContext(
        session=PiSessionRecord(
            session_id="pi-conservative-model-test",
            external_llm_opt_in=True,
            binding=AuthorityBinding(study_context_id="study-1", study_revision=7),
        ),
        allowed_actions=frozenset({"provider_run"}),
    )
    promoted_result = tool_module.execute_tool(
        "easyicu_run",
        {"run_type": "preflight"},
        conservative_model_context,
    )

    assert promoted_result["code"] == "easyicu_full_run_submitted"
    assert promoted_result["details"]["run_id_status"] == "pending_pipeline_start"
    assert submitted_bodies[-1]["run_type"] == "full"
    assert submitted_bodies[-1]["engine"] == "research_agent_pipeline"

    literature_context = ToolExecutionContext(
        session=PiSessionRecord(
            session_id="pi-full-literature-test",
            external_llm_opt_in=True,
            binding=AuthorityBinding(study_context_id="study-1", study_revision=7),
        ),
        allowed_actions=frozenset({"provider_run", "literature"}),
    )
    literature_result = tool_module.execute_tool("easyicu_run", {}, literature_context)

    assert literature_result["code"] == "easyicu_full_run_submitted"
    assert submitted_bodies[-1]["literature_search_authorized"] is True


def test_plan_review_run_projection_cannot_be_mistaken_for_executed_analysis(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    run_row = {
        "run_id": "run-plan-review",
        "study_id": "study-1",
        "gate_status": "blocked",
        "run_status": "human_review_pending",
        "pending_review_reason_codes": ["operator_plan_approval_required"],
        "reportable": False,
        "artifact_names": [
            "agent_plan.json",
            "result_tables.json",
            "manuscript_draft.json",
        ],
    }

    projected = project_run_row(run_row)
    assert projected["execution_phase"] == "plan_review"
    assert projected["human_plan_review_pending"] is True
    assert projected["analysis_executed"] is False
    assert projected["scientific_results_available"] is False
    assert (
        projected["artifact_semantics"]
        == "plan_stage_placeholders_not_analysis_results"
    )

    monkeypatch.setattr(tool_module, "_run_rows", lambda _context: [run_row])
    result = tool_module.execute_tool(
        "easyicu_inspect_run",
        {},
        ToolExecutionContext(
            session=PiSessionRecord(
                session_id="pi-plan-review",
                binding=AuthorityBinding(
                    study_context_id="study-1",
                    study_revision=1,
                    run_id="run-plan-review",
                ),
            )
        ),
    )

    assert "paused at the human plan-review gate" in result["summary"]
    assert "analysis has not executed" in result["summary"]
    assert result["details"]["run"]["analysis_executed"] is False


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
            "primary_exposure": "lactate",
            "covariates": ["age", "sex"],
            "sensitivity_specs": [
                {
                    "spec_id": "landmark_24h",
                    "axis": "timing",
                    "strategy": "landmark",
                    "execution_variables": [],
                    "landmark_hours": 24,
                    "require_alive_at_landmark": True,
                    "exclude_negative_event_times": True,
                }
            ],
            "data_source": {"database": "mimiciv", "path": "/private/export"},
            "cohort": {"cohort_size": 140},
            "literature_authority": {
                "schema_version": "easyicu.web-literature-authority/2",
                "receipt_id": "lit_" + "a" * 24,
                "receipt_sha256": "b" * 64,
                "status": "searched",
                "result_count": 3,
                "searched_at": "2026-08-12T12:00:00+00:00",
                "study_configuration_sha256": "c" * 64,
            },
        }
    )
    assert "/private/export" not in json.dumps(projected)
    assert len(projected["data_source"]["path_digest"]) == 32
    assert projected["primary_exposure"] == "lactate"
    assert projected["covariates"] == ["age", "sex"]
    assert projected["sensitivity_specs"][0]["spec_id"] == "landmark_24h"
    assert projected["literature_authority"]["result_count"] == 3
    assert "/private/" not in json.dumps(projected)

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


def test_run_artifact_tools_emit_path_free_clickable_resources(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    context = ToolExecutionContext(
        session=PiSessionRecord(
            session_id="pi-artifacts",
            binding=AuthorityBinding(run_id="run_20260808"),
        )
    )
    monkeypatch.setattr(
        tool_module.agent_runs,
        "list_run_history",
        lambda **kwargs: {
            "runs": [
                {
                    "run_id": "run_20260808",
                    "project_dir": "/private/owner-only/run",
                }
            ]
        },
    )
    monkeypatch.setattr(
        tool_module.agent_runs,
        "read_run_review",
        lambda project_dir: {
            "ok": True,
            "artifacts": [
                {
                    "name": "table1_summary.json",
                    "bytes": 412,
                    "sha256": "a" * 64,
                    "kind": "json",
                    "path": "/private/owner-only/run/table1_summary.json",
                },
                {
                    "name": "quality_gate.json",
                    "bytes": 233,
                    "sha256": "b" * 64,
                    "kind": "json",
                },
            ],
            "artifact_payloads": {},
        },
    )

    result = tool_module.execute_tool("easyicu_list_artifacts", {}, context)

    assert result["details"]["artifacts"][0]["size"] == 412
    assert result["details"]["artifacts"][0]["media_type"] == "application/json"
    assert result["details"]["resources"][0] == {
        "kind": "research_artifact",
        "run_id": "run_20260808",
        "artifact": "table1_summary.json",
        "label": "table1_summary.json",
        "media_type": "application/json",
        "sha256": "a" * 64,
    }
    assert "project_dir" not in json.dumps(result)
    assert "/private/" not in json.dumps(result)


def test_project_artifact_preview_resolves_authority_and_scrubs_host_paths(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    service = PiCopilotService(
        store_path=tmp_path / "sessions.json",
        gateway=FakeGateway(),
    )
    service.project_store.bind("project-a", "study-a")
    service.project_store.bind("project-b", "study-b")

    def history(*, study_id: str, **kwargs: Any) -> dict[str, Any]:
        return {
            "runs": (
                [{"run_id": "run_20260808", "project_dir": "/private/run-a"}]
                if study_id == "study-a"
                else []
            )
        }

    monkeypatch.setattr(service_module.agent_runs, "list_run_history", history)
    monkeypatch.setattr(
        service_module.agent_runs,
        "read_run_artifact",
        lambda project_dir, artifact_name: {
            "ok": True,
            "artifact": {
                "name": artifact_name,
                "path": f"{project_dir}/{artifact_name}",
                "bytes": 120,
                "sha256": "c" * 64,
                "kind": "json",
            },
            "payload": {
                "status": "ready",
                "source": {"path": "/private/export", "database": "mimiciv"},
                "future_paths": {
                    "artifact_path": "/private/future.json",
                    "output_dir": "/private/output",
                    "cache_file": "/private/cache.bin",
                    "cwd": "/private/work",
                },
                "figures": [{"relative_path": "figures/roc.svg"}],
            },
            "privacy_scan": {"passed": True},
        },
    )
    monkeypatch.setattr(
        service_module.agent_runs,
        "read_run_review",
        lambda project_dir: {
            "ok": True,
            "gate": {"status": "analysis_only"},
            "readiness": {
                "status": "awaiting_human_signoff",
                "signed": False,
                "signoff_stale": False,
                "reportable": False,
            },
        },
    )

    payload = service.get_research_artifact(
        project_id="project-a",
        run_id="run_20260808",
        artifact_name="table1_summary.json",
    )
    encoded = json.dumps(payload)
    assert payload["payload"]["source"] == {"database": "mimiciv"}
    assert payload["payload"]["future_paths"] == {}
    assert payload["payload"]["figures"][0]["relative_path"] == "figures/roc.svg"
    assert payload["governance"] == {
        "authority_class": "easyicu_run_artifact",
        "artifact_integrity": "unsigned",
        "gate_status": "analysis_only",
        "readiness_status": "awaiting_human_signoff",
        "human_signoff": "required",
        "reportable": False,
        "claim_ceiling": "analysis_only",
    }
    assert "project_dir" not in encoded
    assert "/private/" not in encoded

    with pytest.raises(PiCopilotError) as wrong_project:
        service.get_research_artifact(
            project_id="project-b",
            run_id="run_20260808",
            artifact_name="table1_summary.json",
        )
    assert wrong_project.value.code == "pi_research_run_not_found"

    monkeypatch.setattr(
        service_module.agent_runs,
        "read_run_artifact",
        lambda project_dir, artifact_name: {
            "ok": True,
            "artifact": {"name": artifact_name},
            "payload": {"status": "withheld"},
            "privacy_scan": {"passed": False},
        },
    )
    with pytest.raises(PiCopilotError) as privacy_blocked:
        service.get_research_artifact(
            project_id="project-a",
            run_id="run_20260808",
            artifact_name="table1_summary.json",
        )
    assert privacy_blocked.value.code == "pi_research_artifact_privacy_blocked"


def test_project_evidence_preview_is_project_scoped_digest_pinned_and_governed(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    service = PiCopilotService(
        store_path=tmp_path / "sessions.json",
        gateway=FakeGateway(),
    )
    service.project_store.bind("project-a", "study-a")
    digest = "d" * 64
    monkeypatch.setattr(
        service_module.agent_runs,
        "list_run_history",
        lambda **kwargs: {
            "runs": [{"run_id": "run_1", "project_dir": "/private/run-a"}]
            if kwargs.get("study_id") == "study-a"
            else []
        },
    )
    monkeypatch.setattr(
        service_module.agent_runs,
        "read_run_evidence_preview",
        lambda project_dir, evidence_id, expected_sha256: {
            "ok": True,
            "payload": {
                "schema_version": "easyicu.web-evidence-preview/1",
                "evidence_id": evidence_id,
                "sha256": expected_sha256,
                "display_name": "analysis.py",
                "renderer": "code",
                "previewable": True,
                "text": "estimate = 1.25\n",
            },
            "privacy_scan": {"passed": True},
        },
    )
    monkeypatch.setattr(
        service_module.agent_runs,
        "read_run_review",
        lambda project_dir: {
            "ok": True,
            "gate": {"status": "analysis_only"},
            "readiness": {
                "status": "awaiting_human_signoff",
                "signed": False,
                "signoff_stale": False,
                "reportable": False,
            },
        },
    )

    payload = service.get_research_evidence_preview(
        project_id="project-a",
        run_id="run_1",
        evidence_id="code_analysis_1",
        expected_sha256=digest,
    )
    encoded = json.dumps(payload)
    assert payload["payload"]["text"] == "estimate = 1.25\n"
    assert payload["payload"]["sha256"] == digest
    assert payload["governance"]["claim_ceiling"] == "analysis_only"
    assert "project_dir" not in encoded and "/private/" not in encoded

    with pytest.raises(PiCopilotError) as wrong_project:
        service.get_research_evidence_preview(
            project_id="project-b",
            run_id="run_1",
            evidence_id="code_analysis_1",
            expected_sha256=digest,
        )
    assert wrong_project.value.code == "pi_project_not_initialized"

    monkeypatch.setattr(
        service_module.agent_runs,
        "read_run_evidence_preview",
        lambda *args: {
            "ok": True,
            "payload": {},
            "privacy_scan": {"passed": False},
        },
    )
    with pytest.raises(PiCopilotError) as blocked:
        service.get_research_evidence_preview(
            project_id="project-a",
            run_id="run_1",
            evidence_id="code_analysis_1",
            expected_sha256=digest,
        )
    assert blocked.value.code == "pi_research_evidence_privacy_blocked"


def test_project_document_preview_requires_the_current_ledger_digest(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    service = PiCopilotService(
        store_path=tmp_path / "sessions.json",
        gateway=FakeGateway(),
    )
    service.project_store.bind("project-a", "study-a")
    document = b"<!doctype html><title>Bound report</title>"
    digest = hashlib.sha256(document).hexdigest()
    monkeypatch.setattr(
        service_module.agent_runs,
        "list_run_history",
        lambda **_kwargs: {
            "runs": [{"run_id": "run_20260808", "project_dir": "/private/run-a"}]
        },
    )
    monkeypatch.setattr(
        service_module.agent_runs,
        "read_run_artifact_bytes",
        lambda _project_dir, name: {
            "ok": True,
            "name": name,
            "content": document,
            "media_type": "text/html; charset=utf-8",
        },
    )
    review = {
        "ok": True,
        "gate": {"status": "blocked"},
        "readiness": {"status": "blocked", "reportable": False},
        "artifacts": [{"name": "system_validation_report.html", "sha256": digest}],
        "artifact_payloads": {
            "evidence_ledger.json": {
                "artifacts": [
                    {"name": "system_validation_report.html", "sha256": digest}
                ]
            }
        },
    }
    monkeypatch.setattr(
        service_module.agent_runs,
        "read_run_review",
        lambda _project_dir: review,
    )

    loaded = service.get_research_document(
        project_id="project-a",
        run_id="run_20260808",
        document_name="system_validation_report.html",
    )
    assert loaded["content"] == document
    assert loaded["claim_ceiling"] == "engineering_validation_only"

    review["artifact_payloads"]["evidence_ledger.json"]["artifacts"][0][
        "sha256"
    ] = "0" * 64
    with pytest.raises(PiCopilotError) as mismatch:
        service.get_research_document(
            project_id="project-a",
            run_id="run_20260808",
            document_name="system_validation_report.html",
        )
    assert mismatch.value.code == "pi_research_document_digest_mismatch"


def test_project_data_package_preview_is_revision_and_digest_bound(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    service = PiCopilotService(
        store_path=tmp_path / "sessions.json",
        gateway=FakeGateway(),
    )
    service.project_store.bind("project-a", "study-a")
    study = {
        "id": "study-a",
        "revision": 7,
        "data_source": {"path": "/private/export", "database": "miiv"},
    }
    monkeypatch.setattr(service_module.study_contexts, "get_context", lambda _id: study)
    from easyicu.webserver import data_package_review as review_owner

    review_payload = {
        "schema_version": "easyicu.data-package-review/1",
        "study_context_id": "study-a",
        "study_context_revision": 7,
        "status": "ready_for_plan",
        "source": {"database": "miiv"},
        "privacy": {
            "raw_rows_returned": False,
            "host_paths_returned": False,
        },
        "analysis_results_withheld": True,
    }
    review_digest = hashlib.sha256(
        json.dumps(
            review_payload,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    ).hexdigest()
    review_payload["review_sha256"] = review_digest
    monkeypatch.setattr(
        review_owner,
        "build_registered_data_package_review",
        lambda _study: dict(review_payload),
    )

    payload = service.get_data_package_review(
        project_id="project-a",
        study_revision=7,
        review_sha256=review_digest,
    )
    assert payload["payload"]["source"] == {"database": "miiv"}
    assert payload["governance"]["claim_ceiling"] == "pre_analysis_review"
    assert "/private/" not in json.dumps(payload)

    with pytest.raises(PiCopilotError) as stale:
        service.get_data_package_review(
            project_id="project-a", study_revision=6, review_sha256=review_digest
        )
    assert stale.value.code == "pi_data_package_review_snapshot_missing"

    with pytest.raises(PiCopilotError) as drift:
        service.get_data_package_review(
            project_id="project-a", study_revision=7, review_sha256="e" * 64
        )
    assert drift.value.code == "pi_data_package_review_digest_mismatch"


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
