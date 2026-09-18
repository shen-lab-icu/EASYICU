"""Pipeline route and launch authority for the Copilot research workflow.

Split from ``test_pi_copilot_research_workflow.py`` when that module crossed
its size ratchet. These tests cover the HTTP/route boundaries only: which
server-owned mode a launch may enter, which client fields are ignored, and
what a launch/signoff request may not authorize. Owner-facing workflow
projection tests stay in the original module.
"""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Mapping

import pandas as pd
import pytest
from starlette.requests import Request

from easyicu.research_agent.orchestration.workflow import (
    HumanReviewPending,
    HumanReviewRequest,
)
from easyicu.webserver import (
    agent_pipeline_runs,
    agent_runs,
    dataio,
    literature_authority,
    plan_change_requirements,
    provider_adapter,
    research_launch_resume,
    research_launch_scientific,
    research_pipeline_run_preparation,
    research_run_submission,
)
from easyicu.webserver import study_contexts as study_context_owner
from easyicu.webserver.pi_copilot.workflow import (
    active_export_matches_study,
    build_research_workflow_snapshot,
    registered_export_matches_study,
)
from tests.webserver.copilot.research_workflow_fixtures import (
    _acquisition_receipt,
    _foundation_profile as _foundation_profile,
    _write_development_resume_literature,
    _write_development_resume_planner_catalog,
    _write_real_pipeline_fixture,
)
from tests.webserver.copilot.research_workflow_fixtures import (
    complete_study as _complete_study,
    confirmed_cohort_decision as _confirmed_cohort_decision,
)


def _install_pending_review(monkeypatch, entry):
    registry = agent_pipeline_runs.PendingReviewRegistry()
    monkeypatch.setattr(agent_pipeline_runs, "_PENDING_REVIEWS", registry)
    registry.register(entry)
    return registry


def _request() -> Request:
    return Request(
        {
            "type": "http",
            "method": "POST",
            "path": "/api/jobs/agent-run",
            "raw_path": b"/api/jobs/agent-run",
            "headers": [],
            "query_string": b"",
            "scheme": "http",
            "server": ("testserver", 80),
            "client": ("testclient", 123),
        }
    )


def _write_pipeline_export(
    root: Path, *, database: str = "miiv", current_contract: bool = True
) -> Path:
    """Write a prepared export package manifest.

    ``current_contract=False`` reproduces an export written before intake began
    requiring per-file ``concept_ids``: the same directory is laid out correctly
    and passes the layout-only readiness probe, but cannot establish authority.
    """

    root.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({"stay_id": [1], "age": [65]}).to_parquet(
        root / "demographics.parquet", index=False
    )
    # Built in the original key order so the default path serializes the same
    # manifest bytes as before; the file entry joins the binding hash.
    file_entry: dict[str, Any] = {
        "file": "demographics.parquet",
        "module": "demographics",
        "concepts": 1,
    }
    if current_contract:
        file_entry["concept_ids"] = ["age"]
    file_entry["rows"] = 1
    (root / "_manifest.json").write_text(
        json.dumps(
            {
                "database": database,
                "format": "parquet",
                "concept_selection": {
                    "mode": "explicit",
                    "modules": {"demographics": ["age"]},
                },
                "feature_definitions": {"included": False},
                "files": [file_entry],
            }
        ),
        encoding="utf-8",
    )
    return root


def _assume_execution_runtime_ready(monkeypatch: pytest.MonkeyPatch) -> None:
    """An executing launch now probes the container runtime before it starts.

    That gate has its own contract tests in
    ``test_web_execution_runtime_preflight.py``; the launches here are about
    scope and resume authority and must not depend on the host's daemon.
    """

    from easyicu.research_agent.execution import runner as runner_module

    monkeypatch.setattr(
        runner_module,
        "probe_runner_availability",
        lambda kind, **_kwargs: runner_module.RunnerAvailability(
            kind=kind, available=True, image="easyicu-research-agent:test"
        ),
)


_PI_PROVIDER_ENVIRONMENT = {
    "OPENAI_API_KEY": "test-private-provider-key",
    "OPENAI_BASE_URL": "http://127.0.0.1:8317/v1",
    "OPENAI_MODEL": "test-local-model",
    "EASYICU_DISABLE_PROVIDER_ENV_FILE": "1",
}


@pytest.mark.parametrize("suffix", [".csv", ".xlsx"])
def test_pipeline_route_rejects_raw_tabular_files_before_provider_resolution(
    suffix: str,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from easyicu.webserver.routes import agent as agent_route

    raw = tmp_path / "raw"
    raw.mkdir()
    (raw / f"patients{suffix}").write_text("stay_id\n1\n", encoding="utf-8")
    study = {
        **_complete_study(),
        "data_source": {"path": str(raw), "database": "miiv"},
    }
    monkeypatch.setenv("EASYICU_DEVELOPMENT_REVIEWED_EXECUTION", "1")
    monkeypatch.setattr(agent_route.context_store, "get_context", lambda _id: study)
    provider_called = False

    def provider_environment(*_args: Any, **_kwargs: Any) -> dict[str, str]:
        nonlocal provider_called
        provider_called = True
        return dict(_PI_PROVIDER_ENVIRONMENT)

    monkeypatch.setattr(
        agent_route.PiProviderConfigStore,
        "research_agent_environment",
        provider_environment,
    )

    with pytest.raises(Exception) as raised:
        agent_route.jobs_agent_run(
            {
                "path": str(raw),
                "study_context_id": study["id"],
                "engine": "research_agent_pipeline",
                "run_type": "full",
                "credential_source": "pi_verified",
                "external_llm_opt_in": True,
            },
            request=_request(),
        )

    assert raised.value.detail["error"] in {
        "research_pipeline_manifest_required",
        "no_export_files",
    }
    assert provider_called is False


@pytest.mark.parametrize(
    (
        "planner_start_mode",
        "resume_source_job_id",
        "plan_revision_source_run_id",
        "requested_changes",
    ),
    [
        ("fresh", "", "", False),
        ("fresh", "", "", True),
        ("resume_checkpoint", "prior-canary", "", False),
        ("auto", "prior-canary", "run-reviewed-candidate", False),
    ],
)
def test_pipeline_route_ignores_client_project_root_and_uses_pi_workspace(
    planner_start_mode: str,
    resume_source_job_id: str,
    plan_revision_source_run_id: str,
    requested_changes: bool,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from easyicu.webserver.pi_copilot.workspace import ProjectWorkspace
    from easyicu.webserver.routes import agent as agent_route

    export = tmp_path / "raw-mimiciv"
    export.mkdir()
    (export / "patients.csv").write_text("stay_id\n1\n", encoding="utf-8")
    study = {
        **_complete_study(),
        "data_source": {"path": str(export), "database": "miiv"},
    }
    workspace = ProjectWorkspace(tmp_path / "pi-workspace")
    captured: dict[str, Any] = {}
    monkeypatch.setattr(agent_route.context_store, "get_context", lambda _id: study)
    monkeypatch.setattr(
        research_run_submission,
        "research_pipeline_workspace",
        lambda: workspace,
    )
    monkeypatch.setattr(
        research_run_submission,
        "resumable_planner_checkpoint_job_id",
        lambda **_kwargs: resume_source_job_id,
    )
    monkeypatch.setattr(
        research_run_submission,
        "_development_resume_launch_scope",
        lambda **_kwargs: SimpleNamespace(budget_mode="planner_canary", plan_contract=None),
    )
    monkeypatch.setattr(
        agent_route.PiProviderConfigStore,
        "research_agent_environment",
        lambda self, **_kwargs: dict(_PI_PROVIDER_ENVIRONMENT),
    )
    monkeypatch.setattr(agent_route.settings_store, "load_settings", lambda: {"ai_enabled": True})
    monkeypatch.setattr(
        agent_route.capabilities,
        "validate_compute_target",
        lambda _body: {"ok": True, "compute_target": "local"},
    )
    monkeypatch.setattr(
        agent_route.agent_runs,
        "resolve_agent_provider_config",
        lambda **_kwargs: {"provider": "openai", "external": True},
    )
    monkeypatch.setattr(
        agent_route.context_store,
        "build_agent_context_binding",
        lambda *_args, **_kwargs: {},
    )

    def make_runner(**kwargs: Any) -> Any:
        captured.update(kwargs)
        return lambda _job: {"gate": {"status": "blocked"}}

    monkeypatch.setattr(
        agent_route.agent_pipeline_runs,
        "make_research_pipeline_run_runner",
        make_runner,
    )
    monkeypatch.setattr(
        research_run_submission,
        "_submit_job",
        lambda _kind, _runner: SimpleNamespace(id="job-workspace", kind="agent-run", status="queued"),
    )
    monkeypatch.setattr(
        agent_route.context_store,
        "handoff_context",
        lambda *_args, **_kwargs: {"revision": 5},
    )
    monkeypatch.setattr(
        agent_route.capabilities,
        "record_tool_event",
        lambda *_args, **_kwargs: None,
    )

    payload = {
        "path": str(export),
        "study_context_id": study["id"],
        "engine": "research_agent_pipeline",
        "run_type": "full",
        "credential_source": "pi_verified",
        "external_llm_opt_in": True,
        "project_root": str(tmp_path / "client-controlled"),
        "planner_start_mode": planner_start_mode,
    }
    if resume_source_job_id:
        payload["development_resume_source_job_id"] = "client-forged-checkpoint"
    if plan_revision_source_run_id:
        payload["plan_revision_source_run_id"] = plan_revision_source_run_id
    if requested_changes:
        from easyicu.webserver.plan_change_request import PlanChangeRequest

        change = PlanChangeRequest(
            source_run_id="run-current-candidate", user_message="Retain all requested outcomes.",
        )
        result = research_run_submission.submit_research_run(
            research_run_submission.ResearchRunSubmissionRequest(
                study_context_id=study["id"], provider="openai",
                credential_source="pi_verified", external_llm_opt_in=True,
                intent="candidate_plan", planner_start_mode="fresh",
                plan_change_request=change,
            ),
        ).model_dump(mode="json")
        assert captured["plan_change_request"] == change
    else:
        result = agent_route.jobs_agent_run(payload, request=_request())
        assert "plan_change_request" not in captured

    assert result["job_id"] == "job-workspace"
    assert Path(captured["project_root"]) == workspace.project_root(study["id"])
    assert Path(captured["project_root"]) != tmp_path / "client-controlled"
    assert captured["budget_mode"] == "planner_canary"
    assert result["planner_start_mode"] == planner_start_mode
    if resume_source_job_id and not plan_revision_source_run_id:
        assert captured["development_resume_source_job_id"] == resume_source_job_id
        assert result["resume_source_job_id"] == resume_source_job_id
    else:
        assert "development_resume_source_job_id" not in captured
    if plan_revision_source_run_id:
        assert captured["plan_revision_source_run_id"] == plan_revision_source_run_id


def test_monitor_history_merges_default_and_copilot_pipeline_roots(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from easyicu.webserver.routes import agent as agent_route

    pipeline_root = tmp_path / "pi-project"
    pipeline_root.mkdir()
    calls: list[str | None] = []

    class Workspace:
        def existing_project_root(self, project_id: str) -> Path:
            assert project_id == "study-workflow"
            return pipeline_root

    def history(*, study_id: str, project_root: str | None = None, limit: int) -> dict[str, Any]:
        calls.append(project_root)
        if project_root is None:
            rows = [
                {
                    "run_id": "run_preflight",
                    "project_dir": str(tmp_path / "default" / "run_preflight"),
                    "updated_at_epoch": 10,
                }
            ]
        else:
            rows = [
                {
                    "run_id": "run_pipeline",
                    "project_dir": str(pipeline_root / "study-workflow" / "run_pipeline"),
                    "updated_at_epoch": 20,
                }
            ]
        return {"ok": True, "project_root": project_root or "default", "runs": rows, "count": len(rows)}

    monkeypatch.setattr(agent_route, "research_pipeline_workspace", lambda: Workspace())
    monkeypatch.setattr(agent_route.agent_runs, "list_run_history", history)
    result = agent_route.post_agent_run_history(
        {"study_id": "study-workflow", "limit": 50}
    )

    assert calls == [None, str(pipeline_root)]
    assert result["count"] == 2
    assert [row["run_id"] for row in result["runs"]] == [
        "run_pipeline",
        "run_preflight",
    ]


def test_pipeline_route_rejects_client_selected_full_reviewed_mode(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from easyicu.webserver.routes import agent as agent_route

    export = _write_pipeline_export(tmp_path / "export")
    study = {
        **_complete_study(),
        "data_source": {"path": str(export), "database": "miiv"},
    }
    monkeypatch.setattr(agent_route.context_store, "get_context", lambda _id: study)

    with pytest.raises(Exception) as raised:
        agent_route.jobs_agent_run(
            {
                "path": str(export),
                "study_context_id": study["id"],
                "engine": "research_agent_pipeline",
                "run_type": "full",
                "budget_mode": "full_reviewed",
            },
            request=_request(),
        )

    assert raised.value.detail == {
        "error": "research_pipeline_budget_mode_server_owned"
    }


def test_pipeline_development_execution_mode_is_server_owned(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from easyicu.webserver.routes import agent as agent_route

    monkeypatch.delenv("EASYICU_DEVELOPMENT_REVIEWED_EXECUTION", raising=False)
    assert agent_route._server_research_pipeline_budget_mode() == "planner_canary"

    monkeypatch.setenv("EASYICU_DEVELOPMENT_REVIEWED_EXECUTION", "1")
    assert agent_route._server_research_pipeline_budget_mode() == "full_reviewed"

    monkeypatch.setenv("EASYICU_DEVELOPMENT_REVIEWED_EXECUTION", "true")
    with pytest.raises(Exception) as raised:
        agent_route._server_research_pipeline_budget_mode()
    assert raised.value.detail == {
        "error": "research_pipeline_development_mode_invalid"
    }


def test_candidate_plan_click_stays_planner_only_with_or_without_prepared_package(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from easyicu.webserver.routes import agent as agent_route

    monkeypatch.setenv("EASYICU_DEVELOPMENT_REVIEWED_EXECUTION", "1")
    assert agent_route._research_pipeline_budget_mode_for_source(
        prepared_manifest=None,
        metadata_only_planning_authorized=True,
    ) == "planner_canary"
    assert agent_route._research_pipeline_budget_mode_for_source(
        prepared_manifest=tmp_path / "manifest.json",
        metadata_only_planning_authorized=True,
    ) == "planner_canary"
    monkeypatch.delenv("EASYICU_DEVELOPMENT_REVIEWED_EXECUTION", raising=False)
    assert agent_route._research_pipeline_budget_mode_for_source(
        prepared_manifest=tmp_path / "manifest.json",
        metadata_only_planning_authorized=True,
    ) == "planner_canary"
    monkeypatch.setenv("EASYICU_DEVELOPMENT_REVIEWED_EXECUTION", "1")
    assert agent_route._research_pipeline_budget_mode_for_source(
        prepared_manifest=None,
        metadata_only_planning_authorized=False,
    ) == "full_reviewed"


def test_planner_only_plan_requests_package_bound_regeneration() -> None:
    study = _complete_study()
    digest = study_context_owner.scientific_configuration_sha256(study)
    snapshot = build_research_workflow_snapshot(
        study=study,
        active_export_present=True,
        active_job=None,
        latest_run={
            "run_type": "full",
            "run_id": "run-preview-only",
            "engine": "easyicu.research_agent.pipeline",
            "gate_status": "blocked",
            "run_status": "human_review_pending",
            "pending_review_reason_codes": ["operator_plan_approval_required"],
            "artifact_names": ["agent_plan.json", "source_run_manifest.json"],
        },
        plan_review_authority={
            "run_id": "run-preview-only",
            "resumable_here": True,
            "scientific_configuration_sha256": digest,
            "budget_mode": "planner_canary",
        },
    )

    by_id = {row.id: row for row in snapshot.stages}
    assert snapshot.plan_execution_ready is False
    assert snapshot.next_action_code == "plan_execution_upgrade_required"
    assert by_id["plan"].reason_code == "plan_execution_upgrade_required"
    assert by_id["analysis"].reason_code == "plan_execution_upgrade_required"


def test_legacy_scientific_review_requests_fresh_plan_without_reextracting() -> None:
    study = _complete_study()
    digest = study_context_owner.scientific_configuration_sha256(study)
    snapshot = build_research_workflow_snapshot(
        study=study,
        active_export_present=True,
        active_job=None,
        latest_run={
            "run_type": "full",
            "run_id": "run-stale-science-policy",
            "engine": "easyicu.research_agent.pipeline",
            "gate_status": "blocked",
            "run_status": "human_review_pending",
            "pending_review_reason_codes": [
                "scientific_plan_review_policy_stale"
            ],
            "artifact_names": ["agent_plan.json", "source_run_manifest.json"],
        },
        plan_review_authority={
            "run_id": "run-stale-science-policy",
            "resumable_here": True,
            "scientific_configuration_sha256": digest,
            "budget_mode": "full_reviewed",
            "plan_approval_allowed": False,
            "requests": [
                {
                    "reason_code": "scientific_plan_review_policy_stale",
                    "approval_allowed": False,
                }
            ],
        },
    )

    by_id = {row.id: row for row in snapshot.stages}
    assert snapshot.plan_execution_ready is False
    assert snapshot.next_action_code == "scientific_plan_review_policy_stale"
    assert by_id["plan"].reason_code == "scientific_plan_review_policy_stale"
    assert by_id["analysis"].reason_code == "scientific_plan_review_policy_stale"


def test_planner_canary_cannot_be_approved_into_execution(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from easyicu.webserver.routes import agent as agent_route

    monkeypatch.setattr(agent_route.settings_store, "load_settings", lambda: {"ai_enabled": True})
    monkeypatch.setattr(
        agent_route.agent_pipeline_runs,
        "pending_review",
        lambda _run_id: {
            "study_id": "study-workflow",
            "resumable_here": True,
            "budget_mode": "planner_canary",
        },
    )

    with pytest.raises(Exception) as raised:
        agent_route.jobs_agent_run_review(
            {
                "run_id": "run-canary",
                "study_context_id": "study-workflow",
                "decision": "approved",
                "external_llm_opt_in": True,
            },
            request=_request(),
        )

    assert raised.value.detail == {
        "error": "research_pipeline_planner_canary_execution_blocked"
    }


def test_pipeline_bridge_cannot_approve_canary_when_route_is_bypassed(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    request = HumanReviewRequest.create(
        kind="scientific_stop",
        summary="Review canary plan.",
        authority_sha256="a" * 64,
        payload={"reason": "operator_plan_approval_required"},
    )
    pending = HumanReviewPending(
        run_id="run-canary-bypass",
        thread_id="run-canary-bypass",
        run_dir=str(tmp_path / "run-canary-bypass"),
        requests=(request,),
    )
    pipeline_called = False

    class _Pipeline:
        def resume_human_review(self, *_args: Any, **_kwargs: Any) -> Any:
            nonlocal pipeline_called
            pipeline_called = True
            raise AssertionError("canary must not reach execution")

    _install_pending_review(
        monkeypatch,
        agent_pipeline_runs._PendingRun(
            pipeline=_Pipeline(),
            pending=pending,
            wrapper_dir=tmp_path,
            study={"id": "study-canary"},
            provider={},
            acquisition=SimpleNamespace(),
            created_at=1.0,
            budget_mode="planner_canary",
        ),
    )

    with pytest.raises(agent_pipeline_runs.ResearchPipelineRunError) as exc:
        agent_pipeline_runs.resume_research_pipeline(
            run_id=pending.run_id,
            study_context_id="study-canary",
            decision="approved",
            reviewer="server reviewer",
            note="",
            job=SimpleNamespace(emit=lambda _event: None, cancel_requested=False),
        )

    assert exc.value.code == "research_pipeline_planner_canary_execution_blocked"
    assert pipeline_called is False


def test_pipeline_bridge_rejects_paused_legacy_kdigo_plan_before_execution(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    request = HumanReviewRequest.create(
        kind="scientific_stop",
        summary="Review legacy KDIGO plan.",
        authority_sha256="a" * 64,
        payload={"reason": "operator_plan_approval_required"},
    )
    pending = HumanReviewPending(
        run_id="run-legacy-kdigo",
        thread_id="run-legacy-kdigo",
        run_dir=str(tmp_path / "run-legacy-kdigo"),
        requests=(request,),
    )
    pipeline_called = False

    class _Pipeline:
        _scientific_runtime_authorities = SimpleNamespace(
            current_case=SimpleNamespace(exposure_column="aki_stage_max")
        )

        def resume_human_review(self, *_args: Any, **_kwargs: Any) -> Any:
            nonlocal pipeline_called
            pipeline_called = True
            raise AssertionError("legacy KDIGO plan must not reach execution")

    _install_pending_review(
        monkeypatch,
        agent_pipeline_runs._PendingRun(
            pipeline=_Pipeline(),
            pending=pending,
            wrapper_dir=tmp_path,
            study={
                "id": "study-legacy-kdigo",
                "execution_concepts": {"primary_exposure": "aki_stage"},
            },
            provider={},
            acquisition=SimpleNamespace(),
            created_at=1.0,
        ),
    )

    with pytest.raises(agent_pipeline_runs.ResearchPipelineRunError) as exc:
        agent_pipeline_runs.resume_research_pipeline(
            run_id=pending.run_id,
            study_context_id="study-legacy-kdigo",
            decision="approved",
            reviewer="server reviewer",
            note="",
            job=SimpleNamespace(emit=lambda _event: None, cancel_requested=False),
        )

    assert exc.value.code == "research_pipeline_kdigo_observability_authority_missing"
    assert pipeline_called is False


def test_signoff_ignores_client_reviewer_identity(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from easyicu.webserver.routes import agent as agent_route

    captured: dict[str, Any] = {}

    def create_signoff(_project_dir: str, **kwargs: Any) -> dict[str, Any]:
        captured.update(kwargs)
        return {"ok": True}

    monkeypatch.setattr(agent_route.agent_runs, "create_human_signoff", create_signoff)

    agent_route.post_agent_run_signoff(
        {"project_dir": "/server/run", "reviewer": "client-claims-to-be-PI"}
    )

    assert captured["reviewer"] == "easyicu_local_web_operator"


def test_research_pipeline_runner_uses_in_memory_provider_authority(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _assume_execution_runtime_ready(monkeypatch)
    actual_run = tmp_path / "actual-provider-run"
    _write_real_pipeline_fixture(
        actual_run,
        manuscript="# Results\nThe provider-bound result is analysis-only.",
    )
    universe = tmp_path / "universe.parquet"
    universe.write_bytes(b"typed-universe-placeholder")
    acquisition = _acquisition_receipt()
    acquisition.blocked = False
    acquisition.universe_path = universe
    acquisition.cohort_authority_path = None
    acquisition.cohort_authority_ref = None
    acquisition.trajectory_path = None
    acquisition.trajectory_authority_path = None
    acquisition.trajectory_authority_ref = None
    expected_environment = {
        "OPENAI_API_KEY": "test-private-provider-key",
        "OPENAI_BASE_URL": "http://127.0.0.1:8317/v1",
        "OPENAI_MODEL": "test-local-model",
        "EASYICU_DISABLE_PROVIDER_ENV_FILE": "1",
    }
    captured: dict[str, Any] = {}

    def build_client(
        provider: dict[str, Any],
        *,
        request_timeout: float | None = None,
        request_hard_timeout: float | None = None,
        environ: dict[str, str] | None = None,
    ) -> tuple[object, dict[str, Any]]:
        captured["provider"] = dict(provider)
        captured["environment"] = dict(environ or {})
        captured["request_timeout"] = request_timeout
        captured["request_hard_timeout"] = request_hard_timeout
        return object(), {"provider": "openai", "model": "test-local-model"}

    monkeypatch.setattr(
        provider_adapter,
        "build_research_agent_provider_client",
        build_client,
    )
    import easyicu.research_agent as research_agent
    from easyicu.research_agent.acquisition import foundation

    monkeypatch.setattr(
        foundation,
        "acquire_universe_for_question",
        lambda **_kwargs: acquisition,
    )
    monkeypatch.setattr(
        research_pipeline_run_preparation,
        "_data_foundation_profile",
        lambda **_kwargs: _foundation_profile(),
    )

    class FakePipeline:
        def run(self, **_kwargs: Any) -> SimpleNamespace:
            return SimpleNamespace(manifest_path=actual_run / "manifest.json")

    monkeypatch.setattr(
        research_agent.ResearchAgentPipeline,
        "from_config",
        lambda _config, *, services: FakePipeline(),
    )

    export_path = _write_pipeline_export(tmp_path / "export")
    runner = agent_pipeline_runs.make_research_pipeline_run_runner(
        export_path=str(export_path),
        study_context=_complete_study(),
        project_root=str(tmp_path / "projects"),
        provider={"provider": "openai", "external": True},
        provider_environment=expected_environment,
        budget_mode="full_reviewed",
    )

    class Job:
        id = "job-provider-authority"
        cancel_requested = False
        events: list[dict[str, Any]] = []

        def emit(self, event: dict[str, Any]) -> None:
            self.events.append(dict(event))

    result = runner(Job())

    assert captured["environment"] == expected_environment
    assert captured["request_timeout"] is None
    assert captured["request_hard_timeout"] is None
    assert result["provider"]["model"] == "test-local-model"
    assert "test-private-provider-key" not in json.dumps(result)


def test_chitchat_with_client_provider_run_requires_authorization(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """D-P1-1: a tampered client cannot pre-grant a privileged one-shot action.

    The browser's ``full`` access mode ships ``provider_run`` with every
    message; backend text inference is the necessary condition.  Chit-chat
    carrying ``provider_run`` must fail closed with
    ``pi_action_authorization_required``.
    """

    from easyicu.webserver.pi_copilot import tools as tool_module
    from easyicu.webserver.pi_copilot.contracts import (
        AuthorityBinding,
        PiSessionRecord,
        ToolExecutionContext,
    )

    monkeypatch.setattr(
        tool_module, "_bound_context", lambda _binding: _complete_study()
    )

    def _must_not_submit(
        request: Any, *, authorize: Any = None, **kwargs: Any,
    ) -> Any:
        if callable(authorize):
            authorize()
        raise AssertionError("chit-chat must not reach provider submission")

    monkeypatch.setattr(
        research_run_submission, "submit_research_run", _must_not_submit
    )
    context = ToolExecutionContext(
        session=PiSessionRecord(
            session_id="pi-chitchat-provider-run",
            external_llm_opt_in=True,
            binding=AuthorityBinding(
                study_context_id="study-workflow",
                study_revision=4,
            ),
        ),
        user_message="今天天气不错，随便聊聊吧，你最近怎么样？",
        allowed_actions={"provider_run"},
    )
    result = tool_module.execute_tool(
        "easyicu_run", {"run_type": "full"}, context
    )
    assert result["code"] == "pi_action_authorization_required"


def test_service_strips_client_privileged_actions_without_backend_inference() -> None:
    """D-P1-1 service layer: privileged actions come only from inference."""

    # Lightweight unit check of the documented rule without booting a gateway:
    # ordinary actions keep union compatibility, privileged require inference.
    from easyicu.webserver.pi_copilot.service import (
        PRIVILEGED_ONE_SHOT_TURN_ACTIONS,
    )
    from easyicu.webserver.pi_copilot.turn_authority import (
        infer_explicit_turn_actions,
    )

    chitchat = "今天天气不错，随便聊聊吧，你最近怎么样？"
    assert infer_explicit_turn_actions(chitchat) == frozenset()
    client = frozenset({"provider_run", "run", "idea"})
    inferred = infer_explicit_turn_actions(chitchat)
    requested = ((client | inferred) - PRIVILEGED_ONE_SHOT_TURN_ACTIONS) | (
        inferred & PRIVILEGED_ONE_SHOT_TURN_ACTIONS
    )
    assert "provider_run" not in requested
    assert "run" in requested
    assert "idea" in requested
