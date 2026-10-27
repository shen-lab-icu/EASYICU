from __future__ import annotations

import asyncio
import threading
from concurrent.futures import Future
from pathlib import Path
from types import SimpleNamespace
from typing import Any
from unittest.mock import Mock

import pytest
from fastapi import HTTPException
from pydantic import ValidationError

from easyicu.webserver import research_run_submission
from easyicu.webserver.routes import agent as agent_route


def test_host_candidate_plan_control_recovers_server_owned_planner_intent(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        agent_route,
        "build_project_workflow_projection",
        lambda **_kwargs: SimpleNamespace(
            workflow=SimpleNamespace(
                next_action_code="provider_ready_to_generate_plan"
            )
        ),
    )

    assert agent_route._candidate_plan_only_authorized(
        {
            "study_context_id": "study-1",
            "planner_start_mode": "fresh",
        }
    )
    assert agent_route._candidate_plan_only_authorized(
        {
            "study_context_id": "study-1",
            "planner_start_mode": "auto",
        }
    )

    monkeypatch.setattr(
        agent_route,
        "build_project_workflow_projection",
        lambda **_kwargs: SimpleNamespace(
            workflow=SimpleNamespace(
                next_action_code="planner_checkpoint_resume_available"
            )
        ),
    )
    assert agent_route._candidate_plan_only_authorized(
        {
            "study_context_id": "study-1",
            "planner_start_mode": "resume_checkpoint",
        }
    )

    monkeypatch.setattr(
        agent_route,
        "build_project_workflow_projection",
        lambda **_kwargs: SimpleNamespace(
            workflow=SimpleNamespace(
                next_action_code="plan_scientific_changes_required"
            )
        ),
    )
    assert agent_route._candidate_plan_only_authorized(
        {
            "study_context_id": "study-1",
            "planner_start_mode": "auto",
            "plan_revision_source_run_id": "run-review-required",
        }
    )

    monkeypatch.setattr(
        agent_route,
        "build_project_workflow_projection",
        lambda **_kwargs: SimpleNamespace(
            workflow=SimpleNamespace(
                next_action_code="failed_pipeline_requires_fresh_plan"
            )
        ),
    )
    assert agent_route._candidate_plan_only_authorized(
        {
            "study_context_id": "study-1",
            "planner_start_mode": "fresh",
        }
    )
    assert agent_route._candidate_plan_only_authorized(
        {
            "study_context_id": "study-1",
            "planner_start_mode": "auto",
        }
    )

    monkeypatch.setattr(
        agent_route,
        "build_project_workflow_projection",
        lambda **_kwargs: SimpleNamespace(
            workflow=SimpleNamespace(next_action_code="plan_configuration_superseded")
        ),
    )
    assert agent_route._candidate_plan_only_authorized(
        {
            "study_context_id": "study-1",
            "planner_start_mode": "fresh",
        }
    )


def test_failed_plan_regeneration_stays_candidate_only_after_data_preparation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        agent_route,
        "build_project_workflow_projection",
        lambda **_kwargs: SimpleNamespace(
            workflow=SimpleNamespace(
                next_action_code="failed_pipeline_requires_fresh_plan"
            )
        ),
    )
    assert agent_route._candidate_plan_only_authorized(
        {
            "study_context_id": "study-1",
            "planner_start_mode": "fresh",
        }
    )


def test_pipeline_route_delegates_to_submission_owner(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: dict[str, Any] = {}

    def submit(request: Any, **kwargs: Any) -> Any:
        captured["request"] = request
        captured.update(kwargs)
        return research_run_submission.ResearchRunSubmissionReceipt(
            job_id="job-owned",
            kind="agent-run",
            status="running",
            study_context_id=request.study_context_id,
            study_context_revision=2,
            budget_mode="planner_canary",
            planner_start_mode=request.planner_start_mode,
        )

    monkeypatch.setattr(research_run_submission, "submit_research_run", submit)
    body = {
        "engine": "research_agent_pipeline",
        "study_context_id": "study-1",
        "llm_provider": "openai",
        "credential_source": "pi_verified",
        "external_llm_opt_in": True,
        "path": "/not-read-by-route",
    }

    result = agent_route.submit_agent_run(
        body,
        account_environment={"ACCOUNT": "private"},
        metadata_only_planning_authorized=True,
    )

    assert result["job_id"] == "job-owned"
    request = captured.pop("request")
    assert request.study_context_id == "study-1"
    assert request.intent == "candidate_plan"
    assert not hasattr(request, "path")
    assert captured == {"account_environment": {"ACCOUNT": "private"}}


def test_pipeline_route_maps_typed_submission_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def reject(*_args: Any, **_kwargs: Any) -> dict[str, Any]:
        raise research_run_submission.ResearchRunSubmissionError(
            {"error": "job_capacity_exceeded"},
            status_code=429,
        )

    monkeypatch.setattr(research_run_submission, "submit_research_run", reject)

    with pytest.raises(HTTPException) as raised:
        agent_route.submit_agent_run(
            {
                "engine": "research_agent_pipeline",
                "study_context_id": "study-1",
            }
        )

    assert raised.value.status_code == 429
    assert raised.value.detail == {"error": "job_capacity_exceeded"}


def test_submission_request_forbids_adapter_owned_coordinates() -> None:
    with pytest.raises(ValidationError):
        research_run_submission.ResearchRunSubmissionRequest.model_validate(
            {
                "study_context_id": "study-1",
                "provider": "openai",
                "credential_source": "pi_verified",
                "external_llm_opt_in": True,
                "path": "/adapter-must-not-pass-this",
                "budget_mode": "full_reviewed",
                "development_resume_source_job_id": "job-old",
            }
        )


def test_owner_checks_readiness_before_consuming_authorization(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    study = {
        "id": "study-not-ready",
        "revision": 3,
        "question": "",
        "data_source": {"path": "/private/export", "database": "miiv"},
        "cohort": {"preset": "adult_icu"},
    }
    monkeypatch.setattr(
        research_run_submission.context_store,
        "get_context",
        lambda _study_id: study,
    )
    monkeypatch.setattr(
        research_run_submission.dataio,
        "describe_export_source",
        lambda _path: {"ok": True},
    )
    authorizations: list[str] = []
    request = research_run_submission.ResearchRunSubmissionRequest(
        study_context_id=study["id"],
        provider="openai",
        credential_source="pi_verified",
        external_llm_opt_in=True,
    )

    with pytest.raises(research_run_submission.ResearchRunSubmissionError) as raised:
        research_run_submission.submit_research_run(
            request,
            authorize=lambda: authorizations.append("consumed"),
        )

    assert raised.value.code == "study_setup_incomplete"
    assert authorizations == []


def test_submission_receipt_is_frozen_and_path_free() -> None:
    receipt = research_run_submission.ResearchRunSubmissionReceipt(
        job_id="job-1",
        kind="agent-run",
        status="queued",
        study_context_id="study-1",
        study_context_revision=4,
        budget_mode="planner_canary",
        planner_start_mode="auto",
    )

    assert "path" not in receipt.model_dump(mode="json")
    with pytest.raises(ValidationError):
        receipt.job_id = "job-2"


class _StartInterrupted(BaseException):
    """A non-Exception interruption unrelated to interpreter shutdown."""


@pytest.fixture
def start_gate_launch(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    study = {
        "id": "study-start-gate",
        "revision": 3,
        "question": "Describe the synthetic measurements.",
        "data_source": {"path": str(tmp_path / "synthetic-source"), "database": "eicu"},
    }
    monkeypatch.delenv("EASYICU_DEVELOPMENT_REVIEWED_EXECUTION", raising=False)
    monkeypatch.setattr(research_run_submission.context_store, "get_context", lambda _: study)
    monkeypatch.setattr(research_run_submission.dataio, "describe_export_source", lambda _: {"ok": True})
    monkeypatch.setattr(research_run_submission.dataio, "prepared_export_manifest_path", lambda _: None)
    monkeypatch.setattr(
        research_run_submission, "build_research_workflow_snapshot",
        lambda **_: SimpleNamespace(planning_prerequisites_missing=[]),
    )
    monkeypatch.setattr(research_run_submission, "provider_environment_for_agent_run", lambda **_: {})
    monkeypatch.setattr(research_run_submission.settings_store, "load_settings", lambda: {"ai_enabled": True})
    monkeypatch.setattr(research_run_submission.capabilities, "validate_compute_target", lambda _: {"ok": True})
    monkeypatch.setattr(research_run_submission.agent_runs, "resolve_agent_provider_config", lambda **_: {})
    monkeypatch.setattr(research_run_submission.context_store, "build_agent_context_binding", lambda *_, **__: {})
    monkeypatch.setattr(
        research_run_submission, "research_pipeline_workspace",
        lambda: SimpleNamespace(project_root=lambda _: tmp_path),
    )
    callbacks = Mock()
    callbacks.handoff.return_value = {"revision": 4}
    callbacks.base_runner.return_value = {"gate": {"status": "passed"}}
    callbacks.cleanup.return_value = {"context": {"revision": 5}}
    monkeypatch.setattr(research_run_submission.context_store, "handoff_context", callbacks.handoff)
    monkeypatch.setattr(research_run_submission.context_store, "clear_active_job_if", callbacks.cleanup)
    monkeypatch.setattr(research_run_submission.capabilities, "record_tool_event", callbacks.audit)
    monkeypatch.setattr(
        research_run_submission.agent_pipeline_runs, "make_research_pipeline_run_runner",
        lambda **_: callbacks.base_runner,
    )
    job = SimpleNamespace(id="job-start-gate", kind="agent-run", status="queued", cancel_requested=False)
    runner_result: Future[dict] = Future()
    threads: list[threading.Thread] = []

    def enqueue(_kind, runner):
        def run():
            try:
                runner_result.set_result(runner(job))
            except BaseException as exc:
                runner_result.set_exception(exc)

        thread = threading.Thread(target=run, daemon=True)
        threads.append(thread)
        thread.start()
        return job

    monkeypatch.setattr(research_run_submission, "_submit_job", enqueue)

    def submit(*, with_authorize=True):
        return research_run_submission.submit_research_run(
            research_run_submission.ResearchRunSubmissionRequest(
                study_context_id=study["id"],
                provider="mock",
                credential_source="pi_verified",
                external_llm_opt_in=False,
                planner_start_mode="fresh",
            ),
            authorize=callbacks.authorize if with_authorize else None,
        )

    yield SimpleNamespace(
        submit=submit, callbacks=callbacks, runner_result=runner_result, job=job,
    )
    for thread in threads:
        thread.join(timeout=2)
        assert not thread.is_alive(), "the start gate left a runner waiting"


def _assert_start_blocked(launch, code: str) -> None:
    with pytest.raises(RuntimeError, match=f"^agent_run_start_blocked:{code}$"):
        launch.runner_result.result(timeout=2)
    launch.callbacks.base_runner.assert_not_called()
    launch.callbacks.cleanup.assert_not_called()
    launch.callbacks.audit.assert_not_called()


@pytest.mark.parametrize("stage", ["authorize", "handoff"])
@pytest.mark.parametrize(
    "error_type",
    [KeyboardInterrupt, _StartInterrupted, SystemExit, asyncio.CancelledError, GeneratorExit],
)
def test_start_gate_blocks_base_exceptions_and_propagates_them(
    start_gate_launch, stage, error_type,
) -> None:
    launch = start_gate_launch
    error = error_type("synthetic interruption")
    getattr(launch.callbacks, stage).side_effect = error

    with pytest.raises(error_type) as raised:
        launch.submit()

    assert raised.value is error
    _assert_start_blocked(launch, "agent_run_start_unconfirmed")
    if stage == "authorize":
        launch.callbacks.handoff.assert_not_called()
    else:
        launch.callbacks.authorize.assert_called_once_with()


@pytest.mark.parametrize("stage", ["authorize", "handoff"])
def test_start_gate_preserves_typed_submission_failure(start_gate_launch, stage) -> None:
    launch = start_gate_launch
    error = research_run_submission.ResearchRunSubmissionError(
        {"error": "synthetic_authorization_denied", "message": "Synthetic grant refused."},
        status_code=403,
    )
    getattr(launch.callbacks, stage).side_effect = error

    with pytest.raises(research_run_submission.ResearchRunSubmissionError) as raised:
        launch.submit()

    assert raised.value is error
    assert raised.value.status_code == 403
    assert raised.value.detail == error.detail
    _assert_start_blocked(launch, "synthetic_authorization_denied")


@pytest.mark.parametrize("stage", ["authorize", "handoff"])
@pytest.mark.parametrize(
    ("code", "status_code"),
    [("study_context_revision_conflict", 409), ("study_context_not_found", 500)],
)
def test_start_gate_preserves_context_failure_mapping(
    start_gate_launch, stage, code, status_code,
) -> None:
    launch = start_gate_launch
    detail = {"error": code, "message": "Synthetic context refused."}
    getattr(launch.callbacks, stage).side_effect = (
        research_run_submission.context_store.StudyContextError(detail)
    )

    with pytest.raises(research_run_submission.ResearchRunSubmissionError) as raised:
        launch.submit()

    assert raised.value.status_code == status_code
    assert raised.value.detail == {**detail, "job_id": launch.job.id, "job_started": False}
    _assert_start_blocked(launch, code)


@pytest.mark.parametrize(
    ("stage", "code"),
    [("authorize", "agent_run_authorization_failed"), ("handoff", "study_context_active_job_sync_failed")],
)
def test_start_gate_identifies_untyped_failure_stage(start_gate_launch, stage, code) -> None:
    launch = start_gate_launch
    getattr(launch.callbacks, stage).side_effect = RuntimeError("synthetic failure")

    with pytest.raises(research_run_submission.ResearchRunSubmissionError) as raised:
        launch.submit()

    assert raised.value.status_code == 500
    assert raised.value.detail == {
        "error": code, "reason": "RuntimeError", "job_id": launch.job.id, "job_started": False,
    }
    _assert_start_blocked(launch, code)


@pytest.mark.parametrize("with_authorize", [True, False])
def test_start_gate_runs_after_authorization_and_handoff(start_gate_launch, with_authorize) -> None:
    launch = start_gate_launch

    receipt = launch.submit(with_authorize=with_authorize)
    result = launch.runner_result.result(timeout=2)

    assert receipt.job_id == launch.job.id
    assert receipt.study_context_revision == 4
    assert result == {"gate": {"status": "passed"}, "study_context_revision": 5}
    launch.callbacks.base_runner.assert_called_once_with(launch.job)
    names = [call[0] for call in launch.callbacks.mock_calls]
    assert names.index("handoff") < names.index("base_runner") < names.index("cleanup")
    if with_authorize:
        launch.callbacks.authorize.assert_called_once_with()
        assert names.index("authorize") < names.index("handoff")
    else:
        launch.callbacks.authorize.assert_not_called()


@pytest.mark.parametrize(("database", "reason"), [
    ("unregistered_icu", "unknown_source_database"),
    ("eicu_demo", "public_demo_manifest_invalid"),
])
def test_submission_reports_source_proof_refusals_before_enqueuing(start_gate_launch, monkeypatch, database, reason):
    launch = start_gate_launch
    study = research_run_submission.context_store.get_context("fixture")
    study["data_source"]["database"] = database
    enqueue = Mock()
    monkeypatch.setattr(research_run_submission, "_submit_job", enqueue)
    with pytest.raises(research_run_submission.ResearchRunSubmissionError) as caught:
        launch.submit()
    assert caught.value.detail["error"] == reason
    enqueue.assert_not_called()
    launch.callbacks.authorize.assert_not_called()


@pytest.mark.parametrize("source_index", [0, 1])
def test_submission_keeps_the_pinned_demo_clearance_scope(start_gate_launch, tmp_path, monkeypatch, source_index):
    from easyicu.research_agent.providers.client_trust import require_data_handling, source_requires_clearance
    from easyicu.webserver import demo_source_storage as storage
    from easyicu.webserver.demo_source_contracts import SOURCES

    monkeypatch.setenv("EASYICU_DEMO_CACHE_DIR", str(tmp_path / "demo-cache"))
    source = SOURCES[source_index]
    paths = storage.source_paths(source)
    paths.export.mkdir(parents=True)
    (paths.export / "_manifest.json").write_text("{}")
    storage.write_export_marker(paths, source, archive_sha256=source.archive_sha256, export={})
    study = research_run_submission.context_store.get_context("fixture")
    study["data_source"] = {"path": str(paths.export), "database": source.database}
    seen = []

    def resolve(**kwargs):
        seen.append(kwargs["database"])
        require_data_handling(kwargs["database"], "unspecified")
        return {}

    monkeypatch.setattr(research_run_submission.agent_runs, "resolve_agent_provider_config", resolve)
    launch = start_gate_launch
    receipt = launch.submit()
    launch.runner_result.result(timeout=2)
    assert receipt.job_id == launch.job.id
    assert seen == [source.database]
    assert source_requires_clearance("miiv")


@pytest.mark.parametrize("authorized", [True, False])
def test_durable_submission_activates_only_after_original_grant_and_handoff(
    start_gate_launch, monkeypatch, authorized,
):
    from easyicu.webserver import durable_research

    launch = start_gate_launch
    events = []
    captured = {}
    def reserve(payload):
        events.append("reserve")
        captured["payload"] = payload
        return launch.job
    def activate(identity, runner):
        events.append("activate")
        captured["runner"] = runner
    def authorize():
        events.append("authorize")
        if not authorized:
            raise research_run_submission.ResearchRunSubmissionError({"error": "grant_expired"})
    def handoff(*args, **kwargs):
        events.append("handoff")
        return {"revision": 4}
    service = SimpleNamespace(reserve=reserve, activate=activate, abandon=lambda identity: events.append("abandon"))
    monkeypatch.setattr(durable_research, "current_service", lambda: service)
    launch.callbacks.handoff.side_effect = handoff
    request = research_run_submission.ResearchRunSubmissionRequest(
        study_context_id="study-start-gate", provider="mock", credential_source="pi_verified",
        external_llm_opt_in=True, planner_start_mode="fresh",
    )
    if not authorized:
        with pytest.raises(research_run_submission.ResearchRunSubmissionError, match="grant_expired"):
            research_run_submission.submit_research_run(request, authorize=authorize)
        assert events == ["reserve", "authorize", "abandon"]
        launch.callbacks.base_runner.assert_not_called()
    else:
        result = research_run_submission.submit_research_run(request, authorize=authorize)
        assert events == ["reserve", "authorize", "handoff", "activate"]
        assert result.job_id == launch.job.id
        launch.callbacks.base_runner.assert_not_called()
        captured["runner"](launch.job)
        launch.callbacks.base_runner.assert_called_once()
    assert "provider_environment" not in captured["payload"]
