"""An approved run that fails and cannot resume is recorded as failed.

A paused plan's projection says that the plan awaits review. When the approved
resume failed after the workflow had left its pause, the review ended with it,
but the host left that projection in place. The run record kept offering an
approval no one could give; the workflow read the missing review authority as
``plan_review_not_resumable``, whose card said that the study had changed; and
the cause was kept only in a private diagnostic. A failure that leaves the
review open keeps the paused run, so the same plan can be approved again.

Synthetic study, plan and export; no benchmark item.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pandas as pd
import pytest

from easyicu.research_agent.authority.plan_review import PlanReviewAuthority
from easyicu.research_agent.orchestration.workflow import (
    HumanReviewPending,
    HumanReviewRequest,
)
from easyicu.research_agent.schema import AnalysisPlan, AnalysisStep
from easyicu.webserver import agent_pipeline_runs, agent_runs, dataio
from easyicu.webserver import study_contexts as study_context_owner
from easyicu.webserver.pi_copilot.contracts import EXECUTION_RETRY_REPLAYABLE_GATE_REASONS
from easyicu.webserver.pi_copilot.workflow import build_research_workflow_snapshot
from tests.webserver.copilot.research_workflow_fixtures import (
    _acquisition_receipt,
    complete_study,
)

_RUN_ID = "run-approved-then-failed"
_WRAPPER = "run_web_approved_then_failed"


class _FailingPipeline:
    """A pipeline whose approved resume raises; ``resumable`` is its state after."""

    def __init__(self, error: Exception, *, resumable: bool) -> None:
        self._error = error
        self.has_resumable_human_review = resumable

    def resume_human_review(self, decisions, *, run_id, progress_callback=None):
        raise self._error


def _write_export(root: Path) -> Path:
    root.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({"stay_id": [1], "age": [65]}).to_parquet(
        root / "demographics.parquet", index=False
    )
    (root / "_manifest.json").write_text(
        json.dumps(
            {
                "database": "miiv",
                "format": "parquet",
                "concept_selection": {
                    "mode": "explicit",
                    "modules": {"demographics": ["age"]},
                },
                "feature_definitions": {"included": False},
                "files": [
                    {
                        "file": "demographics.parquet",
                        "module": "demographics",
                        "concepts": 1,
                        "concept_ids": ["age"],
                        "rows": 1,
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    return root


def _write_paused_run(run_dir: Path) -> None:
    """A pipeline run paused at plan review: a plan, no analysis yet."""

    run_dir.mkdir(parents=True)
    (run_dir / "run_status.json").write_text("{}", encoding="utf-8")
    plan_bytes = json.dumps(
        {"steps": [{"id": "model", "title": "Fit the specified model"}]}
    ).encode("utf-8")
    (run_dir / "analysis_plan.json").write_bytes(plan_bytes)
    (run_dir / "manifest.json").write_text(
        json.dumps(
            {
                "readiness": {
                    "execution_complete": False,
                    "analysis_validated": False,
                    "evidence_complete": False,
                    "numeric_verified": False,
                    "manuscript_ready": False,
                },
                "current_plan_authority": {
                    "relative_path": "analysis_plan.json",
                    "sha256": hashlib.sha256(plan_bytes).hexdigest(),
                },
            }
        ),
        encoding="utf-8",
    )


def _paused(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    pipeline: _FailingPipeline,
) -> tuple[Path, Path, dict[str, Any]]:
    """Pause one approvable plan and register its live review."""

    monkeypatch.setattr(
        agent_pipeline_runs,
        "_load_pending_scientific_review",
        lambda *_args, **_kwargs: {
            "schema_version": agent_pipeline_runs.CURRENT_SCIENTIFIC_REVIEW_SCHEMA_VERSION,
            "approval_allowed": True,
        },
    )
    export = _write_export(tmp_path / "export")
    study = {
        **complete_study(),
        "data_source": {"path": str(export), "database": "miiv"},
    }
    binding = dict(
        dataio.validate_research_pipeline_source(str(export), database="miiv")[
            "binding"
        ]
    )
    run_dir = tmp_path / "pipeline" / _RUN_ID
    _write_paused_run(run_dir)
    authority = PlanReviewAuthority.create(
        plan=AnalysisPlan(
            research_question="Is an ICU exposure associated with mortality?",
            steps=[
                AnalysisStep(
                    step_id="model",
                    intent="Estimate the association",
                    method="descriptive",
                    inputs=[],
                    expected_outputs=["table:model"],
                )
            ],
        )
    )
    request = HumanReviewRequest.create(
        kind="scientific_stop",
        summary="Review the digest-bound plan before analysis.",
        authority_sha256="a" * 64,
        payload={
            "reason": "operator_plan_approval_required",
            "plan_review_authority": authority.model_dump(mode="json"),
        },
    )
    pending = HumanReviewPending(
        run_id=_RUN_ID,
        thread_id="thread-approved-then-failed",
        run_dir=str(run_dir),
        requests=(request,),
    )
    project_root = tmp_path / "projects"
    wrapper = project_root / str(study["id"]) / _WRAPPER
    agent_pipeline_runs._write_projection(
        wrapper_dir=wrapper,
        study=study,
        provider={"provider": "openai", "model": "test-model"},
        acquisition=_acquisition_receipt(),
        run_dir=run_dir,
        pending=pending,
    )
    hard_stop = agent_pipeline_runs._start_web_provider_hard_stop(
        wrapper_dir=wrapper,
        job_id="approved-then-failed",
        declaration_sha256=study_context_owner.scientific_configuration_sha256(study),
    )
    hard_stop.pause()
    entry = agent_pipeline_runs._PendingRun(
        pipeline=pipeline,
        pending=pending,
        wrapper_dir=wrapper,
        study=study,
        provider={},
        acquisition=_acquisition_receipt(),
        created_at=1.0,
        prepared_package_binding=binding,
        provider_hard_stop=hard_stop,
    )
    registry = agent_pipeline_runs.PendingReviewRegistry()
    monkeypatch.setattr(agent_pipeline_runs, "_PENDING_REVIEWS", registry)
    registry.register(entry)
    return project_root, wrapper, study


def _approve(study: dict[str, Any]) -> agent_pipeline_runs.ResearchPipelineRunError:
    with pytest.raises(agent_pipeline_runs.ResearchPipelineRunError) as caught:
        agent_pipeline_runs.resume_research_pipeline(
            run_id=_RUN_ID,
            study_context_id=study["id"],
            decision="approved",
            reviewer="local reviewer",
            note="",
            job=SimpleNamespace(emit=lambda _event: None, cancel_requested=False),
            current_study_context=study,
        )
    return caught.value


def _row(project_root: Path, study: dict[str, Any]) -> dict[str, Any]:
    history = agent_runs.list_run_history(
        study_id=str(study["id"]), project_root=str(project_root)
    )
    (row,) = history["runs"]
    return row


def test_an_unresumable_failure_records_the_approved_run_as_failed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    project_root, wrapper, study = _paused(
        tmp_path,
        monkeypatch,
        _FailingPipeline(
            ValueError("writer projection rejected /private/run/path"),
            resumable=False,
        ),
    )
    paused = _row(project_root, study)
    assert paused["run_status"] == "human_review_pending"
    assert paused["pending_review_reason_codes"] == ["operator_plan_approval_required"]
    assert paused["plan_available"] is True
    kept = {
        path.name: path.read_bytes()
        for path in wrapper.glob("*.json")
        if path.name
        not in {"quality_gate.json", "source_run_manifest.json", "evidence_ledger.json"}
    }
    assert "agent_plan.json" in kept

    error = _approve(study)

    assert error.code == "research_pipeline_approved_run_failed"
    assert error.details["review_resumable"] is False
    row = _row(project_root, study)
    assert row["run_status"] == "failed"
    assert row["gate_status"] == "blocked"
    assert row["gate_reason"] == "research_pipeline_approved_run_failed"
    assert row["pending_review_reason_codes"] == []
    assert row["plan_approval_allowed"] is False
    # The approved plan stays in the record; only the outcome changed.
    assert row["plan_available"] is True
    for name, raw in kept.items():
        assert (wrapper / name).read_bytes() == raw, name
    assert row["scientific_configuration_sha256"] == paused["scientific_configuration_sha256"]
    manifest = json.loads((wrapper / "source_run_manifest.json").read_text(encoding="utf-8"))
    assert manifest["failure_code"] == "research_pipeline_approved_run_failed"
    assert manifest["diagnostic_available"] is True
    diagnostic = json.loads(
        (wrapper / error.details["diagnostic"]).read_text(encoding="utf-8")
    )
    assert diagnostic["code"] == "research_pipeline_approved_run_failed"
    assert diagnostic["review_resumable"] is False
    # The ledger still describes the files it lists.
    ledger = json.loads((wrapper / "evidence_ledger.json").read_text(encoding="utf-8"))
    for record in ledger["artifacts"]:
        raw = (wrapper / record["name"]).read_bytes()
        assert record["sha256"] == hashlib.sha256(raw).hexdigest(), record["name"]
    for name in ("quality_gate.json", "source_run_manifest.json", "evidence_ledger.json"):
        assert "/private/run/path" not in (wrapper / name).read_text(encoding="utf-8")


def test_the_workflow_offers_a_fresh_plan_not_a_changed_study(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    project_root, _wrapper, study = _paused(
        tmp_path,
        monkeypatch,
        _FailingPipeline(ValueError("writer failed"), resumable=False),
    )
    def next_action(row: dict[str, Any]) -> str:
        return build_research_workflow_snapshot(
            study=study,
            active_export_present=True,
            active_job=None,
            latest_run=row,
            latest_attempt=row,
        ).next_action_code

    # Left as it was, the paused record without its review authority reads
    # as a plan that can no longer be approved.
    assert next_action(_row(project_root, study)) == "plan_review_not_resumable"

    _approve(study)
    row = _row(project_root, study)

    assert next_action(row) == "failed_pipeline_requires_fresh_plan"
    # Retrying the failed step would need recovery state the failure removed,
    # so no retry is offered for the launch to refuse.
    assert row["gate_reason"] not in EXECUTION_RETRY_REPLAYABLE_GATE_REASONS


def test_a_runtime_failure_that_ends_the_review_keeps_its_own_code(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from easyicu.research_agent.execution import runner as runner_module

    unavailable = runner_module.ExecutionRuntimeUnavailableError(
        runner_module.RunnerAvailability(
            kind="docker",
            available=False,
            image="easyicu-research-agent:test",
            reason_code="docker_daemon_unreachable",
        )
    )
    project_root, _wrapper, study = _paused(
        tmp_path, monkeypatch, _FailingPipeline(unavailable, resumable=False)
    )

    error = _approve(study)

    assert error.code == "research_pipeline_execution_runtime_unavailable"
    assert "generate a fresh plan" in str(error)
    row = _row(project_root, study)
    assert row["run_status"] == "failed"
    assert row["gate_reason"] == "research_pipeline_execution_runtime_unavailable"
    assert row["gate_detail"] == {"reason_code": "docker_daemon_unreachable"}


def test_a_failure_that_leaves_the_review_open_keeps_the_paused_run(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    project_root, wrapper, study = _paused(
        tmp_path,
        monkeypatch,
        _FailingPipeline(ValueError("decision not persisted"), resumable=True),
    )
    before = {
        name: (wrapper / name).read_bytes()
        for name in ("quality_gate.json", "source_run_manifest.json", "evidence_ledger.json")
    }

    error = _approve(study)

    assert error.code == "research_pipeline_review_resume_failed"
    assert error.details["review_resumable"] is True
    assert "could not resume" in str(error)
    for name, raw in before.items():
        assert (wrapper / name).read_bytes() == raw, name
    row = _row(project_root, study)
    assert row["run_status"] == "human_review_pending"
    assert row["pending_review_reason_codes"] == ["operator_plan_approval_required"]


def test_only_a_paused_projection_is_closed(tmp_path: Path) -> None:
    """A run that no longer awaits review keeps its record as it is."""

    wrapper = tmp_path / "terminal"
    wrapper.mkdir()
    gate = {"gate": {"status": "pass", "reason": "research_agent_pipeline_complete"}}
    manifest = {"status": "pass"}
    (wrapper / "quality_gate.json").write_text(json.dumps(gate), encoding="utf-8")
    (wrapper / "source_run_manifest.json").write_text(json.dumps(manifest), encoding="utf-8")

    recorded = agent_pipeline_runs._record_unresumable_review_failure(
        wrapper_dir=wrapper,
        code="research_pipeline_approved_run_failed",
        failure_type="error",
        diagnostic="diagnostics/research_pipeline_review_resume_failure.json",
    )

    assert recorded is False
    assert json.loads((wrapper / "quality_gate.json").read_text(encoding="utf-8")) == gate
    assert json.loads((wrapper / "source_run_manifest.json").read_text(encoding="utf-8")) == manifest
