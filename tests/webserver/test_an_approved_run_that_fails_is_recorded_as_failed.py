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
from easyicu.research_agent.planning.progressive_contract import (
    ProgressivePlanCompileError,
)
from easyicu.research_agent.schema import AnalysisPlan, AnalysisStep
from easyicu.webserver import agent_pipeline_runs, agent_runs, dataio
from easyicu.webserver import study_contexts as study_context_owner
from easyicu.webserver.pi_copilot.contracts import (
    EXECUTION_RETRY_REPLAYABLE_GATE_REASONS,
)
from easyicu.webserver.pi_copilot.workflow import build_research_workflow_snapshot
from tests.support.paused_plan_review import _RUN_ID, _FailingPipeline, _paused
from tests.webserver.copilot.research_workflow_fixtures import (
    _acquisition_receipt,
    complete_study,
)


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


def test_a_typed_stop_that_ends_the_review_keeps_its_cause(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # Any owner's typed stop and cause; the record carries the codes only.
    stop = ProgressivePlanCompileError(
        "progressive_resume_outline_authority_mismatch",
        "the resumed outline offers other families than the approved one",
        path="outline",
        cause_code="candidate_families_changed",
    )
    project_root, _wrapper, study = _paused(
        tmp_path, monkeypatch, _FailingPipeline(stop, resumable=False)
    )

    _approve(study)

    row = _row(project_root, study)
    assert row["run_status"] == "failed"
    assert row["gate_detail"] == {
        "reason_code": "progressive_resume_outline_authority_mismatch",
        "cause_code": "candidate_families_changed",
    }

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


def test_an_approved_resume_keeps_the_active_step_through_its_wrapper(tmp_path, monkeypatch):
    from types import SimpleNamespace

    class Pipeline:
        _config = SimpleNamespace(runner_kind="host")
        has_resumable_human_review = False

        def resume_human_review(self, decisions, *, run_id, progress_callback):
            progress_callback({"phase": "execute", "step_id": "absolute_risk_context", "status": "running"})
            raise KeyError("synthetic private column")

    root, wrapper, study = _paused(tmp_path, monkeypatch, Pipeline())
    error = _approve(study)
    assert error.code == "research_pipeline_approved_run_failed"
    receipt = json.loads((wrapper / error.details["diagnostic"]).read_text())
    assert receipt["typed_failure"] == {
        "owner": "easyicu.execution.pipeline_v1", "reason_code": "execution_step_raised",
        "step_id": "absolute_risk_context", "phase": "execute", "runner_kind": "host", "exception_type": "KeyError",
    }
    from easyicu.webserver.pi_copilot.projections import gate_detail_projection
    detail = gate_detail_projection(_row(root, study)["gate_detail"])
    assert detail["gate_detail_step_id"] == "absolute_risk_context"
    assert detail["gate_detail_code"] == "execution_step_raised"
    assert json.loads((wrapper / "source_run_manifest.json").read_text())["analysis_started"] is True
