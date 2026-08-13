"""Human-review resume, egress authority, numeric and table-contract tests.

Origin: 2026-07-27 external review (sixth pass).

Same discipline as the previous rounds: every test drives the **production**
entry point. The recurring lesson from round five was that a test which calls a
primitive directly cannot detect that nobody calls it — and this round's own
review found the sequel to that: a test which hand-builds an input the producer
never emits cannot detect that the consumer reads the wrong field.

So the human-review tests here run the real workflow, take the real pause,
resume through its public channel and let the workflow's
``_human_review_decision_record`` reach the *pipeline's own* recorder closure.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from types import MappingProxyType, SimpleNamespace

import pandas as pd
import pytest


# ---------------------------------------------------------------------------
# P0 — human review: real record shape, resume API, no process-local list
# ---------------------------------------------------------------------------


def _plan_result_with_review(evidence, run_dir):
    """A plan result carrying one error-severity finding that demands review."""

    from easyicu.research_agent.schema import (
        AnalysisPlan,
        AnalysisStep,
        ValidationFinding,
    )

    return SimpleNamespace(
        aborted_result=None,
        evidence=evidence,
        plan=AnalysisPlan(
            research_question="Can this capability be used?",
            steps=[
                AnalysisStep(
                    step_id="step_01",
                    intent="Run the requested registered analysis.",
                )
            ],
        ),
        findings=[
            ValidationFinding(
                validator="capability_gate",
                severity="error",
                message="This plan requests a capability that is not registered.",
                detail={"reason": "capability_review_required"},
            )
        ],
        run_dir=run_dir,
    )


def test_p0_graph_record_is_flat_and_the_recorder_reads_it(tmp_path):
    """The record the workflow emits must be the record the recorder consumes.

    The previous recorder read ``record["request"]["review_id"]``. The workflow
    emits a flat record with a top-level ``review_id``, so every real
    unauthenticated decision under a paper profile raised ``KeyError`` instead
    of the intended controlled refusal. A test that hand-built a nested record
    could not see it.
    """

    from easyicu.research_agent.orchestration.workflow import (
        HumanReviewDecision,
        HumanReviewRequest,
        _human_review_decision_record,
    )

    request = HumanReviewRequest.create(
        kind="capability_request",
        summary="capability review",
        authority_sha256="a" * 64,
        payload={"reason": "capability_review_required"},
    )
    record = _human_review_decision_record(
        request=request,
        decision=HumanReviewDecision(
            review_id=request.review_id,
            authority_sha256=request.authority_sha256,
            decision="approved",
            reviewer="dr-who",
            decided_at="2026-07-27T00:00:00Z",
        ),
        reviewer_identity=None,
    )

    assert "request" not in record
    assert record["review_id"] == request.review_id
    assert record["reviewer_identity_source"] == "unauthenticated_client_claim"


def test_p0_paper_profile_refuses_an_unauthenticated_decision(tmp_path):
    """Drive the production recorder with the production record shape."""

    from easyicu.research_agent import pipeline as pipeline_module
    from easyicu.research_agent.orchestration.workflow import (
        HumanReviewDecision,
        HumanReviewRequest,
        _human_review_decision_record,
    )

    request = HumanReviewRequest.create(
        kind="capability_request",
        summary="capability review",
        authority_sha256="b" * 64,
        payload={},
    )
    record = _human_review_decision_record(
        request=request,
        decision=HumanReviewDecision(
            review_id=request.review_id,
            authority_sha256=request.authority_sha256,
            decision="approved",
            reviewer="claimed-name",
            decided_at="2026-07-27T00:00:00Z",
        ),
        reviewer_identity=None,
    )

    recorder = _capture_production_recorder(
        pipeline_module,
        tmp_path,
        submission_profile="npj_dm",
        profile_version="20260719",
    )
    with pytest.raises(RuntimeError, match="authenticated reviewer identity"):
        recorder([record])
    # The refusal must name the review it refused, which is only possible if it
    # read the field the workflow actually writes.
    try:
        recorder([record])
    except RuntimeError as exc:
        assert request.review_id in str(exc)


def _capture_production_recorder(
    pipeline_module, tmp_path, *, submission_profile=None, profile_version=None
):
    """Return ``run()``'s own recorder closure, without running a pipeline.

    ``run()`` builds the recorder inline and hands it to ``build_pipeline_workflow``,
    so the only way to test *the closure the production path installs* is to
    intercept it at that boundary.
    """

    from easyicu.research_agent.providers.mocks import MockLLMClient

    captured = {}

    def _fake_build(**kwargs):
        captured.update(kwargs)
        raise _StopBuilding()

    class _StopBuilding(Exception):
        pass

    import easyicu.research_agent.orchestration.workflow as workflow_module

    real_build = workflow_module.build_pipeline_workflow
    workflow_module.build_pipeline_workflow = _fake_build
    try:
        frame = pd.DataFrame(
            {
                "stay_id": list(range(30)),
                "sofa2": [i % 5 for i in range(30)],
                "death": [i % 3 == 0 for i in range(30)],
            }
        )
        cohort = tmp_path / "cohort.parquet"
        frame.to_parquet(cohort)
        agent = pipeline_module.ResearchAgentPipeline(
            workdir=tmp_path / "wd",
            llm=MockLLMClient(),
            submission_profile_name=submission_profile,
            submission_profile_version=profile_version,
        )
        try:
            agent.run(question="Does SOFA-2 predict death?", cohort=cohort)
        except _StopBuilding:
            pass
        except Exception:
            if "human_review_recorder" not in captured:
                raise
    finally:
        workflow_module.build_pipeline_workflow = real_build
    assert "human_review_recorder" in captured, "run() no longer wires a recorder"
    return captured["human_review_recorder"]


def test_p0_authority_digest_binds_the_plan_and_its_evidence(tmp_path):
    """An approval must not survive a plan revision or an evidence change.

    Before this, the digest covered the finding text alone, so one signature
    authorised any plan that raised the same finding.
    """

    from easyicu.research_agent.orchestration.workflow import (
        human_review_requests_for_plan,
    )
    from easyicu.research_agent.schema import (
        AnalysisPlan,
        AnalysisStep,
        ValidationFinding,
    )

    finding = ValidationFinding(
        validator="capability_gate",
        severity="error",
        message="This plan requests an unregistered capability.",
        detail={"reason": "capability_review_required"},
    )
    plan_v1 = AnalysisPlan(
        research_question="Does the exposure predict the outcome?",
        analysis_type="descriptive",
        steps=[
            AnalysisStep(
                step_id="s1",
                intent="Summarise the cohort.",
                inputs=["cohort"],
                expected_outputs=["table:summary"],
                method="descriptive_summary",
                icu_rule_refs=["icu:first_24h"],
            )
        ],
        display_labels={"outcome": "Hospital mortality"},
        rationale="Pre-specified descriptive analysis.",
        revision=1,
    )
    plan_v2 = plan_v1.model_copy(update={"revision": 2})

    class _Evidence:
        def __init__(self, sha):
            self._sha = sha

        def records(self):
            return [SimpleNamespace(evidence_id="cohort", sha256=self._sha)]

    base = human_review_requests_for_plan(
        findings=[finding], plan=plan_v1, evidence=_Evidence("a" * 64)
    )
    revised = human_review_requests_for_plan(
        findings=[finding], plan=plan_v2, evidence=_Evidence("a" * 64)
    )
    other_evidence = human_review_requests_for_plan(
        findings=[finding], plan=plan_v1, evidence=_Evidence("b" * 64)
    )

    assert base[0].authority_sha256 != revised[0].authority_sha256
    assert base[0].authority_sha256 != other_evidence[0].authority_sha256
    assert base[0].payload["plan_evidence_sha256"] == {"cohort": "a" * 64}
    authority = base[0].payload["plan_review_authority"]
    assert authority["plan_payload"] == plan_v1.model_dump(mode="json")
    assert (
        authority["plan_sha256"]
        == hashlib.sha256(
            json.dumps(
                plan_v1.model_dump(mode="json"),
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
                allow_nan=False,
            ).encode("utf-8")
        ).hexdigest()
    )


@pytest.mark.parametrize(
    ("field", "replacement"),
    [
        ("research_question", "Does a different exposure predict the outcome?"),
        ("analysis_type", "prediction"),
        ("display_labels", {"outcome": "Thirty-day mortality"}),
        ("rationale", "A revised scientific rationale."),
        ("revision", 2),
    ],
)
def test_review_authority_changes_for_every_plan_level_edit(field, replacement):
    from easyicu.research_agent.orchestration.workflow import (
        human_review_requests_for_plan,
    )
    from easyicu.research_agent.schema import (
        AnalysisPlan,
        AnalysisStep,
        ValidationFinding,
    )

    finding = ValidationFinding(
        validator="capability_gate",
        severity="error",
        message="This plan requests an unregistered capability.",
        detail={"reason": "capability_review_required"},
    )
    plan = AnalysisPlan(
        research_question="Does the exposure predict the outcome?",
        analysis_type="descriptive",
        steps=[
            AnalysisStep(
                step_id="s1",
                intent="Summarise the cohort.",
                inputs=["cohort"],
                expected_outputs=["table:summary"],
                method="descriptive_summary",
                icu_rule_refs=["icu:first_24h"],
            )
        ],
        display_labels={"outcome": "Hospital mortality"},
        rationale="Pre-specified descriptive analysis.",
    )
    changed = plan.model_copy(update={field: replacement})

    original_request = human_review_requests_for_plan(
        findings=[finding],
        plan=plan,
    )[0]
    changed_request = human_review_requests_for_plan(
        findings=[finding],
        plan=changed,
    )[0]

    assert original_request.authority_sha256 != changed_request.authority_sha256


@pytest.mark.parametrize(
    ("field", "replacement"),
    [
        ("intent", "Fit a revised descriptive summary."),
        ("inputs", ["revised_cohort"]),
        ("expected_outputs", ["table:revised_summary"]),
        ("method", "revised_descriptive_summary"),
        ("icu_rule_refs", ["icu:first_48h"]),
    ],
)
def test_review_authority_changes_for_every_step_edit(field, replacement):
    from easyicu.research_agent.orchestration.workflow import (
        human_review_requests_for_plan,
    )
    from easyicu.research_agent.schema import (
        AnalysisPlan,
        AnalysisStep,
        ValidationFinding,
    )

    finding = ValidationFinding(
        validator="capability_gate",
        severity="error",
        message="This plan requests an unregistered capability.",
        detail={"reason": "capability_review_required"},
    )
    step = AnalysisStep(
        step_id="s1",
        intent="Summarise the cohort.",
        inputs=["cohort"],
        expected_outputs=["table:summary"],
        method="descriptive_summary",
        icu_rule_refs=["icu:first_24h"],
    )
    plan = AnalysisPlan(
        research_question="Does the exposure predict the outcome?",
        steps=[step],
    )
    changed_step = step.model_copy(update={field: replacement})
    changed = plan.model_copy(update={"steps": [changed_step]})

    original_request = human_review_requests_for_plan(
        findings=[finding],
        plan=plan,
    )[0]
    changed_request = human_review_requests_for_plan(
        findings=[finding],
        plan=changed,
    )[0]

    assert original_request.authority_sha256 != changed_request.authority_sha256


@pytest.mark.parametrize(
    ("field", "replacement"),
    [
        ("pipeline_config_sha256", "b" * 64),
        ("submission_profile_ref", "npj_dm/20260727"),
        ("capability_activation_sha256", "c" * 64),
        ("run_input_capsule_sha256", "d" * 64),
    ],
)
def test_review_authority_changes_for_execution_identity(field, replacement):
    from easyicu.research_agent.authority.plan_review import ReviewExecutionAuthority
    from easyicu.research_agent.orchestration.workflow import (
        human_review_requests_for_plan,
    )
    from easyicu.research_agent.schema import (
        AnalysisPlan,
        AnalysisStep,
        ValidationFinding,
    )

    finding = ValidationFinding(
        validator="capability_gate",
        severity="error",
        message="This plan requests an unregistered capability.",
        detail={"reason": "capability_review_required"},
    )
    plan = AnalysisPlan(
        research_question="Does the exposure predict the outcome?",
        steps=[AnalysisStep(step_id="s1", intent="Summarise the cohort.")],
    )
    execution = ReviewExecutionAuthority(
        pipeline_config_sha256="a" * 64,
        submission_profile_ref="development/1",
        capability_activation_sha256="e" * 64,
        run_input_capsule_sha256="f" * 64,
    )
    changed_execution = execution.model_copy(update={field: replacement})

    original_request = human_review_requests_for_plan(
        findings=[finding],
        plan=plan,
        execution_authority=execution,
    )[0]
    changed_request = human_review_requests_for_plan(
        findings=[finding],
        plan=plan,
        execution_authority=changed_execution,
    )[0]

    assert original_request.authority_sha256 != changed_request.authority_sha256


def test_p0_run_returns_a_typed_pause_not_a_keyerror():
    """A paused workflow has no result, and returns a typed pause."""

    from easyicu.research_agent.orchestration.workflow import (
        HumanReviewPending,
        HumanReviewRequest,
        WorkflowPaused,
    )
    from easyicu.research_agent.pipeline import ResearchAgentPipeline

    request = HumanReviewRequest.create(
        kind="capability_request",
        summary="needs a signature",
        authority_sha256="c" * 64,
        payload={},
    )
    paused_outcome = WorkflowPaused(requests=(request,))

    agent = ResearchAgentPipeline.__new__(ResearchAgentPipeline)
    agent._pending_human_review = None
    pending = agent._pipeline_result_or_pending(
        paused_outcome,
        workflow=object(),
        run_id="run-xyz",
        run_dir=Path("/tmp/run-xyz"),
    )

    assert isinstance(pending, HumanReviewPending)
    assert pending.review_ids == (request.review_id,)
    assert pending.thread_id == "run-xyz"
    assert agent._pending_human_review is not None


def test_p0_an_unknown_workflow_outcome_is_an_error():
    """Absence of both completion and a pause is an orchestration bug."""

    from easyicu.research_agent.pipeline import ResearchAgentPipeline

    agent = ResearchAgentPipeline.__new__(ResearchAgentPipeline)
    agent._pending_human_review = None
    with pytest.raises(RuntimeError, match="neither a completed result"):
        agent._pipeline_result_or_pending(
            object(),
            workflow=object(),
            run_id="run-1",
            run_dir=Path("/tmp/run-1"),
        )


def test_p0_resume_without_a_pause_is_refused():
    from easyicu.research_agent.pipeline import ResearchAgentPipeline

    agent = ResearchAgentPipeline.__new__(ResearchAgentPipeline)
    agent._pending_human_review = None
    with pytest.raises(RuntimeError, match="no human review is pending"):
        agent.resume_human_review([])


def test_rejected_workflow_clears_the_pipeline_pause(tmp_path):
    """The public pipeline must not keep offering a terminal pause to callers."""

    from easyicu.research_agent.orchestration.workflow import HumanReviewRejected
    from easyicu.research_agent.pipeline import ResearchAgentPipeline

    class _RejectingWorkflow:
        def resume(self, *_args, **_kwargs):
            raise HumanReviewRejected(("review-0123456789abcdef",))

    class _Pending:
        run_id = "20260726T110000_abcdef"
        run_dir = str(tmp_path / "run")
        resumable_here = True

    Path(_Pending.run_dir).mkdir(parents=True)
    agent = ResearchAgentPipeline(workdir=tmp_path / "wd")
    agent._pending_human_review = {
        "workflow": _RejectingWorkflow(),
        "pending": _Pending(),
        "runtime_capabilities": (),
        "runtime_bundle": None,
    }

    with pytest.raises(HumanReviewRejected):
        agent.resume_human_review([])

    assert agent._pending_human_review is None


def test_failed_resume_clears_the_pipeline_pause(tmp_path):
    """A post-approval execution failure leaves no resumable live handoff."""

    from easyicu.research_agent.orchestration.workflow import (
        HumanReviewDecision,
        HumanReviewRequest,
        WorkflowPaused,
        build_pipeline_workflow,
    )
    from easyicu.research_agent.pipeline import ResearchAgentPipeline

    request = HumanReviewRequest.create(
        kind="capability_request",
        summary="Approve the execution environment",
        authority_sha256="f" * 64,
        payload={},
    )

    def _fail_execute(_plan):
        raise RuntimeError("execute failed after approval")

    workflow = build_pipeline_workflow(
        plan_invoker=lambda: SimpleNamespace(aborted_result=None),
        execute_invoker=_fail_execute,
        write_invoker=lambda _plan, _execute: pytest.fail(
            "execution failure reached writing"
        ),
        finalise_invoker=lambda _plan, _execute, _write: pytest.fail(
            "execution failure reached finalisation"
        ),
        human_review_invoker=lambda _plan: (request,),
    )
    assert isinstance(workflow.start(), WorkflowPaused)

    class _Pending:
        run_id = "20260726T111000_abcdef"
        run_dir = str(tmp_path / "run")
        resumable_here = True

    Path(_Pending.run_dir).mkdir(parents=True)
    agent = ResearchAgentPipeline(workdir=tmp_path / "wd")
    agent._pending_human_review = {
        "workflow": workflow,
        "pending": _Pending(),
        "runtime_capabilities": (),
        "runtime_bundle": None,
    }

    with pytest.raises(RuntimeError, match="execute failed after approval"):
        agent.resume_human_review(
            [
                HumanReviewDecision(
                    review_id=request.review_id,
                    authority_sha256=request.authority_sha256,
                    decision="approved",
                    reviewer="reviewer",
                    decided_at="2026-07-27T09:40:00Z",
                )
            ]
        )

    assert workflow.state == "failed"
    assert agent._pending_human_review is None


def test_correctable_resume_error_keeps_the_pipeline_pause(tmp_path):
    """Bad decision input remains retryable while the workflow stays paused."""

    from easyicu.research_agent.pipeline import ResearchAgentPipeline

    class _PausedWorkflow:
        state = "paused"

        def resume(self, *_args, **_kwargs):
            raise ValueError("human review decision authority digest mismatch")

    class _Pending:
        run_id = "20260726T112000_abcdef"
        run_dir = str(tmp_path / "run")
        resumable_here = True

    Path(_Pending.run_dir).mkdir(parents=True)
    agent = ResearchAgentPipeline(workdir=tmp_path / "wd")
    pending_state = {
        "workflow": _PausedWorkflow(),
        "pending": _Pending(),
        "runtime_capabilities": (),
        "runtime_bundle": None,
    }
    agent._pending_human_review = pending_state

    with pytest.raises(ValueError, match="authority digest mismatch"):
        agent.resume_human_review([])

    assert agent._pending_human_review is pending_state


def test_resume_rebinds_progress_to_the_current_transport_job(tmp_path):
    """Post-review execution events belong to the resume job, not the old one."""

    from easyicu.research_agent.pipeline import ResearchAgentPipeline
    from easyicu.research_agent.orchestration.progress import (
        ResumableProgressChannel,
    )

    old_events: list[dict] = []
    current_events: list[dict] = []
    progress_channel = ResumableProgressChannel(old_events.append)

    class _ResumingWorkflow:
        state = "paused"

        def resume(self, *_args, **_kwargs):
            progress_channel.emit("coder", "Generating analysis code.")
            self.state = "completed"
            return "done"

    class _Pending:
        run_id = "20260812T100000_progress"
        run_dir = str(tmp_path / "run")
        resumable_here = True

    Path(_Pending.run_dir).mkdir(parents=True)
    agent = ResearchAgentPipeline(workdir=tmp_path / "wd")
    agent._pending_human_review = {
        "workflow": _ResumingWorkflow(),
        "pending": _Pending(),
        "runtime_capabilities": (),
        "runtime_bundle": None,
        "progress_sink": progress_channel,
    }
    agent._pipeline_result_or_pending = lambda outcome, **_kwargs: outcome

    result = agent.resume_human_review(
        [], progress_callback=current_events.append
    )

    assert result == "done"
    assert old_events == []
    assert len(current_events) == 1
    assert current_events[0]["stage"] == "coder"
    assert current_events[0]["message"] == "Generating analysis code."
    assert current_events[0]["status"] == "running"
    assert current_events[0]["timestamp"]


def test_p0_full_pause_and_resume_through_the_real_workflow(tmp_path):
    """plan → pause → resume → real record → recorder → evidence.

    This is the contract the previous round could not have caught: the record
    that reaches the recorder is produced by ``_human_review_decision_record``,
    not by the test.
    """

    from easyicu.research_agent.authority.evidence_store import EvidenceStore
    from easyicu.research_agent.orchestration.workflow import (
        HumanReviewDecision,
        WorkflowPaused,
        build_pipeline_workflow,
        human_review_requests_for_plan,
    )

    run_dir = tmp_path / "run"
    run_dir.mkdir()
    evidence = EvidenceStore(run_dir)
    plan_result = _plan_result_with_review(evidence, run_dir)
    recorded: list = []

    workflow = build_pipeline_workflow(
        plan_invoker=lambda: plan_result,
        execute_invoker=lambda plan: SimpleNamespace(plan=plan),
        write_invoker=lambda plan, execute: SimpleNamespace(),
        finalise_invoker=lambda plan, execute, write: {"ok": True},
        human_review_invoker=lambda plan: human_review_requests_for_plan(
            findings=plan.findings, plan=plan.plan
        ),
        human_review_recorder=recorded.extend,
        reviewer_identity_resolver=lambda: "sso:reviewer@hospital",
    )

    paused = workflow.start()
    assert isinstance(paused, WorkflowPaused)
    assert len(paused.requests) == 1
    request = paused.requests[0]

    decision = HumanReviewDecision(
        review_id=request.review_id,
        authority_sha256=request.authority_sha256,
        decision="approved",
        reviewer="Dr Reviewer",
        decided_at="2026-07-27T09:00:00Z",
    )
    final = workflow.resume([decision])

    assert final.final_result == {"ok": True}
    assert len(recorded) == 1
    assert recorded[0]["review_id"] == request.review_id
    assert recorded[0]["reviewer_identity"] == "sso:reviewer@hospital"
    assert recorded[0]["reviewer_identity_source"] == "authenticated"


def test_p0_recorder_reopens_evidence_for_direct_diagnostics(tmp_path):
    """A directly exercised recorder must not index an absent live handoff."""

    from easyicu.research_agent import pipeline as pipeline_module

    recorder = _capture_production_recorder(
        pipeline_module,
        tmp_path,
    )
    record = {
        "schema": "easyicu.human_review_decision/1",
        "review_id": "review-0123456789abcdef",
        "authority_sha256": "d" * 64,
        "decision": "approved",
        "reviewer_identity": "sso:reviewer",
        "reviewer_identity_source": "authenticated",
        "server_decided_at": "2026-07-27T09:00:00Z",
    }

    # No pipeline ever ran, so the closure's ``reviewed_plan`` list is empty.
    recorder([record])

    run_dirs = list((tmp_path / "wd").glob("*/human_review_decisions.json"))
    assert run_dirs, "the recorder wrote no decisions file"
    payload = json.loads(run_dirs[0].read_text(encoding="utf-8"))
    assert payload["decisions"][0]["review_id"] == "review-0123456789abcdef"
    assert not list((tmp_path / "wd").glob("*/run_status.json"))


def test_rejected_review_persists_a_run_level_terminal_status(tmp_path):
    """A restarted process can identify rejection without live workflow state."""

    from easyicu.research_agent import pipeline as pipeline_module
    from easyicu.research_agent.authority.evidence_store import EvidenceStore
    from easyicu.research_agent.orchestration.workflow import (
        HumanReviewDecision,
        HumanReviewRejected,
        HumanReviewRequest,
        WorkflowPaused,
        build_pipeline_workflow,
    )

    recorder = _capture_production_recorder(
        pipeline_module,
        tmp_path,
    )
    request = HumanReviewRequest.create(
        kind="scientific_stop",
        summary="Reject an unresolved scientific stop",
        authority_sha256="e" * 64,
        payload={"finding": "positivity_not_established"},
    )
    workflow = build_pipeline_workflow(
        plan_invoker=lambda: SimpleNamespace(aborted_result=None),
        execute_invoker=lambda _plan: pytest.fail("rejection reached execution"),
        write_invoker=lambda _plan, _execute: pytest.fail("rejection reached writing"),
        finalise_invoker=lambda _plan, _execute, _write: pytest.fail(
            "rejection reached finalisation"
        ),
        human_review_invoker=lambda _plan: (request,),
        human_review_recorder=recorder,
        reviewer_identity_resolver=lambda: "sso:reviewer",
    )
    assert isinstance(workflow.start(), WorkflowPaused)
    with pytest.raises(HumanReviewRejected):
        workflow.resume(
            [
                HumanReviewDecision(
                    review_id=request.review_id,
                    authority_sha256=request.authority_sha256,
                    decision="rejected",
                    reviewer="reviewer",
                    decided_at="2026-07-27T09:30:00Z",
                )
            ]
        )
    assert workflow.state == "rejected"

    status_paths = list((tmp_path / "wd").glob("*/run_status.json"))
    assert len(status_paths) == 1
    status_path = status_paths[0]
    payload = json.loads(status_path.read_text(encoding="utf-8"))
    assert payload["status"] == "human_review_rejected"
    assert payload["terminal_reason"] == "operator_rejected"
    assert payload["rejected_review_ids"] == [request.review_id]
    assert payload["gates"]["paper_authorized"] is False
    assert payload["canonical_outputs"]["human_review_decisions"] == (
        "human_review_decisions.json"
    )

    evidence = EvidenceStore(status_path.parent)
    status_record = evidence.get("run_status")
    assert status_record is not None
    assert status_record.generation_mode == "system"
    assert evidence.get("human_review_decisions") is not None


# ---------------------------------------------------------------------------
# P0 — figure egress: host-owned privacy audit, two-phase receipt
# ---------------------------------------------------------------------------


def _contract(figure_id="Figure2", roles=("primary_estimand",), sources=()):
    return SimpleNamespace(
        figure_id=figure_id,
        core_claim="SOFA-2 predicts mortality.",
        statistics_note=None,
        image_integrity_note=None,
        panels=[
            SimpleNamespace(role=role, title=f"Panel {i}", claim="A claim.")
            for i, role in enumerate(roles)
        ],
        source_data=list(sources),
    )


class _FakeEvidence:
    def __init__(self, root, records):
        self.root = root
        self._records = records

    def get(self, evidence_id):
        return self._records.get(evidence_id)


def _record(relative_path, sha=None, root=None):
    """A record whose digest matches the file, the way a real store's does.

    A hand-written placeholder digest would make every audit fail at the
    re-hash step for a reason the test is not about — and would hide whichever
    check it was actually written to exercise. ``sha`` is overridable so the
    digest-mismatch path can be tested deliberately.
    """

    if sha is None and root is not None:
        path = Path(root) / relative_path
        sha = _sha256_of(path) if path.is_file() else "e" * 64
    return SimpleNamespace(
        relative_path=relative_path, sha256=sha or "e" * 64, kind="figure"
    )


def _sha256_of(path):
    digest = hashlib.sha256()
    digest.update(Path(path).read_bytes())
    return digest.hexdigest()


def test_p0_privacy_audit_refuses_a_source_with_a_subject_identifier(tmp_path):
    """An allow-listed panel role does not clear a per-patient source table."""

    from easyicu.research_agent.gates.figure_privacy import audit_figure_privacy
    from easyicu.research_agent.figures.skill import AGGREGATE_ONLY_PANEL_ROLES

    source = tmp_path / "primary.csv"
    source.write_text("stay_id,risk\n1,0.5\n2,0.6\n", encoding="utf-8")
    evidence = _FakeEvidence(
        tmp_path, {"primary": _record("primary.csv", root=tmp_path)}
    )

    audit = audit_figure_privacy(
        contract=_contract(roles=("primary_estimand",), sources=("primary",)),
        evidence=evidence,
        run_dir=tmp_path,
        source_evidence_ids=["primary"],
        allowed_panel_roles=AGGREGATE_ONLY_PANEL_ROLES,
    )

    assert audit.aggregate_only is False
    assert any("stay_id" in reason for reason in audit.reasons)


def test_p0_privacy_audit_clears_an_aggregate_source(tmp_path):
    from easyicu.research_agent.gates.figure_privacy import audit_figure_privacy
    from easyicu.research_agent.figures.skill import AGGREGATE_ONLY_PANEL_ROLES

    source = tmp_path / "summary.json"
    source.write_text(
        json.dumps({"odds_ratio": 1.42, "ci_lower": 1.11, "n_patients": 4_218}),
        encoding="utf-8",
    )
    evidence = _FakeEvidence(
        tmp_path, {"summary": _record("summary.json", root=tmp_path)}
    )

    audit = audit_figure_privacy(
        contract=_contract(roles=("primary_estimand", "calibration")),
        evidence=evidence,
        run_dir=tmp_path,
        source_evidence_ids=["summary"],
        allowed_panel_roles=AGGREGATE_ONLY_PANEL_ROLES,
    )

    assert audit.aggregate_only is True, audit.reasons
    assert audit.as_metadata()["aggregate_only_basis"] == "host_privacy_audit"
    # Honest about the question it did not answer.
    assert audit.mark_count_verified is False


def test_p0_privacy_audit_refuses_a_small_declared_group(tmp_path):
    from easyicu.research_agent.gates.figure_privacy import audit_figure_privacy

    source = tmp_path / "strata.json"
    source.write_text(json.dumps({"strata": [{"n_patients": 3}]}), encoding="utf-8")
    evidence = _FakeEvidence(
        tmp_path, {"strata": _record("strata.json", root=tmp_path)}
    )

    audit = audit_figure_privacy(
        contract=_contract(roles=("heterogeneity",)),
        evidence=evidence,
        run_dir=tmp_path,
        source_evidence_ids=["strata"],
    )

    assert audit.aggregate_only is False
    assert any("group size" in reason for reason in audit.reasons)


def test_p0_privacy_audit_refuses_an_uninspectable_source(tmp_path):
    """What the host cannot open, the host cannot clear."""

    from easyicu.research_agent.gates.figure_privacy import audit_figure_privacy

    source = tmp_path / "opaque.bin"
    source.write_bytes(b"\x00\x01\x02")
    evidence = _FakeEvidence(tmp_path, {"opaque": _record("opaque.bin", root=tmp_path)})

    audit = audit_figure_privacy(
        contract=_contract(),
        evidence=evidence,
        run_dir=tmp_path,
        source_evidence_ids=["opaque"],
    )

    assert audit.aggregate_only is False
    assert any("cannot be inspected" in reason for reason in audit.reasons)


def test_p0_privacy_audit_refuses_a_missing_source(tmp_path):
    from easyicu.research_agent.gates.figure_privacy import audit_figure_privacy

    evidence = _FakeEvidence(tmp_path, {"gone": _record("gone.json", root=tmp_path)})
    audit = audit_figure_privacy(
        contract=_contract(),
        evidence=evidence,
        run_dir=tmp_path,
        source_evidence_ids=["gone"],
    )

    assert audit.aggregate_only is False
    assert any("missing from the run directory" in r for r in audit.reasons)


def test_p0_privacy_audit_refuses_an_identifier_in_rendered_text(tmp_path):
    """Panel titles and claims are drawn into the image."""

    from easyicu.research_agent.gates.figure_privacy import audit_figure_privacy

    source = tmp_path / "summary.json"
    source.write_text(json.dumps({"odds_ratio": 1.42}), encoding="utf-8")
    evidence = _FakeEvidence(
        tmp_path, {"summary": _record("summary.json", root=tmp_path)}
    )
    contract = _contract()
    contract.panels[0].claim = "Index case 30042318 drives the effect."

    audit = audit_figure_privacy(
        contract=contract,
        evidence=evidence,
        run_dir=tmp_path,
        source_evidence_ids=["summary"],
    )

    assert audit.aggregate_only is False
    assert any("identifier-shaped token" in reason for reason in audit.reasons)


def test_p0_role_alone_no_longer_authorizes_egress(tmp_path):
    """The exact bypass the review named: role says validation, data is per-stay."""

    from easyicu.research_agent.gates.figure_privacy import audit_figure_privacy
    from easyicu.research_agent.figures.skill import AGGREGATE_ONLY_PANEL_ROLES

    source = tmp_path / "scatter.csv"
    source.write_text(
        "subject_id,predicted,observed\n1,0.2,0\n2,0.4,1\n", encoding="utf-8"
    )
    evidence = _FakeEvidence(
        tmp_path, {"scatter": _record("scatter.csv", root=tmp_path)}
    )

    audit = audit_figure_privacy(
        contract=_contract(roles=("validation",)),
        evidence=evidence,
        run_dir=tmp_path,
        source_evidence_ids=["scatter"],
        allowed_panel_roles=AGGREGATE_ONLY_PANEL_ROLES,
    )

    assert "validation" in AGGREGATE_ONLY_PANEL_ROLES
    assert audit.aggregate_only is False


def test_p0_egress_receipt_failure_raises_a_typed_blocker(tmp_path):
    from easyicu.research_agent.gates.figure_egress import (
        FigureEgressPolicy,
        FigureEgressReceiptError,
        register_figure_egress_receipt,
    )

    class _Failing:
        def register_file(self, **_kwargs):
            raise OSError("disk full")

    policy = FigureEgressPolicy(allow_external_upload=True)
    policy.record_upload([{"path": "fig.png", "sha256": "f" * 64}])

    with pytest.raises(FigureEgressReceiptError, match="cannot account for"):
        register_figure_egress_receipt(
            policy=policy, evidence=_Failing(), run_dir=tmp_path
        )


def test_p0_egress_receipt_has_two_phases(tmp_path):
    from easyicu.research_agent.gates.figure_egress import (
        FigureEgressPolicy,
        register_figure_egress_receipt,
    )

    registered = []

    class _Store:
        def register_file(self, **kwargs):
            registered.append(kwargs)
            return SimpleNamespace(evidence_id=kwargs["evidence_id"])

    policy = FigureEgressPolicy(allow_external_upload=True)
    register_figure_egress_receipt(
        policy=policy, evidence=_Store(), run_dir=tmp_path, phase="intent"
    )
    policy.record_upload([{"path": "fig.png", "sha256": "a" * 64}])
    register_figure_egress_receipt(
        policy=policy, evidence=_Store(), run_dir=tmp_path, phase="completed"
    )

    ids = [entry["evidence_id"] for entry in registered]
    assert ids == ["figure_egress_authorization_intent", "figure_egress_receipt"]
    intent = json.loads(
        (tmp_path / "figure_egress_authorization_intent.json").read_text("utf-8")
    )
    completed = json.loads((tmp_path / "figure_egress_receipt.json").read_text("utf-8"))
    assert intent["authorized_count"] == 0
    assert completed["authorized_count"] == 1
    # Authorized but never closed out: recorded as unknown, not as a success.
    assert completed["transport_counts"] == {"transport_unknown": 1}


def test_p0_write_phase_does_not_demote_an_egress_receipt_failure():
    """The blanket ``except Exception`` must not swallow this one."""

    import inspect

    from easyicu.research_agent.reporting import write_phase

    source = inspect.getsource(write_phase)
    marker = "except FigureEgressReceiptError:"
    blanket = "except Exception as exc:\n            findings.append("
    assert marker in source
    assert source.index(marker) < source.index(blanket), (
        "the typed egress-receipt handler must precede the blanket handler, "
        "or the blanket one catches it first"
    )


# ---------------------------------------------------------------------------
# MCP — three remaining projection bypasses
# ---------------------------------------------------------------------------


def test_mcp_non_frame_result_is_withheld_not_repr_dumped():
    from easyicu.research_agent.mcp_policy import DisclosurePolicy, summarise_frame

    class _Weird:
        def __repr__(self):
            return "stay_id=30042318 charttime=2180-01-01 lactate=8.1"

    summary = summarise_frame(
        _Weird(),
        policy=DisclosurePolicy(
            patient_data=False, preview_rows=0, include_identifier_columns=False
        ),
    )

    assert summary["unsupported_result"] is True
    assert "repr" not in summary
    assert "30042318" not in json.dumps(summary)


def test_mcp_small_cell_size_is_bounded_not_exact():
    from easyicu.research_agent.mcp_policy import (
        MIN_NON_MISSING_FOR_COLUMN_STATS,
        DisclosurePolicy,
        summarise_frame,
    )

    frame = pd.DataFrame({"lactate": [8.1, 7.2, 6.3] + [None] * 97})
    summary = summarise_frame(
        frame,
        policy=DisclosurePolicy(
            patient_data=False, preview_rows=0, include_identifier_columns=False
        ),
    )

    stats = summary["aggregate_statistics"]["lactate"]
    assert stats["withheld"] is True
    assert stats["non_missing_count"] == f"<{MIN_NON_MISSING_FOR_COLUMN_STATS}"
    # The exact size must not be recoverable from the missingness either.
    assert isinstance(summary["missing_fraction"]["lactate"], str)
    assert summary["missing_fraction"]["lactate"].startswith(">")


def test_mcp_two_small_cells_are_indistinguishable():
    from easyicu.research_agent.mcp_policy import DisclosurePolicy, summarise_frame

    a = pd.DataFrame({"x": [1.0] * 3 + [None] * 97})
    b = pd.DataFrame({"x": [1.0] * 17 + [None] * 83})
    policy = DisclosurePolicy(
        patient_data=False, preview_rows=0, include_identifier_columns=False
    )

    left = summarise_frame(a, policy=policy)
    right = summarise_frame(b, policy=policy)

    assert left["missing_fraction"]["x"] == right["missing_fraction"]["x"]
    assert (
        left["aggregate_statistics"]["x"]["non_missing_count"]
        == right["aggregate_statistics"]["x"]["non_missing_count"]
    )


def test_mcp_auditor_findings_are_projected(monkeypatch, tmp_path):
    """``audit_cohort`` / ``run_validator`` used to return the full dump."""

    from easyicu.research_agent import mcp_server
    from easyicu.research_agent.mcp_policy import MCP_ALLOWED_ROOTS_ENV, MCP_SCOPES_ENV
    from easyicu.research_agent.schema import ValidationFinding

    monkeypatch.setenv(MCP_ALLOWED_ROOTS_ENV, str(tmp_path))
    monkeypatch.setenv(MCP_SCOPES_ENV, "metadata")

    finding = ValidationFinding(
        validator="cohort_auditor",
        severity="error",
        message="Only 3 stays in /Volumes/drive/db have lactate.",
        detail={
            "reason": "sparse_concept",
            "column": "lactate",
            "min_cell_size": 3,
            "path": "/Volumes/drive/db/cohort.parquet",
        },
    )

    projected = mcp_server._safe_finding_payload(finding)

    assert projected["validator"] == "cohort_auditor"
    assert projected["severity"] == "error"
    assert projected["detail"] == {"reason": "sparse_concept", "column": "lactate"}
    assert set(projected["detail_withheld_keys"]) == {"min_cell_size", "path"}
    rendered = json.dumps(projected)
    assert "/Volumes/drive" not in rendered
    assert "Only 3 stays" not in rendered


def test_mcp_default_scopes_are_metadata_only():
    from easyicu.research_agent.mcp_policy import (
        DEFAULT_SCOPES,
        SCOPE_BIND_EVIDENCE,
        SCOPE_METADATA,
        SCOPE_RUN_PIPELINE,
        SCOPE_WRITE_ARTIFACTS,
    )

    assert DEFAULT_SCOPES == frozenset({SCOPE_METADATA})
    for scope in (SCOPE_RUN_PIPELINE, SCOPE_WRITE_ARTIFACTS, SCOPE_BIND_EVIDENCE):
        assert scope not in DEFAULT_SCOPES


# ---------------------------------------------------------------------------
# Numeric contract — misfires, Chinese, and uniform step scoping
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "text",
    [
        "stage 4 patients had higher mortality",
        "Stage 4 patients had higher mortality",
        "grade 3 patients were excluded",
        "class 2 patients",
        "type 1 patients",
        "group 2 patients",
        "arm 2 patients",
        "level 3 patients",
        "day 7 patients",
        "week 2 events",
    ],
)
def test_category_labels_are_not_bound_as_counts(text):
    from easyicu.research_agent.authority.evidence_store import _NUMERIC_IN_PROSE_RE

    assert [m.group("value") for m in _NUMERIC_IN_PROSE_RE.finditer(text)] == []


@pytest.mark.parametrize(
    "text,expected",
    [
        ("42例患者纳入分析", ["42"]),
        ("其中8例死亡", ["8"]),
        ("共17个事件", ["17"]),
        ("纳入 30 名受试者", ["30"]),
        ("the subgroup included 42 patients", ["42"]),
    ],
)
def test_short_counts_bind_in_both_languages(text, expected):
    from easyicu.research_agent.authority.evidence_store import _NUMERIC_IN_PROSE_RE

    assert [m.group("value") for m in _NUMERIC_IN_PROSE_RE.finditer(text)] == expected


def _summary(step_id, **values):
    return {"step_id": step_id, **values}


def test_brier_claim_is_scoped_to_its_own_step():
    from easyicu.research_agent.audits.manuscript_claims import (
        audit_manuscript_numeric_claims,
    )

    manuscript = (
        "The primary model achieved a Brier score of 0.180[^claim_1].\n\n"
        "[^claim_1]: value=0.180; step=step_primary; evidence=step_summary\n"
    )
    records = [
        {
            "step_id": "step_primary",
            "status": "ok",
            "step_summary": {"brier_score": 0.180},
        },
        {
            "step_id": "step_sensitivity",
            "status": "ok",
            "step_summary": {"brier_score": 0.090},
        },
    ]

    assert audit_manuscript_numeric_claims(manuscript, per_step_records=records) == []

    borrowed = manuscript.replace("0.180", "0.090")
    findings = audit_manuscript_numeric_claims(borrowed, per_step_records=records)
    assert [f.detail["metric"] for f in findings] == ["brier_score"]
    assert findings[0].detail["scoped_to_step"] == "step_primary"


def test_prevalence_claim_is_scoped_to_its_own_step():
    from easyicu.research_agent.audits.manuscript_claims import (
        audit_manuscript_numeric_claims,
    )

    manuscript = (
        "Overall mortality was 18.0%[^claim_2].\n\n"
        "[^claim_2]: value=0.180; step=step_primary; evidence=step_summary\n"
    )
    records = [
        {
            "step_id": "step_primary",
            "status": "ok",
            "step_summary": {"event_rate": 0.180},
        },
        {
            "step_id": "step_other",
            "status": "ok",
            "step_summary": {"event_rate": 0.400},
        },
    ]

    assert audit_manuscript_numeric_claims(manuscript, per_step_records=records) == []

    borrowed = manuscript.replace("18.0%", "40.0%")
    findings = audit_manuscript_numeric_claims(borrowed, per_step_records=records)
    assert [f.detail["metric"] for f in findings] == ["baseline_prevalence"]
    assert findings[0].detail["scoped_to_step"] == "step_primary"


def test_an_unbound_claim_still_falls_back_to_match_any():
    """Scoping must not turn every unfootnoted number into a false positive."""

    from easyicu.research_agent.audits.manuscript_claims import (
        audit_manuscript_numeric_claims,
    )

    manuscript = "The model achieved a Brier score of 0.090.\n"
    records = [
        {
            "step_id": "step_primary",
            "status": "ok",
            "step_summary": {"brier_score": 0.180},
        },
        {
            "step_id": "step_sensitivity",
            "status": "ok",
            "step_summary": {"brier_score": 0.090},
        },
    ]

    assert audit_manuscript_numeric_claims(manuscript, per_step_records=records) == []


# ---------------------------------------------------------------------------
# PipelineConfig — freezing the containers, not just the field references
# ---------------------------------------------------------------------------


def test_config_containers_are_immutable(tmp_path):
    from easyicu.research_agent.orchestration.config import PipelineConfig

    runner_kwargs = {"image": "easyicu:1", "env": {"A": "1"}}
    config = PipelineConfig(workdir=tmp_path, runner_kwargs=runner_kwargs)

    assert isinstance(config.runner_kwargs, MappingProxyType)
    with pytest.raises(TypeError):
        config.runner_kwargs["image"] = "attacker:latest"
    with pytest.raises(TypeError):
        config.runner_kwargs["env"]["A"] = "2"

    # Mutating the caller's original dict must not reach the config either.
    runner_kwargs["image"] = "attacker:latest"
    assert config.runner_kwargs["image"] == "easyicu:1"


def test_config_still_expands_as_kwargs(tmp_path):
    from easyicu.research_agent.orchestration.config import PipelineConfig

    config = PipelineConfig(workdir=tmp_path, runner_kwargs={"image": "easyicu:1"})
    kwargs = config.as_kwargs()

    assert kwargs["workdir"] == tmp_path
    assert dict(kwargs["runner_kwargs"]) == {"image": "easyicu:1"}


def test_config_leaves_live_objects_alone(tmp_path):
    """Live collaborators are frozen by reference outside run configuration."""

    import threading

    from easyicu.research_agent.orchestration.config import PipelineConfig
    from easyicu.research_agent.orchestration.services import PipelineServices

    client = SimpleNamespace(lock=threading.Lock())
    config = PipelineConfig(workdir=tmp_path)
    services = PipelineServices(llm=client)

    assert "llm" not in config.as_kwargs()
    assert services.llm is client
    assert services.canonical_payload()["llm"] == "types.SimpleNamespace"


# ---------------------------------------------------------------------------
# P1-8 — dur_unit across rbind / cbind / merge
# ---------------------------------------------------------------------------


def _win(unit, values=(1.0, 2.0)):
    from easyicu.table import WinTbl
    from easyicu.table.duration import set_dur_var_unit

    frame = pd.DataFrame({"id": [1, 2], "time": [0, 1], "dur": list(values)})
    if unit:
        set_dur_var_unit(frame, unit)
    return WinTbl(frame, "id", "time", "dur", dur_unit=unit)


def test_rbind_preserves_a_shared_duration_unit():
    from easyicu.table import rbind_tbl
    from easyicu.table.duration import get_dur_var_unit

    combined = rbind_tbl(_win("hours"), _win("hours", (3.0, 4.0)))

    assert combined.dur_unit == "hours"
    assert get_dur_var_unit(combined.data) == "hours"


def test_rbind_rejects_mixing_minutes_and_hours():
    from easyicu.table import rbind_tbl
    from easyicu.table.duration import DurationUnitError

    with pytest.raises(DurationUnitError, match="different units"):
        rbind_tbl(_win("hours"), _win("minutes", (60.0, 120.0)))


def test_rbind_rejects_an_undeclared_input_under_strict_units():
    from easyicu.table import rbind_tbl
    from easyicu.table.duration import DurationUnitError

    with pytest.raises(DurationUnitError, match="declare no unit"):
        rbind_tbl(_win("hours"), _win(None))


def test_converting_first_makes_the_bind_legal():
    from easyicu.table import WinTbl, rbind_tbl
    from easyicu.table.duration import convert_dur_var_unit

    minutes = _win("minutes", (60.0, 120.0))
    converted = convert_dur_var_unit(
        minutes.data, column="dur", from_unit="minutes", to_unit="hours"
    )
    combined = rbind_tbl(
        _win("hours"), WinTbl(converted, "id", "time", "dur", dur_unit="hours")
    )

    assert combined.dur_unit == "hours"
    assert list(combined.data["dur"]) == [1.0, 2.0, 1.0, 2.0]


def test_timedelta_and_numeric_durations_never_convert():
    from easyicu.table.duration import DurationUnitError, convert_dur_var_unit

    frame = pd.DataFrame({"dur": [1.0]})
    with pytest.raises(DurationUnitError, match="different\n?\\s*representations"):
        convert_dur_var_unit(
            frame, column="dur", from_unit="timedelta", to_unit="hours"
        )


def test_merge_with_a_covariate_table_keeps_the_unit():
    """A joined table with no duration column is not party to the contract."""

    from easyicu.table import merge_lst

    covariates = pd.DataFrame({"id": [1, 2], "time": [0, 1], "age": [61, 74]})
    merged = merge_lst([_win("hours"), covariates], by=["id", "time"])

    assert merged.dur_unit == "hours"


def test_rbind_lst_preserves_the_unit():
    from easyicu.table import rbind_lst

    assert rbind_lst([_win("hours"), _win("hours", (5.0, 6.0))]).dur_unit == "hours"
