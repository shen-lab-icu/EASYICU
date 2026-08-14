"""What a reviewer approved must be what actually executes.

A decision's ``authority_sha256`` proves it answers the request that was made.
It cannot prove the run still holds the plan that request described: the plan
handoff is a live mutable object retained across the pause. These tests cover
the gap between "the signature matches the request" and "the signature matches
what would run", plus the retry path that a partially failed decision write
leaves behind.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from easyicu.research_agent.orchestration.workflow import (
    HumanReviewAuthorityError,
    HumanReviewDecision,
    HumanReviewPending,
    HumanReviewRejected,
    HumanReviewStateDrift,
    PipelineRunOutcome,
    WorkflowCompleted,
    WorkflowPaused,
    build_pipeline_workflow,
    human_review_requests_for_plan,
)
from easyicu.research_agent.schema import (
    AnalysisPlan,
    AnalysisStep,
    ValidationFinding,
)


# ---------------------------------------------------------------------------
# Fixtures: a live handoff whose plan and evidence the test can mutate, exactly
# as anything holding the paused workflow could.
# ---------------------------------------------------------------------------


def _plan() -> AnalysisPlan:
    return AnalysisPlan(
        research_question="Does the approved plan survive the pause?",
        steps=[
            AnalysisStep(
                step_id="01",
                intent="Run the approved analysis.",
                expected_outputs=["table:approved_output"],
            )
        ],
    )


def _stop_finding() -> ValidationFinding:
    return ValidationFinding(
        validator="scientific_stop_auditor",
        severity="error",
        message="Confirm an unresolved scientific stop.",
        detail={"reason": "scientific_stop"},
    )


class _EvidenceStub:
    """Minimal stand-in exposing the one method the authority derivation uses."""

    def __init__(self, digests: dict[str, str] | None = None) -> None:
        self.digests = dict(digests or {"cohort": "a" * 64})
        self.raises: Exception | None = None

    def records(self):
        if self.raises is not None:
            raise self.raises
        return [
            SimpleNamespace(evidence_id=key, sha256=value)
            for key, value in sorted(self.digests.items())
        ]


def _live_workflow(*, execution=None, recorder=None):
    """A workflow whose review requests are derived from a live plan handoff."""

    calls: list[str] = []
    evidence = _EvidenceStub()
    handoff = SimpleNamespace(
        aborted_result=None,
        plan=_plan(),
        findings=[_stop_finding()],
        evidence=evidence,
    )
    identity = {"value": execution}

    def _invoker(plan_result):
        return human_review_requests_for_plan(
            findings=plan_result.findings,
            plan=plan_result.plan,
            evidence=plan_result.evidence,
            execution_authority=identity["value"],
        )

    workflow = build_pipeline_workflow(
        plan_invoker=lambda: calls.append("plan") or handoff,
        execute_invoker=lambda _p: calls.append("execute") or "executed",
        write_invoker=lambda _p, _e: calls.append("write") or "written",
        finalise_invoker=lambda _p, _e, _w: calls.append("finalise") or "final",
        human_review_invoker=_invoker,
        human_review_recorder=recorder,
    )
    return workflow, calls, handoff, evidence, identity


def _approve(request) -> HumanReviewDecision:
    return HumanReviewDecision(
        review_id=request.review_id,
        authority_sha256=request.authority_sha256,
        decision="approved",
        reviewer="maintainer",
        decided_at="2026-07-27T05:00:00Z",
    )


# ---------------------------------------------------------------------------
# P0 — an old approval must not authorize a plan edited after the pause
# ---------------------------------------------------------------------------


def test_an_untouched_plan_resumes_normally() -> None:
    """The negative control: re-derivation must not refuse a legitimate resume."""

    workflow, calls, _handoff, _evidence, _identity = _live_workflow()
    paused = workflow.start()
    assert isinstance(paused, WorkflowPaused)

    completed = workflow.resume([_approve(paused.requests[0])])

    assert isinstance(completed, WorkflowCompleted)
    assert calls == ["plan", "execute", "write", "finalise"]


def test_review_authority_rejects_registered_evidence_byte_drift(tmp_path) -> None:
    from easyicu.research_agent.authority.evidence_store import EvidenceStore

    evidence = EvidenceStore(tmp_path)
    record = evidence.register_json(
        kind="log",
        description="Review-bound evidence.",
        payload={"status": "reviewed"},
        filename="review_bound.json",
        evidence_id="review_bound",
    )
    requests = human_review_requests_for_plan(
        findings=[_stop_finding()],
        plan=_plan(),
        evidence=evidence,
    )
    assert requests

    (tmp_path / record.relative_path).write_text(
        '{"status":"tampered"}',
        encoding="utf-8",
    )

    with pytest.raises(HumanReviewAuthorityError, match="bytes no longer match"):
        human_review_requests_for_plan(
            findings=[_stop_finding()],
            plan=_plan(),
            evidence=evidence,
        )


def test_restored_pause_verifies_review_bound_evidence_without_an_invoker(
    tmp_path,
) -> None:
    from easyicu.research_agent.authority.evidence_store import EvidenceStore

    evidence = EvidenceStore(tmp_path)
    record = evidence.register_text(
        kind="log",
        description="Review-bound evidence.",
        text="reviewed bytes",
        filename="review_bound.txt",
        evidence_id="review_bound",
    )
    plan = _plan()
    requests = human_review_requests_for_plan(
        findings=[_stop_finding()],
        plan=plan,
        evidence=evidence,
    )
    (tmp_path / record.relative_path).write_text("tampered bytes", encoding="utf-8")
    handoff = SimpleNamespace(aborted_result=None, plan=plan, evidence=evidence)
    workflow = build_pipeline_workflow(
        plan_invoker=lambda: handoff,
        execute_invoker=lambda _plan: "executed",
        write_invoker=lambda _plan, _execute: "written",
        finalise_invoker=lambda _plan, _execute, _write: "final",
    )

    with pytest.raises(HumanReviewAuthorityError, match="physical bytes"):
        workflow.restore_paused(plan_result=handoff, requests=requests)

    assert workflow.state == "failed"


def test_a_rewritten_request_payload_is_refused() -> None:
    """`frozen=True` freezes the attributes; the payload dict stays mutable.

    So the drift guard cannot read "what was approved" out of the live request:
    rewrite the embedded authority to describe the new plan and the guard
    compares the new plan against itself. Demonstrated end-to-end before the
    fix -- an unapproved output executed under the original signature.
    """

    workflow, calls, handoff, _evidence, _identity = _live_workflow()
    paused = workflow.start()
    request = paused.requests[0]
    decision = _approve(request)

    handoff.plan.steps[0].expected_outputs.append("table:never_approved")
    fresh = human_review_requests_for_plan(
        findings=handoff.findings,
        plan=handoff.plan,
        evidence=handoff.evidence,
    )[0].payload["plan_review_authority"]
    embedded = request.payload["plan_review_authority"]
    embedded.clear()
    embedded.update(fresh)

    # The frozen fields are untouched, so every digest comparison still agrees.
    assert decision.authority_sha256 == request.authority_sha256

    with pytest.raises(HumanReviewStateDrift, match="were modified after the pause"):
        workflow.resume([decision])

    assert calls == ["plan"]
    assert workflow.state == "failed"


def test_a_swapped_request_tuple_is_refused() -> None:
    """A self-consistent replacement request is still not the one offered.

    Every per-request check compares a decision against the request the engine
    is holding, so replacing both together agrees with itself. Only the
    snapshot taken when the pause was offered can tell.
    """

    workflow, calls, handoff, _evidence, _identity = _live_workflow()
    paused = workflow.start()

    handoff.plan.steps[0].expected_outputs.append("table:never_approved")
    replacement = human_review_requests_for_plan(
        findings=handoff.findings,
        plan=handoff.plan,
        evidence=handoff.evidence,
    )
    workflow._requests = replacement

    with pytest.raises(HumanReviewStateDrift, match="not the one that was offered"):
        workflow.resume([_approve(replacement[0])])

    assert calls == ["plan"]
    assert workflow.state == "failed"


def test_the_approved_side_is_read_from_the_snapshot_not_the_request() -> None:
    """Defence in depth: the comparison is correct without the byte check.

    Two independent guards cover the rewrite. This one asserts the second is
    load-bearing on its own, so a later refactor of either cannot silently
    leave the engine comparing a mutated value against itself.
    """

    workflow, _calls, handoff, _evidence, _identity = _live_workflow()
    paused = workflow.start()
    request = paused.requests[0]

    handoff.plan.steps[0].expected_outputs.append("table:never_approved")
    fresh = human_review_requests_for_plan(
        findings=handoff.findings,
        plan=handoff.plan,
        evidence=handoff.evidence,
    )[0].payload["plan_review_authority"]
    embedded = request.payload["plan_review_authority"]
    embedded.clear()
    embedded.update(fresh)

    with pytest.raises(HumanReviewStateDrift, match="plan changed after"):
        workflow._verify_pause_still_binds_live_state()


def test_the_snapshot_does_not_alias_the_request_payload() -> None:
    """The snapshot must be a copy, not another reference to the same dicts."""

    workflow, _calls, _handoff, _evidence, _identity = _live_workflow()
    paused = workflow.start()

    request = paused.requests[0]
    request.payload["plan_review_authority"]["plan_sha256"] = "0" * 64
    snapshot = workflow._pause_snapshot[0]["payload"]["plan_review_authority"]

    assert snapshot["plan_sha256"] != "0" * 64


def test_a_step_added_after_the_pause_is_refused() -> None:
    """The reported exploit: the digest still matches, the plan does not."""

    workflow, calls, handoff, _evidence, _identity = _live_workflow()
    paused = workflow.start()
    decision = _approve(paused.requests[0])

    handoff.plan.steps[0].expected_outputs.append("table:unreviewed_output")

    with pytest.raises(HumanReviewStateDrift, match="plan changed after"):
        workflow.resume([decision])

    # Nothing executed, and the pause is gone: a caller must not be able to
    # retry into running work no reviewer ever saw.
    assert calls == ["plan"]
    assert workflow.state == "failed"


def test_a_bumped_plan_revision_is_refused() -> None:
    workflow, _calls, handoff, _evidence, _identity = _live_workflow()
    paused = workflow.start()
    decision = _approve(paused.requests[0])

    handoff.plan.revision += 1

    with pytest.raises(HumanReviewStateDrift, match="plan changed after"):
        workflow.resume([decision])


def test_a_whole_replaced_plan_is_refused() -> None:
    workflow, _calls, handoff, _evidence, _identity = _live_workflow()
    paused = workflow.start()
    decision = _approve(paused.requests[0])

    handoff.plan = AnalysisPlan(
        research_question="An entirely different question.",
        steps=[AnalysisStep(step_id="01", intent="Do something else.")],
    )

    with pytest.raises(HumanReviewStateDrift, match="plan changed after"):
        workflow.resume([decision])


def test_a_changed_execution_identity_is_refused() -> None:
    """The reviewer signed off an environment, not only a plan."""

    from easyicu.research_agent.authority.plan_review import ReviewExecutionAuthority

    original = ReviewExecutionAuthority(
        pipeline_config_sha256="1" * 64,
        capability_activation_sha256="2" * 64,
        run_input_capsule_sha256="3" * 64,
    )
    workflow, _calls, _handoff, _evidence, identity = _live_workflow(execution=original)
    paused = workflow.start()
    decision = _approve(paused.requests[0])

    identity["value"] = ReviewExecutionAuthority(
        pipeline_config_sha256="1" * 64,
        capability_activation_sha256="9" * 64,  # a different capability set
        run_input_capsule_sha256="3" * 64,
    )

    with pytest.raises(HumanReviewStateDrift, match="execution identity changed"):
        workflow.resume([decision])


def test_evidence_rewritten_under_the_approval_is_refused() -> None:
    workflow, _calls, _handoff, evidence, _identity = _live_workflow()
    paused = workflow.start()
    decision = _approve(paused.requests[0])

    evidence.digests["cohort"] = "b" * 64

    with pytest.raises(HumanReviewStateDrift, match="changed after review"):
        workflow.resume([decision])


def test_evidence_removed_from_under_the_approval_is_refused() -> None:
    workflow, _calls, _handoff, evidence, _identity = _live_workflow()
    paused = workflow.start()
    decision = _approve(paused.requests[0])

    evidence.digests.clear()

    with pytest.raises(HumanReviewStateDrift, match="no longer in the run"):
        workflow.resume([decision])


def test_new_evidence_added_after_the_pause_is_allowed() -> None:
    """Additions are not drift, and refusing them would deadlock the retry.

    A decision write that fails partway legitimately leaves its own decision
    log registered in the store. If growth counted as tampering, resubmitting
    that decision could never succeed.
    """

    workflow, calls, _handoff, evidence, _identity = _live_workflow()
    paused = workflow.start()
    decision = _approve(paused.requests[0])

    evidence.digests["human_review_decisions"] = "c" * 64

    completed = workflow.resume([decision])

    assert isinstance(completed, WorkflowCompleted)
    assert calls == ["plan", "execute", "write", "finalise"]


def test_a_vanished_review_condition_is_refused() -> None:
    """The finding that required a human is gone, so the approval describes nothing."""

    workflow, _calls, handoff, _evidence, _identity = _live_workflow()
    paused = workflow.start()
    decision = _approve(paused.requests[0])

    handoff.findings.clear()

    with pytest.raises(HumanReviewStateDrift, match="no longer derives"):
        workflow.resume([decision])


def test_an_unreadable_store_leaves_the_pause_resumable() -> None:
    """Cannot-tell is transient; unlike drift it must not burn the Planner work."""

    workflow, calls, _handoff, evidence, _identity = _live_workflow()
    paused = workflow.start()
    decision = _approve(paused.requests[0])

    evidence.raises = OSError("Input/output error")
    with pytest.raises(HumanReviewAuthorityError, match="cannot re-derive"):
        workflow.resume([decision])

    assert workflow.state == "paused"
    assert calls == ["plan"]

    # Recovered: the same decision now goes through.
    evidence.raises = None
    completed = workflow.resume([decision])

    assert isinstance(completed, WorkflowCompleted)


def test_drift_is_checked_before_the_decision_is_recorded() -> None:
    """A refused resume must not leave a recorded approval behind."""

    recorded: list[dict] = []
    workflow, _calls, handoff, _evidence, _identity = _live_workflow(
        recorder=recorded.extend
    )
    paused = workflow.start()
    decision = _approve(paused.requests[0])

    handoff.plan.revision += 1
    with pytest.raises(HumanReviewStateDrift):
        workflow.resume([decision])

    assert recorded == []


# ---------------------------------------------------------------------------
# P1 — a resubmitted decision must present identical bytes
# ---------------------------------------------------------------------------


def _failing_recorder(failures: list[bool], recorded: list[dict]):
    def _record(records):
        if failures:
            failures.pop(0)
            raise OSError("No space left on device")
        recorded.extend(records)

    return _record


def test_a_resubmitted_decision_is_byte_identical() -> None:
    """Otherwise the evidence store refuses the retry it was told to allow.

    ``server_decided_at`` is stamped when the record is built. Rebuilding on
    every attempt changed the decision file's bytes, so its SHA-256 changed,
    so registering it under the fixed ``human_review_decisions`` evidence id
    raised an id-collision error — permanently. The pause was resumable in
    name only.
    """

    seen: list[list[dict]] = []

    def _record(records):
        seen.append([dict(item) for item in records])
        if len(seen) == 1:
            raise OSError("No space left on device")

    workflow, _calls, _handoff, _evidence, _identity = _live_workflow(recorder=_record)
    paused = workflow.start()
    decision = _approve(paused.requests[0])

    with pytest.raises(OSError):
        workflow.resume([decision])
    completed = workflow.resume([decision])

    assert isinstance(completed, WorkflowCompleted)
    assert len(seen) == 2
    assert seen[0] == seen[1], "a retry must present the same bytes, not a new stamp"
    assert seen[0][0]["server_decided_at"] == seen[1][0]["server_decided_at"]


def test_a_different_decision_gets_its_own_fresh_record() -> None:
    """The retry cache must not backdate a decision the operator just made."""

    seen: list[list[dict]] = []

    def _record(records):
        seen.append([dict(item) for item in records])
        if len(seen) == 1:
            raise OSError("No space left on device")

    workflow, _calls, _handoff, _evidence, _identity = _live_workflow(recorder=_record)
    paused = workflow.start()
    request = paused.requests[0]

    with pytest.raises(OSError):
        workflow.resume([_approve(request)])

    rejection = HumanReviewDecision(
        review_id=request.review_id,
        authority_sha256=request.authority_sha256,
        decision="rejected",
        reviewer="maintainer",
        decided_at="2026-07-27T06:00:00Z",
    )
    with pytest.raises(HumanReviewRejected):
        workflow.resume([rejection])

    assert seen[0][0]["decision"] == "approved"
    assert seen[1][0]["decision"] == "rejected"
    assert seen[0] != seen[1]


def test_the_retry_cache_does_not_outlive_the_run() -> None:
    recorded: list[dict] = []
    workflow, _calls, _handoff, _evidence, _identity = _live_workflow(
        recorder=recorded.extend
    )
    paused = workflow.start()

    workflow.resume([_approve(paused.requests[0])])

    assert workflow.state == "completed"
    assert workflow._decision_record_cache == {}


# ---------------------------------------------------------------------------
# P1 — one instance holds one pause; a new run must not silently destroy it
# ---------------------------------------------------------------------------


def test_a_second_run_cannot_discard_a_pause_awaiting_a_human(tmp_path):
    """``_pending_human_review`` is a single slot with no guard on ``run()``.

    A second run overwrote it on pause and cleared it on completion. The
    discarded pause holds a live plan handoff that cannot be rebuilt, so the
    Planner work was gone and ``resume_human_review`` then reported that no
    review was pending at all — while a human was still looking at it. The
    run-level file lock cannot catch this: each run takes a fresh run id.
    """

    import pandas as pd

    from easyicu.research_agent.providers.mocks import MockLLMClient
    from easyicu.research_agent.orchestration.workflow import HumanReviewRequest
    from easyicu.research_agent.pipeline import ResearchAgentPipeline

    request = HumanReviewRequest.create(
        kind="scientific_stop",
        summary="Confirm an unresolved scientific stop",
        authority_sha256="d" * 64,
        payload={"finding": "positivity_not_established"},
    )
    pending = HumanReviewPending(
        run_id="20260727T050000_abcdef",
        thread_id="20260727T050000_abcdef",
        run_dir=str(tmp_path / "run"),
        requests=(request,),
    )
    agent = ResearchAgentPipeline(workdir=tmp_path / "wd", llm=MockLLMClient())
    agent._pending_human_review = {
        "workflow": object(),
        "pending": pending,
        "runtime_capabilities": (),
        "runtime_bundle": None,
    }

    with pytest.raises(RuntimeError, match="paused for human review"):
        agent.run(
            question="Would this run quietly destroy the pending review?",
            cohort=pd.DataFrame({"patient_id": [1, 2], "death": [0, 1]}),
        )

    # Still answerable: the guard refused rather than discarding it.
    assert agent._pending_human_review is not None
    assert agent._pending_human_review["pending"] is pending


# ---------------------------------------------------------------------------
# P2 — the public entry points must admit that a pause is a possible outcome
# ---------------------------------------------------------------------------


def test_the_run_entry_points_are_typed_for_the_pause() -> None:
    import inspect

    from easyicu.research_agent.pipeline import ResearchAgentPipeline

    import easyicu.research_agent.pipeline as pipeline_module

    for name in ("run", "run_async", "run_from_spec", "run_with_graph"):
        # ``eval_str`` because ``run`` is wrapped by the execution-lock and
        # capability-job decorators, which hand back the annotation as a string.
        annotation = inspect.signature(
            getattr(ResearchAgentPipeline, name),
            eval_str=True,
            globals=vars(pipeline_module),
        ).return_annotation
        assert annotation == PipelineRunOutcome, name

    assert HumanReviewPending in PipelineRunOutcome.__args__
