"""General supersession of step-tied findings by per-step success.

When ``per_step_records`` reports a step as ``status="ok"``, any earlier
ValidationFinding tied to that same step_id should be treated as
*superseded* — preserved in the manifest for the audit trail, but not
counted toward the readiness gates (analysis_validated,
numeric_verified, evidence_complete).

This contract was added so that a step which:
* failed mid-pipeline (e.g., upstream 502, coder crash, validator block);
* and then succeeded on resume / replan / deterministic fallback / retry

ends up with the right final tier instead of being permanently dragged
down to AO by the residual failure-era finding.

The fix is intentionally *general*: it applies to every validator name
and every step status that transitions to ``ok``. There is no 502- or
resume-specific special case.

These tests pin the behaviour against future refactors:

1. A finding referencing a step that ultimately succeeded is filtered
   from the active error count.
2. A finding referencing a step that did NOT succeed still counts.
3. A finding with no step_id reference (global error) is never
   superseded by per-step success.
4. ``detail["step_id"]`` is preferred over message-pattern scraping.
5. Multiple message patterns (``for step X``, ``step X failed``) all
   resolve correctly.
6. The full superseded list is exposed for audit inspection.
7. The rule is deterministic — identical inputs produce identical
   partitions.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List

from easyicu.research_agent.evidence import EvidenceStore
from easyicu.research_agent.pipeline_report import (
    _compute_readiness_gates,
    _partition_findings_by_supersession,
    _step_id_referenced_in_finding,
    _successful_step_ids,
)
from easyicu.research_agent.schema import (
    AnalysisPlan,
    AnalysisStep,
    ResearchContext,
    ValidationFinding,
)


def _plan_with_steps(step_ids: List[str]) -> AnalysisPlan:
    """Construct a minimal AnalysisPlan with the named steps."""
    return AnalysisPlan(
        research_question="dummy",
        steps=[
            AnalysisStep(step_id=sid, intent="dummy intent", inputs=[], expected_outputs=[])
            for sid in step_ids
        ],
    )


def _evidence(tmp_path: Path) -> EvidenceStore:
    return EvidenceStore(root=tmp_path)


def _context() -> ResearchContext:
    """Minimal context; ``_compute_readiness_gates`` requires it as a kwarg."""
    return ResearchContext(
        research_question="dummy",
        cohort={
            "cohort_name": "c",
            "database": "miiv",
            "n_patients": 10,
            "n_stays": 10,
        },
        variables=[],
    )


def _ok_record(step_id: str) -> Dict[str, Any]:
    return {"step_id": step_id, "status": "ok"}


def _fail_record(step_id: str, status: str = "coder_failed") -> Dict[str, Any]:
    return {"step_id": step_id, "status": status}


def _coder_failure(step_id: str, msg: str = "coder boom") -> ValidationFinding:
    return ValidationFinding(
        validator="coder",
        severity="error",
        message=f"Coder agent failed for step {step_id}: {msg}",
    )


def test_step_id_extracted_from_for_step_phrase() -> None:
    f = _coder_failure("03_complete_case_robustness", "Error code: 502")
    assert _step_id_referenced_in_finding(f) == "03_complete_case_robustness"


def test_step_id_extracted_from_step_failed_phrase() -> None:
    f = ValidationFinding(
        validator="runner",
        severity="error",
        message="step 04_primary_adjusted_model failed during execution",
    )
    assert _step_id_referenced_in_finding(f) == "04_primary_adjusted_model"


def test_step_id_from_detail_overrides_message_scrape() -> None:
    # Even if the message accidentally mentions a different step, the
    # explicit detail.step_id wins. This is the canonical path for new
    # finding sites.
    f = ValidationFinding(
        validator="coder",
        severity="error",
        message="Coder agent failed for step OLD: stack trace ...",
        detail={"step_id": "NEW"},
    )
    assert _step_id_referenced_in_finding(f) == "NEW"


def test_no_step_id_when_global_finding() -> None:
    f = ValidationFinding(
        validator="pipeline",
        severity="error",
        message="Formal manuscript generation skipped because the execution gate did not pass.",
    )
    assert _step_id_referenced_in_finding(f) is None


def test_successful_step_ids_returns_only_ok_steps() -> None:
    records = [
        _ok_record("01_prep"),
        _fail_record("02_model"),
        _ok_record("03_robust"),
    ]
    assert _successful_step_ids(records) == {"01_prep", "03_robust"}


def test_finding_for_succeeded_step_is_superseded() -> None:
    findings = [_coder_failure("01_prep", "transient 502")]
    active, superseded = _partition_findings_by_supersession(
        findings, success_step_ids={"01_prep"}
    )
    assert active == []
    assert superseded == findings


def test_finding_for_failed_step_remains_active() -> None:
    findings = [_coder_failure("02_model", "real coder bug")]
    active, superseded = _partition_findings_by_supersession(
        findings, success_step_ids={"01_prep"}
    )
    assert active == findings
    assert superseded == []


def test_global_finding_not_superseded_by_per_step_success() -> None:
    f = ValidationFinding(
        validator="pipeline",
        severity="error",
        message="Manuscript scaffold not generated.",
    )
    active, superseded = _partition_findings_by_supersession(
        [f], success_step_ids={"01_prep", "02_model"}
    )
    assert active == [f]
    assert superseded == []


def test_multiple_findings_for_same_succeeded_step_all_superseded() -> None:
    # Real-world case from the mini env retry: original 502 + later
    # timeout + eventual success on resume. All three failure findings
    # should be superseded by the ok status.
    findings = [
        _coder_failure("03_complete_case_robustness", "Error code: 502"),
        _coder_failure("03_complete_case_robustness", "Request timed out."),
        ValidationFinding(
            validator="pipeline",
            severity="error",
            message="step 03_complete_case_robustness failed during execution",
        ),
    ]
    active, superseded = _partition_findings_by_supersession(
        findings, success_step_ids={"03_complete_case_robustness"}
    )
    assert active == []
    assert len(superseded) == 3


def test_gate_treats_superseded_step_finding_as_no_error(tmp_path: Path) -> None:
    """End-to-end: the analysis_validated gate ignores superseded findings.

    This is the regression test for the env-retry → AO problem.
    """
    manuscript = tmp_path / "manuscript_scaffold_bound.md"
    manuscript.write_text("Bound manuscript content here.\n", encoding="utf-8")
    per_step_records: List[Dict[str, Any]] = [
        _ok_record("01_prep"),
        _ok_record("02_model"),
        _ok_record("03_complete_case_robustness"),  # ultimately succeeded
    ]
    findings = [
        # Stale failure from the original pre-resume invocation:
        _coder_failure("03_complete_case_robustness", "Error code: 502"),
        _coder_failure("03_complete_case_robustness", "Request timed out."),
    ]
    plan = _plan_with_steps(["01_prep", "02_model", "03_complete_case_robustness"])
    gates = _compute_readiness_gates(
        context=_context(),
        plan=plan,
        per_step_records=per_step_records,
        findings=findings,
        evidence=_evidence(tmp_path),
        run_dir=tmp_path,
        manuscript_path=manuscript,
        stop_after_analysis=False,
    )
    assert gates["analysis_error_count"] == 0
    assert gates["analysis_errors"] == []
    assert gates["analysis_validated"] is True
    assert gates["execution_complete"] is True
    # Audit trail still surfaces the failures
    assert gates["superseded_error_count"] == 2
    assert len(gates["superseded_errors"]) == 2


def test_gate_keeps_real_failure_when_step_did_not_succeed(tmp_path: Path) -> None:
    """Strict default preserved: a real failure that wasn't resolved
    still fails the gate. The supersession rule does NOT silently
    forgive unresolved errors."""
    manuscript = tmp_path / "manuscript_scaffold_bound.md"
    manuscript.write_text("Bound manuscript content here.\n", encoding="utf-8")
    per_step_records: List[Dict[str, Any]] = [
        _ok_record("01_prep"),
        _fail_record("04_primary_model", status="contract_failed"),
    ]
    findings = [
        ValidationFinding(
            validator="contract",
            severity="error",
            message="step 04_primary_model failed contract: missing effect size.",
        ),
    ]
    plan = _plan_with_steps(["01_prep", "04_primary_model"])
    gates = _compute_readiness_gates(
        context=_context(),
        plan=plan,
        per_step_records=per_step_records,
        findings=findings,
        evidence=_evidence(tmp_path),
        run_dir=tmp_path,
        manuscript_path=manuscript,
        stop_after_analysis=False,
    )
    assert gates["analysis_error_count"] == 1
    assert gates["analysis_validated"] is False
    assert gates["superseded_error_count"] == 0


def test_manuscript_gate_finding_superseded_when_execution_recovers() -> None:
    """A ``manuscript_gate`` finding emitted because execution was
    False at writer-time is stale once execution_complete reaches
    True via resume."""
    f = ValidationFinding(
        validator="manuscript_gate",
        severity="error",
        message=(
            "Formal manuscript generation skipped because the execution gate did not pass. "
            "Review author_review_note.md and the diagnostic artefacts before rerunning."
        ),
    )
    # Execution is now True → finding should be superseded
    active, superseded = _partition_findings_by_supersession(
        [f], success_step_ids=set(), gate_state={"execution_complete": True}
    )
    assert active == []
    assert superseded == [f]
    # Execution still False → finding should remain active (strict)
    active2, superseded2 = _partition_findings_by_supersession(
        [f], success_step_ids=set(), gate_state={"execution_complete": False}
    )
    assert active2 == [f]
    assert superseded2 == []


def test_gate_state_supersession_does_not_match_unrelated_findings() -> None:
    """A finding from a different validator with similar phrasing must
    NOT be superseded just because the gate is True."""
    f = ValidationFinding(
        validator="critic_agent",
        severity="error",
        message="manuscript generation skipped — unrelated wording.",
    )
    active, _ = _partition_findings_by_supersession(
        [f], success_step_ids=set(), gate_state={"execution_complete": True}
    )
    assert active == [f]


def test_finding_for_replanned_away_step_is_superseded() -> None:
    """The replanner sometimes DROPS a failing step and substitutes
    a new one that succeeds. The original failure finding refers
    to a step_id no longer in per_step_records — that finding
    should be superseded by the "no longer in plan" rule."""
    f = _coder_failure("03_old_failing_step", "Error code: 502")
    # The replanner replaced it with 03_new_step (which succeeded)
    known_step_ids = {"01_prep", "02_audit", "03_new_step"}
    success_step_ids = {"01_prep", "02_audit", "03_new_step"}
    active, superseded = _partition_findings_by_supersession(
        [f],
        success_step_ids=success_step_ids,
        known_step_ids=known_step_ids,
    )
    assert active == []
    assert superseded == [f]


def test_replanned_away_supersession_disabled_when_known_step_ids_none() -> None:
    """Backwards-compatible behaviour: callers that pass
    known_step_ids=None get the old (success-only) supersession."""
    f = _coder_failure("03_old_failing_step", "Error code: 502")
    active, superseded = _partition_findings_by_supersession(
        [f],
        success_step_ids=set(),
        known_step_ids=None,
    )
    # Step never succeeded and we don't know it was replanned away
    # (because known_step_ids is None), so finding remains active.
    assert active == [f]
    assert superseded == []


def test_partition_is_deterministic_order_preserving() -> None:
    findings = [
        _coder_failure("a", "x"),
        ValidationFinding(validator="pipeline", severity="error", message="global x"),
        _coder_failure("b", "y"),
        _coder_failure("a", "z"),
    ]
    active1, sup1 = _partition_findings_by_supersession(findings, success_step_ids={"a"})
    active2, sup2 = _partition_findings_by_supersession(findings, success_step_ids={"a"})
    assert active1 == active2
    assert sup1 == sup2
    # Order preserved within each partition
    assert active1[0].validator == "pipeline"
    assert active1[1].message.endswith("y")
    assert len(sup1) == 2  # both "a"-tied findings
