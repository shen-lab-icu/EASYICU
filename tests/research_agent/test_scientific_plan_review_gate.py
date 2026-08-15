from __future__ import annotations

from pathlib import Path

import pytest

from easyicu.research_agent.authority.evidence_store import EvidenceStore
from easyicu.research_agent.orchestration.scientific_plan_review_gate import (
    SCIENTIFIC_PLAN_REVIEW_EVIDENCE_ID,
    ScientificPlanReviewArtifactError,
    persist_or_validate_scientific_plan_review,
    scientific_plan_review_finding,
)
from easyicu.research_agent.planning.scientific_review import (
    PlanScientificFinding,
    PlanScientificReview,
)


def _review(
    *,
    status: str = "changes_required",
    approval_allowed: bool = False,
    generated_at: str = "2026-08-13T00:00:00+00:00",
    plan_sha256: str = "b" * 64,
) -> PlanScientificReview:
    findings = [
        PlanScientificFinding(
            code="SCIENTIFIC_TEST_FINDING",
            severity="blocker",
            dimension="study_design",
            message="The exact plan needs a user-reviewed scientific change.",
            remediation="Revise the plan before approval.",
            remediation_route="study_authority_change",
            requires_user_authorization=True,
            authorization_question="Do you authorize the proposed study change?",
        )
    ]
    return PlanScientificReview(
        status=status,
        approval_allowed=approval_allowed,
        top_journal_candidate=status == "ready_for_approval",
        score=40 if not approval_allowed else 75,
        dimension_scores={"study_design": 40},
        findings=findings,
        facts={"test": True},
        context_sha256="a" * 64,
        plan_sha256=plan_sha256,
        literature_sha256="c" * 64,
        figure_strategy_sha256="d" * 64,
        generated_at=generated_at,
    )


def test_review_artifact_is_registered_then_resume_revalidates_identity(
    tmp_path: Path,
) -> None:
    evidence = EvidenceStore(tmp_path)
    current = _review()

    saved, path = persist_or_validate_scientific_plan_review(
        run_dir=tmp_path,
        evidence=evidence,
        current_review=current,
    )

    assert saved == current
    assert path == tmp_path / "scientific_plan_review.json"
    assert evidence.get(SCIENTIFIC_PLAN_REVIEW_EVIDENCE_ID) is not None

    regenerated = _review(generated_at="2026-08-13T00:01:00+00:00")
    resumed, resumed_path = persist_or_validate_scientific_plan_review(
        run_dir=tmp_path,
        evidence=evidence,
        current_review=regenerated,
    )
    assert resumed == current
    assert resumed_path == path


def test_review_artifact_rejects_resume_time_scientific_identity_drift(
    tmp_path: Path,
) -> None:
    evidence = EvidenceStore(tmp_path)
    persist_or_validate_scientific_plan_review(
        run_dir=tmp_path,
        evidence=evidence,
        current_review=_review(),
    )

    with pytest.raises(ScientificPlanReviewArtifactError) as exc_info:
        persist_or_validate_scientific_plan_review(
            run_dir=tmp_path,
            evidence=evidence,
            current_review=_review(plan_sha256="e" * 64),
        )

    assert exc_info.value.code == "scientific_plan_review_identity_drift"
    assert exc_info.value.path == tmp_path / "scientific_plan_review.json"


@pytest.mark.parametrize(
    ("status", "approval_allowed", "severity", "reason"),
    [
        (
            "changes_required",
            False,
            "error",
            "plan_scientific_changes_required",
        ),
        ("analysis_only", True, "warning", "plan_scientific_review_complete"),
        ("ready_for_approval", True, "info", "plan_scientific_review_complete"),
    ],
)
def test_review_finding_preserves_decision_and_authorization_request(
    status: str,
    approval_allowed: bool,
    severity: str,
    reason: str,
) -> None:
    finding = scientific_plan_review_finding(
        _review(status=status, approval_allowed=approval_allowed)
    )

    assert finding.severity == severity
    assert finding.detail["reason"] == reason
    assert finding.evidence_ids == [SCIENTIFIC_PLAN_REVIEW_EVIDENCE_ID]
    assert finding.detail["user_authorization_requests"] == [
        {
            "code": "SCIENTIFIC_TEST_FINDING",
            "question": "Do you authorize the proposed study change?",
        }
    ]
