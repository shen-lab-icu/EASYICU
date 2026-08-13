"""Persist and project the scientific review that precedes human approval.

The planning owner computes the review.  This orchestration boundary binds the
exact review artifact into EvidenceStore, rejects resume-time drift, and emits
the one structured finding consumed by the workflow.  Pipeline entry code does
not reimplement any of those governance rules.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from ..authority.evidence_store import EvidenceStore
from ..literature import LiteratureBundle
from ..planning.figure_strategy import ArticleFigureStrategy
from ..planning.scientific_review import (
    PlanScientificReview,
    build_plan_scientific_review,
)
from ..schema import AnalysisPlan, ResearchContext, ValidationFinding

SCIENTIFIC_PLAN_REVIEW_EVIDENCE_ID = "scientific_plan_review"
SCIENTIFIC_PLAN_REVIEW_FILENAME = "scientific_plan_review.json"


class ScientificPlanReviewArtifactError(RuntimeError):
    """An existing review artifact cannot authorize the current plan."""

    def __init__(self, *, code: str, path: Path, detail: str) -> None:
        self.code = str(code)
        self.path = Path(path)
        self.detail = str(detail)
        super().__init__(f"{self.code}: {self.path}: {self.detail}")


@dataclass(frozen=True)
class ScientificPlanReviewGate:
    review: PlanScientificReview
    finding: ValidationFinding
    artifact_path: Path
    evidence_id: str = SCIENTIFIC_PLAN_REVIEW_EVIDENCE_ID


def persist_or_validate_scientific_plan_review(
    *,
    run_dir: Path,
    evidence: EvidenceStore,
    current_review: PlanScientificReview,
) -> tuple[PlanScientificReview, Path]:
    """Register a new exact review or revalidate the bound resume artifact."""

    path = Path(run_dir) / SCIENTIFIC_PLAN_REVIEW_FILENAME
    existing_record = evidence.get(SCIENTIFIC_PLAN_REVIEW_EVIDENCE_ID)
    if existing_record is None:
        path.write_text(current_review.model_dump_json(indent=2), encoding="utf-8")
        evidence.register_file(
            kind="log",
            description=(
                "Pre-execution multi-dimensional scientific review of the "
                "exact proposed plan, literature authority and article figure strategy."
            ),
            source_path=path,
            evidence_id=SCIENTIFIC_PLAN_REVIEW_EVIDENCE_ID,
            producer="plan_scientific_review",
            generation_mode="deterministic_skill",
        )
        return current_review, path

    try:
        existing_review = PlanScientificReview.model_validate_json(
            path.read_text(encoding="utf-8")
        )
    except (OSError, UnicodeDecodeError, ValueError) as exc:
        raise ScientificPlanReviewArtifactError(
            code="scientific_plan_review_unreadable",
            path=path,
            detail=str(exc),
        ) from exc
    if existing_review.model_dump(
        mode="json", exclude={"generated_at"}
    ) != current_review.model_dump(mode="json", exclude={"generated_at"}):
        raise ScientificPlanReviewArtifactError(
            code="scientific_plan_review_identity_drift",
            path=path,
            detail=(
                "the existing review does not match the exact current context, "
                "plan, literature, and figure strategy"
            ),
        )
    return existing_review, path


def scientific_plan_review_finding(
    review: PlanScientificReview,
) -> ValidationFinding:
    """Project one reviewed decision into the workflow finding vocabulary."""

    authorization_requests = [
        {"code": item.code, "question": item.authorization_question}
        for item in review.findings
        if item.requires_user_authorization and item.authorization_question
    ]
    return ValidationFinding(
        validator="plan_scientific_review",
        severity=(
            "error"
            if not review.approval_allowed
            else "warning" if review.status == "analysis_only" else "info"
        ),
        message=(
            "The exact analysis plan requires scientific changes before it can be approved."
            if not review.approval_allowed
            else (
                "The exact analysis plan may run only with an analysis-only "
                "claim ceiling unless its major scientific findings are revised."
                if review.status == "analysis_only"
                else "The exact analysis plan is ready for explicit human approval."
            )
        ),
        evidence_ids=[SCIENTIFIC_PLAN_REVIEW_EVIDENCE_ID],
        detail={
            "reason": (
                "plan_scientific_changes_required"
                if not review.approval_allowed
                else "plan_scientific_review_complete"
            ),
            "human_review_required": bool(not review.approval_allowed),
            "approval_allowed": review.approval_allowed,
            "review_status": review.status,
            "review_score": review.score,
            "top_journal_candidate": review.top_journal_candidate,
            "finding_codes": [item.code for item in review.findings],
            "review_evidence_id": SCIENTIFIC_PLAN_REVIEW_EVIDENCE_ID,
            "user_authorization_requests": authorization_requests,
        },
    )


def prepare_scientific_plan_review_gate(
    *,
    context: ResearchContext,
    plan: AnalysisPlan,
    literature: Optional[LiteratureBundle],
    figure_strategy: ArticleFigureStrategy,
    run_dir: Path,
    evidence: EvidenceStore,
) -> ScientificPlanReviewGate:
    """Build, bind, and project the exact review offered to a human."""

    current_review = build_plan_scientific_review(
        context=context,
        plan=plan,
        literature=literature,
        figure_strategy=figure_strategy,
    )
    review, artifact_path = persist_or_validate_scientific_plan_review(
        run_dir=run_dir,
        evidence=evidence,
        current_review=current_review,
    )
    return ScientificPlanReviewGate(
        review=review,
        finding=scientific_plan_review_finding(review),
        artifact_path=artifact_path,
    )


__all__ = [
    "SCIENTIFIC_PLAN_REVIEW_EVIDENCE_ID",
    "SCIENTIFIC_PLAN_REVIEW_FILENAME",
    "ScientificPlanReviewArtifactError",
    "ScientificPlanReviewGate",
    "persist_or_validate_scientific_plan_review",
    "prepare_scientific_plan_review_gate",
    "scientific_plan_review_finding",
]
