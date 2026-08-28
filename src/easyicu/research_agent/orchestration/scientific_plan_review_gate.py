"""Persist and project the scientific review that precedes human approval.

The planning owner computes the review.  This orchestration boundary binds the
exact review artifact into EvidenceStore, rejects resume-time drift, and emits
the one structured finding consumed by the workflow.  Pipeline entry code does
not reimplement any of those governance rules.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

from ..authority.evidence_store import EvidenceStore
from ..contracts.runtime import _PlanPhaseResult
from ..literature import LiteratureBundle
from ..planning.figure_strategy import ArticleFigureStrategy
from ..planning.scientific_review import (
    PlanScientificReview,
    build_plan_scientific_review,
)
from ..planning.literature_design_authority import (
    LiteratureDesignAuthorityError,
    validate_selected_design_against_literature,
)
from ..schema import AnalysisPlan, ResearchContext, ValidationFinding
from ..providers.llm import resolve_role_client
from ..providers.prompts import PROMPT_PACK_VERSION, prompt_pack_files

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


@dataclass(frozen=True)
class PreplanAbortContext:
    run_id: str
    run_dir: Path
    context: Any
    context_path: Path
    agent_context: Any
    evidence: Any
    findings: Any
    llm: Any
    resume_state: Any

    def finish(self, pipeline: Any, *, reason: str) -> _PlanPhaseResult:
        aborted = pipeline._finalise_aborted(
            run_id=self.run_id,
            run_dir=self.run_dir,
            context=self.context,
            context_path=self.context_path,
            evidence=self.evidence,
            findings=self.findings,
            reason=reason,
        )
        return _PlanPhaseResult(
            context=self.context,
            agent_context=self.agent_context,
            context_path=self.context_path,
            evidence=self.evidence,
            findings=self.findings,
            plan=AnalysisPlan(
                research_question=self.context.research_question,
                steps=[],
            ),
            plan_path=self.run_dir / "analysis_plan.json",
            llm_signature=pipeline._llm_signature(self.llm),
            used_mock_llm=any(True for _ in pipeline._iter_mock_clients(self.llm)),
            prompt_version=PROMPT_PACK_VERSION,
            prompt_files=prompt_pack_files(),
            role_resolver=lambda _role: resolve_role_client(self.llm, _role),
            cost_meter=None,
            repro_envelope=None,
            started_at=datetime.now(timezone.utc),
            resume_state=self.resume_state,
            aborted_result=aborted,
        )


def require_strict_planner_route(enabled: bool, skill_obj: Any) -> None:
    if enabled and skill_obj is not None:
        raise LiteratureDesignAuthorityError(
            "literature_design_authority_requires_planner",
            "strict literature-to-design authority cannot be bypassed by a fixed skill plan",
            path="pipeline.skill",
        )


def record_literature_authority_abort(
    findings: Any, emit_progress: Any, run_id: str,
    error: LiteratureDesignAuthorityError,
) -> None:
    findings.append(
        ValidationFinding(
            validator="literature_design_authority",
            severity="error",
            message=(
                "Reviewed literature is not design-ready; stopped before "
                f"Planner Provider use: {error}"
            ),
            evidence_ids=["preplan_literature_bundle"],
            detail={
                "reason": error.reason_code,
                "path": error.path,
                "human_review_required": True,
                "approval_allowed": False,
            },
        )
    )
    emit_progress(
        "hypothesis",
        "Literature-to-design authority is incomplete; aborting before Planner.",
        status="error",
        run_id=run_id,
    )


def fail_if_strict_prompt_compilation_failed(
    enabled: bool, error: Exception
) -> None:
    if enabled:
        raise LiteratureDesignAuthorityError(
            "literature_design_prompt_compilation_failed",
            "reviewed literature could not be compiled into the Planner context",
            path="hypothesis_blueprint",
        ) from error


def append_literature_design_authority_finding(
    findings: Any,
    plan: AnalysisPlan,
    literature: Optional[LiteratureBundle],
) -> None:
    finding = literature_design_authority_finding(plan=plan, literature=literature)
    if finding is not None:
        findings.append(finding)


def literature_design_authority_finding(
    *,
    plan: AnalysisPlan,
    literature: Optional[LiteratureBundle],
) -> Optional[ValidationFinding]:
    """Project an exact post-Plan literature decision failure into workflow."""

    comparison_keys = [
        decision.citation_key
        for decision in (literature.screening_decisions if literature else [])
        if decision.disposition == "include"
        and decision.evidence_role in {"direct_comparator", "design_analogue"}
    ]
    try:
        validate_selected_design_against_literature(
            plan.design_selection,
            design_evidence_cards=(literature.design_evidence_cards if literature else []),
            comparison_keys=comparison_keys,
        )
    except LiteratureDesignAuthorityError as exc:
        return ValidationFinding(
            validator="literature_design_authority",
            severity="error",
            message=str(exc),
            evidence_ids=["analysis_plan", "preplan_literature_bundle"],
            detail={
                "reason": exc.reason_code,
                "path": exc.path,
                "human_review_required": True,
                "approval_allowed": False,
            },
        )
    return None


def persist_or_validate_scientific_plan_review(
    *,
    run_dir: Path,
    evidence: EvidenceStore,
    current_review: PlanScientificReview,
    reuse_existing_review: bool = False,
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
    if reuse_existing_review:
        binding_fields = (
            "context_sha256",
            "plan_sha256",
            "literature_sha256",
            "figure_strategy_sha256",
        )
        if any(
            getattr(existing_review, field) != getattr(current_review, field)
            for field in binding_fields
        ):
            raise ScientificPlanReviewArtifactError(
                code="scientific_plan_review_binding_drift",
                path=path,
                detail=(
                    "the existing review no longer binds the exact current "
                    "context, plan, literature, and figure strategy"
                ),
            )
        return existing_review, path
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
    require_reportable_capability: bool = False,
    reuse_existing_review: bool = False,
) -> ScientificPlanReviewGate:
    """Build, bind, and project the exact review offered to a human."""

    current_review = build_plan_scientific_review(
        context=context,
        plan=plan,
        literature=literature,
        figure_strategy=figure_strategy,
        require_reportable_capability=require_reportable_capability,
    )
    review, artifact_path = persist_or_validate_scientific_plan_review(
        run_dir=run_dir,
        evidence=evidence,
        current_review=current_review,
        reuse_existing_review=reuse_existing_review,
    )
    return ScientificPlanReviewGate(
        review=review,
        finding=scientific_plan_review_finding(review),
        artifact_path=artifact_path,
    )


__all__ = [
    "SCIENTIFIC_PLAN_REVIEW_EVIDENCE_ID",
    "SCIENTIFIC_PLAN_REVIEW_FILENAME",
    "PreplanAbortContext",
    "append_literature_design_authority_finding",
    "ScientificPlanReviewArtifactError",
    "ScientificPlanReviewGate",
    "literature_design_authority_finding",
    "fail_if_strict_prompt_compilation_failed",
    "record_literature_authority_abort",
    "require_strict_planner_route",
    "persist_or_validate_scientific_plan_review",
    "prepare_scientific_plan_review_gate",
    "scientific_plan_review_finding",
]
