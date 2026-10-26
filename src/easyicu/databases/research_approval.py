"""Host-owned source interpretation, bound through existing plan approval.

The plan's companion lives in scientific review facts, whose evidence digest is
already signed by PlanReviewAuthority and ApprovedExecutablePlan. No new
Planner-visible field, approval token, or permissive boolean is introduced.
"""
from __future__ import annotations

from pathlib import Path

from easyicu.databases.research_capabilities import Fact
from easyicu.research_agent.authority.plan_lifecycle import load_approved_executable_plan
from easyicu.research_agent.authority.plan_review import PlanReviewAuthority
from easyicu.research_agent.authority.runtime_artifacts import verified_run_evidence_path
from easyicu.research_agent.canonical_json import canonical_sha256
from .research_plan_requirements import plan_capability_decision, unapplied_substitution_capabilities
from .research_requirements import CapabilityDecision

REVIEW_FIELD = "database_capability_review"
CAPABILITY_APPROVAL_CODES = frozenset({
    "study_capability_refused", "capability_substitution_not_applied",
    "capability_substitution_approval_required", "capability_substitution_review_mismatch",
})


class DatabaseCapabilityReview(Fact):
    plan_sha256: str
    context_sha256: str
    decision: CapabilityDecision
    unapplied_substitutions: tuple[str, ...] = ()


class CapabilityApprovalError(RuntimeError):
    def __init__(self, code: str, decision: CapabilityDecision):
        super().__init__(code)
        self.code = code
        self.details = {"database_capability_decision": decision.model_dump(mode="json")}

    def diagnostic(self) -> dict:
        return {"owner": "easyicu.databases.research_capabilities_v1",
                "reason_code": self.code if self.code in CAPABILITY_APPROVAL_CODES else "study_capability_refused",
                **self.details}


def _companion(context, plan) -> DatabaseCapabilityReview | None:
    decision = plan_capability_decision(context, plan)
    from .research_substitutions import source_binding

    try:
        if decision is None or decision.status != "refusal":
            source_binding(context)
    except (KeyError, ValueError, TypeError) as exc:
        if decision is None:
            from .research_plan_requirements import _constraints

            saved = _constraints(context).get("database_capability_binding") or context.cohort.provenance.get("database_capability_binding")
            decision = CapabilityDecision.model_validate(saved["decision"])
        raise CapabilityApprovalError("capability_substitution_review_mismatch", decision) from exc
    if decision is None:
        return None
    return DatabaseCapabilityReview(
        plan_sha256=canonical_sha256(plan.model_dump(mode="json")),
        context_sha256=canonical_sha256(context.model_dump(mode="json")), decision=decision,
        unapplied_substitutions=unapplied_substitution_capabilities(decision, context, plan),
    )


def bind_database_capability_review(review, *, context, plan):
    companion = _companion(context, plan)
    if companion is None:
        return review
    updates = {"facts": {**review.facts, REVIEW_FIELD: companion.model_dump(mode="json")}}
    if companion.decision.status == "refusal" or companion.unapplied_substitutions:
        updates.update(approval_allowed=False, status="changes_required", top_journal_candidate=False)
    return review.model_copy(update=updates)


def require_approved_database_capabilities(plan_result, run_dir: Path) -> None:
    """Refuse before Execute unless the exact source substitution was approved."""
    companion = _companion(plan_result.context, plan_result.plan)
    if companion is None:
        _refuse_lost_reviewed_substitution(plan_result, run_dir, companion)
        return
    decision = companion.decision
    if decision.status == "refusal":
        raise CapabilityApprovalError("study_capability_refused", decision)
    if companion.unapplied_substitutions:
        raise CapabilityApprovalError("capability_substitution_not_applied", decision)
    if decision.status == "exact":
        _refuse_lost_reviewed_substitution(plan_result, run_dir, companion)
        return
    evidence = plan_result.evidence
    record = evidence.get("scientific_plan_review")
    path = verified_run_evidence_path(run_dir, record) if record is not None else None
    if path is None:
        raise CapabilityApprovalError("capability_substitution_approval_required", decision)
    from easyicu.research_agent.planning.scientific_review import PlanScientificReview

    review = PlanScientificReview.model_validate_json(path.read_bytes())
    if (review.facts.get(REVIEW_FIELD) != companion.model_dump(mode="json")
            or review.plan_sha256 != companion.plan_sha256
            or review.context_sha256 != companion.context_sha256):
        raise CapabilityApprovalError("capability_substitution_review_mismatch", decision)
    if not review.approval_allowed:
        raise CapabilityApprovalError("capability_substitution_approval_required", decision)
    try:
        _verify_existing_approval(plan_result, run_dir, companion, record.sha256)
    except (ValueError, RuntimeError, OSError) as exc:
        raise CapabilityApprovalError("capability_substitution_approval_required", decision) from exc


def _refuse_lost_reviewed_substitution(plan_result, run_dir, companion):
    """Erasing the source or original requirements cannot erase its approval."""
    record = getattr(plan_result, "evidence", {}).get("scientific_plan_review")
    if record is None:
        return
    path = verified_run_evidence_path(run_dir, record)
    if path is None:
        if companion is not None:
            raise CapabilityApprovalError("capability_substitution_review_mismatch", companion.decision)
        raise ValueError("capability_substitution_review_mismatch")
    from easyicu.research_agent.planning.scientific_review import PlanScientificReview

    review = PlanScientificReview.model_validate_json(path.read_bytes())
    saved = review.facts.get(REVIEW_FIELD)
    current = companion.model_dump(mode="json") if companion else None
    if saved and saved.get("decision", {}).get("substitutions") and saved != current:
        raise CapabilityApprovalError("capability_substitution_review_mismatch",
                                      CapabilityDecision.model_validate(saved["decision"]))


def _verify_existing_approval(plan_result, run_dir, companion, review_sha256) -> None:
    from easyicu.research_agent.orchestration.human_review_checkpoint import load_checkpoint

    checkpoint = load_checkpoint(run_dir / "human_review_checkpoint.json", require_pending=False)
    if checkpoint.state not in {"approved_pending_execution", "executing", "completed"}:
        raise ValueError("approval_required")
    if canonical_sha256(checkpoint.plan_handoff.get("plan")) != companion.plan_sha256:
        raise ValueError("approved_plan_changed")
    decisions = {item["review_id"]: item for item in checkpoint.approved_decisions}
    if not decisions or set(decisions) != {item.review_id for item in checkpoint.requests}:
        raise ValueError("decision_set_incomplete")
    authorities = set()
    for request in checkpoint.requests:
        decision = decisions[request.review_id]
        if decision["decision"] != "approved" or decision["authority_sha256"] != request.authority_sha256:
            raise ValueError("request_not_approved")
        authority = PlanReviewAuthority.model_validate(request.payload["plan_review_authority"])
        if authority.plan_sha256 != companion.plan_sha256 or authority.evidence_sha256.get("scientific_plan_review") != review_sha256:
            raise ValueError("source_interpretation_not_approved")
        authorities.add(canonical_sha256(authority.model_dump(mode="json")))
    approved = load_approved_executable_plan(
        run_dir=run_dir, evidence=plan_result.evidence, revision=plan_result.plan.revision,
        expected_plan_sha256=companion.plan_sha256,
        expected_decision_set_sha256=checkpoint.consumed_decision_sha256,
    )
    if authorities != {approved.plan_review_authority_sha256}:
        raise ValueError("approval_authority_mismatch")
