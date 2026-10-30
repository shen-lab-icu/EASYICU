"""Host-owned source interpretation, bound through existing plan approval.

The plan's companion lives in scientific review facts, whose evidence digest is
already signed by PlanReviewAuthority and ApprovedExecutablePlan. No new
Planner-visible field, approval token, or permissive boolean is introduced.
"""
from __future__ import annotations

from pathlib import Path
from dataclasses import dataclass

from easyicu.databases.research_capabilities import Fact
from easyicu.research_agent.authority.plan_lifecycle import load_approved_executable_plan
from easyicu.research_agent.authority.plan_review import PlanReviewAuthority
from easyicu.research_agent.authority.runtime_artifacts import verified_run_evidence_path
from easyicu.research_agent.canonical_json import canonical_sha256
from .research_plan_requirements import plan_capability_decision, unapplied_substitution_capabilities
from .research_requirements import CapabilityDecision

REVIEW_FIELD = "database_capability_review"
TIME_ORIGIN_REVIEW_FIELD = "database_time_origin_review"
CASE_DURATION_REVIEW_FIELD = "database_case_duration_review"
CAPABILITY_APPROVAL_CODES = frozenset({
    "study_capability_refused", "capability_substitution_not_applied",
    "capability_substitution_approval_required", "capability_substitution_review_mismatch",
    "time_origin_execution_contract_unavailable",
    "case_duration_execution_contract_unavailable",
})


class DatabaseCapabilityReview(Fact):
    plan_sha256: str
    context_sha256: str
    decision: CapabilityDecision
    unapplied_substitutions: tuple[str, ...] = ()


@dataclass(frozen=True)
class ApprovedCaseDurationBinding:
    """A runtime result only after the existing human and source gates pass."""

    source_axis: object
    authority: object


class CapabilityApprovalError(RuntimeError):
    def __init__(self, code: str, decision: CapabilityDecision):
        super().__init__(code)
        self.code = code
        self.details = {"database_capability_decision": decision.model_dump(mode="json")}

    def diagnostic(self) -> dict:
        return {"owner": "easyicu.databases.research_capabilities_v1",
                "reason_code": self.code if self.code in CAPABILITY_APPROVAL_CODES else "study_capability_refused",
                **self.details}


def require_approved_case_duration_pre_materialization(
    plan_result, run_dir: Path, *, time_origin_attestor, case_duration_attestor, runtime_authority,
) -> ApprovedCaseDurationBinding:
    """Verify the complete signed plan before creating its first patient universe.

    The verified input export and resolver provenance are attested by host
    services. This function does not read patient rows or accept a pending
    review as user approval.
    """
    from easyicu.databases.research_substitutions import case_duration_selection, time_origin_selection
    from easyicu.databases.time_origin import SourceTimeAxisBinding
    from easyicu.research_agent.authority.case_duration_landmark import CaseDurationLandmarkAuthority
    from easyicu.research_agent.planning.scientific_review import PlanScientificReview

    companion = _companion(plan_result.context, plan_result.plan, runtime_authority=runtime_authority)
    origin, duration = time_origin_selection(plan_result.context), case_duration_selection(plan_result.context)
    if companion is None or origin is None or duration is None:
        raise ValueError("case_duration_execution_contract_unavailable")
    try:
        if (not isinstance(runtime_authority, CaseDurationLandmarkAuthority)
                or time_origin_attestor is None or case_duration_attestor is None
                or companion.decision.status != "substitution" or companion.unapplied_substitutions):
            raise ValueError("case_duration_candidate_unapplied")
        runtime_authority.validate_plan(plan_result.plan)
        runtime_authority.validate_source_selections(origin, duration)
        record = plan_result.evidence.get("scientific_plan_review")
        path = verified_run_evidence_path(run_dir, record) if record is not None else None
        if path is None:
            raise ValueError("review_missing")
        review = PlanScientificReview.model_validate_json(path.read_bytes())
        if (not review.approval_allowed or review.plan_sha256 != companion.plan_sha256
                or review.context_sha256 != companion.context_sha256
                or review.facts.get(REVIEW_FIELD) != companion.model_dump(mode="json")):
            raise ValueError("review_mismatch")
        _verify_existing_approval(plan_result, run_dir, companion, record.sha256)
        selected, axis_receipt = time_origin_attestor(plan_result.context, origin)
        duration_axis, source_proof = case_duration_attestor(plan_result.context, origin)
        origin_fact = review.facts.get(TIME_ORIGIN_REVIEW_FIELD) or {}
        duration_fact = review.facts.get(CASE_DURATION_REVIEW_FIELD) or {}
        if (selected != origin or axis_receipt.to_dict() != duration_axis.to_dict()
                or origin_fact.get("selection") != origin.model_dump(mode="json")
                or origin_fact.get("disposition") != "proposed_not_approved"
                or origin_fact.get("execution_contract_available") is not True
                or origin_fact.get("registered_export_receipt") != axis_receipt.to_dict()
                or duration_fact.get("selection") != duration.model_dump(mode="json")
                or duration_fact.get("execution_contract_available") is not True
                or duration_fact.get("runtime_authority") != runtime_authority.model_dump(mode="json")
                or duration_fact.get("source_provenance") != source_proof.to_dict()
                or source_proof.snapshot_sha256 != runtime_authority.provenance_snapshot_sha256
                or source_proof.implementation_sha256 != runtime_authority.implementation_sha256
                or source_proof.source_config_sha256 != runtime_authority.source_config_sha256):
            raise ValueError("case_duration_review_source_mismatch")
        axis = SourceTimeAxisBinding(**axis_receipt.to_dict())
        if (axis.source_identity_sha256 != runtime_authority.source_identity_sha256
                or axis.export_authority_sha256 != runtime_authority.export_authority_sha256
                or axis.manifest_sha256 != runtime_authority.manifest_sha256
                or axis.selected_origin != runtime_authority.selected_origin):
            raise ValueError("case_duration_source_axis_mismatch")
        return ApprovedCaseDurationBinding(source_axis=axis, authority=runtime_authority)
    except (OSError, ValueError, TypeError, AttributeError, RuntimeError) as exc:
        raise CapabilityApprovalError("case_duration_execution_contract_unavailable", companion.decision) from exc


def _companion(context, plan, *, runtime_authority=None) -> DatabaseCapabilityReview | None:
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
        unapplied_substitutions=unapplied_substitution_capabilities(
            decision, context, plan, runtime_authority=runtime_authority,
        ),
    )


def bind_database_capability_review(review, *, context, plan, time_origin_attestor=None, runtime_authority=None,
                                    case_duration_attestor=None, case_duration_metadata_attestor=None):
    companion = _companion(context, plan, runtime_authority=runtime_authority)
    from .research_substitutions import case_duration_selection, time_origin_selection

    selection = time_origin_selection(context)
    duration = case_duration_selection(context)
    if companion is None:
        return review
    updates = {"facts": {**review.facts, REVIEW_FIELD: companion.model_dump(mode="json")}}
    if duration is not None:
        from easyicu.research_agent.authority.case_duration_landmark import CaseDurationLandmarkAuthority

        proof = None
        metadata_receipt = None
        case_ready = False
        if selection is not None and case_duration_metadata_attestor is not None:
            try:
                candidate = case_duration_metadata_attestor(context, selection)
                if candidate.source_identity_sha256 == selection.source_identity_sha256:
                    metadata_receipt = candidate.to_dict()
            except (OSError, ValueError, TypeError, AttributeError):
                pass
        try:
            if (isinstance(runtime_authority, CaseDurationLandmarkAuthority) and case_duration_attestor is not None
                    and selection is not None and companion.decision.status == "substitution"
                    and not companion.unapplied_substitutions):
                runtime_authority.validate_plan(plan)
                runtime_authority.validate_source_selections(selection, duration)
                axis_receipt, proof = case_duration_attestor(context, selection)
                case_ready = bool(
                    axis_receipt.source_identity_sha256 == runtime_authority.source_identity_sha256
                    and axis_receipt.export_authority_sha256 == runtime_authority.export_authority_sha256
                    and axis_receipt.manifest_sha256 == runtime_authority.manifest_sha256
                    and axis_receipt.selected_origin == runtime_authority.selected_origin
                    and proof.snapshot_sha256 == runtime_authority.provenance_snapshot_sha256
                    and proof.implementation_sha256 == runtime_authority.implementation_sha256
                    and proof.source_config_sha256 == runtime_authority.source_config_sha256
                )
        except (OSError, ValueError, TypeError, AttributeError):
            proof = None
        updates["facts"][CASE_DURATION_REVIEW_FIELD] = {
            "selection": duration.model_dump(mode="json"),
            "disposition": "proposed_not_approved",
            "execution_contract_available": case_ready,
            "runtime_authority": runtime_authority.model_dump(mode="json") if case_ready else None,
            "source_provenance": proof.to_dict() if case_ready else None,
            "metadata_plan_receipt": metadata_receipt,
            "metadata_status": "source_metadata_verified_not_executable" if metadata_receipt else "unavailable",
        }
        if not case_ready:
            updates.update(approval_allowed=False, status="changes_required", top_journal_candidate=False)
    proposed = selection is None and any(
        item.capability in {"time_zero", "icu_los"} for item in companion.decision.missing
    )
    if selection is not None or (proposed and time_origin_attestor is not None):
        # Until materialization, window and method owners verify the same
        # origin and source, a proposed selection is reviewable but cannot
        # be signed as an executable plan.
        try:
            if time_origin_attestor is None:
                receipt = None
            else:
                host_selection, receipt = time_origin_attestor(context, selection)
                if selection is not None and host_selection != selection:
                    raise ValueError("time_origin_export_receipt_mismatch")
                selection = host_selection
        except (OSError, ValueError):
            receipt = None
        if selection is None:
            if companion.decision.status == "refusal" or companion.unapplied_substitutions:
                updates.update(approval_allowed=False, status="changes_required", top_journal_candidate=False)
            return review.model_copy(update=updates)
        if receipt is not None:
            from easyicu.databases.time_origin import SourceTimeAxisBinding

            receipt = SourceTimeAxisBinding(**receipt.to_dict())
            if (receipt.source_identity_sha256 != selection.source_identity_sha256
                    or receipt.selected_origin != selection.selected_origin):
                raise ValueError("time_origin_export_receipt_mismatch")
        executable = bool(selection is not None and not proposed and receipt is not None
                          and companion.decision.status == "substitution"
                          and not companion.unapplied_substitutions)
        updates["facts"][TIME_ORIGIN_REVIEW_FIELD] = {
            "selection": selection.model_dump(mode="json"),
            "disposition": "proposed_not_selected" if proposed else "proposed_not_approved",
            "execution_contract_available": executable,
            "registered_export_receipt": (
                {name: getattr(receipt, name) for name in (
                    "source_identity_sha256", "export_authority_sha256", "manifest_sha256", "selected_origin"
                )} if receipt else None
            ),
        }
        if not executable:
            updates.update(approval_allowed=False, status="changes_required", top_journal_candidate=False)
    if companion.decision.status == "refusal" or companion.unapplied_substitutions:
        updates.update(approval_allowed=False, status="changes_required", top_journal_candidate=False)
    return review.model_copy(update=updates)


def require_approved_database_capabilities(plan_result, run_dir: Path, *, cohort_path=None,
                                           time_origin_attestor=None, runtime_authority=None, case_duration_attestor=None):
    """Refuse before Execute unless the exact source substitution was approved."""
    companion = _companion(plan_result.context, plan_result.plan, runtime_authority=runtime_authority)
    from .research_substitutions import case_duration_selection, time_origin_selection

    selection = time_origin_selection(plan_result.context)
    duration = case_duration_selection(plan_result.context)
    if duration is not None and (runtime_authority is None or case_duration_attestor is None):
        if companion is None:
            raise ValueError("case_duration_execution_contract_unavailable")
        raise CapabilityApprovalError("case_duration_execution_contract_unavailable", companion.decision)
    source_axis = None
    if selection is not None:
        if companion is None:
            raise ValueError("time_origin_execution_contract_unavailable")
        if cohort_path is None or time_origin_attestor is None:
            raise CapabilityApprovalError("time_origin_execution_contract_unavailable", companion.decision)
        source_axis = _require_bound_time_origin_materialization(
            plan_result, run_dir, companion, selection, cohort_path, time_origin_attestor,
        )
    if companion is None:
        _refuse_lost_reviewed_substitution(plan_result, run_dir, companion)
        return
    decision = companion.decision
    if decision.status == "refusal":
        raise CapabilityApprovalError("study_capability_refused", decision)
    if companion.unapplied_substitutions:
        raise CapabilityApprovalError("capability_substitution_not_applied", decision)
    from .research_substitutions import approximate_variables_used, source_binding

    approximate = approximate_variables_used(plan_result.context, plan_result.plan)
    if approximate:
        binding = source_binding(plan_result.context)
        if binding is None or any(name not in binding.get("approximate_covariates", {}) for name in approximate):
            raise CapabilityApprovalError("capability_substitution_not_applied", decision)
    if decision.status == "exact" and not approximate:
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
    if duration is not None:
        from easyicu.research_agent.authority.case_duration_landmark import CaseDurationLandmarkAuthority
        try:
            fact = review.facts.get(CASE_DURATION_REVIEW_FIELD) or {}
            if (not isinstance(runtime_authority, CaseDurationLandmarkAuthority) or case_duration_attestor is None
                    or source_axis is None or fact.get("execution_contract_available") is not True
                    or fact.get("selection") != duration.model_dump(mode="json")
                    or fact.get("runtime_authority") != runtime_authority.model_dump(mode="json")):
                raise ValueError("case_duration_review_mismatch")
            runtime_authority.validate_plan(plan_result.plan)
            runtime_authority.validate_source_selections(selection, duration)
            axis_receipt, source_proof = case_duration_attestor(plan_result.context, selection)
            if (axis_receipt.to_dict() != {
                    "source_identity_sha256": source_axis.source_identity_sha256,
                    "export_authority_sha256": source_axis.export_authority_sha256,
                    "manifest_sha256": source_axis.manifest_sha256,
                    "selected_origin": source_axis.selected_origin,
                } or fact.get("source_provenance") != source_proof.to_dict()
                    or source_proof.snapshot_sha256 != runtime_authority.provenance_snapshot_sha256
                    or source_proof.implementation_sha256 != runtime_authority.implementation_sha256
                    or source_proof.source_config_sha256 != runtime_authority.source_config_sha256):
                raise ValueError("case_duration_source_mismatch")
            return ApprovedCaseDurationBinding(source_axis=source_axis, authority=runtime_authority)
        except (OSError, ValueError, TypeError, AttributeError) as exc:
            raise CapabilityApprovalError("case_duration_execution_contract_unavailable", decision) from exc
    return source_axis


def _require_bound_time_origin_materialization(plan_result, run_dir, companion, selection, cohort_path, attestor):
    """Recheck the approved export and the actual sealed cohort axis before execution."""
    from easyicu.research_agent.intake.materialized_metadata import load_verified_materialized_cohort_authority
    from easyicu.research_agent.planning.scientific_review import PlanScientificReview
    from easyicu.databases.time_origin import SourceTimeAxisBinding

    try:
        record = plan_result.evidence.get("scientific_plan_review")
        path = verified_run_evidence_path(run_dir, record) if record is not None else None
        if path is None:
            raise ValueError("review_missing")
        review = PlanScientificReview.model_validate_json(path.read_bytes())
        if (review.facts.get(REVIEW_FIELD) != companion.model_dump(mode="json")
                or review.plan_sha256 != companion.plan_sha256
                or review.context_sha256 != companion.context_sha256):
            raise ValueError("approved_plan_mismatch")
        _verify_existing_approval(plan_result, run_dir, companion, record.sha256)
        fact = review.facts.get(TIME_ORIGIN_REVIEW_FIELD) or {}
        current_selection, current_receipt = attestor(plan_result.context, selection)
        receipt = current_receipt.to_dict()
        sealed = load_verified_materialized_cohort_authority(Path(cohort_path))
        actual = sealed.authority.semantic_provenance.get("source_time_axis")
        if (current_selection != selection or not review.approval_allowed
                or fact.get("execution_contract_available") is not True
                or fact.get("selection") != selection.model_dump(mode="json")
                or fact.get("registered_export_receipt") != receipt or actual != receipt):
            raise ValueError("time_origin_evidence_mismatch")
        return SourceTimeAxisBinding(**receipt)
    except (OSError, ValueError, TypeError, AttributeError) as exc:
        raise CapabilityApprovalError("time_origin_execution_contract_unavailable", companion.decision) from exc


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
