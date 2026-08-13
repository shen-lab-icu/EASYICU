"""Fail-closed execution-graph gate for proposed plan revisions."""

from __future__ import annotations

from typing import Sequence

from ..contracts.declared_product import primary_analysis_cohort_plan_findings
from ..contracts.runtime import ValidationFinding
from ..plan_utils import (
    PlanShapeValidationError,
    endpoint_contract_findings,
    validate_final_plan_shape,
    _typed_plan_dag_findings,
)
from .adjustment_authority import (
    AdjustmentAuthorityError,
    validate_plan_against_adjustment_authority,
)
from .literature_bindings import validate_literature_citation_bindings
from ..schema import AnalysisPlan, ResearchContext
from ..trajectory.plan_contract import trajectory_plan_dag_findings


def replan_candidate_contract_findings(
    *,
    plan: AnalysisPlan,
    context: ResearchContext,
    allowed_literature_citation_keys: Sequence[str] = (),
    direct_comparator_literature_keys: Sequence[str] = (),
    owner_declaration_findings: Sequence[ValidationFinding] = (),
) -> list[ValidationFinding]:
    """Return execution-blocking graph defects in one proposed revision.

    Execution-owned selector diagnostics arrive as immutable findings instead
    of making the planning package import the execution implementation.
    """

    findings = [
        *endpoint_contract_findings(plan, context=context, severity="error"),
        *_typed_plan_dag_findings(plan),
        *primary_analysis_cohort_plan_findings(plan=plan),
        *trajectory_plan_dag_findings(plan=plan, context=context),
        *owner_declaration_findings,
    ]
    try:
        validate_literature_citation_bindings(
            plan,
            allowed_literature_citation_keys,
            context=context,
            direct_comparator_keys=direct_comparator_literature_keys,
        )
    except ValueError as exc:
        findings.append(
            ValidationFinding(
                validator="replanner_literature_authority",
                severity="error",
                message=str(exc),
                detail={"reason": "replan_literature_authority_invalid"},
            )
        )
    try:
        validate_plan_against_adjustment_authority(plan=plan, context=context)
    except AdjustmentAuthorityError as exc:
        findings.append(
            ValidationFinding(
                validator="replanner_adjustment_authority",
                severity="error",
                message=str(exc),
                detail={"reason": "replan_adjustment_authority_invalid"},
            )
        )
    try:
        validate_final_plan_shape(plan)
    except PlanShapeValidationError as exc:
        findings.append(
            ValidationFinding(
                validator="replanner_final_plan_shape",
                severity="error",
                message=str(exc),
                detail={
                    "reason": exc.reason,
                    "step_ids": list(exc.step_ids),
                },
            )
        )
    return findings


def replan_candidate_rejection_finding(
    *,
    contract_errors: list[ValidationFinding],
    trigger: str,
    candidate_revision: int,
) -> ValidationFinding:
    """Summarize a rejected invalid candidate without activating its errors."""

    return ValidationFinding(
        validator="replanner",
        severity="warning",
        message=(
            "Rejected a proposed plan revision because its typed execution "
            "graph was invalid; retained the current plan."
        ),
        detail={
            "reason": "replan_candidate_execution_graph_invalid",
            "trigger": trigger,
            "candidate_revision": candidate_revision,
            "contract_findings": [
                {
                    "validator": finding.validator,
                    "message": finding.message,
                    "detail": finding.detail,
                }
                for finding in contract_errors
            ],
        },
    )


def partition_replan_candidate_findings(
    *,
    normalization_findings: list[ValidationFinding],
    contract_findings: list[ValidationFinding],
) -> tuple[list[ValidationFinding], list[ValidationFinding]]:
    """Keep rejected-candidate errors diagnostic instead of active.

    Candidate normalization may itself surface an error before the final typed
    DAG gate repeats the same defect. If the candidate is rejected and the
    current plan remains authoritative, persisting either error as an active
    run finding incorrectly blocks an otherwise valid execution. Preserve
    non-error normalization findings, and return a de-duplicated error list for
    the rejection warning's diagnostic payload.
    """

    active = [
        finding for finding in normalization_findings if finding.severity != "error"
    ]
    errors: list[ValidationFinding] = []
    seen: set[tuple[str, str, str]] = set()
    for finding in (*normalization_findings, *contract_findings):
        if finding.severity != "error":
            continue
        key = (
            str(finding.validator),
            str(finding.message),
            repr(finding.detail),
        )
        if key in seen:
            continue
        seen.add(key)
        errors.append(finding)
    return active, errors


__all__ = [
    "partition_replan_candidate_findings",
    "replan_candidate_contract_findings",
    "replan_candidate_rejection_finding",
]
