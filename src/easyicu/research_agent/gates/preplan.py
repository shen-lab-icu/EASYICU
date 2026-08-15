"""Host-owned data gates that run before Planner/provider execution."""

from __future__ import annotations

from pathlib import Path
from typing import Sequence

from ..audits.validators import CohortAuditor
from ..research_context.temporal_semantics import (
    primary_exposure_time_anchor_alignment,
)
from ..schema import ResearchContext, ValidationFinding
from .data_answerability import analysis_answerability_findings


def clinical_time_authority_findings(
    context: ResearchContext,
) -> list[ValidationFinding]:
    """Fail before Provider work when the exposure cannot honor time zero.

    A declared disease/event anchor and a physical ICU-admission observation
    window are different coordinates even if both contain the same number of
    hours. The comparison target is the concept owner's typed clinical-
    definition anchor. The gate reports both owner-issued identities and leaves
    revision to StudyContext/concept authority; the Planner cannot repair it.
    """

    alignment = primary_exposure_time_anchor_alignment(context)
    if alignment.status not in {"mismatch", "declared_only"}:
        return []
    mismatch = alignment.status == "mismatch"
    code = (
        "primary_exposure_time_anchor_mismatch"
        if mismatch
        else "primary_exposure_time_anchor_unverified"
    )
    message = (
        "The declared primary-exposure time anchor "
        f"`{alignment.declared_anchor}` conflicts with the materialized "
        f"owner-issued definition anchor `{alignment.definition_anchor}`."
        if mismatch
        else (
            "The study declares primary-exposure time anchor "
            f"`{alignment.declared_anchor}`, but the exposure does not carry a "
            "verifiable owner-issued clinical-definition anchor identity."
        )
    )
    return [
        ValidationFinding(
            validator="clinical_time_authority_gate",
            severity="error",
            message=(
                message
                + " Revise the study or materialization contract; do not ask the "
                "Planner to infer a clinical definition from an observation "
                "window or reconcile two clinical definitions."
            ),
            evidence_ids=[],
            detail={
                "kind": code,
                **alignment.to_dict(),
                "required_action": (
                    "create_new_study_or_materialization_authority_with_matching_anchor"
                ),
                "provider_called": False,
            },
        )
    ]


def preplan_data_findings(
    *,
    context: ResearchContext,
    cohort_path: Path,
) -> list[ValidationFinding]:
    """Return integrity and proven scientific-infeasibility findings."""

    return [
        *CohortAuditor().audit(context=context, cohort_path=cohort_path),
        *clinical_time_authority_findings(context),
        *analysis_answerability_findings(context),
    ]


def preplan_data_failure_reason(findings: Sequence[ValidationFinding]) -> str:
    """Classify a blocking pre-plan result without hiding its typed findings."""

    if any(
        finding.severity == "error"
        and finding.validator == "clinical_time_authority_gate"
        for finding in findings
    ):
        return "clinical_time_authority_failed"
    if any(
        finding.severity == "error" and finding.validator == "data_answerability_gate"
        for finding in findings
    ):
        return "data_answerability_failed"
    return "cohort_audit_failed"


__all__ = [
    "clinical_time_authority_findings",
    "preplan_data_failure_reason",
    "preplan_data_findings",
]
