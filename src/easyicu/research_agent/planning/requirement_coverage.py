"""Digest-bound coverage projection for explicit study requirements.

The scientific-review owner has already decided whether a typed outcome,
sensitivity, or exact covariate is covered, and the question-requirements
owner (``planning.question_requirements``) whether an analysis the question
asks for is.  This module records that decision as a stable row and binds it
to the exact context and plan.  It does not infer requirements from prose or
grant execution/reporting authority.

The structured adjustment reading is a required subset independent of exact
rosters. Its coverage is checked here against primary fitted covariate terms.
"""

from __future__ import annotations

from typing import Literal, Mapping, Optional, Sequence

from pydantic import BaseModel, ConfigDict, Field, model_validator


RequirementKind = Literal["outcome", "sensitivity", "covariate", "question"]
RequirementCoverageStatus = Literal["covered", "missing", "unsupported"]


class PlanRequirementCoverageRecord(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)

    requirement_id: str = Field(min_length=3, max_length=256)
    kind: RequirementKind
    #: What a question requirement asks for (``benchmark``, ``subgroup`` ...);
    #: set on question rows only.
    question_kind: Optional[str] = Field(default=None, pattern=r"^[a-z][a-z_]{2,31}$")
    concept_identity: str = Field(min_length=1, max_length=256)
    source_ref: str = Field(min_length=3, max_length=512)
    status: RequirementCoverageStatus
    owner_step_ids: tuple[str, ...] = ()
    reason_code: str = Field(min_length=3, max_length=160)

    @model_validator(mode="after")
    def _owner_status_is_consistent(self) -> "PlanRequirementCoverageRecord":
        if len(self.owner_step_ids) != len(set(self.owner_step_ids)):
            raise ValueError("requirement coverage owner step ids must be unique")
        if self.status == "covered" and not self.owner_step_ids:
            raise ValueError("covered requirement must name at least one owner step")
        if self.status != "covered" and self.owner_step_ids:
            raise ValueError("uncovered requirement cannot name an owner step")
        if (self.kind == "question") != (self.question_kind is not None):
            raise ValueError(
                "a question requirement row states what the question asks for"
            )
        return self


class PlanRequirementCoverage(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)

    schema_version: Literal["easyicu.plan_requirement_coverage/1"] = (
        "easyicu.plan_requirement_coverage/1"
    )
    context_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    plan_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    records: tuple[PlanRequirementCoverageRecord, ...]
    complete: bool

    @model_validator(mode="after")
    def _records_are_unique_and_complete(self) -> "PlanRequirementCoverage":
        identities = [record.requirement_id for record in self.records]
        if len(identities) != len(set(identities)):
            raise ValueError("plan requirement coverage ids must be unique")
        expected = all(record.status == "covered" for record in self.records)
        if self.complete != expected:
            raise ValueError("plan requirement coverage completeness drifted")
        return self


def build_plan_requirement_coverage(
    *,
    context_sha256: str,
    plan_sha256: str,
    requested_outcomes: Sequence[str],
    outcome_owner_step_ids: Mapping[str, Sequence[str]],
    sensitivity_spec_ids: Sequence[str],
    sensitivity_owner_step_ids: Mapping[str, Sequence[str]],
    unsupported_sensitivity_spec_ids: Sequence[str],
    exact_covariates: Sequence[str],
    covariate_owner_step_ids: Mapping[str, Sequence[str]],
) -> PlanRequirementCoverage:
    """Build rows only from typed requirements and prior owner validation."""

    records: list[PlanRequirementCoverageRecord] = []
    for outcome in dict.fromkeys(str(value).strip() for value in requested_outcomes):
        if not outcome:
            continue
        owners = tuple(dict.fromkeys(outcome_owner_step_ids.get(outcome, ())))
        records.append(
            PlanRequirementCoverageRecord(
                requirement_id=f"outcome:{outcome}",
                kind="outcome",
                concept_identity=outcome,
                source_ref="research_context.json.cohort.requested_outcome_columns",
                status="covered" if owners else "missing",
                owner_step_ids=owners,
                reason_code=(
                    "typed_plan_owner_present"
                    if owners
                    else "typed_plan_owner_missing"
                ),
            )
        )

    unsupported = set(unsupported_sensitivity_spec_ids)
    for spec_id in dict.fromkeys(str(value).strip() for value in sensitivity_spec_ids):
        if not spec_id:
            continue
        owners = tuple(dict.fromkeys(sensitivity_owner_step_ids.get(spec_id, ())))
        if owners:
            status: RequirementCoverageStatus = "covered"
            reason_code = "typed_plan_owner_present"
        elif spec_id in unsupported:
            status = "unsupported"
            reason_code = "runtime_capability_unavailable"
        else:
            status = "missing"
            reason_code = "typed_plan_owner_missing"
        records.append(
            PlanRequirementCoverageRecord(
                requirement_id=f"sensitivity:{spec_id}",
                kind="sensitivity",
                concept_identity=spec_id,
                source_ref=(
                    "research_context.json.user_preferences.sensitivity_specs"
                ),
                status=status,
                owner_step_ids=owners,
                reason_code=reason_code,
            )
        )

    for covariate in dict.fromkeys(str(value).strip() for value in exact_covariates):
        if not covariate:
            continue
        owners = tuple(dict.fromkeys(covariate_owner_step_ids.get(covariate, ())))
        records.append(
            PlanRequirementCoverageRecord(
                requirement_id=f"covariate:{covariate}",
                kind="covariate",
                concept_identity=covariate,
                source_ref="research_context.json.user_preferences.covariates",
                status="covered" if owners else "missing",
                owner_step_ids=owners,
                reason_code=(
                    "typed_plan_owner_present"
                    if owners
                    else "typed_plan_owner_missing"
                ),
            )
        )

    return PlanRequirementCoverage(
        context_sha256=context_sha256,
        plan_sha256=plan_sha256,
        records=tuple(records),
        complete=all(record.status == "covered" for record in records),
    )


def stated_adjustment_coverage(context, plan) -> PlanRequirementCoverage:
    """Only primary fitted covariate terms cover the question's required subset.

    A value appearing in Table 1, inputs, robustness or an exposure term is
    not adjustment. Every primary model must keep it; added covariates remain
    governed by the existing rationale and temporal-authority owners.
    """

    from ..canonical_json import canonical_sha256
    from .adjustment_authority import (
        context_stated_adjustments, stated_adjustment_columns,
    )

    primary = [
        (step.step_id, requirement)
        for step in plan.steps if step.planned_analysis_role == "primary"
        for requirement in step.model_requirements
        if requirement.analysis_role == "primary"
    ]
    records = []
    for adjustment in context_stated_adjustments(context):
        identity = adjustment.concept or adjustment.variable
        columns = set(stated_adjustment_columns(context, adjustment))
        owners = [
            step_id for step_id, requirement in primary
            if columns.intersection(requirement.covariates or ())
            and columns.intersection(
                term.name for term in requirement.model_terms or () if term.role == "covariate"
            )
        ]
        unavailable = adjustment.source_status == "unavailable"
        # Without a concept no column can be checked; the review shows it.
        unresolved = adjustment.concept is None
        covered = (
            bool(primary) and len(owners) == len(primary)
            and not unavailable and not unresolved
        )
        if unavailable:
            status, reason = "unsupported", "source_concept_unavailable"
        elif unresolved:
            status, reason = "unsupported", "stated_adjustment_concept_unresolved"
        elif covered:
            status, reason = "covered", "typed_primary_adjustment_present"
        else:
            status, reason = "missing", "typed_primary_adjustment_missing"
        records.append(PlanRequirementCoverageRecord(
            requirement_id=f"stated_adjustment:{identity}", kind="covariate",
            concept_identity=identity,
            source_ref="research_context.json.user_preferences.data_constraints.stated_adjustments",
            status=status,
            owner_step_ids=tuple(dict.fromkeys(owners)) if covered else (),
            reason_code=reason,
        ))
    # Several clauses may state the same concept; it remains one requirement.
    unique = {record.requirement_id: record for record in records}
    return PlanRequirementCoverage(
        context_sha256=canonical_sha256(context.model_dump(mode="json")),
        plan_sha256=canonical_sha256(plan.model_dump(mode="json")),
        records=tuple(unique.values()),
        complete=all(record.status == "covered" for record in unique.values()),
    )


__all__ = [
    "PlanRequirementCoverage",
    "PlanRequirementCoverageRecord",
    "build_plan_requirement_coverage",
]
