"""Digest-bound coverage projection for explicit study requirements.

The scientific-review owner has already decided whether a typed outcome,
sensitivity, or exact covariate is covered.  This module records that decision
as a stable row and binds it to the exact context and plan.  It does not infer
requirements from prose or grant execution/reporting authority.
"""

from __future__ import annotations

from typing import Literal, Mapping, Sequence

from pydantic import BaseModel, ConfigDict, Field, model_validator


RequirementKind = Literal["outcome", "sensitivity", "covariate"]
RequirementCoverageStatus = Literal["covered", "missing", "unsupported"]


class PlanRequirementCoverageRecord(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)

    requirement_id: str = Field(min_length=3, max_length=256)
    kind: RequirementKind
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


__all__ = [
    "PlanRequirementCoverage",
    "PlanRequirementCoverageRecord",
    "build_plan_requirement_coverage",
]
