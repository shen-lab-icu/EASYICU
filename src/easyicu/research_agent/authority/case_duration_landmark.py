"""Versioned design contract for a proposed SIC case-duration landmark OLS.

This contract is not an execution grant. It retains the clinical distinction
between the requested ICU LOS and a case-closure proxy; the host must still
bind a verified source, materialize the risk set and obtain human approval.
"""
from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator
import pandas as pd

from easyicu.databases.research_requirements import CaseDurationSelection, TimeOriginSelection
from easyicu.research_agent.canonical_json import canonical_sha256
from easyicu.research_agent.contracts.association_execution import association_execution_verdict


class CaseDurationLandmarkAuthority(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)

    schema_version: Literal["easyicu.case_duration_landmark_authority/1"] = "easyicu.case_duration_landmark_authority/1"
    authority_kind: Literal["case_duration_landmark_ols"] = "case_duration_landmark_ols"
    requested_outcome: Literal["los_icu"] = "los_icu"
    proposed_outcome: Literal["care_unit_to_case_closure_days"] = "care_unit_to_case_closure_days"
    requested_origin: Literal["icu_admission"] = "icu_admission"
    selected_origin: Literal["first_intensive_or_intermediate_care_transfer"] = "first_intensive_or_intermediate_care_transfer"
    duration_unit: Literal["days"] = "days"
    death_status_column: Literal["death_during_case"] = "death_during_case"
    death_time_column: Literal["death_time_during_case_hours"] = "death_time_during_case_hours"
    exposure_column: str = Field(min_length=1)
    source_identity_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    export_authority_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    manifest_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    provenance_snapshot_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    implementation_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    source_config_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    model_requirement_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    landmark_hours: Literal[24] = 24
    exposure_window_start_hours: Literal[0] = 0
    exposure_window_end_hours: Literal[24] = 24
    exposure_window_end_exclusive: Literal[True] = True
    risk_policy: Literal["case_open_and_known_alive_strictly_after_landmark"] = "case_open_and_known_alive_strictly_after_landmark"
    execution_contract_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")

    @model_validator(mode="after")
    def _verify_digest(self):
        if canonical_sha256(self.model_dump(mode="json", exclude={"execution_contract_sha256"})) != self.execution_contract_sha256:
            raise ValueError("case_duration_landmark_authority_digest_mismatch")
        return self

    def validate_source_selections(self, origin: TimeOriginSelection, outcome: CaseDurationSelection) -> None:
        if (origin.requested_origin != self.requested_origin or origin.selected_origin != self.selected_origin
                or outcome.requested_outcome != self.requested_outcome or outcome.proposed_outcome != self.proposed_outcome
                or origin.source_identity_sha256 != self.source_identity_sha256
                or outcome.source_identity_sha256 != self.source_identity_sha256
                or outcome.start_field != origin.source_field or outcome.end_field.table != "cases"
                or outcome.end_field.column != "TimeOfStay"):
            raise ValueError("case_duration_landmark_source_mismatch")

    def validate_primary_step(self, step) -> None:
        """Confirm the deterministic OLS executor owns exactly this model."""
        if not association_execution_verdict(step).claimed or len(step.model_requirements) != 1:
            raise ValueError("case_duration_landmark_executor_unavailable")
        requirement = step.model_requirements[0]
        if (requirement.outcome != self.proposed_outcome or requirement.outcome_type != "continuous"
                or requirement.exposure_source != self.exposure_column
                or canonical_sha256(requirement.model_dump(mode="json")) != self.model_requirement_sha256):
            raise ValueError("case_duration_landmark_model_mismatch")

    @property
    def plan_rule_ref(self) -> str:
        return f"scientific_runtime_contract:{self.execution_contract_sha256}"

    def validate_plan(self, plan) -> None:
        """Require a named, reviewed risk-set step before the exact OLS model."""
        if (plan.analysis_type not in {"association", "association_study"} or len(plan.steps) != 2
                or plan.cohort is None or plan.cohort.inclusion or plan.cohort.exclusion
                or plan.cohort.unapplied_population_criteria):
            raise ValueError("case_duration_landmark_plan_scope_mismatch")
        risk = [step for step in plan.steps if step.method == "signed_care_unit_landmark_cohort"]
        primary = [step for step in plan.steps if step.planned_analysis_role == "primary"]
        if len(risk) != 1 or len(primary) != 1 or plan.steps.index(risk[0]) >= plan.steps.index(primary[0]):
            raise ValueError("case_duration_landmark_plan_step_mismatch")
        risk_step, primary_step = risk[0], primary[0]
        if (risk_step.expected_outputs != ["artifact:analysis_cohort"]
                or set(risk_step.inputs) != {
                    self.proposed_outcome, self.death_status_column, self.death_time_column, self.exposure_column,
                }
                or self.plan_rule_ref not in risk_step.icu_rule_refs
                or self.plan_rule_ref not in primary_step.icu_rule_refs
                or "artifact:analysis_cohort" not in primary_step.inputs):
            raise ValueError("case_duration_landmark_plan_binding_mismatch")
        self.validate_primary_step(primary_step)

    def bind_plan(self, plan):
        """Validate the stated plan; never invent or select a risk set."""
        self.validate_plan(plan)
        return plan


def propose_case_duration_landmark_authority(*, origin, outcome, source_axis, source_provenance, primary_step):
    """Create a digest-bound candidate; explicit human review remains required."""
    if len(primary_step.model_requirements) != 1:
        raise ValueError("case_duration_landmark_executor_unavailable")
    requirement = primary_step.model_requirements[0]
    fields = {
        "exposure_column": requirement.exposure_source,
        "source_identity_sha256": source_axis.source_identity_sha256,
        "export_authority_sha256": source_axis.export_authority_sha256,
        "manifest_sha256": source_axis.manifest_sha256,
        "provenance_snapshot_sha256": source_provenance.snapshot_sha256,
        "implementation_sha256": source_provenance.implementation_sha256,
        "source_config_sha256": source_provenance.source_config_sha256,
        "model_requirement_sha256": canonical_sha256(requirement.model_dump(mode="json")),
    }
    draft = CaseDurationLandmarkAuthority.model_construct(**fields)
    payload = draft.model_dump(mode="json", exclude={"execution_contract_sha256"})
    candidate = CaseDurationLandmarkAuthority.model_validate({
        **payload, "execution_contract_sha256": canonical_sha256(payload),
    })
    candidate.validate_source_selections(origin, outcome)
    candidate.validate_primary_step(primary_step)
    if source_axis.selected_origin != candidate.selected_origin:
        raise ValueError("case_duration_landmark_source_mismatch")
    return candidate


def apply_case_duration_landmark(frame, *, authority, source_axis, primary_step):
    """Apply the exact proposal mechanically after the host's approval gate.

    This pure helper does not check approval, open data, or write evidence;
    only a host that has independently verified the signed plan may call it.
    The returned counts do not contain patient identifiers.
    """
    from easyicu.databases.time_origin import SourceTimeAxisBinding
    from .care_unit_landmark import case_duration_landmark_risk_set

    if not isinstance(authority, CaseDurationLandmarkAuthority) or not isinstance(source_axis, SourceTimeAxisBinding):
        raise ValueError("case_duration_landmark_authority_missing")
    expected = (authority.source_identity_sha256, authority.export_authority_sha256,
                authority.manifest_sha256, authority.selected_origin)
    actual = (source_axis.source_identity_sha256, source_axis.export_authority_sha256,
              source_axis.manifest_sha256, source_axis.selected_origin)
    if expected != actual:
        raise ValueError("case_duration_landmark_source_mismatch")
    authority.validate_primary_step(primary_step)
    columns = (authority.proposed_outcome, authority.death_status_column, authority.death_time_column,
               authority.exposure_column)
    if not isinstance(frame, pd.DataFrame) or any(name not in frame.columns for name in columns):
        raise ValueError("case_duration_landmark_input_missing")
    duration = pd.to_numeric(frame[authority.proposed_outcome], errors="coerce") * 24.0
    risk = case_duration_landmark_risk_set(
        duration_hours=duration, death_status=frame[authority.death_status_column],
        death_time_hours=frame[authority.death_time_column], landmark_hours=authority.landmark_hours,
    )
    return frame.loc[risk.eligible].copy(), {
        "input_rows": len(frame), "eligible_rows": int(risk.eligible.sum()),
        "unidentifiable_rows": int(risk.unidentifiable.sum()),
        "closed_or_dead_at_landmark_rows": int(risk.closed_or_dead_at_landmark.sum()),
    }
