"""Digest-bound current-run scientific authority for caller-reviewed cases.

The shared engine does not know E2 or H2.  It knows two small execution shapes:
one frozen landmark spline association and one source-feasibility result that
must fail closed without estimating an effect.  A benchmark-local compiler
chooses the columns and scientific constants, seals them, and passes the
immutable value object here.  Plan validation and deterministic executors then
consume the same object, so signed rules cannot stop at prompt prose.
"""

from __future__ import annotations

import hashlib
import json
from typing import Annotated, Any, Literal, Mapping, Union

from pydantic import BaseModel, ConfigDict, Field, TypeAdapter, model_validator

from ..contracts.cohort_product_keys import sole_typed_cohort_input
from ..schema import AnalysisPlan, AnalysisStep


class CurrentCaseScientificAuthorityError(ValueError):
    """The plan drifted from a caller-reviewed current-run contract."""


def _canonical_bytes(value: Mapping[str, Any]) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")


def _plain(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _plain(child) for key, child in value.items()}
    if isinstance(value, (list, tuple)):
        return [_plain(child) for child in value]
    return value


def _normalise(value: Any) -> str:
    return "_".join(
        part
        for part in "".join(
            char.lower() if char.isalnum() else " " for char in str(value or "")
        ).split()
        if part
    )


class _AuthorityBase(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)

    protocol_content_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    execution_contract_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")

    @property
    def plan_rule_ref(self) -> str:
        return f"scientific_runtime_contract:{self.execution_contract_sha256}"

    def _verify_digest(self) -> None:
        body = self.model_dump(mode="json", exclude={"execution_contract_sha256"})
        observed = hashlib.sha256(_canonical_bytes(body)).hexdigest()
        if observed != self.execution_contract_sha256:
            raise ValueError("current-run scientific execution-contract digest mismatch")

    def _require_rule_ref(self, step: AnalysisStep) -> None:
        if self.plan_rule_ref not in set(step.icu_rule_refs):
            raise CurrentCaseScientificAuthorityError(
                "governed plan step lacks the signed scientific runtime digest"
            )


class LandmarkSplineRuntimeAuthority(_AuthorityBase):
    schema_version: Literal["easyicu.landmark_spline_runtime_authority/1"]
    authority_kind: Literal["landmark_spline_association"]
    plan_method: Literal["signed_landmark_restricted_cubic_spline"]
    plan_intent: str
    plan_outputs: tuple[str, ...]
    exposure_column: str
    outcome_column: str
    outcome_time_column: str
    observation_duration_column: str
    observation_duration_unit: Literal["days"]
    landmark_hours: Literal[24]
    required_adjustment_columns: tuple[str, ...]
    categorical_adjustment_columns: tuple[str, ...]
    spline_knot_quantiles: tuple[float, float, float]
    spline_reference: Literal["median_in_primary_population"]
    curve_quantile_range: tuple[float, float]
    curve_points: int = Field(ge=5, le=201)
    linear_sensitivity_per_unit: Literal[1.0]
    interpretation: Literal["descriptive_prognostic_association_not_causal"]

    @model_validator(mode="after")
    def _closed_contract(self) -> "LandmarkSplineRuntimeAuthority":
        if self.spline_knot_quantiles != (0.10, 0.50, 0.90):
            raise ValueError("landmark spline authority requires frozen 10/50/90 knots")
        if self.curve_quantile_range != (0.10, 0.90):
            raise ValueError("landmark spline curve must span the frozen boundary knots")
        if len(self.required_adjustment_columns) != len(
            set(self.required_adjustment_columns)
        ):
            raise ValueError("landmark adjustment columns must be unique")
        if not set(self.categorical_adjustment_columns).issubset(
            self.required_adjustment_columns
        ):
            raise ValueError(
                "categorical adjustments must be members of the adjustment set"
            )
        self._verify_digest()
        return self

    @property
    def required_columns(self) -> tuple[str, ...]:
        return tuple(
            dict.fromkeys(
                (
                    self.exposure_column,
                    self.outcome_column,
                    self.outcome_time_column,
                    self.observation_duration_column,
                    *self.required_adjustment_columns,
                )
            )
        )

    def governed_step(self, plan: AnalysisPlan) -> AnalysisStep:
        primary = [
            step for step in plan.steps if step.planned_analysis_role == "primary"
        ]
        if len(primary) != 1:
            raise CurrentCaseScientificAuthorityError(
                "landmark spline authority requires exactly one primary step"
            )
        step = primary[0]
        issues: list[str] = []
        if step.method != self.plan_method:
            issues.append("method")
        if step.intent != self.plan_intent:
            issues.append("intent")
        if step.scientific_capability != "association_freeform_v1":
            issues.append("scientific_capability")
        if tuple(step.expected_outputs) != self.plan_outputs:
            issues.append("expected_outputs")
        if not set(self.required_columns).issubset(step.inputs):
            issues.append("required_inputs")
        if not sole_typed_cohort_input(step):
            issues.append("typed_cohort_input")
        if step.model_requirements:
            issues.append("model_requirements")
        if issues:
            raise CurrentCaseScientificAuthorityError(
                "landmark spline plan drifted from signed authority: "
                + ", ".join(issues)
            )
        self._require_rule_ref(step)
        return step

    def validate_plan(self, plan: AnalysisPlan) -> None:
        self.governed_step(plan)


class SourceFeasibilityRuntimeAuthority(_AuthorityBase):
    schema_version: Literal["easyicu.source_feasibility_runtime_authority/1"]
    authority_kind: Literal["source_feasibility_fail_closed"]
    plan_method: Literal["signed_source_feasibility_fail_closed"]
    plan_intent: str
    plan_outputs: tuple[str, ...]
    source: str
    audited_window_hours: tuple[Literal[0], Literal[24]]
    decision: Literal["fail_closed"]
    reason_code: Literal["H2_VERIFIED_NON_USE_UNAVAILABLE"]
    verified_non_use_available: Literal[False]
    binary_control_arm_authorized: Literal[False]
    causal_contrast_authorized: Literal[False]
    forbidden_plan_tokens: tuple[str, ...]
    future_design_authorized: Literal[False]

    @model_validator(mode="after")
    def _closed_contract(self) -> "SourceFeasibilityRuntimeAuthority":
        if not self.forbidden_plan_tokens:
            raise ValueError("source feasibility authority needs forbidden plan tokens")
        if len(self.forbidden_plan_tokens) != len(set(self.forbidden_plan_tokens)):
            raise ValueError("source feasibility forbidden tokens must be unique")
        self._verify_digest()
        return self

    def governed_step(self, plan: AnalysisPlan) -> AnalysisStep:
        if len(plan.steps) != 1:
            raise CurrentCaseScientificAuthorityError(
                "source feasibility-only plan contains forbidden additional steps"
            )
        owners = [step for step in plan.steps if step.method == self.plan_method]
        if len(owners) != 1:
            raise CurrentCaseScientificAuthorityError(
                "source feasibility authority requires exactly one fail-closed owner"
            )
        if any(step.planned_analysis_role == "primary" for step in plan.steps):
            raise CurrentCaseScientificAuthorityError(
                "source feasibility-only plan must not declare a primary effect step"
            )
        step = owners[0]
        issues: list[str] = []
        if step.intent != self.plan_intent:
            issues.append("intent")
        if tuple(step.expected_outputs) != self.plan_outputs:
            issues.append("expected_outputs")
        if step.model_requirements or step.family_primary_result_requirement is not None:
            issues.append("effect_model_contract")
        if issues:
            raise CurrentCaseScientificAuthorityError(
                "source feasibility plan drifted from signed authority: "
                + ", ".join(issues)
            )
        self._require_rule_ref(step)
        forbidden = set(self.forbidden_plan_tokens)
        for candidate in plan.steps:
            if candidate is step:
                # The signed owner uses negated phrases such as "no effect estimate".
                # Its fields were already checked for exact equality above, so scan
                # only additional steps for prohibited work.
                continue
            fields = (
                candidate.method,
                candidate.intent,
                *candidate.expected_outputs,
            )
            observed = {_normalise(value) for value in fields if value}
            for value in observed:
                if any(
                    token == value
                    or value.startswith(token + "_")
                    or value.endswith("_" + token)
                    or ("_" + token + "_") in ("_" + value + "_")
                    for token in forbidden
                ):
                    raise CurrentCaseScientificAuthorityError(
                        "source feasibility-only plan requested a forbidden current-run "
                        f"action: {value}"
                    )
        return step

    def validate_plan(self, plan: AnalysisPlan) -> None:
        self.governed_step(plan)


CurrentCaseScientificRuntimeAuthority = Annotated[
    Union[LandmarkSplineRuntimeAuthority, SourceFeasibilityRuntimeAuthority],
    Field(discriminator="authority_kind"),
]
_AUTHORITY_ADAPTER = TypeAdapter(CurrentCaseScientificRuntimeAuthority)


def load_current_case_scientific_runtime_authority(
    value: CurrentCaseScientificRuntimeAuthority | Mapping[str, Any],
) -> CurrentCaseScientificRuntimeAuthority:
    if isinstance(
        value, (LandmarkSplineRuntimeAuthority, SourceFeasibilityRuntimeAuthority)
    ):
        return value
    return _AUTHORITY_ADAPTER.validate_json(
        json.dumps(_plain(value), sort_keys=True), strict=True
    )


def build_current_case_scientific_runtime_authority(
    value: Mapping[str, Any],
) -> CurrentCaseScientificRuntimeAuthority:
    body = dict(value)
    if "execution_contract_sha256" in body:
        raise ValueError("execution contract digest is host-generated")
    body["execution_contract_sha256"] = hashlib.sha256(
        _canonical_bytes(body)
    ).hexdigest()
    return load_current_case_scientific_runtime_authority(body)


__all__ = [
    "CurrentCaseScientificAuthorityError",
    "CurrentCaseScientificRuntimeAuthority",
    "LandmarkSplineRuntimeAuthority",
    "SourceFeasibilityRuntimeAuthority",
    "build_current_case_scientific_runtime_authority",
    "load_current_case_scientific_runtime_authority",
]
