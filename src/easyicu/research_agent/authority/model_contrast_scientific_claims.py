"""Compile bounded model contrasts from a deterministic reporting projection.

The projection consumes sealed aggregate model products, never patient rows.
Its explicit population and contrast coordinates prevent a display anchor from
acquiring authority for a whole nonlinear association or another risk set.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from .numeric_claim_identity import NumericEffectScale, NumericEstimand


class ModelContrast(BaseModel):
    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)

    kind: Literal["spline_point", "linear_increment", "covariate_form_point"]
    target_column: str | None = Field(default=None, min_length=1)
    sensitivity_spec_id: str | None = Field(default=None, min_length=1)
    source_evidence_id: str = Field(min_length=1)
    value: float
    reference: float | None = None
    estimate: float = Field(gt=0)
    lower: float = Field(gt=0)
    upper: float = Field(gt=0)

    @field_validator("value", "reference", "estimate", "lower", "upper", mode="before")
    @classmethod
    def _not_boolean(cls, value):
        if isinstance(value, bool):
            raise ValueError("model contrast numeric values cannot be booleans")
        return value

    @model_validator(mode="after")
    def _bounded(self):
        if not self.lower <= self.estimate <= self.upper:
            raise ValueError("model contrast interval must contain the estimate")
        if self.kind in {"spline_point", "covariate_form_point"} and self.reference is None:
            raise ValueError("spline point requires its reference")
        if self.kind == "linear_increment" and (
            self.reference is not None or self.value <= 0
        ):
            raise ValueError("linear increment requires a positive increment only")
        if self.kind == "covariate_form_point":
            if self.target_column is None or self.sensitivity_spec_id is None:
                raise ValueError("covariate-form contrast requires its target and reviewed spec")
        elif self.target_column is not None or self.sensitivity_spec_id is not None:
            raise ValueError("primary/linear contrasts cannot acquire a covariate-form identity")
        return self


class ModelContrastReporting(BaseModel):
    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)

    schema_version: Literal["easyicu.model_contrast_reporting/1"]
    execution_owner: Literal["landmark_spline_robustness_executor_v1"]
    interpretation: Literal["descriptive_prognostic_association_not_causal"]
    runtime_projection_sha256: str = Field(pattern=r"^[a-f0-9]{64}$")
    exposure: str = Field(min_length=1)
    outcome: str = Field(min_length=1)
    exposure_unit: str = Field(min_length=1)
    landmark_hours: float = Field(gt=0)
    population_rule: Literal["alive_and_under_observation_at_landmark_with_valid_exposure"]
    n: int = Field(gt=0, strict=True)
    events: int = Field(ge=0, strict=True)
    adjustment_columns: list[str]
    confidence_level: Literal[0.95]
    interval_method: Literal["wald_log_odds"]
    variance_estimator: Literal["model_based", "patient_cluster_robust"]
    contrasts: list[ModelContrast] = Field(min_length=3)

    @model_validator(mode="after")
    def _coherent(self):
        if self.events > self.n:
            raise ValueError("model contrast events exceed the analysis population")
        if any(not x.strip() for x in self.adjustment_columns) or len(
            set(self.adjustment_columns)
        ) != len(self.adjustment_columns):
            raise ValueError("model contrast adjustment columns must be unique and nonempty")
        points = [x for x in self.contrasts if x.kind == "spline_point"]
        increments = [x for x in self.contrasts if x.kind == "linear_increment"]
        if len(points) < 2 or len(increments) != 1:
            raise ValueError("model contrast projection requires spline points and one linear sensitivity")
        if len({x.reference for x in points}) != 1 or len(
            {x.value for x in points}
        ) != len(points):
            raise ValueError("model contrast points require unique values and a shared reference")
        alternatives = [x for x in self.contrasts if x.kind == "covariate_form_point"]
        for spec_id in {x.sensitivity_spec_id for x in alternatives}:
            group = [x for x in alternatives if x.sensitivity_spec_id == spec_id]
            if (
                len(group) != len(points)
                or len({x.target_column for x in group}) != 1
                or group[0].target_column not in self.adjustment_columns
                or group[0].target_column == self.exposure
                or len({x.source_evidence_id for x in group}) != 1
                or {(x.value, x.reference) for x in group} != {(x.value, x.reference) for x in points}
            ):
                raise ValueError("covariate-form points must retain the primary contrast coordinates and one adjustment target")
        return self


def model_contrast_numeric_identities(summary: dict) -> dict[
    str, tuple[NumericEffectScale, NumericEstimand]
]:
    """Map exact sealed result fields to the scale declared by this owner.

    The /1 contract uses generic estimate/lower/upper keys. Their OR and
    confidence-limit identities come from the validated Wald-log-odds
    contract, not field-name guessing or another result in the summary.
    """

    derive_model_contrast_claim_payloads(summary)
    roles = {
        "estimate": NumericEstimand.POINT_ESTIMATE,
        "lower": NumericEstimand.CONFIDENCE_INTERVAL_LOWER,
        "upper": NumericEstimand.CONFIDENCE_INTERVAL_UPPER,
    }
    return {
        f"reportable_model_contrasts.contrasts[{index}].{field}": (
            NumericEffectScale.ODDS_RATIO, role,
        )
        for index, _ in enumerate(summary["reportable_model_contrasts"]["contrasts"])
        for field, role in roles.items()
    }


def derive_model_contrast_claim_payloads(summary: dict) -> list[dict]:
    """Reject incomplete projections; derive no global curve or causal claim."""

    if (
        summary.get("status") != "ok"
        or summary.get("analysis_family") != "robustness_sensitivity"
        or summary.get("authority_kind") != "signed_landmark_spline_robustness"
        or summary.get("primary_effect_is_nonlinear_curve_summary") is not False
    ):
        raise ValueError("model contrast reporting requires its native bounded projection")
    report = ModelContrastReporting.model_validate(summary["reportable_model_contrasts"])
    if (
        report.runtime_projection_sha256 != summary.get("runtime_projection_sha256")
        or report.n != summary.get("complete_case_n")
    ):
        raise ValueError("model contrast reporting disagrees with runtime or population")
    bindings = summary.get("input_bindings")
    if not isinstance(bindings, list):
        raise ValueError("model contrast reporting lacks consumed source bindings")
    source_ids = {
        binding.get("evidence_id") for binding in bindings
        if isinstance(binding, dict) and binding.get("loaded") is True
        and isinstance(binding.get("sha256"), str)
        and len(binding["sha256"]) == 64
    }
    if any(x.source_evidence_id not in source_ids for x in report.contrasts):
        raise ValueError("model contrast reporting source was not consumed")
    alternatives = [x for x in report.contrasts if x.kind == "covariate_form_point"]
    effect_sources = summary.get("functional_form_effect_sources", [])
    if alternatives and not isinstance(effect_sources, list):
        raise ValueError("covariate-form claims lack the consumed effect products")
    for contrast in alternatives:
        matches = [item for item in effect_sources if isinstance(item, dict)
                   and item.get("spec_id") == contrast.sensitivity_spec_id
                   and item.get("target_column") == contrast.target_column
                   and item.get("contrast_evidence_id") == contrast.source_evidence_id]
        if len(matches) != 1 or matches[0].get("curve_evidence_id") not in source_ids:
            raise ValueError("covariate-form claim requires both consumed curve and point products")
    upper = max((x for x in report.contrasts if x.kind == "spline_point"), key=lambda x: x.value)
    if (upper.estimate, upper.lower, upper.upper) != tuple(
        summary.get(x) for x in ("primary_or", "primary_ci_low", "primary_ci_high")
    ):
        raise ValueError("model contrast reporting disagrees with its display anchor")
    population = (
        f"the {report.n} complete-case records alive and under observation at the "
        f"{report.landmark_hours:g}-hour landmark with valid exposure"
    )
    payloads = []
    for index, contrast in enumerate(report.contrasts):
        point = contrast.kind != "linear_increment"
        coordinate = (
            f"{report.exposure} at {contrast.value:g} versus {contrast.reference:g} {report.exposure_unit}"
            if point else f"a {contrast.value:g} {report.exposure_unit} increase in {report.exposure}"
        )
        scope = (
            "this point contrast only, not a summary of the nonlinear curve"
            if point else "prespecified linear functional-form sensitivity"
        )
        if contrast.kind == "covariate_form_point":
            reader_target = contrast.target_column.replace("_", " ")
            scope = (
                f"prespecified sensitivity: {reader_target} modeled with its reviewed restricted cubic spline "
                "instead of a linear adjustment term; this exposure point contrast only"
            )
        variance = report.variance_estimator.replace("_", " ")
        payloads.append({
            "claim_id": f"model_contrast_{index + 1}",
            "claim_type": "association",
            "exposure": coordinate,
            "outcome": report.outcome,
            "direction": (
                "positive" if contrast.lower > 1 else
                "negative" if contrast.upper < 1 else "no_clear_association"
            ),
            "estimand": f"odds ratio; {scope}; {variance} Wald interval; noncausal association",
            "population": population,
            "analysis_role": "primary" if contrast.kind == "spline_point" else "sensitivity",
            "status": "supported",
            "adjusted_for": report.adjustment_columns,
            "point_estimate": contrast.estimate,
            "interval_lower": contrast.lower,
            "interval_upper": contrast.upper,
        })
    return payloads
