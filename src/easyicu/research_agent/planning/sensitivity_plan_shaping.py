"""Project exact typed sensitivity authority into executable plan steps.

The user-reviewed :class:`PrespecifiedSensitivitySpec` owns the scientific
choice.  This module only closes a missing execution coordinate for a binary
adjusted-association plan.  It never invents a sensitivity, merges axes, or
claims that an agent-coded method is a deterministic host adapter.
"""

from __future__ import annotations

import re
from typing import Sequence

from ..authority.current_case_scientific_runtime import (
    CurrentCaseScientificRuntimeAuthority,
    LandmarkSplineRuntimeAuthority,
)
from ..contracts.association_execution import (
    ASSOCIATION_BINARY_SENSITIVITY_CAPABILITY_ID,
    ASSOCIATION_BINARY_SENSITIVITY_PARENT_PRODUCT,
)
from ..schema import (
    AnalysisPlan,
    AnalysisStep,
    ArtifactConsumptionContract,
    LiteratureDesignBinding,
    ResearchContext,
    ValidationFinding,
)
from .sensitivity_authority import EXECUTABLE_METHODS_BY_STRATEGY, PrespecifiedSensitivitySpec


_PREFERRED_METHOD_BY_STRATEGY = {
    "landmark": "landmark_analysis",
    "first_stay": "first_stay_association",
    "cluster_robust": "cluster_robust_association",
    "restricted_cubic_spline": "restricted_cubic_spline_sensitivity",
    "linear_per_unit": "linear_per_unit_sensitivity",
    "fractional_polynomial": "fractional_polynomial_sensitivity",
    "categorical": "categorical_functional_form_sensitivity",
    "complete_case": "complete_case_sensitivity",
    "multiple_imputation": "multiple_imputation_sensitivity",
    "inverse_probability_weighting": ("inverse_probability_weighting_sensitivity"),
}


def _method_head(step: AnalysisStep) -> str:
    return str(step.method or "").strip().casefold().split("(", 1)[0].strip()


def _safe_product_id(spec_id: str, occupied: set[str]) -> str:
    base = re.sub(r"[^a-z0-9]+", "_", str(spec_id).casefold()).strip("_")
    base = f"sensitivity_{base}"[:120].rstrip("_")
    output = f"table:{base}"
    suffix = 2
    while output in occupied:
        trimmed = base[: max(1, 120 - len(str(suffix)) - 1)].rstrip("_")
        output = f"table:{trimmed}_{suffix}"
        suffix += 1
    return output


def _locked_complete_case_spec_ids(plan: AnalysisPlan) -> set[str]:
    replay_ids = {
        spec_id
        for step in plan.steps
        if _method_head(step) == "robustness_sensitivity"
        and step.robustness_replay_spec is not None
        for spec_id in step.sensitivity_spec_ids
    }
    return {
        spec.spec_id
        for spec in plan.robustness_specs
        if spec.spec_id in replay_ids
        and spec.axis == "missing"
        and str((spec.missing_override or {}).get("strategy") or "").strip().casefold()
        == "complete_case"
    }


def _copied_literature_authority(
    steps: Sequence[AnalysisStep],
    *,
    spec_id: str,
) -> tuple[list[str], list[LiteratureDesignBinding]]:
    bindings_by_key: dict[str, LiteratureDesignBinding] = {}
    for step in steps:
        if spec_id not in step.sensitivity_spec_ids:
            continue
        for binding in step.literature_design_bindings:
            # One AnalysisStep may carry at most one design binding per source.
            # Prefer the latest spec-bound step, which is normally the
            # robustness coordinate rather than the primary model.
            bindings_by_key[binding.citation_key] = binding
    bindings = list(bindings_by_key.values())
    return [binding.citation_key for binding in bindings], bindings


def ensure_prespecified_sensitivity_steps(
    *,
    plan: AnalysisPlan,
    context: ResearchContext,
    runtime_authority: CurrentCaseScientificRuntimeAuthority | None = None,
) -> tuple[AnalysisPlan, list[ValidationFinding]]:
    """Add only missing, exact binary-association sensitivity coordinates.

    The resulting steps retain an ``analysis_only`` agent-coded capability.
    Registering a future deterministic Method Adapter may replace their
    execution owner, but this projection does not pretend one exists today.
    """

    preferences = context.user_preferences
    specs = list(preferences.sensitivity_specs if preferences is not None else ())
    if not specs:
        return plan, []
    primary_steps = [
        step
        for step in plan.steps
        if step.planned_analysis_role == "primary"
        and _method_head(step) == "adjusted_association_models"
        and ASSOCIATION_BINARY_SENSITIVITY_PARENT_PRODUCT in set(step.expected_outputs)
        and len(step.model_requirements) == 1
        and step.model_requirements[0].outcome_type == "binary"
    ]
    if len(primary_steps) != 1:
        return plan, []
    primary = primary_steps[0]
    # A declared id or prose alone is insufficient. Only the actual runtime
    # owner, with the exact same time coordinates, can fulfill this obligation
    # through the primary fit. Final runtime binding/validation remains required.
    primary_covered = {
        spec.spec_id
        for spec in specs
        if _primary_landmark_covers_spec(primary, spec, runtime_authority)
    }
    already_executed = {
        spec_id
        for step in plan.steps
        for spec_id in step.sensitivity_spec_ids
        for spec in specs
        if spec.spec_id == spec_id
        and _method_head(step) in EXECUTABLE_METHODS_BY_STRATEGY[spec.strategy]
    }
    already_executed.update(_locked_complete_case_spec_ids(plan))
    already_executed.update(primary_covered)
    missing = [
        spec
        for spec in specs
        if spec.spec_id not in already_executed
        and spec.strategy in _PREFERRED_METHOD_BY_STRATEGY
    ]
    coverage_findings = [
        ValidationFinding(
            validator="prespecified_sensitivity_plan_shaping",
            severity="warning",
            message="The exact landmark obligation is assigned to the runtime-bound primary fit, not a duplicate sensitivity analysis.",
            detail={
                "reason_code": "typed_landmark_obligation_owned_by_primary",
                "step_id": primary.step_id,
                "spec_ids": sorted(primary_covered),
                "execution_contract_sha256": runtime_authority.execution_contract_sha256,
            },
        )
    ] if primary_covered and runtime_authority is not None else []
    if not missing:
        return plan, coverage_findings

    requirement = primary.model_requirements[0]
    dependence_group_source = (
        getattr(requirement.dependence, "group_source", None)
        if requirement.dependence is not None
        else None
    )
    if isinstance(requirement.dependence, dict):
        dependence_group_source = requirement.dependence.get("group_source")
    raw_inputs = list(
        dict.fromkeys(
            [
                requirement.exposure_source,
                requirement.outcome,
                *requirement.covariates,
                *([dependence_group_source] if dependence_group_source else []),
            ]
        )
    )
    cohort_inputs = [
        value
        for value in primary.inputs
        if str(value).startswith(("artifact:", "dataset:", "table:"))
        and str(value) != ASSOCIATION_BINARY_SENSITIVITY_PARENT_PRODUCT
    ]
    occupied_outputs = {
        str(output) for step in plan.steps for output in step.expected_outputs
    }
    occupied_ids = {str(step.step_id) for step in plan.steps}
    inserted: list[AnalysisStep] = []
    findings: list[ValidationFinding] = list(coverage_findings)
    for spec in missing:
        base_id = re.sub(
            r"[^a-z0-9]+", "_", f"sensitivity_{spec.spec_id}".casefold()
        ).strip("_")[:72]
        step_id = base_id
        suffix = 2
        while step_id in occupied_ids:
            step_id = f"{base_id[:68]}_{suffix}"
            suffix += 1
        occupied_ids.add(step_id)
        output = _safe_product_id(spec.spec_id, occupied_outputs)
        occupied_outputs.add(output)
        citation_keys, bindings = _copied_literature_authority(
            plan.steps,
            spec_id=spec.spec_id,
        )
        step = AnalysisStep(
            step_id=step_id,
            planned_analysis_role="sensitivity",
            intent=(
                "Execute the exact user-reviewed sensitivity specification "
                f"{spec.spec_id!r} using strategy {spec.strategy!r}; preserve "
                "the primary binary outcome, exposure contrast, adjustment "
                "roster, and analysis-only claim ceiling."
            ),
            method=_PREFERRED_METHOD_BY_STRATEGY[spec.strategy],
            inputs=list(
                dict.fromkeys(
                    [
                        *raw_inputs,
                        *spec.execution_variables,
                        *cohort_inputs,
                        ASSOCIATION_BINARY_SENSITIVITY_PARENT_PRODUCT,
                    ]
                )
            ),
            expected_outputs=[output],
            sensitivity_spec_ids=[spec.spec_id],
            scientific_capability=ASSOCIATION_BINARY_SENSITIVITY_CAPABILITY_ID,
            literature_citation_keys=citation_keys,
            literature_design_bindings=bindings,
            input_consumption_contracts=[
                ArtifactConsumptionContract(
                    input_key=ASSOCIATION_BINARY_SENSITIVITY_PARENT_PRODUCT,
                    mode="all_rows",
                )
            ],
        )
        inserted.append(step)
        findings.append(
            ValidationFinding(
                validator="prespecified_sensitivity_plan_shaping",
                severity="warning",
                message=(
                    "Projected an omitted typed sensitivity obligation into "
                    f"an explicit analysis-only step {step_id!r}."
                ),
                detail={
                    "reason_code": "typed_sensitivity_execution_step_projected",
                    "step_id": step_id,
                    "spec_id": spec.spec_id,
                    "strategy": spec.strategy,
                    "method": step.method,
                    "claim_ceiling": "analysis_only",
                    "deterministic_method_adapter": False,
                },
            )
        )

    primary_index = plan.steps.index(primary)
    steps = [
        *plan.steps[: primary_index + 1],
        *inserted,
        *plan.steps[primary_index + 1 :],
    ]
    return plan.model_copy(update={"steps": steps}), findings


def _primary_landmark_covers_spec(
    primary: AnalysisStep,
    spec: PrespecifiedSensitivitySpec,
    authority: CurrentCaseScientificRuntimeAuthority | None,
) -> bool:
    if not isinstance(authority, LandmarkSplineRuntimeAuthority):
        return False
    requirement = primary.model_requirements[0]
    return (
        spec.spec_id in primary.sensitivity_spec_ids
        and spec.strategy == "landmark"
        and spec.landmark_hours == authority.landmark_hours
        and spec.require_alive_at_landmark
        and spec.exclude_negative_event_times
        and spec.event_time_variable == authority.outcome_time_column
        and spec.observation_duration_variable == authority.observation_duration_column
        and spec.observation_duration_unit == authority.observation_duration_unit
        and set(spec.execution_variables) == {
            authority.outcome_time_column, authority.observation_duration_column
        }
        and requirement.exposure_source == authority.exposure_column
        and requirement.outcome == authority.outcome_column
        and set(requirement.covariates) == set(authority.required_adjustment_columns)
    )


__all__ = ["ensure_prespecified_sensitivity_steps"]
