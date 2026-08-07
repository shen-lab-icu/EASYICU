"""Exact claim boundary for the host-owned primary survival executor."""

from __future__ import annotations

from typing import Any, Iterable

from .model_tokens import normalise_model_contract_token
from .ownership_verdict import OwnershipVerdict
from .survival import (
    SURVIVAL_ANALYSIS_RECEIPT_PRODUCT,
    SURVIVAL_PH_DIAGNOSTIC_PRODUCT,
)


SURVIVAL_PRIMARY_ANALYSIS_KIND = "survival_primary_cox"


def survival_execution_verdict(
    *,
    requirement: Any,
    planned_analysis_role: Any,
    expected_outputs: Iterable[Any],
    inputs: Iterable[Any],
) -> OwnershipVerdict:
    """Return one decision used by plan validation and runtime selection."""

    if (
        requirement is None
        or getattr(requirement, "analysis_family", None) != "survival"
        or planned_analysis_role != "primary"
    ):
        return OwnershipVerdict.wrong_shape(
            SURVIVAL_PRIMARY_ANALYSIS_KIND,
            reason="the step is not the declared primary survival result",
        )
    expected = {
        requirement.expected_result_product,
        SURVIVAL_ANALYSIS_RECEIPT_PRODUCT,
        SURVIVAL_PH_DIAGNOSTIC_PRODUCT,
    }
    if set(expected_outputs or ()) != expected:
        return OwnershipVerdict.wrong_shape(
            SURVIVAL_PRIMARY_ANALYSIS_KIND,
            reason=(
                "the primary survival owner emits exactly the result, PH "
                "diagnostic and host receipt products"
            ),
        )
    typed_inputs = {
        str(value or "").strip()
        for value in inputs or ()
        if ":" in str(value or "").strip()
    }
    input_product = str(getattr(requirement, "input_product", None) or "").strip()
    if not input_product or typed_inputs != {input_product}:
        return OwnershipVerdict.incomplete_declaration(
            SURVIVAL_PRIMARY_ANALYSIS_KIND,
            missing=("one exact digest-bound input_product",),
            reason=(
                "a primary survival result must declare and consume exactly one "
                "typed analysis-frame product"
            ),
        )
    unsupported: list[str] = []
    if "cox" not in normalise_model_contract_token(requirement.estimator):
        unsupported.append("estimator")
    if normalise_model_contract_token(requirement.effect_scale) != "hazard_ratio":
        unsupported.append("effect_scale")
    if normalise_model_contract_token(requirement.effect_measure) != "hazard_ratio":
        unsupported.append("effect_measure")
    if normalise_model_contract_token(requirement.uncertainty_method) not in {
        "wald_95ci",
        "wald_95_ci",
    }:
        unsupported.append("uncertainty_method")
    if normalise_model_contract_token(requirement.competing_risk_strategy) not in {
        "none",
        "no_competing_risk",
    }:
        unsupported.append("competing_risk_strategy")
    if "schoenfeld" not in normalise_model_contract_token(
        requirement.proportional_hazards_diagnostic
    ):
        unsupported.append("proportional_hazards_diagnostic")
    if requirement.exposure_encoding != "numeric_linear":
        unsupported.append("exposure_encoding")
    if requirement.missing_data_policy != "complete_case":
        unsupported.append("missing_data_policy")
    if requirement.event_value is None or requirement.event_value <= 0:
        unsupported.append("event_value")
    if requirement.time_horizon_value is None:
        unsupported.append("time_horizon_value")
    if requirement.covariates is None:
        unsupported.append("covariates")
    if unsupported:
        return OwnershipVerdict.wrong_shape(
            SURVIVAL_PRIMARY_ANALYSIS_KIND,
            reason="unsupported survival contract fields: " + ", ".join(unsupported),
        )
    return OwnershipVerdict.claim(
        SURVIVAL_PRIMARY_ANALYSIS_KIND,
        reason="one fully declared, digest-bound Cox primary model",
    )


__all__ = ["SURVIVAL_PRIMARY_ANALYSIS_KIND", "survival_execution_verdict"]
