"""Exact claim boundary for the host-owned descriptive primary executor.

The broad ``descriptive_epidemiology`` family also contains agent-authored
Table One and measurement-audit steps.  Those steps must remain
``analysis_only``.  This module identifies the much narrower primary contract
that the host can execute and validate without choosing any science: one
typed cohort, one fully declared exposure/outcome distribution specification,
and an explicit descriptive-only claim ceiling.

It is dependency-neutral so planning, execution and readiness ask the same
owner predicate rather than independently inferring ownership from a method
label or an output name.
"""

from __future__ import annotations

from typing import Any, Mapping

from .cohort_product_keys import sole_typed_cohort_input
from .ownership_verdict import OwnershipVerdict


DESCRIPTIVE_EXPOSURE_OUTCOME_CAPABILITY_ID = (
    "descriptive_exposure_outcome_distribution_v1"
)
EXPOSURE_OUTCOME_DISTRIBUTION_ANALYSIS_KIND = "exposure_outcome_distribution"
EXPOSURE_OUTCOME_DISTRIBUTION_OUTPUT = "table:exposure_outcome_distribution"


def exposure_outcome_distribution_execution_verdict(step: Any) -> OwnershipVerdict:
    """Claim only the exact host-executable descriptive result contract."""

    spec = getattr(step, "exposure_outcome_distribution_spec", None)
    role = str(getattr(step, "planned_analysis_role", "") or "").strip().casefold()
    method = str(getattr(step, "method", "") or "").strip().casefold()
    claim = getattr(step, "descriptive_claim", None)
    primary_is_descriptive = bool(
        role == "primary"
        and method == "descriptive"
        and claim is not None
        and getattr(claim, "claim_ceiling", None) == "descriptive_only"
    )
    declared_outputs = [
        str(value or "").strip()
        for value in getattr(step, "expected_outputs", None) or []
    ]
    if spec is None:
        return OwnershipVerdict.wrong_shape(
            EXPOSURE_OUTCOME_DISTRIBUTION_ANALYSIS_KIND,
            reason="the step has no typed exposure/outcome distribution specification",
        )
    if str(getattr(spec, "schema_version", "") or "") != (
        "easyicu.exposure_outcome_distribution/2"
    ):
        return OwnershipVerdict.wrong_shape(
            EXPOSURE_OUTCOME_DISTRIBUTION_ANALYSIS_KIND,
            reason="the step does not declare exposure_outcome_distribution/2",
        )
    if method not in {"descriptive", "distribution"} or not (
        role == "auxiliary" or primary_is_descriptive
    ):
        return OwnershipVerdict.wrong_shape(
            EXPOSURE_OUTCOME_DISTRIBUTION_ANALYSIS_KIND,
            reason=(
                "the step is neither an auxiliary distribution nor a primary "
                "descriptive-only result"
            ),
        )
    if declared_outputs != [EXPOSURE_OUTCOME_DISTRIBUTION_OUTPUT]:
        return OwnershipVerdict.wrong_shape(
            EXPOSURE_OUTCOME_DISTRIBUTION_ANALYSIS_KIND,
            reason="the step does not promise exactly the owned distribution product",
        )
    if not sole_typed_cohort_input(step):
        return OwnershipVerdict.wrong_shape(
            EXPOSURE_OUTCOME_DISTRIBUTION_ANALYSIS_KIND,
            reason="the step does not consume exactly one typed cohort authority",
        )
    declared_capability = str(
        getattr(step, "scientific_capability", "") or ""
    ).strip()
    if (
        list(getattr(step, "model_requirements", None) or [])
        or getattr(step, "family_primary_result_requirement", None) is not None
        or declared_capability
        not in {"", DESCRIPTIVE_EXPOSURE_OUTCOME_CAPABILITY_ID}
        or getattr(step, "table_one_spec", None) is not None
        or getattr(step, "trajectory_stability_spec", None) is not None
    ):
        return OwnershipVerdict.wrong_shape(
            EXPOSURE_OUTCOME_DISTRIBUTION_ANALYSIS_KIND,
            reason="the step also declares a different scientific owner contract",
        )
    return OwnershipVerdict.claim(
        EXPOSURE_OUTCOME_DISTRIBUTION_ANALYSIS_KIND,
        reason=(
            "the exact typed cohort, exposure/outcome distribution/2 design and "
            "descriptive-only claim ceiling are host executable"
        ),
    )


def exposure_outcome_distribution_result_receipt_valid(summary: Any) -> bool:
    """Validate the closed, noncausal result identity emitted by the owner.

    This does not recompute estimates; the deterministic executor and numeric
    evidence gates own that work.  It proves that the primary result presented
    to final readiness is the exact descriptive receipt this capability
    registers, rather than an arbitrary table carrying the same method label.
    """

    if not isinstance(summary, Mapping):
        return False
    estimates = summary.get("descriptive_estimates")
    if not isinstance(estimates, Mapping):
        return False
    if not (
        summary.get("status") == "ok"
        and summary.get("interpretation_class")
        == "exposure_outcome_distribution"
        and summary.get("analysis_role") == "primary"
        and summary.get("analysis_set") == "bound_typed_cohort"
        and summary.get("interpretation_ceiling")
        == "descriptive_unadjusted_not_causal"
        and summary.get("adjusted_effect") is None
        and estimates.get("schema_version")
        == "easyicu.exposure_outcome_descriptive_estimates/1"
        and estimates.get("analysis_role") == "primary"
        and estimates.get("analysis_set") == "bound_typed_cohort"
        and estimates.get("interpretation_ceiling")
        == "descriptive_unadjusted_not_causal"
    ):
        return False
    absolute_risks = estimates.get("outcome_absolute_risks")
    prevalence = estimates.get("exposure_prevalence")
    if not (
        isinstance(prevalence, list)
        and prevalence
        and all(isinstance(row, Mapping) for row in prevalence)
        and isinstance(absolute_risks, list)
        and absolute_risks
        and all(isinstance(row, Mapping) for row in absolute_risks)
    ):
        return False
    contrast = estimates.get("risk_difference")
    if contrast is None:
        return True
    return bool(
        isinstance(contrast, Mapping)
        and contrast.get("direction") == "comparison_minus_reference"
        and contrast.get("interpretation_ceiling")
        == "descriptive_unadjusted_not_causal"
    )


__all__ = [
    "DESCRIPTIVE_EXPOSURE_OUTCOME_CAPABILITY_ID",
    "EXPOSURE_OUTCOME_DISTRIBUTION_ANALYSIS_KIND",
    "EXPOSURE_OUTCOME_DISTRIBUTION_OUTPUT",
    "exposure_outcome_distribution_execution_verdict",
    "exposure_outcome_distribution_result_receipt_valid",
]
