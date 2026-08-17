"""Fresh Planner-only typed product declarations.

This owner keeps rules that the model can repair out of the broad Agent parser
while leaving ``AnalysisPlan`` compatible with digest-verified historical
plans.  Scientific choices are declared here; runtime owners only execute the
resulting immutable contracts.
"""

from __future__ import annotations

from ..schema import AnalysisPlan, ResearchContext
from ..contracts.descriptive_execution import (
    DESCRIPTIVE_EXPOSURE_OUTCOME_CAPABILITY_ID,
)
from .scientific_review import post_baseline_exposure


class PlannerOutputContractError(ValueError):
    """A fresh Planner response omitted required typed scientific authority."""


def _method_head(value: object) -> str:
    return str(value or "").strip().casefold().split("(", 1)[0].strip()


def missing_post_baseline_descriptive_claims(
    *,
    plan: AnalysisPlan,
    context: ResearchContext,
) -> tuple[str, ...]:
    """Return primary descriptive estimators missing their typed claim ceiling.

    The ceiling is required only when the sealed exposure window extends after
    the clinical anchor.  It is deliberately limited to the exact deterministic
    descriptive product shapes accepted by scientific review; association,
    temporal-model, and auxiliary distribution steps are not relabelled.
    """

    if not post_baseline_exposure(context)[0]:
        return ()
    missing: list[str] = []
    for step in plan.steps:
        if step.planned_analysis_role != "primary":
            continue
        shape = (_method_head(step.method), tuple(step.expected_outputs))
        is_descriptive_product = bool(
            shape
            in {
                (
                    "descriptive",
                    ("table:exposure_outcome_distribution",),
                ),
                (
                    "descriptive_distribution",
                    ("table:distribution_prevalence",),
                ),
                (
                    "descriptive_distribution_summary",
                    ("table:distribution_prevalence",),
                ),
            }
        )
        if not is_descriptive_product:
            continue
        if (
            step.model_requirements
            or step.family_primary_result_requirement is not None
            or step.scientific_capability
            not in {None, DESCRIPTIVE_EXPOSURE_OUTCOME_CAPABILITY_ID}
        ):
            continue
        if step.descriptive_claim is None:
            missing.append(step.step_id)
    return tuple(missing)


def validate_fresh_planner_typed_product_specs(
    plan: AnalysisPlan,
    *,
    context: ResearchContext,
) -> None:
    """Reject ambiguous typed products in a newly generated Planner response."""

    missing_distribution_specs = [
        step.step_id
        for step in plan.steps
        if "table:exposure_outcome_distribution" in step.expected_outputs
        and step.exposure_outcome_distribution_spec is None
    ]
    if missing_distribution_specs:
        raise ValueError(
            "Planner exposure/outcome distribution steps must declare "
            "exposure_outcome_distribution_spec; missing for "
            f"{missing_distribution_specs!r}. The exposure, outcome, event "
            "value and denominator policy are scientific choices and are not "
            "inferred from column names or input order."
        )
    missing_table_one_specs = [
        step.step_id
        for step in plan.steps
        if "table:table_one" in step.expected_outputs and step.table_one_spec is None
    ]
    if missing_table_one_specs:
        raise ValueError(
            "Planner Table 1 steps must declare table_one_spec; missing for "
            f"{missing_table_one_specs!r}. Use table:cohort_summary for an "
            "ungrouped descriptive table."
        )
    missing_claims = missing_post_baseline_descriptive_claims(
        plan=plan,
        context=context,
    )
    if missing_claims:
        raise PlannerOutputContractError(
            "Planner primary descriptive steps for a post-baseline exposure "
            "must declare descriptive_claim with claim_ceiling="
            "'descriptive_only' and unresolved_limitations="
            "['post_baseline_exposure_opportunity_unresolved']; missing for "
            f"{list(missing_claims)!r}. Descriptive prose does not create this "
            "machine-verifiable claim ceiling."
        )


__all__ = [
    "PlannerOutputContractError",
    "missing_post_baseline_descriptive_claims",
    "validate_fresh_planner_typed_product_specs",
]
