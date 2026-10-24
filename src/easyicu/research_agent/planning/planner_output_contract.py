"""Fresh Planner-only typed product declarations and accepted-roster checks.

This owner checks accepted-roster availability before a provider request and
keeps repairable omissions inside the Planner's local retry. ``AnalysisPlan``
remains compatible with digest-verified historical plans; shared runtime
readers consume baseline coverage without loading Planner schemas or retries.
"""

from __future__ import annotations

import json
from typing import Any, Mapping, Sequence

from ..schema import AnalysisPlan, ResearchContext
from ..contracts.descriptive_execution import (
    DESCRIPTIVE_EXPOSURE_OUTCOME_CAPABILITY_ID,
)
from ..contracts.table_one_semantics import validate_table_one_semantic_roles
from .baseline_requirements import baseline_outline_coverage
from .progressive_contract import ProgressivePlanCompileError
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
    for step in plan.steps:
        if step.table_one_spec is not None:
            validate_table_one_semantic_roles(step.table_one_spec, context)
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


def validate_baseline_materialization(context: ResearchContext) -> None:
    """Refuse unavailable accepted coordinates before any provider request.

    A Planner can repair an omitted available row, but cannot materialize a
    missing clinical coordinate. Keep that host failure outside parse retries.
    """

    coverage = baseline_outline_coverage(context, ())
    unavailable = [table for table in coverage["tables"] if table["unavailable_coordinates"]]
    if unavailable:
        names = sorted({name for table in unavailable for name in table["unavailable_coordinates"]})
        raise ProgressivePlanCompileError(
            "progressive_accepted_baseline_materialization_missing",
            "Accepted baseline coordinates are unavailable after extraction: " + ", ".join(names),
            step_id=unavailable[0]["source_step_id"], path="accepted_baseline_requirements",
            concepts=names, findings=unavailable,
        )


def validate_baseline_outline(context: ResearchContext, steps: Sequence[Mapping[str, Any]]) -> None:
    """Keep repairable roster omissions inside the outline's local retry."""

    baseline = baseline_outline_coverage(context, steps)
    if baseline["status"] == "incomplete":
        raise ProgressivePlanCompileError(
            "progressive_outline_accepted_baseline_incomplete",
            "The outline must preserve each accepted baseline roster in one "
            "table_one step (or an auxiliary summary for an ungrouped roster) "
            "before foundation or step materialization. Include the required "
            "grouping and one available clinical-value column per required variable; "
            "measurement metadata and mentions in an audit or prose do not satisfy "
            "this requirement. " + json.dumps(baseline, ensure_ascii=False, sort_keys=True),
            path="steps", findings=(baseline,),
        )


__all__ = [
    "PlannerOutputContractError",
    "missing_post_baseline_descriptive_claims",
    "validate_fresh_planner_typed_product_specs",
    "validate_baseline_materialization",
    "validate_baseline_outline",
]
