"""Fresh Planner-only typed product declarations.

This owner keeps rules that the model can repair out of the broad Agent parser
while leaving ``AnalysisPlan`` compatible with digest-verified historical
plans.  Scientific choices are declared here; runtime owners only execute the
resulting immutable contracts.
"""

from __future__ import annotations

from ..schema import AnalysisPlan
from .planner_measurement_audit import validate_planner_measurement_audit_specs


def validate_fresh_planner_typed_product_specs(plan: AnalysisPlan) -> None:
    """Reject ambiguous typed products in a newly generated Planner response."""

    validate_planner_measurement_audit_specs(plan)
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


__all__ = ["validate_fresh_planner_typed_product_specs"]
