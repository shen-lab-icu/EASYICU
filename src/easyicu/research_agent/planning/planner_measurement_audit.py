"""Planner-only contract for deterministic measurement-audit declarations.

Recorded plans may predate ``MeasurementAuditSpec`` and remain replayable.
Fresh Planner output has no such ambiguity allowance: when it proposes a
count-only measurement/missingness audit, it must say which closed audit each
declared table represents so renderers and readers do not infer science from a
product label.
"""

from __future__ import annotations

import re

from ..contracts.declared_product import typed_product
from ..schema import AnalysisPlan, AnalysisStep


class PlannerMeasurementAuditContractError(ValueError):
    """Fresh Planner output omitted typed measurement-audit authority."""


_COUNT_AUDIT_TOKENS = frozenset(
    {
        "audit",
        "availability",
        "completeness",
        "component",
        "data",
        "denominator",
        "denominators",
        "event",
        "flow",
        "measurement",
        "missingness",
        "process",
        "quality",
        "source",
        "timing",
    }
)
_RICH_ANALYSIS_TOKENS = frozenset(
    {
        "bias",
        "estimation",
        "model",
        "modeling",
        "modelling",
        "reconciliation",
        "repair",
        "representation",
        "trajectory",
        "validation",
    }
)


def _method_tokens(value: object) -> frozenset[str]:
    return frozenset(re.findall(r"[a-z0-9]+", str(value or "").casefold()))


def _is_count_only_measurement_audit(step: AnalysisStep) -> bool:
    tokens = _method_tokens(step.method)
    if (
        "audit" not in tokens
        or not tokens.intersection(
            {"measurement", "missingness", "availability", "quality"}
        )
        or tokens.intersection(_RICH_ANALYSIS_TOKENS)
        or not tokens <= _COUNT_AUDIT_TOKENS
    ):
        return False
    outputs = [typed_product(value) for value in step.expected_outputs]
    return bool(outputs) and all(
        product is not None and product[0] == "table" for product in outputs
    )


def missing_planner_measurement_audit_specs(
    plan: AnalysisPlan,
) -> tuple[str, ...]:
    """Return fresh count-only audit steps lacking their typed product map."""

    return tuple(
        str(step.step_id)
        for step in plan.steps
        if _is_count_only_measurement_audit(step)
        and step.measurement_audit_spec is None
    )


def validate_planner_measurement_audit_specs(plan: AnalysisPlan) -> None:
    """Reject ambiguous fresh audit tables before plan review and digesting."""

    missing = missing_planner_measurement_audit_specs(plan)
    if not missing:
        return
    raise PlannerMeasurementAuditContractError(
        "Planner count-only measurement/missingness audit steps must declare "
        "measurement_audit_spec.products with one exact product_id per table "
        "and one legal audit kind; missing for "
        f"{list(missing)!r}. Product labels and figure prose do not establish "
        "measurement_missingness, measurement_process, source availability, "
        "event timing, component completeness, analytic denominators, or "
        "cohort-flow authority."
    )


__all__ = [
    "PlannerMeasurementAuditContractError",
    "missing_planner_measurement_audit_specs",
    "validate_planner_measurement_audit_specs",
]
