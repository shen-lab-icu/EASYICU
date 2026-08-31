"""Final fail-closed invariants for a host-shaped analysis plan.

Planner output is validated before host transformations, but dependency
closure, mixed-output splitting, and step capping can still damage the final
graph.  This dependency-light owner validates only the post-shaping structure;
it does not choose scientific methods or repair a plan.
"""

from __future__ import annotations

import re
from typing import Sequence

from ..contracts.declared_product import typed_product
from ..plan_utils import (
    _cap_plan_preserving_figure_steps,
    _clustering_contract_applies,
    _cohort_definition_contract_findings,
    _cohort_definition_is_empty,
    _ensure_publication_figure_step_in_plan,
    _enforce_advanced_plan_contract,
    _plan_expects_analysis_cohort,
    _prediction_contract_applies,
    _split_table_and_figure_outputs_in_plan,
    _typed_plan_dag_findings,
)
from .endpoint_contract import endpoint_contract_findings
from .figure_plan_shaping import (
    augment_report_typed_product_inputs as _augment_report_typed_product_inputs,
)
from ..schema import AnalysisPlan


class PlanShapeValidationError(ValueError):
    """A host-shaped plan is structurally unsafe to review or execute."""

    def __init__(self, *, reason: str, step_ids: Sequence[str]) -> None:
        self.reason = str(reason)
        self.step_ids = tuple(str(step_id) for step_id in step_ids)
        super().__init__(f"{self.reason}: " + ", ".join(self.step_ids))


def _method_head(method: str) -> str:
    normalized = re.sub(
        r"[^a-z0-9]+", "_", str(method or "").strip().lower()
    ).strip("_")
    return normalized.split("_with_", 1)[0]


def validate_final_plan_shape(plan: AnalysisPlan) -> None:
    """Reject empty renderers and duplicate typed figure ownership."""

    invalid_step_ids = [
        str(step.step_id)
        for step in plan.steps or []
        if _method_head(str(step.method or "")) == "visualization"
        and not any(
            (product := typed_product(output)) is not None
            and product[0] == "figure"
            for output in step.expected_outputs or []
        )
    ]
    if invalid_step_ids:
        raise PlanShapeValidationError(
            reason="visualization_without_typed_figure_output",
            step_ids=invalid_step_ids,
        )

    figure_owners: dict[tuple[str, str], list[str]] = {}
    for step in plan.steps or []:
        for output in step.expected_outputs or []:
            product = typed_product(output)
            if product is not None and product[0] == "figure":
                figure_owners.setdefault(product, []).append(str(step.step_id))
    duplicate_owner_ids = [
        step_id
        for owners in figure_owners.values()
        if len(owners) > 1
        for step_id in owners
    ]
    if duplicate_owner_ids:
        raise PlanShapeValidationError(
            reason="duplicate_typed_figure_output",
            step_ids=duplicate_owner_ids,
        )


__all__ = [
    "PlanShapeValidationError",
    "_augment_report_typed_product_inputs",
    "_cap_plan_preserving_figure_steps",
    "_clustering_contract_applies",
    "_cohort_definition_contract_findings",
    "_cohort_definition_is_empty",
    "_ensure_publication_figure_step_in_plan",
    "_enforce_advanced_plan_contract",
    "_plan_expects_analysis_cohort",
    "_prediction_contract_applies",
    "_split_table_and_figure_outputs_in_plan",
    "_typed_plan_dag_findings",
    "endpoint_contract_findings",
    "validate_final_plan_shape",
]
