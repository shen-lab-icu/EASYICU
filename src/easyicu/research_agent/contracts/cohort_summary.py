"""Closed descriptive-summary ownership shared by planning and execution."""

from __future__ import annotations

from typing import TYPE_CHECKING

from .cohort_product_keys import sole_typed_cohort_input

if TYPE_CHECKING:
    from ..schema import AnalysisStep


def declared_summary_columns(step: AnalysisStep) -> tuple[str, ...]:
    return tuple(
        str(value).strip()
        for value in step.inputs
        if str(value).strip() and ":" not in str(value).strip()
    )


def is_descriptive_cohort_summary_step(step: AnalysisStep) -> bool:
    """Own only a complete, auxiliary, count/descriptive-only contract."""

    columns = declared_summary_columns(step)
    return bool(
        str(step.method or "").strip().casefold()
        in {"descriptive_cohort_summary", "descriptive"}
        and str(step.planned_analysis_role or "").strip().casefold() == "auxiliary"
        and list(step.expected_outputs or []) == ["table:cohort_summary"]
        and columns
        and len(columns) == len(set(columns))
        and sole_typed_cohort_input(step) != ""
        and not step.model_requirements
        and step.table_one_spec is None
        and step.trajectory_stability_spec is None
    )

