"""Closed descriptive-summary ownership shared by planning and execution."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Mapping

from .cohort_product_keys import sole_typed_cohort_input

if TYPE_CHECKING:
    from ..schema import AnalysisStep


DESCRIPTIVE_SUMMARY_ACTION = "descriptive.descriptive_summary"
DESCRIPTIVE_SUMMARY_OUTPUTS = ("table:cohort_summary", "table:distribution_summary")
DESCRIPTIVE_SUMMARY_ANALYSIS_KIND = "descriptive_cohort_summary"


def declared_summary_columns(step: AnalysisStep) -> tuple[str, ...]:
    return tuple(
        str(value).strip()
        for value in step.inputs
        if str(value).strip() and ":" not in str(value).strip()
    )


def is_primary_descriptive_summary_step(step: AnalysisStep) -> bool:
    """Own empirical measurement summaries, never a model or inferred endpoint.

    Identity inputs may accompany the measurements; their typed context role
    excludes them from the summarized roster at execution.
    """

    columns = declared_summary_columns(step)
    return bool(
        step.planned_analysis_role == "primary"
        and step.scientific_action_id == DESCRIPTIVE_SUMMARY_ACTION
        and step.scientific_capability in {None, "descriptive_measurement_v1"}
        and tuple(step.expected_outputs) == DESCRIPTIVE_SUMMARY_OUTPUTS
        and columns and len(columns) == len(set(columns))
        and sole_typed_cohort_input(step)
        and not step.model_requirements
        and not any((
            step.table_one_spec, step.trajectory_stability_spec,
            step.exposure_outcome_distribution_spec, step.cohort_definition_spec,
            step.measurement_audit_spec, step.robustness_replay_spec,
            step.family_primary_result_requirement, step.runtime_outcome_contract,
        ))
    )


def descriptive_summary_result_receipt_valid(summary: Any) -> bool:
    """Recognize the validated empirical result, without granting paper authority."""

    if not isinstance(summary, Mapping):
        return False
    receipt = summary.get("descriptive_summary_validation")
    if not isinstance(receipt, Mapping):
        return False
    return bool(
        summary.get("status") == "ok"
        and summary.get("analysis_role") == "primary"
        and summary.get("interpretation_ceiling") == "empirical_stay_distribution"
        and summary.get("adjusted_effect") is None
        and summary.get("paper_authorization_allowed") is False
        and receipt.get("schema_version") == "easyicu.descriptive_summary_validation/1"
        and receipt.get("validated") is True
        and receipt.get("source_rows") == summary.get("cohort_n")
        and receipt.get("columns") == summary.get("declared_columns")
        and bool(receipt.get("columns"))
        and all(
            isinstance(receipt.get(key), str)
            and len(receipt[key]) == 64
            and all(char in "0123456789abcdef" for char in receipt[key])
            for key in ("cohort_summary_sha256", "distribution_summary_sha256", "source_sha256")
        )
    )


def is_descriptive_cohort_summary_step(step: AnalysisStep) -> bool:
    """Own the primary empirical contract or a complete auxiliary summary."""

    if is_primary_descriptive_summary_step(step):
        return True
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
