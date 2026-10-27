"""Closed descriptive-summary ownership shared by planning and execution."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Mapping, Sequence

from ..authority.declared_levels import closed_planning_levels_for
from .cohort_product_keys import sole_typed_cohort_input
from .continuous_outcome_metadata import continuous_summary_columns
from .table_one_semantics import table_one_variable_kind

if TYPE_CHECKING:
    from ..schema import AnalysisStep, ConceptDescriptor, ResearchContext


DESCRIPTIVE_SUMMARY_ACTION = "descriptive.descriptive_summary"
DESCRIPTIVE_SUMMARY_OUTPUTS = ("table:cohort_summary", "table:distribution_summary")
DESCRIPTIVE_SUMMARY_ANALYSIS_KIND = "descriptive_cohort_summary"


def declared_summary_columns(step: AnalysisStep) -> tuple[str, ...]:
    return tuple(
        str(value).strip()
        for value in step.inputs
        if str(value).strip() and ":" not in str(value).strip()
    )


def primary_descriptive_summary_scope(
    *, declared_columns: Sequence[str], variables: Mapping[str, ConceptDescriptor],
    grouping: str | None = None, planning_context: ResearchContext | None = None,
) -> tuple[str | None, str]:
    """Bind one numeric measurement and, optionally, its declared categories.

    Identity roles are lineage only. Grouped outcomes share the continuous
    summary owner's metadata rule, including its row-free factor deferral;
    execution must supply closed levels instead of that planning allowance.
    """

    if not declared_columns or any(name not in variables for name in declared_columns):
        raise ValueError("Primary descriptive summary needs known typed columns")
    measured = [name for name in declared_columns if variables[name].role.value not in {"id", "index"}]
    if grouping in declared_columns and grouping not in measured:
        raise ValueError("Primary descriptive grouping cannot be an identity column")
    if len(measured) == 2 and not grouping:
        categorical = [name for name in measured if table_one_variable_kind(
            variables[name], closed_planning_levels_for(name=name, variables=dict(variables)),
        ) != "continuous"]
        if len(categorical) == 1:
            grouping = categorical[0]
    if len(measured) == 2 and grouping in measured:
        group, outcome = continuous_summary_columns(
            grouping=grouping, raw_inputs=measured, variables=variables,
            planning_context=planning_context,
        )
        return group, outcome
    if len(measured) != 1:
        raise ValueError("Primary descriptive summary needs one measurement and an optional declared grouping")
    variable = variables[measured[0]]
    if (
        not any(token in str(variable.dtype).lower() for token in ("float", "int", "double", "numeric"))
        or (variable.observed_domain or {}).get("levels")
        or table_one_variable_kind(variable, ()) != "continuous"
    ):
        raise ValueError("Primary empirical summary requires numeric measurements")
    return None, measured[0]


def primary_descriptive_summary_scope_error(
    step: AnalysisStep, *, context: ResearchContext,
) -> str | None:
    """Why the executor could not bind this exact primary summary, else ``None``.

    The question is the executor's own, asked of the planning context (with its
    row-free factor deferral). Any other step is not this owner's to judge.
    """

    if not is_primary_descriptive_summary_step(step):
        return None
    try:
        primary_descriptive_summary_scope(
            declared_columns=declared_summary_columns(step),
            variables={variable.name: variable for variable in context.variables},
            grouping=context.primary_exposure, planning_context=context,
        )
    except ValueError as exc:
        return str(exc)
    return None


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
        and receipt.get("grouping_variable") == summary.get("grouping_variable")
        and receipt.get("grouping_levels") == summary.get("grouping_levels")
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
