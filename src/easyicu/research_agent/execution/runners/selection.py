"""Select trusted executors for complete Planner-owned contracts."""

from __future__ import annotations

from dataclasses import dataclass

from ...authority.plausibility import FlagOnlyPlausibilityScope
from ...schema import AnalysisPlan, AnalysisStep
from .cohort_summary_executor import (
    cohort_summary_executor_code,
    cohort_summary_executor_owns_step,
)
from .deterministic_missingness import (
    is_compact_missingness_measurement_contract,
    missingness_audit_cohort_input_key,
    missingness_audit_executor_owns_step,
    missingness_measurement_audit_code,
    source_availability_audit_executor_owns_step,
)
from .missingness_measurement_figure_executor import (
    MISSINGNESS_MEASUREMENT_FIGURE_INPUTS,
    missingness_measurement_figure_executor_code,
    missingness_measurement_figure_executor_owns_step,
)
from .prevalence_outcome_figure_executor import (
    PREVALENCE_OUTCOME_FIGURE_INPUT,
    prevalence_outcome_figure_executor_code,
    prevalence_outcome_figure_executor_owns_step,
)
from .prevalence_mortality_figure_executor import (
    PREVALENCE_MORTALITY_FIGURE_INPUTS,
    prevalence_mortality_figure_executor_code,
    prevalence_mortality_figure_executor_owns_step,
)
from .table_one_executor import table_one_executor_code, table_one_executor_owns_step
from .trajectory_stability_executor import (
    STABILITY_EXECUTOR_INPUTS,
    trajectory_stability_executor_code,
    trajectory_stability_executor_owns_step,
)

__all__ = ["StandardExecutorSelection", "select_standard_executor"]


@dataclass(frozen=True, slots=True)
class StandardExecutorSelection:
    """One deterministic implementation of already-fixed Planner science."""

    analysis_kind: str
    selection_reason: str
    progress_message: str
    code: str
    consumed_input_keys: tuple[str, ...]


def select_standard_executor(
    step: AnalysisStep,
    *,
    plan: AnalysisPlan,
    plausibility_scope: FlagOnlyPlausibilityScope | None = None,
) -> StandardExecutorSelection | None:
    """Select by exact typed contract, never by prose or benchmark identity."""

    if plausibility_scope is not None:
        plausibility_scope.require_step(step.step_id)
    receipt_required = bool(
        plausibility_scope is not None
        and plausibility_scope.expected_columns
    )

    if cohort_summary_executor_owns_step(step):
        if receipt_required:
            return None
        typed_cohort_inputs = tuple(
            str(value or "").strip()
            for value in step.inputs
            if str(value or "").strip().startswith("cohort:")
            or str(value or "").strip() == "artifact:analysis_cohort"
        )
        return StandardExecutorSelection(
            analysis_kind="descriptive_cohort_summary",
            selection_reason="cohort_summary_contract_preflight",
            progress_message="Using planner-scoped cohort summary executor",
            code=cohort_summary_executor_code(step),
            consumed_input_keys=typed_cohort_inputs,
        )
    if prevalence_outcome_figure_executor_owns_step(step):
        if receipt_required:
            return None
        return StandardExecutorSelection(
            analysis_kind="prevalence_outcome_figure",
            selection_reason="prevalence_outcome_figure_contract_preflight",
            progress_message="Using planner-scoped prevalence/outcome figure executor",
            code=prevalence_outcome_figure_executor_code(step),
            consumed_input_keys=(PREVALENCE_OUTCOME_FIGURE_INPUT,),
        )
    if prevalence_mortality_figure_executor_owns_step(step):
        if receipt_required:
            return None
        return StandardExecutorSelection(
            analysis_kind="prevalence_mortality_figure",
            selection_reason="prevalence_mortality_figure_contract_preflight",
            progress_message=(
                "Using planner-scoped prevalence/mortality figure executor"
            ),
            code=prevalence_mortality_figure_executor_code(
                step,
                display_labels=plan.display_labels,
            ),
            consumed_input_keys=PREVALENCE_MORTALITY_FIGURE_INPUTS,
        )
    if missingness_measurement_figure_executor_owns_step(step):
        if receipt_required:
            return None
        return StandardExecutorSelection(
            analysis_kind="missingness_measurement_figure",
            selection_reason="missingness_measurement_figure_contract_preflight",
            progress_message=(
                "Using planner-scoped missingness/measurement figure executor"
            ),
            code=missingness_measurement_figure_executor_code(step),
            consumed_input_keys=MISSINGNESS_MEASUREMENT_FIGURE_INPUTS,
        )
    if table_one_executor_owns_step(step):
        typed_cohort_inputs = tuple(
            str(value or "").strip()
            for value in step.inputs
            if str(value or "").strip().startswith("cohort:")
            or str(value or "").strip() == "artifact:analysis_cohort"
        )
        return StandardExecutorSelection(
            analysis_kind="grouped_table_one",
            selection_reason="table_one_spec_preflight",
            progress_message="Using planner-specified grouped Table 1 executor",
            code=table_one_executor_code(
                step,
                plausibility_scope=plausibility_scope,
            ),
            consumed_input_keys=typed_cohort_inputs,
        )
    if missingness_audit_executor_owns_step(step):
        source_availability = source_availability_audit_executor_owns_step(step)
        compact_measurement = is_compact_missingness_measurement_contract(
            step.method,
            step.expected_outputs,
        )
        typed_cohort_input = missingness_audit_cohort_input_key(step)
        return StandardExecutorSelection(
            analysis_kind=(
                "missingness_source_availability_audit"
                if source_availability
                else (
                    "missingness_measurement_audit"
                    if compact_measurement
                    else "missingness_complete_case_audit"
                )
            ),
            selection_reason=(
                "missingness_source_availability_contract_preflight"
                if source_availability
                else (
                    "missingness_measurement_contract_preflight"
                    if compact_measurement
                    else "missingness_complete_case_contract_preflight"
                )
            ),
            progress_message="Using planner-specified missingness audit executor",
            code=missingness_measurement_audit_code(
                step,
                plausibility_scope=plausibility_scope,
            ),
            consumed_input_keys=(
                (typed_cohort_input,) if typed_cohort_input is not None else ()
            ),
        )
    if trajectory_stability_executor_owns_step(step, plan=plan):
        if receipt_required:
            return None
        return StandardExecutorSelection(
            analysis_kind="trajectory_cluster_stability",
            selection_reason="trajectory_stability_spec_preflight",
            progress_message="Using planner-specified trajectory stability executor",
            code=trajectory_stability_executor_code(step, plan=plan),
            consumed_input_keys=tuple(sorted(STABILITY_EXECUTOR_INPUTS)),
        )
    return None
