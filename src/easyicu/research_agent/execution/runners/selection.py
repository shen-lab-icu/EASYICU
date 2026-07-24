"""Select trusted executors for complete Planner-owned contracts."""

from __future__ import annotations

from dataclasses import dataclass

from ...schema import AnalysisPlan, AnalysisStep
from .deterministic_missingness import (
    missingness_audit_executor_owns_step,
    missingness_measurement_audit_code,
    source_availability_audit_executor_owns_step,
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
) -> StandardExecutorSelection | None:
    """Select by exact typed contract, never by prose or benchmark identity."""

    if table_one_executor_owns_step(step):
        return StandardExecutorSelection(
            analysis_kind="grouped_table_one",
            selection_reason="table_one_spec_preflight",
            progress_message="Using planner-specified grouped Table 1 executor",
            code=table_one_executor_code(step),
            consumed_input_keys=tuple(
                value
                for value in step.inputs
                if str(value).strip() == "artifact:analysis_cohort"
            ),
        )
    if missingness_audit_executor_owns_step(step):
        source_availability = source_availability_audit_executor_owns_step(step)
        return StandardExecutorSelection(
            analysis_kind=(
                "missingness_source_availability_audit"
                if source_availability
                else "missingness_complete_case_audit"
            ),
            selection_reason=(
                "missingness_source_availability_contract_preflight"
                if source_availability
                else "missingness_complete_case_contract_preflight"
            ),
            progress_message="Using planner-specified missingness audit executor",
            code=missingness_measurement_audit_code(),
            consumed_input_keys=tuple(
                value
                for value in step.inputs
                if str(value).strip() == "artifact:analysis_cohort"
            ),
        )
    if trajectory_stability_executor_owns_step(step, plan=plan):
        return StandardExecutorSelection(
            analysis_kind="trajectory_cluster_stability",
            selection_reason="trajectory_stability_spec_preflight",
            progress_message="Using planner-specified trajectory stability executor",
            code=trajectory_stability_executor_code(step, plan=plan),
            consumed_input_keys=tuple(sorted(STABILITY_EXECUTOR_INPUTS)),
        )
    return None
