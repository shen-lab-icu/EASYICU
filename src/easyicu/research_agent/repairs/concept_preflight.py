"""Ordered deterministic repairs applied before concept-level branching."""

from __future__ import annotations

from typing import Sequence

from ..schema import ValidationFinding
from .binary_feasibility import patch_binary_domain_before_authored_feasibility
from .cohort_row_count import patch_hardcoded_execution_cohort_row_count
from .domain_guards import patch_llm_proven_domain_guards
from .host_helper_result import (
    patch_closed_counts_level_column,
    patch_table_one_planner_spec,
)
from .host_helper_failure import patch_host_validation_helper_reraise
from .local_binding import patch_local_read_before_assignment_hoist
from .runtime_context_levels import patch_runtime_context_opaque_levels


def patch_concept_preflight_repairs(
    code: str,
    *,
    findings: Sequence[ValidationFinding],
) -> tuple[str, list[str]]:
    """Apply the three independent pre-branch guards in stable order."""

    repaired = code
    names: list[str] = []
    candidate = patch_closed_counts_level_column(
        repaired,
        findings=findings,
    )
    if candidate != repaired:
        repaired = candidate
        names.append("closed_counts_level_column_v1")
    candidate = patch_table_one_planner_spec(repaired, findings=findings)
    if candidate != repaired:
        repaired = candidate
        names.append("table_one_planner_spec_binding_v1")
    candidate = patch_binary_domain_before_authored_feasibility(
        repaired,
        repair_findings=findings,
    )
    if candidate != repaired:
        repaired = candidate
        names.append("binary_domain_authored_feasibility_v1")
    candidate = patch_local_read_before_assignment_hoist(
        repaired,
        repair_findings=findings,
    )
    if candidate != repaired:
        repaired = candidate
        names.append("local_read_before_assignment_hoist_v1")
    candidate = patch_llm_proven_domain_guards(repaired, findings=findings)
    if candidate != repaired:
        repaired = candidate
        names.append("llm_proven_numeric_domain_guards_v1")
    candidate = patch_host_validation_helper_reraise(repaired, findings=findings)
    if candidate != repaired:
        repaired = candidate
        names.append("host_validation_helper_reraise_v1")
    candidate = patch_hardcoded_execution_cohort_row_count(
        repaired,
        findings=findings,
    )
    if candidate != repaired:
        repaired = candidate
        names.append("execution_cohort_runtime_row_count_v1")
    candidate = patch_runtime_context_opaque_levels(
        repaired,
        findings=findings,
    )
    if candidate != repaired:
        repaired = candidate
        names.append("runtime_context_private_levels_v1")
    return repaired, names


__all__ = ["patch_concept_preflight_repairs"]
