"""Ordered deterministic repairs for mechanical preflight findings."""

from __future__ import annotations

import ast
from typing import Sequence

from ..schema import ValidationFinding
from .ast_rewrites import _absolute_offset
from .binary_feasibility import patch_binary_domain_before_authored_feasibility
from .cohort_row_count import patch_hardcoded_execution_cohort_row_count
from .domain_guards import patch_llm_proven_domain_guards
from .host_helper_result import (
    patch_closed_counts_level_column,
    patch_host_validation_helper_reraise,
    patch_table_one_planner_spec,
)
from .local_binding import patch_local_read_before_assignment_hoist
from .percentage_identity import patch_bound_percentage_identity_guards
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
    candidate = patch_bound_percentage_identity_guards(
        repaired,
        findings=findings,
    )
    if candidate != repaired:
        repaired = candidate
        names.append("bound_percentage_identity_guard_v1")
    return repaired, names



def patch_statsmodels_interval_method_label(
    code: str,
    *,
    findings: Sequence[ValidationFinding],
) -> str:
    """Replace only the exact literals authorized by the host finding."""

    coordinates: dict[tuple[int, int, int, int], tuple[str, str]] = {}
    for finding in findings:
        detail = finding.detail or {}
        if (
            finding.validator != "mechanical_code_preflight"
            or detail.get("reason") != "confidence_interval_method_mislabeled"
        ):
            continue
        for occurrence in detail.get("occurrences") or []:
            if not isinstance(occurrence, dict):
                continue
            try:
                coordinate = (
                    int(occurrence["line"]),
                    int(occurrence["column"]),
                    int(occurrence["end_line"]),
                    int(occurrence["end_column"]),
                )
            except (KeyError, TypeError, ValueError):
                continue
            coordinates[coordinate] = (
                str(occurrence.get("reported") or ""),
                str(occurrence.get("expected") or ""),
            )
    if not coordinates:
        return code

    try:
        tree = ast.parse(code)
    except SyntaxError:
        return code
    replacements: list[tuple[int, int, str]] = []
    for node in ast.walk(tree):
        if not (
            isinstance(node, ast.Constant)
            and isinstance(node.value, str)
            and node.end_lineno is not None
            and node.end_col_offset is not None
        ):
            continue
        coordinate = (
            int(node.lineno),
            int(node.col_offset),
            int(node.end_lineno),
            int(node.end_col_offset),
        )
        labels = coordinates.get(coordinate)
        if labels is None or node.value != labels[0] or not labels[1]:
            continue
        start = _absolute_offset(code, node.lineno, node.col_offset)
        end = _absolute_offset(code, node.end_lineno, node.end_col_offset)
        replacements.append((start, end, repr(labels[1])))

    if len(replacements) != len(coordinates):
        return code
    repaired = code
    for start, end, replacement in sorted(replacements, reverse=True):
        repaired = repaired[:start] + replacement + repaired[end:]
    try:
        ast.parse(repaired)
    except SyntaxError:
        return code
    return repaired



def patch_preflight_repairs(
    code: str,
    *,
    findings: Sequence[ValidationFinding],
) -> tuple[str, list[str]]:
    """Apply independent coordinate-bound preflight repairs in stable order."""

    repaired, names = patch_concept_preflight_repairs(code, findings=findings)
    candidate = patch_statsmodels_interval_method_label(repaired, findings=findings)
    if candidate != repaired:
        repaired = candidate
        names.append("statsmodels_interval_method_label_v1")
    return repaired, names


__all__ = ["patch_preflight_repairs"]
