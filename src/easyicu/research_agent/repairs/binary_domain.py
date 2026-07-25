"""Fail-closed repair for a host-proven binary primary-exposure domain."""

from __future__ import annotations

import ast
from typing import Sequence

from ..schema import ResearchContext, ValidationFinding


def _references_name(node: ast.AST, name: str) -> bool:
    return any(
        isinstance(child, ast.Name)
        and isinstance(child.ctx, ast.Load)
        and child.id == name
        for child in ast.walk(node)
    )


def _is_zero(node: ast.AST) -> bool:
    return (
        isinstance(node, ast.Constant)
        and not isinstance(node.value, bool)
        and node.value in {0, 0.0}
    )


def patch_observed_binary_primary_exposure_guard(
    code: str,
    *,
    context: ResearchContext,
    repair_findings: Sequence[ValidationFinding],
) -> str:
    """Guard one thresholded primary exposure whose host domain is binary.

    The model-origin finding only nominates a variable.  It does not authorize
    the domain.  The transformation proceeds only when ResearchContext itself
    identifies that exact primary exposure as observed numeric binary and the
    candidate has one unguarded alias that partitions values at zero.  The
    inserted check can only fail closed; it does not recode a value or choose a
    cohort, threshold, model, or estimand.
    """

    primary_exposure = str(context.primary_exposure or "").strip()
    if not primary_exposure:
        return code
    descriptors = [
        descriptor
        for descriptor in context.variables
        if str(descriptor.name or "").strip() == primary_exposure
    ]
    if len(descriptors) != 1 or not bool(
        (descriptors[0].observed_domain or {}).get("is_binary") is True
    ):
        return code
    nominated = False
    for finding in repair_findings:
        if finding.validator != "llm_concept_auditor" or finding.severity != "error":
            continue
        variables = (finding.detail or {}).get("variables")
        if isinstance(variables, list) and variables == [primary_exposure]:
            nominated = True
            break
    if not nominated:
        return code
    try:
        tree = ast.parse(code)
    except SyntaxError:
        return code
    assignments: list[tuple[ast.Assign, str]] = []
    for node in ast.walk(tree):
        if not (
            isinstance(node, ast.Assign)
            and len(node.targets) == 1
            and isinstance(node.targets[0], ast.Name)
            and isinstance(node.value, ast.Subscript)
            and isinstance(node.value.slice, ast.Constant)
            and node.value.slice.value == primary_exposure
        ):
            continue
        assignments.append((node, node.targets[0].id))
    if len(assignments) != 1:
        return code
    assignment, alias = assignments[0]
    if "_easyicu_observed_binary_primary_exposure_guard_v1" in code:
        return code
    lower_partition = False
    upper_partition = False
    exact_domain_check = False
    for node in ast.walk(tree):
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute):
            if node.func.attr == "isin" and _references_name(node.func.value, alias):
                if len(node.args) == 1 and isinstance(
                    node.args[0], (ast.List, ast.Tuple)
                ):
                    values = [
                        item.value
                        for item in node.args[0].elts
                        if isinstance(item, ast.Constant)
                        and not isinstance(item.value, bool)
                    ]
                    if values == [0, 1]:
                        exact_domain_check = True
        if not isinstance(node, ast.Compare) or len(node.ops) != 1:
            continue
        if len(node.comparators) != 1 or not _is_zero(node.comparators[0]):
            continue
        if not _references_name(node.left, alias):
            continue
        lower_partition = lower_partition or isinstance(node.ops[0], ast.LtE)
        upper_partition = upper_partition or isinstance(node.ops[0], ast.Gt)
    if exact_domain_check or not (lower_partition and upper_partition):
        return code
    if assignment.end_lineno is None:
        return code
    lines = code.splitlines(keepends=True)
    insertion_index = int(assignment.end_lineno)
    source_line = lines[int(assignment.lineno) - 1]
    indent = source_line[: len(source_line) - len(source_line.lstrip())]
    guard = (
        f"{indent}# _easyicu_observed_binary_primary_exposure_guard_v1\n"
        f"{indent}if not bool({alias}.dropna().isin([0, 1]).all()):\n"
        f'{indent}    raise RuntimeError("Host-bound binary primary exposure '
        'violates the exact {0,1} domain.")\n'
    )
    repaired = (
        "".join(lines[:insertion_index]) + guard + "".join(lines[insertion_index:])
    )
    try:
        ast.parse(repaired)
    except SyntaxError:
        return code
    return repaired


__all__ = ["patch_observed_binary_primary_exposure_guard"]
