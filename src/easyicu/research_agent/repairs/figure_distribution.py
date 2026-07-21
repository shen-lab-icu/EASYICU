"""Narrow repairs for generated categorical-distribution renderers."""

from __future__ import annotations

import ast
import re
from typing import Any, Mapping, Optional

_EMPTY_DISTRIBUTION_MARKERS = (
    "no supported ",
    "categorical distribution rows were found",
    "explicitly supported distribution statistic",
)
_KNOWN_DISTRIBUTION_ROLES = {
    "distribution",
    "category_distribution",
    "level_distribution",
    "frequency",
    "count",
    "percentage",
    "prevalence",
}


def _subscript_key(node: ast.AST) -> Optional[str]:
    if not isinstance(node, ast.Subscript):
        return None
    value = node.slice
    return (
        value.value
        if isinstance(value, ast.Constant) and isinstance(value.value, str)
        else None
    )


def _has_category_nonnull_guard(tree: ast.AST) -> bool:
    return any(
        isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr == "notna"
        and _subscript_key(node.func.value) == "category"
        for node in ast.walk(tree)
    )


def _used_as_statistic_role_set(tree: ast.AST, name: str) -> bool:
    for node in ast.walk(tree):
        if not (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and node.func.attr == "isin"
            and len(node.args) == 1
            and isinstance(node.args[0], ast.Name)
            and node.args[0].id == name
        ):
            continue
        receiver = node.func.value
        if isinstance(receiver, ast.Subscript):
            key = _subscript_key(receiver)
            if key is not None and "statistic" in key.lower():
                return True
    return False


def _set_assignment_candidates(tree: ast.AST) -> list[tuple[ast.Name, ast.Set]]:
    candidates: list[tuple[ast.Name, ast.Set]] = []
    for node in ast.walk(tree):
        if not (
            isinstance(node, ast.Assign)
            and len(node.targets) == 1
            and isinstance(node.targets[0], ast.Name)
            and isinstance(node.value, ast.Set)
        ):
            continue
        values = {
            element.value
            for element in node.value.elts
            if isinstance(element, ast.Constant) and isinstance(element.value, str)
        }
        if len(values) != len(node.value.elts):
            continue
        if "clinical_bin" in values or len(values & _KNOWN_DISTRIBUTION_ROLES) < 3:
            continue
        candidates.append((node.targets[0], node.value))
    return candidates


def _append_set_literal(code: str, node: ast.Set) -> Optional[str]:
    if node.end_lineno is None or node.end_col_offset is None:
        return None
    lines = code.splitlines(keepends=True)
    start = sum(len(line) for line in lines[: node.lineno - 1]) + node.col_offset
    end = sum(len(line) for line in lines[: node.end_lineno - 1]) + node.end_col_offset
    segment = code[start:end]
    segment_lines = segment.splitlines()
    if len(segment_lines) >= 2 and segment_lines[-1].strip() == "}":
        item_indent = re.match(r"\s*", segment_lines[1]).group(0)
        replacement = "\n".join(
            [*segment_lines[:-1], f'{item_indent}"clinical_bin",', segment_lines[-1]]
        )
    elif segment.endswith("}"):
        replacement = segment[:-1].rstrip() + ', "clinical_bin"}'
    else:
        return None
    repaired = code[:start] + replacement + code[end:]
    try:
        ast.parse(repaired)
    except SyntaxError:
        return None
    return repaired


def patch_categorical_distribution_clinical_bin_role(
    code: str,
    step_summary: Mapping[str, Any],
) -> Optional[str]:
    """Admit the host's closed ``clinical_bin`` role in one failed renderer.

    The patch is authorized only when the renderer itself reported that its
    categorical role allowlist selected no rows, the source contains one
    unambiguous literal role set used against a statistic column, and category
    rows remain protected by a non-null guard.  It changes no values, bins,
    labels, denominators, or reconciliation checks.
    """

    summary_text = str(step_summary).lower()
    if not all(marker in summary_text for marker in _EMPTY_DISTRIBUTION_MARKERS):
        return None
    try:
        tree = ast.parse(code)
    except SyntaxError:
        return None
    if not _has_category_nonnull_guard(tree):
        return None
    candidates = [
        set_node
        for name_node, set_node in _set_assignment_candidates(tree)
        if _used_as_statistic_role_set(tree, name_node.id)
    ]
    if len(candidates) != 1:
        return None
    return _append_set_literal(code, candidates[0])


__all__ = ["patch_categorical_distribution_clinical_bin_role"]
