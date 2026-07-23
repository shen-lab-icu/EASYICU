"""Narrow repair for confusing consumer scope with physical table schema."""

from __future__ import annotations

import ast
from typing import Optional

_PHYSICAL_SCOPE_ERROR = (
    "locked cohort columns do not match the exact declared input scope"
)


def _physical_columns_owner(node: ast.AST) -> Optional[ast.AST]:
    """Return ``df`` from ``list/set/tuple(df.columns)`` or ``df.columns``."""

    candidate = node
    if (
        isinstance(candidate, ast.Call)
        and isinstance(candidate.func, ast.Name)
        and candidate.func.id in {"list", "set", "tuple"}
        and len(candidate.args) == 1
        and not candidate.keywords
    ):
        candidate = candidate.args[0]
    if isinstance(candidate, ast.Attribute) and candidate.attr == "columns":
        return candidate.value
    return None


def patch_raw_input_physical_superset_guard(code: str, run_log: str) -> str:
    """Replace one proven closed-world column assertion with a presence check.

    ``planner_declared_inputs`` is consumer authority, while ``COHORT_PARQUET`` is
    a host-locked physical source that may legitimately contain more columns.
    This repair changes only an authored ``require`` whose own diagnostic and the
    runner log both name the exact failure. It never selects, drops, or transforms
    data and retains fail-closed behavior when a declared input is absent.
    """

    if _PHYSICAL_SCOPE_ERROR not in str(run_log or "").lower():
        return code
    try:
        tree = ast.parse(code)
    except SyntaxError:
        return code

    lines = code.splitlines(keepends=True)
    line_starts: list[int] = []
    offset = 0
    for line in lines:
        line_starts.append(offset)
        offset += len(line)

    def _absolute_offset(lineno: int, utf8_col: int) -> int:
        line = lines[lineno - 1]
        char_col = len(line.encode("utf-8")[:utf8_col].decode("utf-8"))
        return line_starts[lineno - 1] + char_col

    replacements: list[tuple[int, int, str]] = []
    for node in ast.walk(tree):
        if not (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Name)
            and node.func.id == "require"
            and len(node.args) >= 2
            and isinstance(node.args[1], ast.Constant)
            and isinstance(node.args[1].value, str)
            and _PHYSICAL_SCOPE_ERROR in node.args[1].value.lower()
        ):
            continue
        condition = node.args[0]
        binding: tuple[ast.AST, ast.AST] | None = None
        for comparison in ast.walk(condition):
            if not (
                isinstance(comparison, ast.Compare)
                and len(comparison.ops) == 1
                and isinstance(comparison.ops[0], ast.Eq)
                and len(comparison.comparators) == 1
            ):
                continue
            left_owner = _physical_columns_owner(comparison.left)
            right_owner = _physical_columns_owner(comparison.comparators[0])
            if (left_owner is None) == (right_owner is None):
                continue
            frame_node = left_owner if left_owner is not None else right_owner
            expected_node = (
                comparison.comparators[0]
                if left_owner is not None
                else comparison.left
            )
            binding = (frame_node, expected_node)
            break
        if binding is None:
            continue
        frame_node, expected_node = binding
        frame_source = ast.get_source_segment(code, frame_node)
        expected_source = ast.get_source_segment(code, expected_node)
        if not frame_source or not expected_source:
            continue
        replacement = (
            "all(_easyicu_input in "
            f"({frame_source}).columns for _easyicu_input in ({expected_source}))"
        )
        replacements.append(
            (
                _absolute_offset(condition.lineno, condition.col_offset),
                _absolute_offset(condition.end_lineno, condition.end_col_offset),
                replacement,
            )
        )

    if not replacements:
        return code
    repaired = code
    for start, end, replacement in sorted(replacements, reverse=True):
        repaired = repaired[:start] + replacement + repaired[end:]
    try:
        ast.parse(repaired)
    except SyntaxError:
        return code
    return repaired


__all__ = ["patch_raw_input_physical_superset_guard"]
