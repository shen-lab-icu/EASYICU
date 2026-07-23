"""Repair hard-coded execution-cohort row-count assertions.

The research context describes the run input before a Planner-locked cohort is
materialised.  Downstream generated code must therefore not treat that initial
cardinality as the current ``COHORT_PARQUET`` cardinality.  The runner exposes
the exact current value through the host-owned ``EASYICU_COHORT_ROWS``
coordinate; this module replaces only constants identified by the mechanical
gate as comparisons against the loaded execution cohort.
"""

from __future__ import annotations

import ast
from typing import Sequence

from ..schema import ValidationFinding

_HOST_ROW_COUNT_EXPRESSION = 'int(__import__("os").environ["EASYICU_COHORT_ROWS"])'


def patch_hardcoded_execution_cohort_row_count(
    code: str,
    *,
    findings: Sequence[ValidationFinding],
) -> str:
    """Replace exact gate-identified literals with the host runtime coordinate."""

    coordinates: list[tuple[int, int, int, int]] = []
    for finding in findings:
        detail = finding.detail or {}
        if (
            finding.validator != "mechanical_code_preflight"
            or finding.severity != "error"
            or detail.get("reason") != "execution_cohort_row_count_hardcoded"
        ):
            continue
        values = (
            detail.get("line"),
            detail.get("column"),
            detail.get("end_line"),
            detail.get("end_column"),
        )
        if not all(
            isinstance(value, int) and not isinstance(value, bool) and int(value) >= 0
            for value in values
        ):
            return code
        if int(values[0]) <= 0 or int(values[2]) <= 0:
            return code
        coordinates.append(tuple(int(value) for value in values))
    if not coordinates or len(coordinates) != len(set(coordinates)):
        return code

    try:
        tree = ast.parse(code)
    except SyntaxError:
        return code
    constants = {
        (
            int(node.lineno),
            int(node.col_offset),
            int(node.end_lineno),
            int(node.end_col_offset),
        ): node
        for node in ast.walk(tree)
        if isinstance(node, ast.Constant)
        and isinstance(node.value, int)
        and not isinstance(node.value, bool)
        and node.value > 0
        and node.end_lineno is not None
        and node.end_col_offset is not None
    }
    if any(coordinate not in constants for coordinate in coordinates):
        return code

    lines = code.splitlines(keepends=True)
    starts: list[int] = []
    offset = 0
    for line in lines:
        starts.append(offset)
        offset += len(line)

    def _absolute_offset(line: int, utf8_column: int) -> int:
        source_line = lines[line - 1]
        character_column = len(
            source_line.encode("utf-8")[:utf8_column].decode("utf-8")
        )
        return starts[line - 1] + character_column

    repaired = code
    replacements = [
        (
            _absolute_offset(line, column),
            _absolute_offset(end_line, end_column),
            _HOST_ROW_COUNT_EXPRESSION,
        )
        for line, column, end_line, end_column in coordinates
    ]
    for start, end, replacement in sorted(replacements, reverse=True):
        repaired = repaired[:start] + replacement + repaired[end:]
    try:
        ast.parse(repaired)
    except SyntaxError:
        return code
    return repaired


__all__ = ["patch_hardcoded_execution_cohort_row_count"]
