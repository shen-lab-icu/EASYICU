"""Closed repair for direct reads of resolved typed-input identity rows."""

from __future__ import annotations

import ast
from typing import Sequence

from ..schema import ValidationFinding


def patch_direct_resolved_input_identity_key(
    code: str,
    *,
    findings: Sequence[ValidationFinding],
) -> str:
    """Read the exact host identity key without changing input selection."""

    matching = [
        finding
        for finding in findings
        if (finding.detail or {}).get("reason") == "resolved_input_key_not_materialized"
        and isinstance((finding.detail or {}).get("binding_name"), str)
    ]
    if len(matching) != 1:
        return code
    detail = matching[0].detail or {}
    binding_name = detail.get("binding_name")
    occurrences = detail.get("access_occurrences")
    if (
        not isinstance(binding_name, str)
        or not binding_name.isidentifier()
        or not isinstance(occurrences, list)
        or not occurrences
    ):
        return code
    coordinates: list[tuple[int, int, int, int]] = []
    for occurrence in occurrences:
        if not isinstance(occurrence, dict):
            return code
        values = (
            occurrence.get("line"),
            occurrence.get("column"),
            occurrence.get("end_line"),
            occurrence.get("end_column"),
        )
        if not all(
            isinstance(value, int) and not isinstance(value, bool) and value >= 0
            for value in values
        ):
            return code
        coordinates.append(tuple(int(value) for value in values))
    if len(coordinates) != len(set(coordinates)):
        return code

    try:
        tree = ast.parse(code)
    except SyntaxError:
        return code
    candidate_coordinates: set[tuple[int, int, int, int]] = set()
    for node in ast.walk(tree):
        if (
            getattr(node, "end_lineno", None) is None
            or getattr(node, "end_col_offset", None) is None
        ):
            continue
        direct_subscript = (
            isinstance(node, ast.Subscript)
            and isinstance(node.value, ast.Name)
            and node.value.id == binding_name
            and isinstance(node.slice, ast.Constant)
            and node.slice.value == "input_key"
        )
        direct_get = (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and node.func.attr == "get"
            and isinstance(node.func.value, ast.Name)
            and node.func.value.id == binding_name
            and node.args
            and isinstance(node.args[0], ast.Constant)
            and node.args[0].value == "input_key"
        )
        if direct_subscript or direct_get:
            candidate_coordinates.add(
                (
                    int(node.lineno),
                    int(node.col_offset),
                    int(node.end_lineno),
                    int(node.end_col_offset),
                )
            )
    if not set(coordinates) <= candidate_coordinates:
        return code

    lines = code.splitlines(keepends=True)
    line_starts: list[int] = []
    offset = 0
    for line in lines:
        line_starts.append(offset)
        offset += len(line)

    def absolute_offset(lineno: int, utf8_col: int) -> int:
        line = lines[lineno - 1]
        char_col = len(line.encode("utf-8")[:utf8_col].decode("utf-8"))
        return line_starts[lineno - 1] + char_col

    repaired = code
    replacement = f'{binding_name}["identity_row"]["input_key"]'
    for line, column, end_line, end_column in sorted(coordinates, reverse=True):
        start = absolute_offset(line, column)
        end = absolute_offset(end_line, end_column)
        repaired = repaired[:start] + replacement + repaired[end:]
    try:
        ast.parse(repaired)
    except SyntaxError:
        return code
    return repaired


__all__ = ["patch_direct_resolved_input_identity_key"]
