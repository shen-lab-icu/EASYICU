"""Bind outbound-safe categorical placeholders to local context levels."""

from __future__ import annotations

import ast
from typing import Sequence

from ..schema import ValidationFinding


def patch_runtime_context_opaque_levels(
    code: str,
    *,
    findings: Sequence[ValidationFinding],
) -> str:
    """Rewrite only gate-identified projection keys to local runtime keys."""

    replacements: list[tuple[tuple[int, int, int, int], str, str]] = []
    for finding in findings:
        detail = finding.detail or {}
        if (
            finding.validator != "mechanical_code_preflight"
            or finding.severity != "error"
            or detail.get("reason") != "runtime_context_opaque_levels_projection"
        ):
            continue
        for field, expected, replacement in (
            ("observed_shape_key", "observed_shape", "observed_domain"),
            ("opaque_levels_key", "opaque_levels", "levels"),
        ):
            coordinate = detail.get(field)
            if not isinstance(coordinate, dict):
                return code
            values = (
                coordinate.get("line"),
                coordinate.get("column"),
                coordinate.get("end_line"),
                coordinate.get("end_column"),
            )
            if not all(
                isinstance(value, int)
                and not isinstance(value, bool)
                and int(value) >= 0
                for value in values
            ):
                return code
            if int(values[0]) <= 0 or int(values[2]) <= 0:
                return code
            replacements.append(
                (tuple(int(value) for value in values), expected, replacement)
            )
    if not replacements or len(replacements) != len(
        {coordinate for coordinate, _, _ in replacements}
    ):
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
        and isinstance(node.value, str)
        and node.end_lineno is not None
        and node.end_col_offset is not None
    }
    if any(
        coordinate not in constants or constants[coordinate].value != expected
        for coordinate, expected, _ in replacements
    ):
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
    absolute_replacements = [
        (
            _absolute_offset(coordinate[0], coordinate[1]),
            _absolute_offset(coordinate[2], coordinate[3]),
            repr(replacement),
        )
        for coordinate, _, replacement in replacements
    ]
    for start, end, replacement in sorted(absolute_replacements, reverse=True):
        repaired = repaired[:start] + replacement + repaired[end:]
    try:
        ast.parse(repaired)
    except SyntaxError:
        return code
    return repaired


__all__ = ["patch_runtime_context_opaque_levels"]
