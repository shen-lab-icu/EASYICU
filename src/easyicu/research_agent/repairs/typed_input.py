"""Science-neutral source repairs for host-issued typed-input bindings."""

from __future__ import annotations

import ast
from typing import Sequence

from ..gates.typed_input import resolved_input_shadowed_by_cohort_env_findings
from ..schema import ValidationFinding


def patch_resolved_input_relative_path_root(
    code: str,
    *,
    repair_findings: Sequence[ValidationFinding],
) -> str:
    """Join host-issued run-relative input paths to ``EASYICU_RUN_DIR``."""

    coordinates: list[tuple[int, int, int, int]] = []
    matching_findings = 0
    for finding in repair_findings:
        detail = finding.detail or {}
        if detail.get("reason") != "resolved_input_relative_path_wrong_root":
            continue
        matching_findings += 1
        occurrences = detail.get("occurrences")
        if not isinstance(occurrences, list) or not occurrences:
            return code
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
    if (
        matching_findings != 1
        or not coordinates
        or len(coordinates) != len(set(coordinates))
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
        and node.value == "EASYICU_EVIDENCE_DIR"
        and node.end_lineno is not None
        and node.end_col_offset is not None
    }
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

    replacements: list[tuple[int, int]] = []
    for coordinate in coordinates:
        node = constants.get(coordinate)
        if node is None:
            return code
        replacements.append(
            (
                absolute_offset(coordinate[0], coordinate[1]),
                absolute_offset(coordinate[2], coordinate[3]),
            )
        )
    repaired = code
    replacement = repr("EASYICU_RUN_DIR")
    for start, end in sorted(replacements, reverse=True):
        repaired = repaired[:start] + replacement + repaired[end:]
    try:
        ast.parse(repaired)
    except SyntaxError:
        return code
    return repaired


def patch_resolved_input_cohort_env_shadow(
    code: str,
    *,
    repair_findings: Sequence[ValidationFinding],
) -> str:
    """Keep an exact manifest-bound artifact as the physical typed input."""

    try:
        tree = ast.parse(code)
    except SyntaxError:
        return code
    authoritative = resolved_input_shadowed_by_cohort_env_findings(tree)
    supplied = [
        finding
        for finding in repair_findings
        if (finding.detail or {}).get("reason")
        == "resolved_typed_input_shadowed_by_cohort_env"
    ]
    if (
        len(authoritative) != 1
        or len(supplied) != 1
        or supplied[0].detail != authoritative[0].detail
    ):
        return code
    occurrences = authoritative[0].detail.get("occurrences")
    if not isinstance(occurrences, list) or not occurrences:
        return code
    calls = {
        (
            int(node.lineno),
            int(node.col_offset),
            int(node.end_lineno),
            int(node.end_col_offset),
        ): node
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and node.end_lineno is not None
        and node.end_col_offset is not None
    }
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

    replacements: list[tuple[int, int, str]] = []
    for occurrence in occurrences:
        if not isinstance(occurrence, dict):
            return code
        values = (
            occurrence.get("line"),
            occurrence.get("column"),
            occurrence.get("end_line"),
            occurrence.get("end_column"),
        )
        replacement_name = occurrence.get("replacement_name")
        if (
            not all(
                isinstance(value, int) and not isinstance(value, bool) and value >= 0
                for value in values
            )
            or not isinstance(replacement_name, str)
            or not replacement_name.isidentifier()
        ):
            return code
        coordinate = tuple(int(value) for value in values)
        call = calls.get(coordinate)
        if (
            call is None
            or not isinstance(call.func, ast.Name)
            or call.func.id != "Path"
        ):
            return code
        replacements.append(
            (
                absolute_offset(coordinate[0], coordinate[1]),
                absolute_offset(coordinate[2], coordinate[3]),
                replacement_name,
            )
        )
    repaired = code
    for start, end, replacement in sorted(replacements, reverse=True):
        repaired = repaired[:start] + replacement + repaired[end:]
    try:
        ast.parse(repaired)
    except SyntaxError:
        return code
    return repaired


__all__ = [
    "patch_resolved_input_cohort_env_shadow",
    "patch_resolved_input_relative_path_root",
]
