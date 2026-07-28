"""Coordinate-bound repair for confidence-interval method metadata."""

from __future__ import annotations

import ast
from typing import Sequence

from ..schema import ValidationFinding


def _line_offsets(source: str) -> list[int]:
    offsets = [0]
    for line in source.splitlines(keepends=True):
        offsets.append(offsets[-1] + len(line))
    return offsets


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
    offsets = _line_offsets(code)
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
        start = offsets[node.lineno - 1] + node.col_offset
        end = offsets[node.end_lineno - 1] + node.end_col_offset
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


__all__ = ["patch_statsmodels_interval_method_label"]
