"""Repair a figure product registered to a stem instead of an export file."""

from __future__ import annotations

import ast
from collections.abc import Collection
from typing import Mapping


def _absolute_offset(code: str, line: int, byte_column: int) -> int:
    lines = code.splitlines(keepends=True)
    prefix = "".join(lines[: line - 1])
    line_text = lines[line - 1]
    encoded_prefix = line_text.encode("utf-8")[:byte_column]
    return len(prefix) + len(encoded_prefix.decode("utf-8"))


def patch_figure_output_registration(
    code: str,
    *,
    coordinates: Collection[Mapping[str, object]],
) -> str:
    """Point diagnosed figure-product entries at their primary export file."""

    wanted = {
        (int(item["line"]), str(item["name"]), str(item["field"]))
        for item in coordinates
        if isinstance(item.get("line"), int)
        and not isinstance(item.get("line"), bool)
        and int(item["line"]) > 0
        and isinstance(item.get("name"), str)
        and isinstance(item.get("field"), str)
    }
    if not wanted:
        return code
    try:
        tree = ast.parse(code)
    except SyntaxError:
        return code
    replacements: list[tuple[int, int, str]] = []
    for summary in [node for node in ast.walk(tree) if isinstance(node, ast.Dict)]:
        fields = {
            str(key.value): value
            for key, value in zip(summary.keys, summary.values)
            if isinstance(key, ast.Constant) and isinstance(key.value, str)
        }
        output_map = fields.get("output_files")
        figure_files = fields.get("figure_files")
        if not isinstance(output_map, ast.Dict) or not isinstance(figure_files, ast.Name):
            continue
        for key, value in zip(output_map.keys, output_map.values):
            if not (
                isinstance(key, ast.Constant)
                and isinstance(key.value, str)
                and isinstance(value, ast.Name)
                and value.end_lineno is not None
                and value.end_col_offset is not None
            ):
                continue
            coordinate = (int(value.lineno), figure_files.id, str(key.value))
            if coordinate not in wanted:
                continue
            replacements.append(
                (
                    _absolute_offset(code, value.lineno, value.col_offset),
                    _absolute_offset(code, value.end_lineno, value.end_col_offset),
                    f"{figure_files.id}[0]",
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


__all__ = ["patch_figure_output_registration"]
