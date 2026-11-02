"""Coordinate-bound AST repairs for diagnosed mechanical code errors."""

from __future__ import annotations

import ast
from collections.abc import Collection
from typing import Mapping


def _absolute_offset(code: str, line: int, byte_column: int) -> int:
    """Translate Python AST's UTF-8 byte column into a string offset."""

    lines = code.splitlines(keepends=True)
    prefix = "".join(lines[: line - 1])
    line_text = lines[line - 1]
    encoded_prefix = line_text.encode("utf-8")[:byte_column]
    return len(prefix) + len(encoded_prefix.decode("utf-8"))



def patch_literal_mapping_access(
    code: str,
    *,
    coordinates: Collection[Mapping[str, object]],
) -> str:
    """Replace ``mapping[integer][known_key]`` with direct known-key access."""

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
    for node in ast.walk(tree):
        if not (
            isinstance(node, ast.Subscript)
            and isinstance(node.slice, ast.Constant)
            and isinstance(node.slice.value, str)
            and isinstance(node.value, ast.Subscript)
            and isinstance(node.value.value, ast.Name)
            and isinstance(node.value.slice, ast.Constant)
            and isinstance(node.value.slice.value, int)
            and node.end_lineno is not None
            and node.end_col_offset is not None
        ):
            continue
        coordinate = (
            int(node.lineno),
            node.value.value.id,
            str(node.slice.value),
        )
        if coordinate not in wanted:
            continue
        replacements.append(
            (
                _absolute_offset(code, node.lineno, node.col_offset),
                _absolute_offset(code, node.end_lineno, node.end_col_offset),
                f"{coordinate[1]}[{coordinate[2]!r}]",
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



def patch_pandas_numeric_container(
    code: str,
    *,
    finding_lines: Collection[int],
) -> str:
    """Wrap only diagnosed ``pd.to_numeric`` assignment results as Series."""

    try:
        tree = ast.parse(code)
    except SyntaxError:
        return code
    lines = {
        int(line)
        for line in finding_lines
        if isinstance(line, int) and not isinstance(line, bool) and line > 0
    }
    if not lines:
        return code

    replacements: list[tuple[int, int, str]] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Assign):
            value = node.value
        elif isinstance(node, ast.AnnAssign):
            value = node.value
        else:
            continue
        if (
            value is None
            or int(getattr(node, "lineno", -1)) not in lines
            or not isinstance(value, ast.Call)
            or not isinstance(value.func, ast.Attribute)
            or value.func.attr != "to_numeric"
            or not isinstance(value.func.value, ast.Name)
            or value.func.value.id not in {"pd", "pandas"}
            or value.end_lineno is None
            or value.end_col_offset is None
        ):
            continue
        source = ast.get_source_segment(code, value)
        if not source:
            continue
        alias = value.func.value.id
        replacements.append(
            (
                _absolute_offset(code, value.lineno, value.col_offset),
                _absolute_offset(code, value.end_lineno, value.end_col_offset),
                f"{alias}.Series({source})",
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



__all__ = [
    "patch_figure_output_registration",
    "patch_literal_mapping_access",
    "patch_pandas_numeric_container",
]
