"""Repair a proved invalid double subscript on a literal mapping."""

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


__all__ = ["patch_literal_mapping_access"]
