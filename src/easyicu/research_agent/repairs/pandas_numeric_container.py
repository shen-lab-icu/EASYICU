"""Science-neutral repair for pandas numeric container compatibility."""

from __future__ import annotations

import ast
from collections.abc import Collection


def _absolute_offset(code: str, line: int, byte_column: int) -> int:
    """Translate Python AST's UTF-8 byte column into a string offset."""

    lines = code.splitlines(keepends=True)
    prefix = "".join(lines[: line - 1])
    line_text = lines[line - 1]
    encoded_prefix = line_text.encode("utf-8")[:byte_column]
    return len(prefix) + len(encoded_prefix.decode("utf-8"))


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


__all__ = ["patch_pandas_numeric_container"]
