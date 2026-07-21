"""Repair an exact binary sample guard after authored feasibility proof."""

from __future__ import annotations

import ast
from typing import Sequence

from ..schema import ValidationFinding


def patch_binary_domain_before_authored_feasibility(
    code: str,
    *,
    repair_findings: Sequence[ValidationFinding],
) -> str:
    """Let singleton observed domains reach an existing two-group fit gate."""

    matching = [
        finding
        for finding in repair_findings
        if finding.validator == "mechanical_code_preflight"
        and finding.severity == "error"
        and (finding.detail or {}).get("reason")
        == "binary_domain_preempts_authored_feasibility"
    ]
    if len(matching) != 1:
        return code
    detail = matching[0].detail or {}
    function_name = detail.get("function")
    levels_name = detail.get("levels_name")
    line = detail.get("line")
    if not (
        isinstance(function_name, str)
        and function_name.isidentifier()
        and isinstance(levels_name, str)
        and levels_name.isidentifier()
        and isinstance(line, int)
        and not isinstance(line, bool)
    ):
        return code
    try:
        tree = ast.parse(code)
    except SyntaxError:
        return code
    matches: list[ast.Compare] = []
    for function in ast.walk(tree):
        if not (
            isinstance(function, (ast.FunctionDef, ast.AsyncFunctionDef))
            and function.name == function_name
        ):
            continue
        matches.extend(
            node
            for node in ast.walk(function)
            if isinstance(node, ast.Compare)
            and int(node.lineno) == line
            and isinstance(node.left, ast.Name)
            and node.left.id == levels_name
            and len(node.ops) == 1
            and isinstance(node.ops[0], ast.NotEq)
            and len(node.comparators) == 1
            and isinstance(node.comparators[0], (ast.List, ast.Tuple))
            and [
                item.value
                for item in node.comparators[0].elts
                if isinstance(item, ast.Constant) and not isinstance(item.value, bool)
            ]
            == [0, 1]
        )
    if len(matches) != 1 or matches[0].end_lineno is None:
        return code
    comparison = matches[0]
    lines = code.splitlines(keepends=True)
    line_starts: list[int] = []
    offset = 0
    for source_line in lines:
        line_starts.append(offset)
        offset += len(source_line)

    def absolute_offset(lineno: int, utf8_col: int) -> int:
        source_line = lines[lineno - 1]
        char_col = len(source_line.encode("utf-8")[:utf8_col].decode("utf-8"))
        return line_starts[lineno - 1] + char_col

    start = absolute_offset(int(comparison.lineno), int(comparison.col_offset))
    end = absolute_offset(int(comparison.end_lineno), int(comparison.end_col_offset))
    replacement = f"not set({levels_name}).issubset({{0, 1}})"
    repaired = code[:start] + replacement + code[end:]
    try:
        ast.parse(repaired)
    except SyntaxError:
        return code
    return repaired


__all__ = ["patch_binary_domain_before_authored_feasibility"]
