"""Narrow deterministic repairs for categorical runtime scaffolds."""

from __future__ import annotations

import ast


def _offsets(code: str) -> tuple[list[str], list[int]]:
    lines = code.splitlines(keepends=True)
    starts: list[int] = []
    cursor = 0
    for line in lines:
        starts.append(cursor)
        cursor += len(line)
    return lines, starts


def patch_categorical_declared_order_check(code: str, run_log: str) -> str:
    """Read declared categorical levels instead of first-observation order.

    ``Series.unique()`` preserves row encounter order, not categorical order.
    For one exact ``list(series.astype(...).dropna().unique())`` assignment
    whose value is immediately compared with a literal ordered-level list,
    read ``series.cat.categories`` instead.  The cut points, labels, rows,
    model, and expected ordered levels are unchanged.
    """

    lowered = (run_log or "").lower()
    if "did not produce" not in lowered or "ordered levels" not in lowered:
        return code
    try:
        tree = ast.parse(code)
    except SyntaxError:
        return code

    assignments: dict[str, ast.Assign] = {}
    for node in tree.body:
        if (
            isinstance(node, ast.Assign)
            and len(node.targets) == 1
            and isinstance(node.targets[0], ast.Name)
        ):
            assignments[node.targets[0].id] = node

    candidates: list[tuple[ast.Assign, ast.AST]] = []
    for node in tree.body:
        if not (
            isinstance(node, ast.If)
            and isinstance(node.test, ast.Compare)
            and len(node.test.ops) == 1
            and isinstance(node.test.ops[0], ast.NotEq)
            and len(node.test.comparators) == 1
            and isinstance(node.test.left, ast.Name)
            and isinstance(node.test.comparators[0], (ast.List, ast.Tuple))
            and node.test.comparators[0].elts
            and all(
                isinstance(value, ast.Constant)
                and isinstance(value.value, (str, int, float))
                for value in node.test.comparators[0].elts
            )
        ):
            continue
        assignment = assignments.get(node.test.left.id)
        if assignment is None:
            continue
        value = assignment.value
        if not (
            isinstance(value, ast.Call)
            and isinstance(value.func, ast.Name)
            and value.func.id == "list"
            and len(value.args) == 1
            and not value.keywords
        ):
            continue
        unique_call = value.args[0]
        if not (
            isinstance(unique_call, ast.Call)
            and isinstance(unique_call.func, ast.Attribute)
            and unique_call.func.attr == "unique"
            and not unique_call.args
            and not unique_call.keywords
            and isinstance(unique_call.func.value, ast.Call)
            and isinstance(unique_call.func.value.func, ast.Attribute)
            and unique_call.func.value.func.attr == "dropna"
            and not unique_call.func.value.args
            and not unique_call.func.value.keywords
            and isinstance(unique_call.func.value.func.value, ast.Call)
            and isinstance(unique_call.func.value.func.value.func, ast.Attribute)
            and unique_call.func.value.func.value.func.attr == "astype"
        ):
            continue
        astype_call = unique_call.func.value.func.value
        if not (
            len(astype_call.args) == 1
            and isinstance(astype_call.args[0], ast.Constant)
            and astype_call.args[0].value == "string"
            and not astype_call.keywords
        ):
            continue
        series = astype_call.func.value
        candidates.append((assignment, series))
    if len(candidates) != 1:
        return code

    assignment, series = candidates[0]
    series_source = ast.get_source_segment(code, series)
    coordinates = (
        assignment.value.lineno,
        assignment.value.col_offset,
        assignment.value.end_lineno,
        assignment.value.end_col_offset,
    )
    if not series_source or not all(isinstance(value, int) for value in coordinates):
        return code
    lineno, col, end_lineno, end_col = coordinates
    lines, starts = _offsets(code)
    start = starts[lineno - 1] + col
    end = starts[end_lineno - 1] + end_col
    replacement = f"list(map(str, {series_source}.cat.categories))"
    repaired = code[:start] + replacement + code[end:]
    try:
        ast.parse(repaired)
    except SyntaxError:
        return code
    return repaired


__all__ = ["patch_categorical_declared_order_check"]
