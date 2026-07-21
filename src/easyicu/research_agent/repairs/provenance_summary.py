"""Narrow deterministic repairs for host provenance summary envelopes."""

from __future__ import annotations

import ast
from typing import Optional


def _lexical_scope(
    node: ast.AST,
    parents: dict[ast.AST, ast.AST],
) -> ast.AST:
    current = node
    while current in parents:
        current = parents[current]
        if isinstance(current, (ast.Module, ast.FunctionDef, ast.AsyncFunctionDef)):
            return current
    return current


def patch_direct_host_provenance_summary(code: str) -> str:
    """Wrap one direct host receipt in the required source/checks envelope.

    The candidate must import the exact host helper, bind its returned mapping
    once, and place that mapping directly in the unique step summary.  The
    transformation does not derive or alter any receipt value.
    """

    try:
        tree = ast.parse(code)
    except SyntaxError:
        return code
    exact_imports = [
        node
        for node in tree.body
        if isinstance(node, ast.ImportFrom)
        and node.level == 0
        and node.module == "easyicu.research_agent.methods.descriptive_inputs"
        and any(
            alias.name == "measurement_provenance_receipt" and alias.asname is None
            for alias in node.names
        )
    ]
    if len(exact_imports) != 1:
        return code
    parents = {
        child: parent
        for parent in ast.walk(tree)
        for child in ast.iter_child_nodes(parent)
    }
    summary_values: list[ast.Name] = []
    for node in ast.walk(tree):
        if not (
            isinstance(node, ast.Assign)
            and len(node.targets) == 1
            and isinstance(node.targets[0], ast.Name)
            and node.targets[0].id == "step_summary"
            and isinstance(node.value, ast.Dict)
        ):
            continue
        for key, value in zip(node.value.keys, node.value.values, strict=True):
            if (
                isinstance(key, ast.Constant)
                and key.value == "measurement_provenance_audit"
                and isinstance(value, ast.Name)
            ):
                summary_values.append(value)
    if len(summary_values) != 1:
        return code
    summary_value = summary_values[0]
    receipt_name = summary_value.id
    scope = _lexical_scope(summary_value, parents)
    assignments = [
        node
        for node in ast.walk(scope)
        if isinstance(node, ast.Assign)
        and _lexical_scope(node, parents) is scope
        and len(node.targets) == 1
        and isinstance(node.targets[0], ast.Name)
        and node.targets[0].id == receipt_name
        and isinstance(node.value, ast.Call)
        and isinstance(node.value.func, ast.Name)
        and node.value.func.id == "measurement_provenance_receipt"
        and len(node.value.args) == 1
        and not any(keyword.arg is None for keyword in node.value.keywords)
        and {keyword.arg for keyword in node.value.keywords}
        == {"measured_column", "count_column"}
    ]
    if len(assignments) != 1:
        return code
    loads = [
        node
        for node in ast.walk(scope)
        if isinstance(node, ast.Name)
        and isinstance(node.ctx, ast.Load)
        and node.id == receipt_name
        and _lexical_scope(node, parents) is scope
    ]

    def _allowed_none_guard(load: ast.Name) -> bool:
        current: Optional[ast.AST] = load
        compare: Optional[ast.Compare] = None
        while current in parents and current is not scope:
            current = parents[current]
            if isinstance(current, ast.Compare):
                compare = current
            if isinstance(current, ast.If):
                return bool(
                    compare is not None
                    and current.test is compare
                    and len(compare.ops) == 1
                    and isinstance(compare.ops[0], ast.Is)
                    and len(compare.comparators) == 1
                    and isinstance(compare.comparators[0], ast.Constant)
                    and compare.comparators[0].value is None
                    and current.body
                    and isinstance(current.body[0], ast.Raise)
                )
        return False

    if any(
        load is not summary_value and not _allowed_none_guard(load) for load in loads
    ):
        return code
    lines = code.splitlines(keepends=True)
    line_starts: list[int] = []
    offset = 0
    for line in lines:
        line_starts.append(offset)
        offset += len(line)

    def _absolute_offset(lineno: int, utf8_col: int) -> int:
        line = lines[lineno - 1]
        char_col = len(line.encode("utf-8")[:utf8_col].decode("utf-8"))
        return line_starts[lineno - 1] + char_col

    if summary_value.end_lineno is None or summary_value.end_col_offset is None:
        return code
    start = _absolute_offset(summary_value.lineno, summary_value.col_offset)
    end = _absolute_offset(summary_value.end_lineno, summary_value.end_col_offset)
    replacement = '{"source": "COHORT_PARQUET", "checks": [' + receipt_name + "]}"
    repaired = code[:start] + replacement + code[end:]
    try:
        ast.parse(repaired)
    except SyntaxError:
        return code
    return repaired


__all__ = ["patch_direct_host_provenance_summary"]
