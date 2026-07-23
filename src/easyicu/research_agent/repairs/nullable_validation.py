"""Fail-safe repair for unused nullable fields in generated figure validators."""

from __future__ import annotations

import ast
import re
from typing import Any, Mapping

_ERROR_RE = re.compile(
    r"^(?P<table>[A-Za-z_][A-Za-z0-9_]*)\."
    r"(?P<column>[A-Za-z_][A-Za-z0-9_]*) contains non-finite values$"
)


def _strings(node: ast.AST | None) -> list[str] | None:
    if not isinstance(node, (ast.List, ast.Tuple)):
        return None
    if any(
        not isinstance(item, ast.Constant) or not isinstance(item.value, str)
        for item in node.elts
    ):
        return None
    return [item.value for item in node.elts]


def _assignment(node: ast.AST) -> tuple[str, ast.AST] | None:
    if isinstance(node, ast.Assign) and len(node.targets) == 1:
        target, value = node.targets[0], node.value
    elif isinstance(node, ast.AnnAssign) and node.value is not None:
        target, value = node.target, node.value
    else:
        return None
    if not isinstance(target, ast.Name):
        return None
    return target.id, value


def _within(
    node: ast.AST,
    ancestor: ast.AST,
    parents: Mapping[ast.AST, ast.AST],
) -> bool:
    current = node
    while current in parents:
        current = parents[current]
        if current is ancestor:
            return True
    return False


def _validation_loop(
    tree: ast.AST,
    *,
    list_name: str,
    table_name: str,
) -> ast.For | None:
    matches: list[ast.For] = []
    for loop in ast.walk(tree):
        if (
            not isinstance(loop, ast.For)
            or not isinstance(loop.target, ast.Name)
            or not isinstance(loop.iter, ast.Name)
            or loop.iter.id != list_name
            or len(loop.body) != 1
            or not isinstance(loop.body[0], ast.Expr)
            or not isinstance(loop.body[0].value, ast.Call)
        ):
            continue
        call = loop.body[0].value
        if (
            not isinstance(call.func, ast.Name)
            or call.func.id != "validate_numeric_series"
            or len(call.args) != 2
            or not isinstance(call.args[0], ast.Subscript)
            or not isinstance(call.args[0].value, ast.Name)
            or call.args[0].value.id != table_name
            or not isinstance(call.args[0].slice, ast.Name)
            or call.args[0].slice.id != loop.target.id
            or not isinstance(call.args[1], ast.JoinedStr)
        ):
            continue
        label = call.args[1].values
        if (
            len(label) == 2
            and isinstance(label[0], ast.Constant)
            and label[0].value == f"{table_name}."
            and isinstance(label[1], ast.FormattedValue)
            and isinstance(label[1].value, ast.Name)
            and label[1].value.id == loop.target.id
        ):
            matches.append(loop)
    return matches[0] if len(matches) == 1 else None


def _replace_node(code: str, node: ast.AST, replacement: str) -> str | None:
    if not all(
        hasattr(node, attribute)
        for attribute in ("lineno", "col_offset", "end_lineno", "end_col_offset")
    ):
        return None
    lines = code.splitlines(keepends=True)

    def character_offset(line_number: int, byte_column: int) -> int:
        line = lines[line_number - 1]
        character_column = len(line.encode("utf-8")[:byte_column].decode("utf-8"))
        return sum(len(part) for part in lines[: line_number - 1]) + character_column

    start = character_offset(node.lineno, node.col_offset)
    end = character_offset(node.end_lineno, node.end_col_offset)
    return code[:start] + replacement + code[end:]


def patch_unused_nullable_numeric_validation(
    code: str,
    step_summary: Mapping[str, Any],
) -> tuple[str, str] | None:
    """Remove one proven-unused nullable field from a numeric validation loop.

    The source data and every value used by a calculation, plot, or claim stay
    untouched. The repair requires an exact script-authored failure, a literal
    required-schema declaration, and a narrow AST proof that the column's only
    other occurrence is the dedicated validation loop.
    """

    error = str(
        step_summary.get("error")
        or step_summary.get("error_message")
        or step_summary.get("note")
        or ""
    ).strip()
    match = _ERROR_RE.fullmatch(error)
    if match is None:
        return None
    table_name, column_name = match.group("table"), match.group("column")
    try:
        tree = ast.parse(code)
    except SyntaxError:
        return None

    parents = {
        child: parent
        for parent in ast.walk(tree)
        for child in ast.iter_child_nodes(parent)
    }
    assignments = [
        item for node in ast.walk(tree) if (item := _assignment(node)) is not None
    ]
    required_values = [
        value
        for name, value in assignments
        if name.startswith("required_")
        and name.endswith("_columns")
        and (items := _strings(value)) is not None
        and column_name in items
    ]
    if not required_values:
        return None

    for list_name, list_value in assignments:
        columns = _strings(list_value)
        if columns is None or column_name not in columns or len(columns) < 2:
            continue
        loop = _validation_loop(tree, list_name=list_name, table_name=table_name)
        if loop is None:
            continue
        list_loads = [
            node
            for node in ast.walk(tree)
            if isinstance(node, ast.Name)
            and isinstance(node.ctx, ast.Load)
            and node.id == list_name
        ]
        if list_loads != [loop.iter]:
            continue

        ignored = [list_value, *required_values]
        if any(
            isinstance(node, ast.Constant)
            and node.value == column_name
            and not any(
                node is root or _within(node, root, parents) for root in ignored
            )
            for node in ast.walk(tree)
        ):
            continue

        row_aliases = {table_name}
        for alias, value in assignments:
            if (
                isinstance(value, ast.Subscript)
                and isinstance(value.value, ast.Attribute)
                and value.value.attr in {"iloc", "loc"}
                and isinstance(value.value.value, ast.Name)
                and value.value.value.id == table_name
            ):
                row_aliases.add(alias)
        if any(
            isinstance(node, ast.Subscript)
            and not _within(node, loop, parents)
            and isinstance(node.value, ast.Name)
            and node.value.id in row_aliases
            and not isinstance(node.slice, ast.Constant)
            for node in ast.walk(tree)
        ):
            continue

        remaining = [column for column in columns if column != column_name]
        replacement = ast.unparse(
            ast.List(
                elts=[ast.Constant(value=column) for column in remaining],
                ctx=ast.Load(),
            )
        )
        repaired = _replace_node(code, list_value, replacement)
        if repaired is None:
            continue
        try:
            ast.parse(repaired)
        except SyntaxError:
            continue
        repair_name = "unused_nullable_numeric_validation_v1"
        return repair_name, repaired
    return None


__all__ = ["patch_unused_nullable_numeric_validation"]
