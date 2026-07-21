"""Fail-closed repair for one proven Pandas merge suffix collision."""

from __future__ import annotations

import ast
import re

_KEY_ERROR = re.compile(r"KeyError:\s*['\"](?P<name>[A-Za-z_]\w*)['\"]")
_SENTINELS = {
    "_easyicu_merge_source_v1",
    "_easyicu_merge_left_values_v1",
    "_easyicu_merge_right_values_v1",
}


def _right_projection(
    tree: ast.Module,
    *,
    right_name: str,
    key_name: str,
) -> str | None:
    matches: list[str] = []
    for node in ast.walk(tree):
        if not (
            isinstance(node, ast.Assign)
            and len(node.targets) == 1
            and isinstance(node.targets[0], ast.Name)
            and node.targets[0].id == right_name
            and isinstance(node.value, ast.Call)
            and isinstance(node.value.func, ast.Attribute)
            and node.value.func.attr == "copy"
            and not node.value.args
            and not node.value.keywords
            and isinstance(node.value.func.value, ast.Subscript)
            and isinstance(node.value.func.value.slice, ast.List)
        ):
            continue
        elements = node.value.func.value.slice.elts
        if (
            len(elements) == 2
            and isinstance(elements[0], ast.Constant)
            and elements[0].value == key_name
            and isinstance(elements[1], ast.Name)
        ):
            matches.append(elements[1].id)
    return matches[0] if len(matches) == 1 else None


def patch_pandas_merge_dynamic_column_collision(code: str, run_log: str) -> str:
    """Guard and remove a duplicate dynamic column only after exact equality."""

    if _KEY_ERROR.search(run_log or "") is None:
        return code
    try:
        tree = ast.parse(code)
    except SyntaxError:
        return code
    names = {node.id for node in ast.walk(tree) if isinstance(node, ast.Name)}
    if names & _SENTINELS:
        return code
    candidates: list[tuple[ast.Assign, ast.Name, str, str, str, str]] = []
    for node in ast.walk(tree):
        if not (
            isinstance(node, ast.Assign)
            and len(node.targets) == 1
            and isinstance(node.targets[0], ast.Name)
            and isinstance(node.value, ast.Call)
            and isinstance(node.value.func, ast.Attribute)
            and node.value.func.attr == "merge"
            and isinstance(node.value.func.value, ast.Name)
            and len(node.value.args) == 1
            and isinstance(node.value.args[0], ast.Name)
        ):
            continue
        on_values = [
            keyword.value.value
            for keyword in node.value.keywords
            if keyword.arg == "on"
            and isinstance(keyword.value, ast.Constant)
            and isinstance(keyword.value.value, str)
        ]
        if len(on_values) != 1 or not str(on_values[0]).isidentifier():
            continue
        left_name = node.value.func.value.id
        right_name = node.value.args[0].id
        result_name = node.targets[0].id
        key_name = str(on_values[0])
        column_name = _right_projection(
            tree,
            right_name=right_name,
            key_name=key_name,
        )
        if column_name is None:
            continue
        dynamic_reads = [
            subscript
            for subscript in ast.walk(tree)
            if isinstance(subscript, ast.Subscript)
            and isinstance(subscript.value, ast.Name)
            and subscript.value.id == result_name
            and isinstance(subscript.slice, ast.Name)
            and subscript.slice.id == column_name
            and int(subscript.lineno) > int(node.lineno)
        ]
        if not dynamic_reads:
            continue
        candidates.append(
            (node, node.value.func.value, left_name, right_name, key_name, column_name)
        )
    if len(candidates) != 1:
        return code
    statement, receiver, left_name, right_name, key_name, column_name = candidates[0]
    if statement.end_lineno is None or receiver.end_lineno is None:
        return code
    lines = code.splitlines(keepends=True)
    source_line = lines[int(receiver.lineno) - 1]
    receiver_start = len(source_line.encode("utf-8")[: receiver.col_offset].decode())
    receiver_end = len(source_line.encode("utf-8")[: receiver.end_col_offset].decode())
    if source_line[receiver_start:receiver_end] != left_name:
        return code
    lines[int(receiver.lineno) - 1] = (
        source_line[:receiver_start]
        + "_easyicu_merge_source_v1"
        + source_line[receiver_end:]
    )
    indent = " " * int(statement.col_offset)
    guard = (
        f"{indent}_easyicu_merge_source_v1 = {left_name}\n"
        f"{indent}if {column_name} in {left_name}.columns and "
        f"{column_name} in {right_name}.columns:\n"
        f"{indent}    _easyicu_merge_left_values_v1 = "
        f"{left_name}.set_index({key_name!r})[{column_name}].sort_index()\n"
        f"{indent}    _easyicu_merge_right_values_v1 = "
        f"{right_name}.set_index({key_name!r})[{column_name}].sort_index()\n"
        f"{indent}    if not _easyicu_merge_left_values_v1.equals("
        "_easyicu_merge_right_values_v1):\n"
        f"{indent}        raise RuntimeError("
        "'Duplicate merge column disagrees across typed inputs')\n"
        f"{indent}    _easyicu_merge_source_v1 = "
        f"{left_name}.drop(columns=[{column_name}])\n"
    )
    insertion = int(statement.lineno) - 1
    repaired = "".join(lines[:insertion] + [guard] + lines[insertion:])
    try:
        ast.parse(repaired)
    except SyntaxError:
        return code
    return repaired


__all__ = ["patch_pandas_merge_dynamic_column_collision"]
