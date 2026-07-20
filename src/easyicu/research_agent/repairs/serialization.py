"""Narrow deterministic repairs for generated JSON serialization hooks."""

from __future__ import annotations

import ast
from typing import Optional


def _patch_validation_finding_json_default(code: str) -> Optional[str]:
    """Teach an existing JSON default hook to serialize host findings."""

    try:
        tree = ast.parse(code)
    except SyntaxError:
        return None
    functions = [
        node
        for node in ast.walk(tree)
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
        and node.name == "json_default"
        and len(node.args.args) == 1
        and not node.args.posonlyargs
        and node.args.vararg is None
        and node.args.kwarg is None
    ]
    if len(functions) != 1:
        return None
    function = functions[0]
    if any(
        isinstance(node, ast.Name)
        and node.id == "hasattr"
        and isinstance(node.ctx, (ast.Store, ast.Del))
        for node in ast.walk(tree)
    ):
        return None
    if any(
        isinstance(node, ast.Attribute) and node.attr == "model_dump"
        for node in ast.walk(function)
    ):
        return None
    json_default_is_used = any(
        isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr in {"dump", "dumps"}
        and any(
            keyword.arg == "default"
            and isinstance(keyword.value, ast.Name)
            and keyword.value.id == "json_default"
            for keyword in node.keywords
        )
        for node in ast.walk(tree)
    )
    if not json_default_is_used or not function.body:
        return None

    body_index = int(
        isinstance(function.body[0], ast.Expr)
        and isinstance(function.body[0].value, ast.Constant)
        and isinstance(function.body[0].value.value, str)
    )
    lines = code.splitlines(keepends=True)
    line_starts: list[int] = []
    offset = 0
    for line in lines:
        line_starts.append(offset)
        offset += len(line)
    if body_index < len(function.body):
        target = function.body[body_index]
        insertion_offset = line_starts[int(target.lineno) - 1]
        indentation = " " * int(target.col_offset)
    else:
        last = function.body[-1]
        if last.end_lineno is None:
            return None
        insertion_offset = (
            line_starts[int(last.end_lineno)]
            if int(last.end_lineno) < len(line_starts)
            else len(code)
        )
        indentation = " " * (int(function.col_offset) + 4)
    value_name = function.args.args[0].arg
    insertion = (
        f'{indentation}if hasattr({value_name}, "model_dump"):\n'
        f"{indentation}    return {value_name}.model_dump()\n"
    )
    repaired = code[:insertion_offset] + insertion + code[insertion_offset:]
    try:
        ast.parse(repaired)
    except SyntaxError:
        return None
    return repaired


def validation_finding_json_runner_repair(
    *,
    code: str,
    run_log: str,
    previous_repair: str | None,
) -> tuple[str, str] | None:
    """Return the exact post-error serialization repair, if authorized by shape."""

    repair_id = "validation_finding_json_default_v1"
    if (
        repair_id == previous_repair
        or "object of type validationfinding is not json serializable"
        not in str(run_log or "").lower()
    ):
        return None
    repaired = _patch_validation_finding_json_default(code)
    return (repair_id, repaired) if repaired is not None else None


__all__ = ["validation_finding_json_runner_repair"]
