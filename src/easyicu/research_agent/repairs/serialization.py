"""Narrow deterministic repairs for generated JSON serialization hooks."""

from __future__ import annotations

import ast
from typing import Optional


def _source_offsets(code: str) -> list[int]:
    offsets = []
    cursor = 0
    for line in code.splitlines(keepends=True):
        offsets.append(cursor)
        cursor += len(line)
    return offsets


def _node_span(code: str, node: ast.AST) -> tuple[int, int] | None:
    if not all(
        hasattr(node, field)
        for field in ("lineno", "col_offset", "end_lineno", "end_col_offset")
    ):
        return None
    offsets = _source_offsets(code)
    try:
        start = offsets[int(node.lineno) - 1] + int(node.col_offset)
        end = offsets[int(node.end_lineno) - 1] + int(node.end_col_offset)
    except (IndexError, TypeError, ValueError):
        return None
    return start, end


def _values_container_name(node: ast.AST) -> str | None:
    """Return ``mapping`` for ``mapping.values()`` or ``sorted(...values())``."""

    candidate = node
    if (
        isinstance(candidate, ast.Call)
        and isinstance(candidate.func, ast.Name)
        and candidate.func.id == "sorted"
        and len(candidate.args) == 1
    ):
        candidate = candidate.args[0]
    if not (
        isinstance(candidate, ast.Call)
        and isinstance(candidate.func, ast.Attribute)
        and candidate.func.attr == "values"
        and not candidate.args
        and isinstance(candidate.func.value, ast.Name)
    ):
        return None
    return candidate.func.value.id


def _patch_sklearn_runtime_object_diagnostics(code: str) -> Optional[str]:
    """Keep fitted runtime objects out of a JSON diagnostic projection.

    Generated clustering scripts sometimes retain a fitted estimator and its
    assignment vector in an in-memory fit registry, then append that entire
    registry row to ``step_summary`` diagnostics.  The objects are required for
    execution but are not diagnostic JSON fields.  Patch only the uniquely
    proven registry-values -> append -> summary-dict shape and preserve the
    numeric/scalar diagnostics unchanged.
    """

    try:
        tree = ast.parse(code)
    except SyntaxError:
        return None

    registry_keys: dict[str, set[str]] = {}
    for node in ast.walk(tree):
        if not (
            isinstance(node, ast.Assign)
            and len(node.targets) == 1
            and isinstance(node.targets[0], ast.Subscript)
            and isinstance(node.targets[0].value, ast.Name)
            and isinstance(node.value, ast.Dict)
        ):
            continue
        keys = {
            str(key.value)
            for key in node.value.keys
            if isinstance(key, ast.Constant) and isinstance(key.value, str)
        }
        if {"model", "labels"}.issubset(keys):
            registry_keys[node.targets[0].value.id] = keys

    candidates: list[tuple[ast.Name, str]] = []
    for node in ast.walk(tree):
        if not (
            isinstance(node, ast.For)
            and isinstance(node.target, ast.Name)
            and (container := _values_container_name(node.iter)) in registry_keys
        ):
            continue
        loop_name = node.target.id
        for child in ast.walk(node):
            if not (
                isinstance(child, ast.Call)
                and isinstance(child.func, ast.Attribute)
                and child.func.attr == "append"
                and isinstance(child.func.value, ast.Name)
                and len(child.args) == 1
                and isinstance(child.args[0], ast.Name)
                and child.args[0].id == loop_name
            ):
                continue
            diagnostics_name = child.func.value.id
            summary_bound = any(
                isinstance(value, ast.Name) and value.id == diagnostics_name
                for mapping in ast.walk(tree)
                if isinstance(mapping, ast.Dict)
                for value in mapping.values
            )
            if summary_bound:
                candidates.append((child.args[0], loop_name))

    if len(candidates) != 1:
        return None
    argument, loop_name = candidates[0]
    span = _node_span(code, argument)
    if span is None:
        return None
    replacement = (
        f"{{key: value for key, value in {loop_name}.items() "
        'if key not in {"model", "labels"}}'
    )
    repaired = code[: span[0]] + replacement + code[span[1] :]
    try:
        ast.parse(repaired)
    except SyntaxError:
        return None
    return repaired


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


def serialization_runner_repair(
    *,
    code: str,
    run_log: str,
    previous_repair: str | None,
) -> tuple[str, str] | None:
    """Return one narrowly proven serialization repair for a runtime failure."""

    validation_repair = validation_finding_json_runner_repair(
        code=code,
        run_log=run_log,
        previous_repair=previous_repair,
    )
    if validation_repair is not None:
        return validation_repair

    repair_id = "sklearn_runtime_object_diagnostics_v1"
    lowered = str(run_log or "").lower()
    if repair_id == previous_repair or not (
        "sklearn." in lowered
        and ("unsupported json value" in lowered or "not json serializable" in lowered)
    ):
        return None
    repaired = _patch_sklearn_runtime_object_diagnostics(code)
    return (repair_id, repaired) if repaired is not None else None


__all__ = ["serialization_runner_repair", "validation_finding_json_runner_repair"]
