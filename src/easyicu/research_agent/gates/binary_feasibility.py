"""Detect an exact-domain guard that preempts authored feasibility handling."""

from __future__ import annotations

import ast

from ..schema import ValidationFinding


def _references(node: ast.AST, name: str) -> bool:
    return any(
        isinstance(child, ast.Name)
        and isinstance(child.ctx, ast.Load)
        and child.id == name
        for child in ast.walk(node)
    )


def _is_treatment_reference(
    node: ast.AST,
    *,
    name: str,
    materialized: set[tuple[str, str]],
) -> bool:
    if isinstance(node, ast.Name):
        return isinstance(node.ctx, ast.Load) and node.id == name
    if not isinstance(node, ast.Subscript):
        return False
    frame_name = None
    if isinstance(node.value, ast.Name):
        frame_name = node.value.id
    elif (
        isinstance(node.value, ast.Attribute)
        and node.value.attr in {"at", "loc"}
        and isinstance(node.value.value, ast.Name)
    ):
        frame_name = node.value.value.id
    columns = [
        child.value
        for child in ast.walk(node.slice)
        if isinstance(child, ast.Constant) and isinstance(child.value, str)
    ]
    return bool(
        frame_name is not None
        and any((frame_name, str(column)) in materialized for column in columns)
    )


def _references_treatment(
    node: ast.AST,
    *,
    name: str,
    materialized: set[tuple[str, str]],
) -> bool:
    return any(
        _is_treatment_reference(child, name=name, materialized=materialized)
        for child in ast.walk(node)
    )


def _one_minus_treatment(
    node: ast.AST,
    *,
    name: str,
    materialized: set[tuple[str, str]],
) -> bool:
    return any(
        isinstance(child, ast.BinOp)
        and isinstance(child.op, ast.Sub)
        and isinstance(child.left, ast.Constant)
        and child.left.value == 1
        and _is_treatment_reference(
            child.right,
            name=name,
            materialized=materialized,
        )
        for child in ast.walk(node)
    )


def _assigns_non_none(body: list[ast.stmt]) -> set[str]:
    return {
        child.targets[0].id
        for statement in body
        for child in ast.walk(statement)
        if isinstance(child, ast.Assign)
        and len(child.targets) == 1
        and isinstance(child.targets[0], ast.Name)
        and not (isinstance(child.value, ast.Constant) and child.value.value is None)
    }


def binary_feasibility_guard_findings(tree: ast.Module) -> list[ValidationFinding]:
    """Return one finding only when two-group feasibility is already authored."""

    guards: list[tuple[ast.FunctionDef | ast.AsyncFunctionDef, ast.Compare, str]] = []
    for function in (
        node
        for node in ast.walk(tree)
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
    ):
        for node in ast.walk(function):
            if not (
                isinstance(node, ast.If)
                and isinstance(node.test, ast.Compare)
                and len(node.test.ops) == 1
                and isinstance(node.test.ops[0], ast.NotEq)
                and isinstance(node.test.left, ast.Name)
                and len(node.test.comparators) == 1
                and isinstance(node.test.comparators[0], (ast.List, ast.Tuple))
                and [
                    item.value
                    for item in node.test.comparators[0].elts
                    if isinstance(item, ast.Constant)
                    and not isinstance(item.value, bool)
                ]
                == [0, 1]
                and any(isinstance(child, ast.Raise) for child in node.body)
            ):
                continue
            levels_name = node.test.left.id
            assigned_from_observed_levels = any(
                isinstance(candidate, ast.Assign)
                and len(candidate.targets) == 1
                and isinstance(candidate.targets[0], ast.Name)
                and candidate.targets[0].id == levels_name
                and isinstance(candidate.value, ast.Call)
                and any(
                    isinstance(child, ast.Attribute) and child.attr == "unique"
                    for child in ast.walk(candidate.value)
                )
                for candidate in ast.walk(function)
            )
            if assigned_from_observed_levels:
                guards.append((function, node.test, levels_name))
    if len(guards) != 1:
        return []
    function, comparison, levels_name = guards[0]
    calls = [
        (node.targets[0].id, node)
        for node in ast.walk(tree)
        if isinstance(node, ast.Assign)
        and len(node.targets) == 1
        and isinstance(node.targets[0], ast.Name)
        and isinstance(node.value, ast.Call)
        and isinstance(node.value.func, ast.Name)
        and node.value.func.id == function.name
        and int(node.lineno) > int(function.end_lineno or function.lineno)
    ]
    proven_calls: list[tuple[str, str]] = []
    for treatment_name, call in calls:
        materialized = {
            (node.targets[0].value.id, str(node.targets[0].slice.value))
            for node in ast.walk(tree)
            if isinstance(node, ast.Assign)
            and len(node.targets) == 1
            and isinstance(node.targets[0], ast.Subscript)
            and isinstance(node.targets[0].value, ast.Name)
            and isinstance(node.targets[0].slice, ast.Constant)
            and isinstance(node.targets[0].slice.value, str)
            and isinstance(node.value, ast.Name)
            and node.value.id == treatment_name
        }
        treated_notes: set[str] = set()
        untreated_notes: set[str] = set()
        for node in ast.walk(tree):
            if not isinstance(node, ast.If) or int(node.lineno) <= int(call.lineno):
                continue
            if not _references_treatment(
                node.test,
                name=treatment_name,
                materialized=materialized,
            ):
                continue
            assigned = _assigns_non_none(node.body)
            if _one_minus_treatment(
                node.test,
                name=treatment_name,
                materialized=materialized,
            ):
                untreated_notes.update(assigned)
            elif any(
                isinstance(child, ast.Call)
                and isinstance(child.func, ast.Attribute)
                and child.func.attr == "sum"
                and _references_treatment(
                    child.func.value,
                    name=treatment_name,
                    materialized=materialized,
                )
                for child in ast.walk(node.test)
            ):
                treated_notes.update(assigned)
        for feasibility_name in treated_notes & untreated_notes:
            proven_calls.append((treatment_name, feasibility_name))
    if len(proven_calls) != 1:
        return []
    treatment_name, feasibility_name = proven_calls[0]
    fit_gate = any(
        isinstance(node, ast.If)
        and isinstance(node.test, ast.Compare)
        and isinstance(node.test.left, ast.Name)
        and node.test.left.id == feasibility_name
        and len(node.test.ops) == 1
        and isinstance(node.test.ops[0], ast.Is)
        and len(node.test.comparators) == 1
        and isinstance(node.test.comparators[0], ast.Constant)
        and node.test.comparators[0].value is None
        and any(
            isinstance(child, ast.Call)
            and isinstance(child.func, ast.Attribute)
            and child.func.attr == "fit"
            for statement in node.body
            for child in ast.walk(statement)
        )
        for node in ast.walk(tree)
    )
    if not fit_gate:
        return []
    return [
        ValidationFinding(
            validator="mechanical_code_preflight",
            severity="error",
            message=(
                "The binary type guard requires both sample levels before the "
                "script's own treated/untreated feasibility gate can run."
            ),
            detail={
                "reason": "binary_domain_preempts_authored_feasibility",
                "function": function.name,
                "line": int(comparison.lineno),
                "levels_name": levels_name,
                "treatment_name": treatment_name,
                "feasibility_name": feasibility_name,
            },
        )
    ]


__all__ = ["binary_feasibility_guard_findings"]
