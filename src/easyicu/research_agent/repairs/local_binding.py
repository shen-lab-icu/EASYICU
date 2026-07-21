"""Narrow repair for one proven local read-before-assignment defect."""

from __future__ import annotations

import ast
from typing import Sequence

from ..schema import ValidationFinding


def _scope_body(tree: ast.Module, function_name: str) -> list[ast.stmt] | None:
    if function_name == "<module>":
        return tree.body
    functions = [
        node
        for node in ast.walk(tree)
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
        and node.name == function_name
    ]
    return functions[0].body if len(functions) == 1 else None


def _stored_names(node: ast.AST) -> set[str]:
    return {
        child.id
        for child in ast.walk(node)
        if isinstance(child, ast.Name) and isinstance(child.ctx, ast.Store)
    }


def patch_local_read_before_assignment_hoist(
    code: str,
    *,
    repair_findings: Sequence[ValidationFinding],
) -> str:
    """Move one existing simple assignment before its first proven read.

    No expression is synthesized.  The exact assignment is relocated only
    within one lexical statement list, and only when none of its dependencies
    are rebound between the destination and source coordinates.
    """

    matching = [
        finding
        for finding in repair_findings
        if finding.validator == "mechanical_code_preflight"
        and finding.severity == "error"
        and (finding.detail or {}).get("reason") == "local_read_before_assignment"
    ]
    if len(matching) != 1:
        return code
    detail = matching[0].detail or {}
    name = detail.get("name")
    function_name = detail.get("function")
    first_use_line = detail.get("first_use_line")
    first_assignment_line = detail.get("first_assignment_line")
    if not (
        isinstance(name, str)
        and name.isidentifier()
        and isinstance(function_name, str)
        and isinstance(first_use_line, int)
        and not isinstance(first_use_line, bool)
        and isinstance(first_assignment_line, int)
        and not isinstance(first_assignment_line, bool)
        and 0 < first_use_line < first_assignment_line
    ):
        return code
    try:
        tree = ast.parse(code)
    except SyntaxError:
        return code
    body = _scope_body(tree, function_name)
    if body is None:
        return code
    assignments = [
        statement
        for statement in body
        if isinstance(statement, ast.Assign)
        and int(statement.lineno) == first_assignment_line
        and len(statement.targets) == 1
        and isinstance(statement.targets[0], ast.Name)
        and statement.targets[0].id == name
        and isinstance(statement.value, ast.Call)
        and not any(
            isinstance(node, (ast.Await, ast.NamedExpr, ast.Yield, ast.YieldFrom))
            for node in ast.walk(statement.value)
        )
    ]
    if len(assignments) != 1:
        return code
    assignment = assignments[0]
    reads = [
        node
        for statement in body
        for node in ast.walk(statement)
        if isinstance(node, ast.Name)
        and isinstance(node.ctx, ast.Load)
        and node.id == name
    ]
    if not reads or min(int(node.lineno) for node in reads) != first_use_line:
        return code
    if (
        sum(
            1
            for statement in body
            for node in ast.walk(statement)
            if isinstance(node, ast.Name)
            and isinstance(node.ctx, ast.Store)
            and node.id == name
        )
        != 1
    ):
        return code
    destination_candidates = [
        statement
        for statement in body
        if int(statement.lineno)
        <= first_use_line
        <= int(statement.end_lineno or statement.lineno)
    ]
    if len(destination_candidates) != 1:
        return code
    destination = destination_candidates[0]
    try:
        destination_index = body.index(destination)
        assignment_index = body.index(assignment)
    except ValueError:
        return code
    if assignment_index <= destination_index:
        return code
    rhs_dependencies = {
        node.id
        for node in ast.walk(assignment.value)
        if isinstance(node, ast.Name) and isinstance(node.ctx, ast.Load)
    }
    if not rhs_dependencies or any(
        _stored_names(statement) & rhs_dependencies
        for statement in body[destination_index:assignment_index]
    ):
        return code
    if assignment.end_lineno is None:
        return code
    lines = code.splitlines(keepends=True)
    assignment_start = int(assignment.lineno) - 1
    assignment_end = int(assignment.end_lineno)
    destination_start = int(destination.lineno) - 1
    assignment_lines = lines[assignment_start:assignment_end]
    if not assignment_lines:
        return code
    without_assignment = lines[:assignment_start] + lines[assignment_end:]
    repaired = "".join(
        without_assignment[:destination_start]
        + assignment_lines
        + without_assignment[destination_start:]
    )
    try:
        ast.parse(repaired)
    except SyntaxError:
        return code
    return repaired


__all__ = ["patch_local_read_before_assignment_hoist"]
