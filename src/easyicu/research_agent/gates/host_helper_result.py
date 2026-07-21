"""Contracts for consuming stable host-helper result objects."""

from __future__ import annotations

import ast
from typing import Optional

from ..schema import AnalysisStep, ValidationFinding

_SCOPES = (ast.Module, ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)
_HELPER_MODULE = "easyicu.research_agent.methods.descriptive_inputs"


def _call_name(node: ast.AST) -> str:
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        prefix = _call_name(node.value)
        return f"{prefix}.{node.attr}" if prefix else node.attr
    return ""


def _scope_id(
    node: ast.AST,
    *,
    parents: dict[int, ast.AST],
    tree: ast.Module,
) -> int:
    current: Optional[ast.AST] = node
    while current is not None:
        current = parents.get(id(current))
        if isinstance(current, _SCOPES):
            return id(current)
    return id(tree)


def closed_counts_table_index_findings(
    tree: ast.Module,
) -> list[ValidationFinding]:
    """Reject a proven closed-counts RangeIndex used as category levels.

    The helper returns an explicit ``level`` column and an ordinary RangeIndex.
    Authority is claimed only when the exact helper import, unique direct result
    assignment, ``.table`` projection, and a literal declared-level comparison
    are all statically linked in one lexical scope. Generic DataFrame index use
    remains outside this contract.
    """

    parents = {
        id(child): parent
        for parent in ast.walk(tree)
        for child in ast.iter_child_nodes(parent)
    }
    direct_names: set[str] = set()
    module_names: set[str] = set()
    for node in tree.body:
        if (
            isinstance(node, ast.ImportFrom)
            and node.level == 0
            and node.module == _HELPER_MODULE
        ):
            direct_names.update(
                alias.asname or alias.name
                for alias in node.names
                if alias.name == "closed_categorical_counts"
            )
        elif isinstance(node, ast.Import):
            module_names.update(
                alias.asname or alias.name
                for alias in node.names
                if alias.name == _HELPER_MODULE
            )
    if not direct_names and not module_names:
        return []

    def _is_exact_helper_call(node: ast.AST, scope_id: int) -> bool:
        if not isinstance(node, ast.Call):
            return False
        call_name = _call_name(node.func)
        roots = {name for name in direct_names if call_name == name} | {
            name
            for name in module_names
            if call_name == f"{name}.closed_categorical_counts"
        }
        if len(roots) != 1:
            return False
        root = next(iter(roots))
        for candidate in ast.walk(tree):
            if _scope_id(candidate, parents=parents, tree=tree) != scope_id:
                continue
            if isinstance(candidate, ast.arg) and candidate.arg == root:
                return False
            if (
                isinstance(candidate, ast.Name)
                and isinstance(candidate.ctx, (ast.Store, ast.Del))
                and candidate.id == root
            ):
                return False
        return True

    result_assignments: dict[tuple[int, str], list[ast.Assign]] = {}
    table_assignments: dict[tuple[int, str], list[tuple[ast.Assign, str]]] = {}
    for node in ast.walk(tree):
        if not isinstance(node, ast.Assign) or len(node.targets) != 1:
            continue
        target = node.targets[0]
        if not isinstance(target, ast.Name):
            continue
        scope_id = _scope_id(node, parents=parents, tree=tree)
        if _is_exact_helper_call(node.value, scope_id):
            result_assignments.setdefault((scope_id, target.id), []).append(node)
        elif (
            isinstance(node.value, ast.Attribute)
            and node.value.attr == "table"
            and isinstance(node.value.value, ast.Name)
        ):
            table_assignments.setdefault((scope_id, target.id), []).append(
                (node, node.value.value.id)
            )

    findings: list[ValidationFinding] = []
    for node in ast.walk(tree):
        if not (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Name)
            and node.func.id == "list"
            and len(node.args) == 1
            and not node.keywords
            and isinstance(node.args[0], ast.Attribute)
            and node.args[0].attr == "index"
        ):
            continue
        scope_id = _scope_id(node, parents=parents, tree=tree)
        assignment = parents.get(id(node))
        if not (
            isinstance(assignment, ast.Assign)
            and len(assignment.targets) == 1
            and isinstance(assignment.targets[0], ast.Name)
        ):
            continue
        levels_name = assignment.targets[0].id
        compared = False
        for comparison in ast.walk(tree):
            if not (
                isinstance(comparison, ast.Compare)
                and comparison.lineno > node.lineno
                and _scope_id(comparison, parents=parents, tree=tree) == scope_id
                and len(comparison.ops) == 1
                and len(comparison.comparators) == 1
            ):
                continue
            operands = [comparison.left, comparison.comparators[0]]
            has_set_of_result = any(
                isinstance(operand, ast.Call)
                and isinstance(operand.func, ast.Name)
                and operand.func.id == "set"
                and len(operand.args) == 1
                and not operand.keywords
                and isinstance(operand.args[0], ast.Name)
                and operand.args[0].id == levels_name
                for operand in operands
            )
            has_literal_levels = any(
                isinstance(operand, (ast.Set, ast.List, ast.Tuple))
                and all(isinstance(item, ast.Constant) for item in operand.elts)
                for operand in operands
            )
            if has_set_of_result and has_literal_levels:
                compared = True
                break
        if not compared:
            continue
        table_expression = node.args[0].value
        result_name = ""
        table_name = ""
        if isinstance(table_expression, ast.Name):
            table_name = table_expression.id
            assignments = table_assignments.get((scope_id, table_name), [])
            if len(assignments) == 1 and assignments[0][0].lineno < node.lineno:
                result_name = assignments[0][1]
        elif (
            isinstance(table_expression, ast.Attribute)
            and table_expression.attr == "table"
            and isinstance(table_expression.value, ast.Name)
        ):
            result_name = table_expression.value.id
        result_sites = result_assignments.get((scope_id, result_name), [])
        if len(result_sites) != 1 or result_sites[0].lineno >= node.lineno:
            continue
        findings.append(
            ValidationFinding(
                validator="mechanical_code_preflight",
                severity="error",
                message=(
                    "The stable closed-categorical-counts result stores category "
                    "labels in its explicit 'level' column; its table index is an "
                    "ordinary row index and cannot be used as declared levels."
                ),
                detail={
                    "reason": "closed_counts_table_index_used_as_levels",
                    "helper_name": "closed_categorical_counts",
                    "line": int(node.args[0].lineno),
                    "result_name": result_name,
                    "table_name": table_name,
                },
            )
        )
    return findings


def table_one_spec_binding_findings(
    tree: ast.Module,
    step: AnalysisStep,
) -> list[ValidationFinding]:
    """Require the SDK call to consume the exact Planner-owned Table 1 spec."""

    if step.table_one_spec is None:
        return []
    call_names: set[str] = set()
    for node in tree.body:
        if (
            isinstance(node, ast.ImportFrom)
            and node.level == 0
            and node.module == "easyicu.research_agent.methods.table_one"
        ):
            call_names.update(
                alias.asname or alias.name
                for alias in node.names
                if alias.name == "build_grouped_table_one"
            )
    if not call_names:
        return []
    expected = step.table_one_spec.model_dump(mode="json")
    assignments: dict[str, list[ast.Assign]] = {}
    for node in ast.walk(tree):
        if (
            isinstance(node, ast.Assign)
            and len(node.targets) == 1
            and isinstance(node.targets[0], ast.Name)
        ):
            assignments.setdefault(node.targets[0].id, []).append(node)
    findings: list[ValidationFinding] = []
    for node in ast.walk(tree):
        if not (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Name)
            and node.func.id in call_names
            and len(node.args) >= 2
            and isinstance(node.args[1], ast.Name)
        ):
            continue
        spec_name = node.args[1].id
        sites = assignments.get(spec_name, [])
        if len(sites) != 1 or sites[0].lineno >= node.lineno:
            continue
        try:
            actual = ast.literal_eval(sites[0].value)
        except (ValueError, TypeError, SyntaxError):
            actual = None
        if actual == expected:
            continue
        findings.append(
            ValidationFinding(
                validator="mechanical_code_preflight",
                severity="error",
                message=(
                    "The grouped Table 1 SDK call must consume the exact "
                    "Planner-owned table_one_spec instead of a reconstructed "
                    "or extended local schema."
                ),
                detail={
                    "reason": "table_one_spec_not_planner_owned",
                    "helper_name": "build_grouped_table_one",
                    "line": int(sites[0].value.lineno),
                    "spec_name": spec_name,
                    "expected_spec": expected,
                },
            )
        )
    return findings


__all__ = [
    "closed_counts_table_index_findings",
    "table_one_spec_binding_findings",
]
