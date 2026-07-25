"""Fail-closed domain guards authorized by exact concept-audit findings."""

from __future__ import annotations

import ast
from typing import Sequence

from ..schema import ValidationFinding

_MARKER = "# EASYICU_HOST_DOMAIN_GUARDS_V1"
_ORDINAL_PROBLEM = "invalid ordinal levels are not rejected before rounding"
_BINARY_PROBLEM = "binary domain validation is missing"


def _coordinates(
    findings: Sequence[ValidationFinding],
) -> tuple[set[str], set[str]] | None:
    ordinal: set[str] = set()
    binary: set[str] = set()
    for finding in findings:
        detail = finding.detail or {}
        if not (
            finding.validator == "llm_concept_auditor"
            and finding.severity == "error"
            and detail.get("issue_code") == "other"
        ):
            continue
        variable = detail.get("variable")
        problem = detail.get("problem")
        if not isinstance(variable, str) or not variable.strip():
            continue
        if problem == _ORDINAL_PROBLEM:
            ordinal.add(variable)
        elif problem == _BINARY_PROBLEM:
            binary.add(variable)
    if not ordinal and not binary:
        return None
    if ordinal & binary:
        return None
    return ordinal, binary


def _strict_numeric_binding(
    tree: ast.Module,
    *,
    variable: str,
) -> tuple[str, ast.Assign] | None:
    matches: list[tuple[str, ast.Assign]] = []
    for node in tree.body:
        if not (
            isinstance(node, ast.Assign)
            and len(node.targets) == 1
            and isinstance(node.targets[0], ast.Name)
            and isinstance(node.value, ast.Call)
            and isinstance(node.value.func, ast.Name)
            and node.value.func.id == "strict_numeric"
            and len(node.value.args) == 2
            and not node.value.keywords
            and isinstance(node.value.args[0], ast.Subscript)
            and isinstance(node.value.args[0].value, ast.Name)
            and node.value.args[0].value.id == "df"
            and isinstance(node.value.args[0].slice, ast.Constant)
            and node.value.args[0].slice.value == variable
            and isinstance(node.value.args[1], ast.Constant)
            and node.value.args[1].value == variable
        ):
            continue
        matches.append((node.targets[0].id, node))
    return matches[0] if len(matches) == 1 else None


def _ordinal_round_assignment(
    tree: ast.Module,
    *,
    numeric_name: str,
) -> ast.Assign | None:
    matches: list[ast.Assign] = []
    for node in tree.body:
        if not isinstance(node, ast.Assign):
            continue
        rounds = [
            call
            for call in ast.walk(node.value)
            if isinstance(call, ast.Call)
            and isinstance(call.func, ast.Attribute)
            and call.func.attr == "round"
            and isinstance(call.func.value, ast.Name)
            and call.func.value.id == numeric_name
            and not call.args
            and not call.keywords
        ]
        if rounds:
            matches.append(node)
    return matches[0] if len(matches) == 1 else None


def _has_authored_range_guard(
    tree: ast.Module,
    *,
    numeric_name: str,
    after_line: int,
    before_line: int,
) -> bool:
    candidates = []
    for node in tree.body:
        if not (
            isinstance(node, ast.If)
            and after_line < node.lineno < before_line
            and any(
                isinstance(name, ast.Name) and name.id == numeric_name
                for name in ast.walk(node.test)
            )
        ):
            continue
        comparisons = [
            item
            for item in ast.walk(node.test)
            if isinstance(item, ast.Compare)
            and any(
                isinstance(name, ast.Name) and name.id == numeric_name
                for name in ast.walk(item)
            )
            and any(
                isinstance(op, (ast.Lt, ast.LtE, ast.Gt, ast.GtE)) for op in item.ops
            )
        ]
        if len(comparisons) >= 2:
            candidates.append(node)
    return len(candidates) == 1


def _literal_list_bindings(tree: ast.Module, *, variable: str) -> set[str]:
    names: set[str] = set()
    for node in tree.body:
        if not (
            isinstance(node, ast.Assign)
            and len(node.targets) == 1
            and isinstance(node.targets[0], ast.Name)
            and isinstance(node.value, ast.List)
        ):
            continue
        values = [
            item.value
            for item in node.value.elts
            if isinstance(item, ast.Constant) and isinstance(item.value, str)
        ]
        if len(values) == len(node.value.elts) and variable in values:
            names.add(node.targets[0].id)
    return names


def _strict_numeric_loop(
    tree: ast.Module,
    *,
    list_names: set[str],
) -> ast.For | None:
    matches: list[ast.For] = []
    for node in tree.body:
        if not (
            isinstance(node, ast.For)
            and isinstance(node.target, ast.Name)
            and any(
                isinstance(name, ast.Name) and name.id in list_names
                for name in ast.walk(node.iter)
            )
        ):
            continue
        name = node.target.id
        exact_assignments = [
            item
            for item in node.body
            if isinstance(item, ast.Assign)
            and len(item.targets) == 1
            and isinstance(item.targets[0], ast.Subscript)
            and isinstance(item.targets[0].value, ast.Name)
            and item.targets[0].value.id == "df"
            and isinstance(item.targets[0].slice, ast.Name)
            and item.targets[0].slice.id == name
            and isinstance(item.value, ast.Call)
            and isinstance(item.value.func, ast.Name)
            and item.value.func.id == "strict_numeric"
            and len(item.value.args) == 2
            and not item.value.keywords
        ]
        if len(exact_assignments) == 1:
            matches.append(node)
    return matches[0] if len(matches) == 1 else None


def patch_llm_proven_domain_guards(
    code: str,
    *,
    findings: Sequence[ValidationFinding],
) -> str:
    """Insert only auditor-authorized rejection guards around existing values."""

    coordinates = _coordinates(findings)
    if coordinates is None or _MARKER in code:
        return code
    ordinal, binary = coordinates
    try:
        tree = ast.parse(code)
    except SyntaxError:
        return code

    insertions: dict[int, list[str]] = {}
    for variable in sorted(ordinal):
        binding = _strict_numeric_binding(tree, variable=variable)
        if binding is None:
            return code
        numeric_name, numeric_assignment = binding
        rounding = _ordinal_round_assignment(tree, numeric_name=numeric_name)
        if rounding is None or not _has_authored_range_guard(
            tree,
            numeric_name=numeric_name,
            after_line=numeric_assignment.lineno,
            before_line=rounding.lineno,
        ):
            return code
        insertions.setdefault(rounding.lineno, []).extend(
            [
                f"if not {numeric_name}.dropna().mod(1).eq(0).all():",
                f"    raise RuntimeError({variable!r} + ' contains non-integer ordinal values')",
            ]
        )

    binary_loops: dict[int, tuple[ast.For, list[str]]] = {}
    for variable in sorted(binary):
        list_names = _literal_list_bindings(tree, variable=variable)
        loop = _strict_numeric_loop(tree, list_names=list_names)
        if not list_names or loop is None or loop.end_lineno is None:
            return code
        binary_loops.setdefault(loop.end_lineno, (loop, []))[1].extend(
            [
                f"if not set(df[{variable!r}].dropna().unique()).issubset({{0, 1}}):",
                f"    raise RuntimeError({variable!r} + ' must be binary')",
            ]
        )

    lines = code.splitlines(keepends=True)
    for end_line, (_loop, guard_lines) in binary_loops.items():
        insertions.setdefault(end_line + 1, []).extend(guard_lines)
    if not insertions:
        return code
    for line_number in sorted(insertions, reverse=True):
        payload = insertions[line_number]
        if line_number == min(insertions):
            payload = [_MARKER, *payload]
        lines[line_number - 1 : line_number - 1] = [f"{line}\n" for line in payload]
    repaired = "".join(lines)
    try:
        ast.parse(repaired)
    except SyntaxError:
        return code
    return repaired


__all__ = ["patch_llm_proven_domain_guards"]
