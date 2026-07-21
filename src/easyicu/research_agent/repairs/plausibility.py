"""Deterministic repairs for typed plausibility-range policy findings."""

from __future__ import annotations

import ast
from typing import Optional, Sequence

from ..schema import ValidationFinding


def patch_flag_only_plausibility_range_rejection(
    code: str,
    *,
    repair_findings: Sequence[ValidationFinding],
) -> str:
    """Remove one auditor-proven hard rejection of a flag-only range.

    The ConceptDescriptor plausibility range is not an exclusion contract.  A
    deterministic repair is permitted only when the typed auditor identifies
    one variable and the script contains exactly one adjacent
    ``range-mask = lower | upper`` / ``if range-mask.any(): raise`` pair.  Any
    additional use, side effect, ambiguity, or non-literal boundary leaves the
    code unchanged for provider repair.
    """

    matching: list[set[str] | None] = []
    for finding in repair_findings:
        detail = finding.detail or {}
        if (
            finding.validator == "llm_concept_auditor"
            and detail.get("issue_code") == "plausibility_range_exclusion_required"
            and detail.get("value_class") == "finite_outside_plausibility_range"
        ):
            raw_variable = detail.get("variable")
            if isinstance(raw_variable, str) and raw_variable.isidentifier():
                matching.append({raw_variable})
            elif (
                isinstance(raw_variable, list)
                and raw_variable
                and all(isinstance(item, str) and item.strip() for item in raw_variable)
            ):
                # The auditor may report a semantic column family while the
                # generated code validates it through one loop-local Series.
                # In that case the structural match below must be globally
                # unique because no local Python name can be inferred safely.
                matching.append(None)
    if len(matching) != 1:
        return code
    accepted_variables = matching[0]

    try:
        tree = ast.parse(code)
    except SyntaxError:
        return code

    def numeric_literal(node: ast.AST) -> bool:
        if isinstance(node, ast.Constant):
            return isinstance(node.value, (int, float)) and not isinstance(
                node.value, bool
            )
        return (
            isinstance(node, ast.UnaryOp)
            and isinstance(node.op, (ast.UAdd, ast.USub))
            and numeric_literal(node.operand)
        )

    def bound_side(node: ast.AST) -> Optional[tuple[str, str]]:
        if not (
            isinstance(node, ast.Compare)
            and len(node.ops) == 1
            and len(node.comparators) == 1
        ):
            return None
        left, right = node.left, node.comparators[0]
        operator = node.ops[0]
        if isinstance(left, ast.Name) and numeric_literal(right):
            if isinstance(operator, (ast.Lt, ast.LtE)):
                return "lower", left.id
            if isinstance(operator, (ast.Gt, ast.GtE)):
                return "upper", left.id
        if numeric_literal(left) and isinstance(right, ast.Name):
            if isinstance(operator, (ast.Gt, ast.GtE)):
                return "lower", right.id
            if isinstance(operator, (ast.Lt, ast.LtE)):
                return "upper", right.id
        return None

    def range_variable(node: ast.AST) -> Optional[str]:
        if not (isinstance(node, ast.BinOp) and isinstance(node.op, ast.BitOr)):
            return None
        bounds = (bound_side(node.left), bound_side(node.right))
        if any(bound is None for bound in bounds):
            return None
        assert bounds[0] is not None and bounds[1] is not None
        if {bounds[0][0], bounds[1][0]} != {"lower", "upper"}:
            return None
        if bounds[0][1] != bounds[1][1]:
            return None
        return bounds[0][1]

    def range_mask_variable(node: ast.AST) -> Optional[str]:
        direct = range_variable(node)
        if direct is not None:
            return direct
        if not (isinstance(node, ast.BinOp) and isinstance(node.op, ast.BitAnd)):
            return None
        for availability, bounds in ((node.left, node.right), (node.right, node.left)):
            variable = range_variable(bounds)
            if (
                variable is not None
                and isinstance(availability, ast.Call)
                and isinstance(availability.func, ast.Attribute)
                and availability.func.attr == "notna"
                and isinstance(availability.func.value, ast.Name)
                and availability.func.value.id == variable
                and not availability.args
                and not availability.keywords
            ):
                return variable
        return None

    def is_raise_guard(node: ast.stmt, mask_name: str) -> bool:
        if not (
            isinstance(node, ast.If)
            and len(node.body) == 1
            and isinstance(node.body[0], ast.Raise)
            and not node.orelse
        ):
            return False
        test = node.test
        if (
            isinstance(test, ast.Call)
            and isinstance(test.func, ast.Name)
            and test.func.id == "bool"
            and len(test.args) == 1
            and not test.keywords
        ):
            test = test.args[0]
        return (
            isinstance(test, ast.Call)
            and isinstance(test.func, ast.Attribute)
            and test.func.attr == "any"
            and isinstance(test.func.value, ast.Name)
            and test.func.value.id == mask_name
            and not test.args
            and not test.keywords
        )

    candidates: list[tuple[ast.Assign, ast.If]] = []
    for parent in ast.walk(tree):
        for field in ("body", "orelse", "finalbody"):
            statements = getattr(parent, field, None)
            if not isinstance(statements, list):
                continue
            for first, second in zip(statements, statements[1:]):
                range_variable_name = (
                    range_mask_variable(first.value)
                    if isinstance(first, ast.Assign)
                    else None
                )
                if not (
                    isinstance(first, ast.Assign)
                    and len(first.targets) == 1
                    and isinstance(first.targets[0], ast.Name)
                    and range_variable_name is not None
                    and (
                        accepted_variables is None
                        or range_variable_name in accepted_variables
                    )
                ):
                    continue
                mask_name = first.targets[0].id
                if not is_raise_guard(second, mask_name):
                    continue
                loads = [
                    node
                    for node in ast.walk(tree)
                    if isinstance(node, ast.Name)
                    and isinstance(node.ctx, ast.Load)
                    and node.id == mask_name
                ]
                if len(loads) == 1:
                    candidates.append((first, second))
    if len(candidates) != 1:
        return code
    assignment, guard = candidates[0]
    if assignment.end_lineno is None or guard.end_lineno is None:
        return code
    lines = code.splitlines(keepends=True)
    source_line = lines[assignment.lineno - 1]
    indent = source_line[: len(source_line) - len(source_line.lstrip(" \t"))]
    replacement = f"{indent}# _easyicu_flag_only_plausibility_range_retained_v1\n"
    lines[assignment.lineno - 1 : guard.end_lineno] = [replacement]
    repaired = "".join(lines)
    try:
        ast.parse(repaired)
    except SyntaxError:
        return code
    return repaired


__all__ = ["patch_flag_only_plausibility_range_rejection"]
