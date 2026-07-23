"""Deterministic guards for bound percentage/count identities."""

from __future__ import annotations

import ast
import re
from collections import defaultdict
from typing import Sequence

from ..schema import ValidationFinding


def _normalise(value: object) -> str:
    return re.sub(r"[^a-z0-9]+", "_", str(value or "").lower()).strip("_")


def _column_role(column: str) -> str | None:
    name = _normalise(column)
    tokens = set(name.split("_"))
    if tokens & {"percentage", "percent", "pct"}:
        return "percentage"
    if "denominator" in tokens or name in {"n_total", "total_n"}:
        return "denominator"
    if (
        name in {"n", "count", "numerator", "missing_n", "n_complete", "event_n"}
        or name.endswith("_count")
        or name.endswith("_numerator")
    ):
        return "numerator"
    return None


def _finding_variables(
    findings: Sequence[ValidationFinding],
) -> set[str] | None:
    relevant = []
    for finding in findings:
        validator = getattr(finding, "validator", None)
        severity = getattr(finding, "severity", None)
        detail = getattr(finding, "detail", None)
        if isinstance(finding, dict):
            validator = finding.get("validator")
            severity = finding.get("severity")
            detail = finding.get("detail")
        if (
            validator != "llm_concept_auditor"
            or severity != "error"
            or not isinstance(detail, dict)
        ):
            continue
        variables = detail.get("variables")
        if not isinstance(variables, list) or not variables:
            continue
        parsed = {
            str(value).strip()
            for value in variables
            if re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", str(value).strip())
        }
        if len(parsed) != len(variables):
            return None
        relevant.append(parsed)
    if len(relevant) != 1:
        return None
    return relevant[0]


def patch_bound_percentage_identity_guards(
    code: str,
    *,
    findings: Sequence[ValidationFinding],
) -> str:
    """Check every auditor-named bound percentage against counts.

    The patch is available only when each named variable is the direct result
    of ``finite_numeric(frame, column, input_key, ...)`` and that exact frame
    and input key also provide one unambiguous numerator and denominator.
    It inserts validation only; it never recomputes or replaces a displayed
    value.
    """

    marker = "_easyicu_expected_percentage_"
    if marker in code:
        return code
    variables = _finding_variables(findings)
    if not variables:
        return code
    try:
        tree = ast.parse(code)
    except SyntaxError:
        return code
    if not any(
        isinstance(node, ast.Import)
        and any(alias.name == "numpy" and alias.asname == "np" for alias in node.names)
        for node in tree.body
    ):
        return code

    calls: list[dict[str, object]] = []
    for node in ast.walk(tree):
        if not (
            isinstance(node, ast.Assign)
            and len(node.targets) == 1
            and isinstance(node.targets[0], ast.Name)
            and isinstance(node.value, ast.Call)
            and (
                (
                    isinstance(node.value.func, ast.Name)
                    and node.value.func.id == "finite_numeric"
                )
                or (
                    isinstance(node.value.func, ast.Attribute)
                    and node.value.func.attr == "finite_numeric"
                )
            )
            and len(node.value.args) >= 3
            and isinstance(node.value.args[0], ast.Name)
            and isinstance(node.value.args[1], ast.Constant)
            and isinstance(node.value.args[1].value, str)
            and isinstance(node.value.args[2], ast.Constant)
            and isinstance(node.value.args[2].value, str)
        ):
            continue
        role = _column_role(node.value.args[1].value)
        if role is None:
            continue
        calls.append(
            {
                "assignment": node,
                "target": node.targets[0].id,
                "frame": node.value.args[0].id,
                "column": node.value.args[1].value,
                "input_key": node.value.args[2].value,
                "role": role,
            }
        )

    percentages = {
        str(item["target"]): item
        for item in calls
        if item["role"] == "percentage" and item["target"] in variables
    }
    if set(percentages) != variables:
        return code

    groups: dict[tuple[str, str], list[dict[str, object]]] = defaultdict(list)
    for item in calls:
        groups[(str(item["frame"]), str(item["input_key"]))].append(item)

    lines = code.splitlines(keepends=True)
    insertions: dict[int, list[str]] = defaultdict(list)
    for index, target in enumerate(sorted(percentages)):
        percentage = percentages[target]
        siblings = groups[(str(percentage["frame"]), str(percentage["input_key"]))]
        numerators = [item for item in siblings if item["role"] == "numerator"]
        denominators = [item for item in siblings if item["role"] == "denominator"]
        if len(numerators) != 1 or len(denominators) != 1:
            return code
        numerator = numerators[0]
        denominator = denominators[0]
        assignments = [
            percentage["assignment"],
            numerator["assignment"],
            denominator["assignment"],
        ]
        if any(
            not isinstance(assignment, ast.Assign) or assignment.end_lineno is None
            for assignment in assignments
        ):
            return code
        anchor = max(assignments, key=lambda assignment: assignment.end_lineno)
        if (
            len(
                {
                    assignment.col_offset
                    for assignment in assignments
                    if isinstance(assignment, ast.Assign)
                }
            )
            != 1
        ):
            return code
        indent = " " * anchor.col_offset
        expected = f"_easyicu_expected_percentage_{index}"
        denominator_name = str(denominator["target"])
        numerator_name = str(numerator["target"])
        input_key = str(percentage["input_key"])
        guard = (
            f"{indent}if np.any({denominator_name} <= 0):\n"
            f"{indent}    raise RuntimeError("
            f'{input_key!r} + " has a non-positive percentage denominator")\n'
            f"{indent}{expected} = 100.0 * {numerator_name} / {denominator_name}\n"
            f"{indent}if not np.allclose(\n"
            f"{indent}    {target}, {expected}, rtol=0.0, atol=1e-6, "
            f"equal_nan=False\n"
            f"{indent}):\n"
            f"{indent}    raise RuntimeError("
            f'{input_key!r} + " percentage disagrees with numerator/denominator")\n'
        )
        insert_at = sum(len(line) for line in lines[: anchor.end_lineno])
        insertions[insert_at].append(guard)

    repaired = code
    for insert_at, guards in sorted(insertions.items(), reverse=True):
        repaired = repaired[:insert_at] + "".join(guards) + repaired[insert_at:]
    try:
        ast.parse(repaired)
    except SyntaxError:
        return code
    return repaired


__all__ = ["patch_bound_percentage_identity_guards"]
