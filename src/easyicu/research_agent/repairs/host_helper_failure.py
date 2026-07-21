"""Fail-closed repair for caught host validation-helper errors."""

from __future__ import annotations

import ast
from typing import Sequence

from ..schema import ValidationFinding

_SENTINEL = "_easyicu_host_validation_helper_reraise_v1"


def _call_name(node: ast.AST) -> str:
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        prefix = _call_name(node.value)
        return f"{prefix}.{node.attr}" if prefix else node.attr
    return ""


def patch_host_validation_helper_reraise(
    code: str,
    *,
    findings: Sequence[ValidationFinding],
) -> str:
    """Re-raise at an exact handler named by deterministic preflight.

    The repair neither changes the successful path nor supplies a fallback
    value.  It only prevents a caught host-owned validation failure from being
    converted into apparently usable output.
    """

    if _SENTINEL in code:
        return code
    coordinates: list[tuple[int, frozenset[str]]] = []
    for finding in findings:
        detail = finding.detail or {}
        if (
            finding.validator != "mechanical_code_preflight"
            or finding.severity != "error"
            or detail.get("reason") != "host_validation_helper_error_swallowed"
        ):
            continue
        line = detail.get("line")
        helper_names = detail.get("helper_names")
        if not isinstance(line, int) or not isinstance(helper_names, list):
            return code
        closed_helpers = frozenset(
            str(name) for name in helper_names if str(name).isidentifier()
        )
        if not closed_helpers or len(closed_helpers) != len(helper_names):
            return code
        coordinates.append((line, closed_helpers))
    if len(coordinates) != 1:
        return code
    try:
        tree = ast.parse(code)
    except SyntaxError:
        return code
    line, helper_names = coordinates[0]
    matches: list[ast.ExceptHandler] = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Try):
            continue
        called_helpers = {
            _call_name(candidate.func).split(".")[-1]
            for statement in node.body
            for candidate in ast.walk(statement)
            if isinstance(candidate, ast.Call)
        }
        if not helper_names <= called_helpers:
            continue
        for handler in node.handlers:
            if handler.lineno != line or not handler.body:
                continue
            if isinstance(handler.body[0], ast.Raise):
                return code
            matches.append(handler)
    if len(matches) != 1:
        return code
    handler = matches[0]
    lines = code.splitlines(keepends=True)
    first_line = lines[handler.body[0].lineno - 1]
    indent = first_line[: len(first_line) - len(first_line.lstrip())]
    lines.insert(
        handler.body[0].lineno - 1,
        f"{indent}# {_SENTINEL}\n{indent}raise\n",
    )
    repaired = "".join(lines)
    try:
        ast.parse(repaired)
    except SyntaxError:
        return code
    return repaired


__all__ = ["patch_host_validation_helper_reraise"]
