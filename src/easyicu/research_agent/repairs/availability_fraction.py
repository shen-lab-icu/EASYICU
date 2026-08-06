"""Narrow repair for an audited availability-fraction denominator.

This module owns one representation contract: when generated code labels a
quantity as availability and divides ``n_nonmissing`` by ``n``, an explicit
concept-audit finding may establish that the only available total is the sum
of the non-missing and missing components.  The repair changes that one
derived denominator and its adjacent summary labels.  It does not select rows,
features, clusters, models, or source artifacts.
"""

from __future__ import annotations

import ast
from collections.abc import Mapping, Sequence
from typing import Any


_REPAIR_ID = "availability_fraction_component_denominator_v1"
_DENOMINATOR_LABEL = "n_nonmissing + missing_n"
_DEFINITION_LABEL = "n_nonmissing / (n_nonmissing + missing_n)"
_RECONCILIATION_LABEL = (
    "availability denominator reconstructed exactly as n_nonmissing + missing_n"
)


def _finding_payload(finding: Any) -> tuple[str, Mapping[str, Any]]:
    if isinstance(finding, Mapping):
        message = str(finding.get("message") or "")
        detail = finding.get("detail")
    else:
        message = str(getattr(finding, "message", "") or "")
        detail = getattr(finding, "detail", None)
    return message, detail if isinstance(detail, Mapping) else {}


def _audit_names_component_denominator(
    *,
    audit_messages: Sequence[str],
    repair_findings: Sequence[Any],
) -> bool:
    texts = [str(message or "") for message in audit_messages]
    variables: set[str] = set()
    for finding in repair_findings:
        message, detail = _finding_payload(finding)
        texts.extend((message, str(detail.get("reason") or "")))
        raw_variables = detail.get("variables")
        if isinstance(raw_variables, Sequence) and not isinstance(
            raw_variables, (str, bytes)
        ):
            variables.update(str(value).strip().lower() for value in raw_variables)

    normalized = " ".join(texts).lower()
    named_in_text = all(
        marker in normalized for marker in ("availability", "n_nonmissing", "missing_n")
    )
    named_in_detail = {"n_nonmissing", "missing_n"} <= variables and (
        "availability" in normalized or "availability_fraction" in variables
    )
    return (named_in_text or named_in_detail) and any(
        marker in normalized
        for marker in ("reconcil", "denominator", "percent", "fraction")
    )


def _named_string_subscript(node: ast.AST) -> tuple[str, str] | None:
    if not isinstance(node, ast.Subscript) or not isinstance(node.value, ast.Name):
        return None
    if not isinstance(node.slice, ast.Constant) or not isinstance(
        node.slice.value, str
    ):
        return None
    return node.value.id, node.slice.value


def _line_offsets(code: str) -> list[int]:
    offsets: list[int] = []
    offset = 0
    for line in code.splitlines(keepends=True):
        offsets.append(offset)
        offset += len(line)
    if not offsets:
        offsets.append(0)
    return offsets


def _absolute_offset(code: str, offsets: Sequence[int], line: int, col: int) -> int:
    source_line = code.splitlines(keepends=True)[line - 1]
    char_col = len(source_line.encode("utf-8")[:col].decode("utf-8"))
    return offsets[line - 1] + char_col


def _node_span(
    code: str,
    offsets: Sequence[int],
    node: ast.AST,
) -> tuple[int, int] | None:
    coordinates = (
        getattr(node, "lineno", None),
        getattr(node, "col_offset", None),
        getattr(node, "end_lineno", None),
        getattr(node, "end_col_offset", None),
    )
    if any(value is None for value in coordinates):
        return None
    line, col, end_line, end_col = coordinates
    return (
        _absolute_offset(code, offsets, int(line), int(col)),
        _absolute_offset(code, offsets, int(end_line), int(end_col)),
    )


def _summary_label_replacements(
    tree: ast.AST,
) -> list[tuple[ast.AST, str]]:
    replacements: list[tuple[ast.AST, str]] = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Dict):
            continue
        for key, value in zip(node.keys, node.values):
            if (
                not isinstance(key, ast.Constant)
                or not isinstance(key.value, str)
                or not isinstance(value, ast.Constant)
                or not isinstance(value.value, str)
            ):
                continue
            if key.value == "availability_denominator" and value.value == "n":
                replacements.append((value, _DENOMINATOR_LABEL))
            elif (
                key.value == "availability_definition"
                and value.value == "n_nonmissing / n"
            ):
                replacements.append((value, _DEFINITION_LABEL))
            elif (
                key.value == "missing_n_reconciliation"
                and "not imposed" in value.value.lower()
                and "n_nonmissing" in value.value
                and "missing_n" in value.value
            ):
                replacements.append((value, _RECONCILIATION_LABEL))
    return replacements


def patch_availability_fraction_component_denominator(
    code: str,
    *,
    audit_messages: Sequence[str],
    repair_findings: Sequence[Any] = (),
) -> tuple[str, str | None]:
    """Use the two audited count components as one availability denominator.

    The transform is deliberately fail-closed: exactly one matching division
    must exist, both operands must come from the same named frame, and that
    frame must already consume ``missing_n`` elsewhere in the script.
    """

    if not _audit_names_component_denominator(
        audit_messages=audit_messages,
        repair_findings=repair_findings,
    ):
        return code, None
    try:
        tree = ast.parse(code)
    except SyntaxError:
        return code, None

    divisions: list[tuple[ast.BinOp, str]] = []
    available_subscripts = {
        binding
        for node in ast.walk(tree)
        if (binding := _named_string_subscript(node)) is not None
    }
    for node in ast.walk(tree):
        if not isinstance(node, ast.BinOp) or not isinstance(node.op, ast.Div):
            continue
        numerator = _named_string_subscript(node.left)
        denominator = _named_string_subscript(node.right)
        if (
            numerator is None
            or denominator is None
            or numerator[0] != denominator[0]
            or numerator[1] != "n_nonmissing"
            or denominator[1] != "n"
            or (numerator[0], "missing_n") not in available_subscripts
        ):
            continue
        divisions.append((node, numerator[0]))
    if len(divisions) != 1:
        return code, None

    division, frame_name = divisions[0]
    offsets = _line_offsets(code)
    denominator_span = _node_span(code, offsets, division.right)
    if denominator_span is None:
        return code, None
    replacements: list[tuple[int, int, str]] = [
        (
            denominator_span[0],
            denominator_span[1],
            f'({frame_name}["n_nonmissing"] + {frame_name}["missing_n"])',
        )
    ]
    for node, label in _summary_label_replacements(tree):
        span = _node_span(code, offsets, node)
        if span is None:
            return code, None
        replacements.append((span[0], span[1], repr(label)))

    repaired = code
    for start, end, replacement in sorted(replacements, reverse=True):
        repaired = repaired[:start] + replacement + repaired[end:]
    try:
        ast.parse(repaired)
    except SyntaxError:
        return code, None
    return repaired, _REPAIR_ID


__all__ = ["patch_availability_fraction_component_denominator"]
