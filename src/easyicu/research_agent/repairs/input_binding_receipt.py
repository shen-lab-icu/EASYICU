"""Mechanical repairs for generated typed-input binding receipts."""

from __future__ import annotations

import ast
from typing import Any, Mapping, Sequence


def _finding_detail(finding: Any) -> tuple[object, object]:
    if isinstance(finding, Mapping):
        return finding.get("validator"), finding.get("detail")
    return getattr(finding, "validator", None), getattr(finding, "detail", None)


def _requests_loaded_only_repair(findings: Sequence[Any]) -> bool:
    matched = False
    for finding in findings:
        validator, detail = _finding_detail(finding)
        if not (
            validator == "step_summary_integrity"
            and isinstance(detail, Mapping)
            and detail.get("issue") == "input_binding_load_contract_invalid"
        ):
            continue
        invalid_fields = detail.get("invalid_fields")
        if invalid_fields != ["loaded"]:
            return False
        matched = True
    return matched


def _literal_fields(node: ast.Dict) -> dict[str, ast.AST] | None:
    fields: dict[str, ast.AST] = {}
    for key, value in zip(node.keys, node.values, strict=True):
        if not (
            isinstance(key, ast.Constant)
            and isinstance(key.value, str)
            and key.value not in fields
        ):
            return None
        fields[key.value] = value
    return fields


def _counts_returned_frame(expression: ast.AST, frame_name: str) -> bool:
    return any(
        isinstance(candidate, ast.Call)
        and isinstance(candidate.func, ast.Name)
        and candidate.func.id == "len"
        and len(candidate.args) == 1
        and isinstance(candidate.args[0], ast.Name)
        and candidate.args[0].id == frame_name
        for candidate in ast.walk(expression)
    )


def patch_missing_loaded_input_binding_receipt(
    code: str,
    *,
    findings: Sequence[Any],
) -> str:
    """Add ``loaded=True`` only to a receipt proven to return its counted frame.

    The host has already verified the emitted row count and reports that the
    sole malformed field is ``loaded``.  Static proof additionally requires a
    helper to return ``(frame, receipt)`` while the receipt's row count is
    derived from ``len(frame)``.  Ambiguous receipt construction is left for
    the ordinary repair path.
    """

    if not _requests_loaded_only_repair(findings):
        return code
    try:
        tree = ast.parse(code)
    except SyntaxError:
        return code

    candidates: list[ast.Dict] = []
    for returned in (
        node for node in ast.walk(tree) if isinstance(node, ast.Return)
    ):
        value = returned.value
        if not isinstance(value, ast.Tuple) or len(value.elts) != 2:
            continue
        frame, receipt = value.elts
        if not isinstance(frame, ast.Name) or not isinstance(receipt, ast.Dict):
            continue
        fields = _literal_fields(receipt)
        if fields is None or "loaded" in fields:
            continue
        if not {"input_key", "evidence_id", "sha256", "row_count"}.issubset(fields):
            continue
        if not _counts_returned_frame(fields["row_count"], frame.id):
            continue
        candidates.append(receipt)
    if not candidates:
        return code

    lines = code.splitlines(keepends=True)
    line_starts: list[int] = []
    offset = 0
    for line in lines:
        line_starts.append(offset)
        offset += len(line)

    def absolute_offset(node: ast.AST) -> int:
        line = lines[node.lineno - 1]
        char_col = len(line.encode("utf-8")[: node.col_offset].decode("utf-8"))
        return line_starts[node.lineno - 1] + char_col

    repaired = code
    for candidate in sorted(candidates, key=absolute_offset, reverse=True):
        start = absolute_offset(candidate)
        if repaired[start : start + 1] != "{":
            return code
        repaired = repaired[: start + 1] + '"loaded": True, ' + repaired[start + 1 :]
    try:
        ast.parse(repaired)
    except SyntaxError:
        return code
    return repaired


__all__ = ["patch_missing_loaded_input_binding_receipt"]
