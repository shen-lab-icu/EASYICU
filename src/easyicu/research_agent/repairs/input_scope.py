"""Narrow repair for confusing consumer scope with physical table schema."""

from __future__ import annotations

import ast
import re
from typing import Optional

_PHYSICAL_SCOPE_ERROR = (
    "locked cohort columns do not match the exact declared input scope"
)
_RAW_CONTRACT_MAPPING_ERROR = "attributeerror: 'str' object has no attribute 'get'"
_MISSING_RAW_CONTRACT_ERROR = re.compile(
    r"missing raw input contract for [A-Za-z_][A-Za-z0-9_]*",
    re.IGNORECASE,
)


def _physical_columns_owner(node: ast.AST) -> Optional[ast.AST]:
    """Return ``df`` from ``list/set/tuple(df.columns)`` or ``df.columns``."""

    candidate = node
    if (
        isinstance(candidate, ast.Call)
        and isinstance(candidate.func, ast.Name)
        and candidate.func.id in {"list", "set", "tuple"}
        and len(candidate.args) == 1
        and not candidate.keywords
    ):
        candidate = candidate.args[0]
    if isinstance(candidate, ast.Attribute) and candidate.attr == "columns":
        return candidate.value
    return None


def patch_raw_input_physical_superset_guard(code: str, run_log: str) -> str:
    """Replace one proven closed-world column assertion with a presence check.

    ``planner_declared_inputs`` is consumer authority, while ``COHORT_PARQUET`` is
    a host-locked physical source that may legitimately contain more columns.
    This repair changes only an authored ``require`` whose own diagnostic and the
    runner log both name the exact failure. It never selects, drops, or transforms
    data and retains fail-closed behavior when a declared input is absent.
    """

    if _PHYSICAL_SCOPE_ERROR not in str(run_log or "").lower():
        return code
    try:
        tree = ast.parse(code)
    except SyntaxError:
        return code

    lines = code.splitlines(keepends=True)
    line_starts: list[int] = []
    offset = 0
    for line in lines:
        line_starts.append(offset)
        offset += len(line)

    def _absolute_offset(lineno: int, utf8_col: int) -> int:
        line = lines[lineno - 1]
        char_col = len(line.encode("utf-8")[:utf8_col].decode("utf-8"))
        return line_starts[lineno - 1] + char_col

    replacements: list[tuple[int, int, str]] = []
    for node in ast.walk(tree):
        if not (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Name)
            and node.func.id == "require"
            and len(node.args) >= 2
            and isinstance(node.args[1], ast.Constant)
            and isinstance(node.args[1].value, str)
            and _PHYSICAL_SCOPE_ERROR in node.args[1].value.lower()
        ):
            continue
        condition = node.args[0]
        binding: tuple[ast.AST, ast.AST] | None = None
        for comparison in ast.walk(condition):
            if not (
                isinstance(comparison, ast.Compare)
                and len(comparison.ops) == 1
                and isinstance(comparison.ops[0], ast.Eq)
                and len(comparison.comparators) == 1
            ):
                continue
            left_owner = _physical_columns_owner(comparison.left)
            right_owner = _physical_columns_owner(comparison.comparators[0])
            if (left_owner is None) == (right_owner is None):
                continue
            frame_node = left_owner if left_owner is not None else right_owner
            expected_node = (
                comparison.comparators[0]
                if left_owner is not None
                else comparison.left
            )
            binding = (frame_node, expected_node)
            break
        if binding is None:
            continue
        frame_node, expected_node = binding
        frame_source = ast.get_source_segment(code, frame_node)
        expected_source = ast.get_source_segment(code, expected_node)
        if not frame_source or not expected_source:
            continue
        replacement = (
            "all(_easyicu_input in "
            f"({frame_source}).columns for _easyicu_input in ({expected_source}))"
        )
        replacements.append(
            (
                _absolute_offset(condition.lineno, condition.col_offset),
                _absolute_offset(condition.end_lineno, condition.end_col_offset),
                replacement,
            )
        )

    if not replacements:
        return code
    repaired = code
    for start, end, replacement in sorted(replacements, reverse=True):
        repaired = repaired[:start] + replacement + repaired[end:]
    try:
        ast.parse(repaired)
    except SyntaxError:
        return code
    return repaired


def patch_raw_contract_mapping_iteration(code: str, run_log: str) -> str:
    """Iterate values of the host's column-keyed raw-contract mapping.

    The resolved-input schema defines ``contracts`` as a JSON object keyed by
    column. Generated code occasionally treats that object as a list of
    contract records, so Python yields string keys and ``contract.get(...)``
    fails. Only the exact traceback and one unambiguous AST shape authorize
    this syntax-only adapter.
    """

    if _RAW_CONTRACT_MAPPING_ERROR not in str(run_log or "").lower():
        return code
    try:
        tree = ast.parse(code)
    except SyntaxError:
        return code

    contract_mapping_names: set[str] = set()
    for node in ast.walk(tree):
        if not (
            isinstance(node, ast.Assign)
            and len(node.targets) == 1
            and isinstance(node.targets[0], ast.Name)
            and isinstance(node.value, ast.Call)
            and isinstance(node.value.func, ast.Attribute)
            and node.value.func.attr == "get"
            and node.value.args
            and isinstance(node.value.args[0], ast.Constant)
            and node.value.args[0].value == "contracts"
        ):
            continue
        contract_mapping_names.add(node.targets[0].id)

    candidates: list[ast.Name] = []
    for node in ast.walk(tree):
        if not (
            isinstance(node, ast.For)
            and isinstance(node.target, ast.Name)
            and isinstance(node.iter, ast.Name)
            and node.iter.id in contract_mapping_names
        ):
            continue
        item_name = node.target.id
        reads_contract_column = any(
            isinstance(child, ast.Call)
            and isinstance(child.func, ast.Attribute)
            and child.func.attr == "get"
            and isinstance(child.func.value, ast.Name)
            and child.func.value.id == item_name
            and child.args
            and isinstance(child.args[0], ast.Constant)
            and child.args[0].value == "column"
            for statement in node.body
            for child in ast.walk(statement)
        )
        if reads_contract_column:
            candidates.append(node.iter)
    if len(candidates) != 1:
        return code

    iterable = candidates[0]
    lines = code.splitlines(keepends=True)
    line_starts: list[int] = []
    offset = 0
    for line in lines:
        line_starts.append(offset)
        offset += len(line)

    def _absolute_offset(lineno: int, utf8_col: int) -> int:
        line = lines[lineno - 1]
        char_col = len(line.encode("utf-8")[:utf8_col].decode("utf-8"))
        return line_starts[lineno - 1] + char_col

    start = _absolute_offset(iterable.lineno, iterable.col_offset)
    end = _absolute_offset(iterable.end_lineno, iterable.end_col_offset)
    repaired = code[:start] + f"{iterable.id}.values()" + code[end:]
    try:
        ast.parse(repaired)
    except SyntaxError:
        return code
    return repaired


def patch_raw_contract_document_fallback(code: str, run_log: str) -> str:
    """Preserve an unwrapped resolved-input document as its own manifest.

    Resolved-input schema 2.1 is emitted directly at the document root, while
    archived manifests may still be wrapped under ``manifest``. Generated code
    sometimes uses ``document.get("manifest", {})`` and thereby discards every
    host-issued raw contract in the current unwrapped shape. Only the exact
    missing-contract failure and one unambiguous three-level contract lookup
    authorize replacing that empty fallback with the parsed document itself.
    """

    if _MISSING_RAW_CONTRACT_ERROR.search(str(run_log or "")) is None:
        return code
    try:
        tree = ast.parse(code)
    except SyntaxError:
        return code

    candidates: list[tuple[ast.Dict, str]] = []
    for assignment in ast.walk(tree):
        if not (
            isinstance(assignment, ast.Assign)
            and len(assignment.targets) == 1
            and isinstance(assignment.targets[0], ast.Name)
        ):
            continue
        lookup_keys = {
            str(call.args[0].value)
            for call in ast.walk(assignment.value)
            if isinstance(call, ast.Call)
            and isinstance(call.func, ast.Attribute)
            and call.func.attr == "get"
            and call.args
            and isinstance(call.args[0], ast.Constant)
            and isinstance(call.args[0].value, str)
        }
        if not {"manifest", "raw_input_contracts", "contracts"} <= lookup_keys:
            continue
        for call in ast.walk(assignment.value):
            if not (
                isinstance(call, ast.Call)
                and isinstance(call.func, ast.Attribute)
                and call.func.attr == "get"
                and isinstance(call.func.value, ast.Name)
                and len(call.args) == 2
                and not call.keywords
                and isinstance(call.args[0], ast.Constant)
                and call.args[0].value == "manifest"
                and isinstance(call.args[1], ast.Dict)
                and not call.args[1].keys
            ):
                continue
            candidates.append((call.args[1], call.func.value.id))
    if len(candidates) != 1:
        return code

    fallback, document_name = candidates[0]
    lines = code.splitlines(keepends=True)
    line_starts: list[int] = []
    offset = 0
    for line in lines:
        line_starts.append(offset)
        offset += len(line)

    def _absolute_offset(lineno: int, utf8_col: int) -> int:
        line = lines[lineno - 1]
        char_col = len(line.encode("utf-8")[:utf8_col].decode("utf-8"))
        return line_starts[lineno - 1] + char_col

    start = _absolute_offset(fallback.lineno, fallback.col_offset)
    end = _absolute_offset(fallback.end_lineno, fallback.end_col_offset)
    repaired = code[:start] + document_name + code[end:]
    try:
        ast.parse(repaired)
    except SyntaxError:
        return code
    return repaired


__all__ = [
    "patch_raw_contract_document_fallback",
    "patch_raw_contract_mapping_iteration",
    "patch_raw_input_physical_superset_guard",
]
