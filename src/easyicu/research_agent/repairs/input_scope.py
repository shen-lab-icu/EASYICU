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
# Keyed on the host schema path rather than on any authored wording: code that
# asserts the wrong container type reports the failure in its own prose
# ("... is missing", "... is not a list"), which is not a stable trigger.
_RAW_CONTRACT_SHAPE_ERROR = re.compile(
    r"raw_input_contracts(?:\.|\[\s*['\"])contracts",
    re.IGNORECASE,
)
_PLANNER_RAW_INPUT_PROJECTION_ERROR = "indexerror: list index out of range"


def _planner_declared_input_names(tree: ast.AST) -> set[str]:
    """Return locals bound directly to ``planner_declared_inputs``."""

    names: set[str] = set()
    for node in ast.walk(tree):
        if not (
            isinstance(node, ast.Assign)
            and len(node.targets) == 1
            and isinstance(node.targets[0], ast.Name)
        ):
            continue
        value = node.value
        is_declared_input_lookup = (
            isinstance(value, ast.Subscript)
            and isinstance(value.slice, ast.Constant)
            and value.slice.value == "planner_declared_inputs"
        ) or (
            isinstance(value, ast.Call)
            and isinstance(value.func, ast.Attribute)
            and value.func.attr == "get"
            and value.args
            and isinstance(value.args[0], ast.Constant)
            and value.args[0].value == "planner_declared_inputs"
        )
        if is_declared_input_lookup:
            names.add(node.targets[0].id)
    return names


def patch_planner_declared_raw_input_projection(code: str, run_log: str) -> str:
    """Project raw coordinates without indexing a nonexistent ``kind:`` suffix.

    The resolved-input contract is deliberately mixed: typed products use
    ``kind:name`` while raw columns are bare names. Generated code has
    occasionally filtered out known product prefixes and then evaluated
    ``item.split(":", 1)[1]`` for the remaining bare names. This exact shape
    raises before analysis starts. Replace one unambiguous set-comprehension
    with the schema-owned projection (tokens without a colon), preserving every
    declared raw name and introducing no data or scientific choice.
    """

    if _PLANNER_RAW_INPUT_PROJECTION_ERROR not in str(run_log or "").lower():
        return code
    try:
        tree = ast.parse(code)
    except SyntaxError:
        return code

    declared_input_names = _planner_declared_input_names(tree)
    candidates: list[tuple[ast.SetComp, str, str]] = []
    for node in ast.walk(tree):
        if not (
            isinstance(node, ast.SetComp)
            and len(node.generators) == 1
            and isinstance(node.generators[0].target, ast.Name)
            and isinstance(node.generators[0].iter, ast.Name)
            and node.generators[0].iter.id in declared_input_names
            and isinstance(node.elt, ast.Subscript)
            and isinstance(node.elt.slice, ast.Constant)
            and node.elt.slice.value == 1
            and isinstance(node.elt.value, ast.Call)
            and isinstance(node.elt.value.func, ast.Attribute)
            and node.elt.value.func.attr == "split"
            and isinstance(node.elt.value.func.value, ast.Name)
            and node.elt.value.func.value.id == node.generators[0].target.id
            and len(node.elt.value.args) == 2
            and isinstance(node.elt.value.args[0], ast.Constant)
            and node.elt.value.args[0].value == ":"
            and isinstance(node.elt.value.args[1], ast.Constant)
            and node.elt.value.args[1].value == 1
        ):
            continue
        candidates.append(
            (
                node,
                node.generators[0].target.id,
                node.generators[0].iter.id,
            )
        )
    if len(candidates) != 1:
        return code

    candidate, item_name, iterable_name = candidates[0]
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

    start = _absolute_offset(candidate.lineno, candidate.col_offset)
    end = _absolute_offset(candidate.end_lineno, candidate.end_col_offset)
    replacement = (
        f"{{{item_name} for {item_name} in {iterable_name} "
        f"if ':' not in {item_name}}}"
    )
    repaired = code[:start] + replacement + code[end:]
    try:
        ast.parse(repaired)
    except SyntaxError:
        return code
    return repaired


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


def _contract_mapping_names(tree: ast.AST) -> set[str]:
    """Return names bound to the host's ``contracts`` mapping."""

    names: set[str] = set()
    for node in ast.walk(tree):
        if (
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
            names.add(node.targets[0].id)
    return names


def patch_raw_contract_list_type_assertion(code: str, run_log: str) -> str:
    """Accept the host's column-keyed mapping where code asserted a list.

    ``raw_input_contracts.contracts`` is a JSON object keyed by resolved
    column. Generated code sometimes guards it with ``isinstance(contracts,
    list)`` and then rebuilds a by-column mapping by iterating records. The
    guard fires first, so the step dies reporting the contracts as *missing*
    when they are present under the documented shape — the previous
    ``AttributeError`` adapter never sees the traceback.

    This repair rewrites only how the container is read: the ``list`` type
    assertion becomes ``dict`` and the matching iteration walks ``.values()``.
    It never touches the cohort, variables, model, estimand, or any numeric
    value, and it keeps every downstream per-column presence check intact, so
    a genuinely absent contract still fails closed.
    """

    if _RAW_CONTRACT_SHAPE_ERROR.search(str(run_log or "")) is None:
        return code
    try:
        tree = ast.parse(code)
    except SyntaxError:
        return code

    mapping_names = _contract_mapping_names(tree)
    if not mapping_names:
        return code

    # The list assertion: exactly one ``isinstance(<contracts>, list)``.
    type_assertions: list[ast.Name] = []
    for node in ast.walk(tree):
        if (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Name)
            and node.func.id == "isinstance"
            and len(node.args) == 2
            and not node.keywords
            and isinstance(node.args[0], ast.Name)
            and node.args[0].id in mapping_names
            and isinstance(node.args[1], ast.Name)
            and node.args[1].id == "list"
        ):
            type_assertions.append(node.args[1])
    if len(type_assertions) != 1:
        return code

    # The matching record iteration, recognised only when the loop body reads
    # a per-contract ``column`` — the shape that re-keys an already-keyed map.
    iterations: list[ast.Name] = []
    for node in ast.walk(tree):
        if not (
            isinstance(node, ast.For)
            and isinstance(node.target, ast.Name)
            and isinstance(node.iter, ast.Name)
            and node.iter.id in mapping_names
        ):
            continue
        item_name = node.target.id
        reads_column = any(
            (
                isinstance(child, ast.Call)
                and isinstance(child.func, ast.Attribute)
                and child.func.attr == "get"
                and isinstance(child.func.value, ast.Name)
                and child.func.value.id == item_name
                and child.args
                and isinstance(child.args[0], ast.Constant)
                and child.args[0].value == "column"
            )
            or (
                isinstance(child, ast.Subscript)
                and isinstance(child.value, ast.Name)
                and child.value.id == item_name
                and isinstance(child.slice, ast.Constant)
                and child.slice.value == "column"
            )
            for statement in node.body
            for child in ast.walk(statement)
        )
        if reads_column:
            iterations.append(node.iter)
    if len(iterations) != 1:
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

    replacements = [
        (
            _absolute_offset(type_assertions[0].lineno, type_assertions[0].col_offset),
            _absolute_offset(
                type_assertions[0].end_lineno, type_assertions[0].end_col_offset
            ),
            "dict",
        ),
        (
            _absolute_offset(iterations[0].lineno, iterations[0].col_offset),
            _absolute_offset(iterations[0].end_lineno, iterations[0].end_col_offset),
            f"{iterations[0].id}.values()",
        ),
    ]
    repaired = code
    for start, end, replacement in sorted(replacements, reverse=True):
        repaired = repaired[:start] + replacement + repaired[end:]
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
    "patch_planner_declared_raw_input_projection",
    "patch_raw_contract_document_fallback",
    "patch_raw_contract_list_type_assertion",
    "patch_raw_contract_mapping_iteration",
    "patch_raw_input_physical_superset_guard",
]
