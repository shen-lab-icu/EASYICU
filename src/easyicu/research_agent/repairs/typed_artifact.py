"""Deterministic adapters for digest-bound non-tabular typed products."""

from __future__ import annotations

import ast
import copy
import re
from collections.abc import Mapping
from typing import Any

_JSON_LOADER_ANCHOR = """    if suffix == ".csv":
        return pd.read_csv(path)
    raise ValueError(f"Unsupported tabular input format: {path}")
"""

_JSON_LOADER_REPLACEMENT = """    if suffix == ".csv":
        return pd.read_csv(path)
    if suffix == ".json":
        with path.open("r", encoding="utf-8") as handle:
            payload = json.load(handle)
        if not isinstance(payload, dict):
            raise ValueError(f"Bound JSON input must contain an object: {path}")
        return pd.DataFrame([payload])
    raise ValueError(f"Unsupported tabular input format: {path}")
"""

_SCHEMA_ANCHOR = """    expected_columns = contract.get("columns")
    if not isinstance(expected_columns, list) or not expected_columns:
        raise ValueError(f"Missing product_contract.columns for {expected_key}")
"""

_SCHEMA_REPLACEMENT = """    expected_columns = contract.get("columns")
    if (
        (not isinstance(expected_columns, list) or not expected_columns)
        and path.suffix.lower() == ".json"
        and contract.get("schema_version") == "easyicu.host_typed_product.v1"
    ):
        expected_columns = list(table.columns)
    elif not isinstance(expected_columns, list) or not expected_columns:
        raise ValueError(f"Missing product_contract.columns for {expected_key}")
"""


def patch_resolved_json_document_adapter(code: str, run_log: str) -> str:
    """Adapt one proven v1 JSON product without weakening table contracts.

    The repair activates only after the runner reports that a digest-bound
    ``.json`` artifact reached the generated tabular loader.  It preserves the
    existing CSV/Parquet paths and permits a JSON *object* only when the host
    product contract is the non-tabular v1 shape.  Newer tabular contracts
    still require their exact declared columns.
    """

    lowered = (run_log or "").lower()
    if "unsupported tabular input format:" not in lowered or ".json" not in lowered:
        return code
    if code.count(_JSON_LOADER_ANCHOR) != 1 or code.count(_SCHEMA_ANCHOR) != 1:
        return code

    repaired = code.replace(
        _JSON_LOADER_ANCHOR,
        _JSON_LOADER_REPLACEMENT,
        1,
    ).replace(
        _SCHEMA_ANCHOR,
        _SCHEMA_REPLACEMENT,
        1,
    )
    try:
        ast.parse(repaired)
    except SyntaxError:
        return code
    return repaired


_NON_TABULAR_CONTRACT_ERROR = re.compile(
    r"^(?:ValueError:\s*)?Typed input (?P<input_key>[^\s{}]+) "
    r"lacks a tabular product contract\s*$",
    re.MULTILINE,
)


def _source_offsets(code: str) -> tuple[list[str], list[int]]:
    lines = code.splitlines(keepends=True)
    starts: list[int] = []
    cursor = 0
    for line in lines:
        starts.append(cursor)
        cursor += len(line)
    return lines, starts


def _node_span(
    node: ast.AST, *, lines: list[str], starts: list[int]
) -> tuple[int, int] | None:
    coordinates = (
        getattr(node, "lineno", None),
        getattr(node, "col_offset", None),
        getattr(node, "end_lineno", None),
        getattr(node, "end_col_offset", None),
    )
    if not all(isinstance(value, int) for value in coordinates):
        return None
    lineno, col, end_lineno, end_col = coordinates
    assert isinstance(lineno, int)
    assert isinstance(col, int)
    assert isinstance(end_lineno, int)
    assert isinstance(end_col, int)
    if not (1 <= lineno <= len(lines) and 1 <= end_lineno <= len(lines)):
        return None
    return starts[lineno - 1] + col, starts[end_lineno - 1] + end_col


def _loaded_binding_names(tree: ast.Module, input_key: str) -> tuple[str, str] | None:
    matches: list[tuple[str, str]] = []
    for node in tree.body:
        if not (
            isinstance(node, ast.Assign)
            and len(node.targets) == 1
            and isinstance(node.targets[0], (ast.Tuple, ast.List))
            and len(node.targets[0].elts) == 2
            and all(isinstance(elt, ast.Name) for elt in node.targets[0].elts)
            and isinstance(node.value, ast.Call)
            and isinstance(node.value.func, ast.Name)
            and node.value.func.id == "load_bound_input"
            and len(node.value.args) == 2
            and isinstance(node.value.args[1], ast.Constant)
            and node.value.args[1].value == input_key
        ):
            continue
        first, second = node.targets[0].elts
        assert isinstance(first, ast.Name) and isinstance(second, ast.Name)
        matches.append((first.id, second.id))
    return matches[0] if len(matches) == 1 else None


def _names(node: ast.AST) -> set[str]:
    return {child.id for child in ast.walk(node) if isinstance(child, ast.Name)}


def _assigned_names(node: ast.AST) -> set[str]:
    names: set[str] = set()
    for child in ast.walk(node):
        if isinstance(child, ast.Name) and isinstance(child.ctx, (ast.Store, ast.Del)):
            names.add(child.id)
    return names


def _flatten_boolean(node: ast.AST, op_type: type[ast.operator]) -> list[ast.AST]:
    if isinstance(node, ast.BinOp) and isinstance(node.op, op_type):
        return [
            *_flatten_boolean(node.left, op_type),
            *_flatten_boolean(node.right, op_type),
        ]
    return [node]


def _rebuild_boolean(parts: list[ast.AST], op_type: type[ast.operator]) -> ast.AST:
    result = parts[0]
    for part in parts[1:]:
        result = ast.BinOp(left=result, op=op_type(), right=part)
    return result


class _DropCompanionMaskTerms(ast.NodeTransformer):
    def __init__(self, tainted: set[str]) -> None:
        self.tainted = tainted
        self.changed = False
        self.invalid = False

    def visit_BinOp(self, node: ast.BinOp) -> ast.AST:
        node = copy.deepcopy(node)
        if isinstance(node.op, (ast.BitAnd, ast.BitOr)):
            op_type = type(node.op)
            parts = _flatten_boolean(node, op_type)
            kept: list[ast.AST] = []
            for part in parts:
                part = self.visit(part)
                part_names = _names(part)
                if part_names & self.tainted:
                    if part_names - self.tainted:
                        self.invalid = True
                        return node
                    self.changed = True
                    continue
                kept.append(part)
            if not kept:
                self.invalid = True
                return node
            return ast.copy_location(_rebuild_boolean(kept, op_type), node)
        return self.generic_visit(node)


def _trusted_non_tabular_binding(
    resolved_input_bindings: Mapping[str, Any], input_key: str
) -> bool:
    binding = resolved_input_bindings.get(input_key)
    if not isinstance(binding, Mapping):
        return False
    contract = binding.get("product_contract")
    relative_path = binding.get("relative_path")
    return bool(
        isinstance(contract, Mapping)
        and contract.get("schema_version") == "easyicu.host_typed_product.v1"
        and not contract.get("columns")
        and isinstance(relative_path, str)
        and relative_path.lower().endswith(".json")
        and binding.get("evidence_kind") in {"log", "metadata", "json"}
    )


def patch_non_tabular_companion_row_gate(
    code: str,
    run_log: str,
    *,
    resolved_input_bindings: Mapping[str, Any] | None,
) -> str:
    """Keep a digest-bound summary document out of patient-level selection.

    The runtime error alone is not authority for this structural change.  The
    repair requires the host-resolved binding to prove that the named product
    is a v1 JSON document with no tabular columns.  It then removes only the
    scaffold that tries to treat that document as a row-keyed dataframe.  The
    locked cohort, its row-level validity predicates, model formula, covariates,
    outcomes, constants, and outputs are left unchanged.
    """

    match = _NON_TABULAR_CONTRACT_ERROR.search(run_log or "")
    if match is None or not isinstance(resolved_input_bindings, Mapping):
        return code
    input_key = match.group("input_key")
    if not _trusted_non_tabular_binding(resolved_input_bindings, input_key):
        return code
    try:
        tree = ast.parse(code)
    except SyntaxError:
        return code
    loaded_names = _loaded_binding_names(tree, input_key)
    if loaded_names is None:
        return code
    binding_name, object_name = loaded_names

    contract_assignments: list[ast.Assign] = []
    contract_names: set[str] = set()
    for node in tree.body:
        if not (
            isinstance(node, ast.Assign)
            and isinstance(node.value, ast.Call)
            and isinstance(node.value.func, ast.Name)
            and node.value.func.id == "require_tabular_contract"
            and len(node.value.args) == 2
            and isinstance(node.value.args[0], ast.Name)
            and node.value.args[0].id == binding_name
            and isinstance(node.value.args[1], ast.Constant)
            and node.value.args[1].value == input_key
        ):
            continue
        contract_assignments.append(node)
        contract_names.update(_assigned_names(node))
    if len(contract_assignments) != 1 or not contract_names:
        return code

    tainted = {object_name, *contract_names}
    removable: list[ast.stmt] = [contract_assignments[0]]
    merge_replacement: tuple[ast.Assign, str] | None = None
    boolean_rewrites: dict[ast.stmt, ast.AST] = {}

    # Resolve the small, module-level dataframe scaffold to a fixed point.
    changed = True
    while changed:
        changed = False
        for statement in tree.body:
            if statement in removable:
                continue
            if isinstance(statement, ast.Assign):
                value_names = _names(statement.value)
                target_names = _assigned_names(statement)
                if not (value_names & tainted):
                    continue
                transformer = _DropCompanionMaskTerms(tainted)
                rewritten = transformer.visit(copy.deepcopy(statement))
                if transformer.invalid:
                    return code
                if transformer.changed and not transformer.invalid:
                    ast.fix_missing_locations(rewritten)
                    boolean_rewrites[statement] = rewritten
                    continue
                if (
                    isinstance(statement.value, ast.Call)
                    and isinstance(statement.value.func, ast.Attribute)
                    and statement.value.func.attr == "merge"
                    and isinstance(statement.value.func.value, ast.Name)
                    and any(_names(arg) & tainted for arg in statement.value.args)
                    and len(target_names) == 1
                ):
                    base_name = statement.value.func.value.id
                    target_name = next(iter(target_names))
                    candidate = f"{target_name} = {base_name}.copy()"
                    if (
                        merge_replacement is not None
                        and merge_replacement[0] is not statement
                    ):
                        return code
                    merge_replacement = (statement, candidate)
                    continue
                # Keep the binding-consumption record; only its misleading
                # row_count key is rewritten below.
                if object_name in value_names and not target_names:
                    continue
                removable.append(statement)
                new_names = target_names - tainted
                if new_names:
                    tainted.update(new_names)
                    changed = True
            elif isinstance(statement, ast.If) and _names(statement.test) & tainted:
                removable.append(statement)
                new_names = _assigned_names(statement) - tainted
                if new_names:
                    tainted.update(new_names)
                    changed = True
    if merge_replacement is None:
        return code

    edits: list[tuple[int, int, str]] = []
    lines, starts = _source_offsets(code)
    for statement in removable:
        span = _node_span(statement, lines=lines, starts=starts)
        if span is None:
            return code
        start, end = span
        if end < len(code) and code[end : end + 1] == "\n":
            end += 1
        edits.append((start, end, ""))

    merge_statement, merge_source = merge_replacement
    merge_span = _node_span(merge_statement, lines=lines, starts=starts)
    if merge_span is None:
        return code
    edits.append((*merge_span, merge_source))

    # A non-tabular document has no row count.  Preserve evidence that it was
    # loaded while naming the observed mapping cardinality honestly.
    dict_key_edits = 0
    for node in ast.walk(tree):
        if not isinstance(node, ast.Dict):
            continue
        input_key_match = any(
            isinstance(key, ast.Constant)
            and key.value == "input_key"
            and isinstance(value, ast.Constant)
            and value.value == input_key
            for key, value in zip(node.keys, node.values)
        )
        if not input_key_match:
            continue
        for key, value in zip(node.keys, node.values):
            if not (
                isinstance(key, ast.Constant)
                and key.value == "row_count"
                and object_name in _names(value)
            ):
                continue
            span = _node_span(key, lines=lines, starts=starts)
            if span is None:
                return code
            edits.append((*span, '"document_key_count"'))
            dict_key_edits += 1
    if dict_key_edits != 1:
        return code

    # Remove the now-untrusted companion selector from conjunctions and
    # disjunctions, but refuse mixed expressions where it cannot be detached
    # as one complete boolean term.
    rewritten_statements = 0
    for statement, rewritten in boolean_rewrites.items():
        if statement in removable or statement is merge_statement:
            return code
        span = _node_span(statement, lines=lines, starts=starts)
        if span is None:
            return code
        edits.append((*span, ast.unparse(rewritten)))
        rewritten_statements += 1
    if rewritten_statements < 1:
        return code

    ordered = sorted(edits)
    if any(
        end > next_start
        for (_, end, _), (next_start, _, _) in zip(ordered, ordered[1:])
    ):
        return code
    repaired = code
    for start, end, replacement in reversed(ordered):
        repaired = repaired[:start] + replacement + repaired[end:]
    try:
        ast.parse(repaired)
    except SyntaxError:
        return code
    return repaired


__all__ = [
    "patch_non_tabular_companion_row_gate",
    "patch_resolved_json_document_adapter",
]
