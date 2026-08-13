"""Host-schema-aware deterministic gates for typed tabular products."""

from __future__ import annotations

import ast
from typing import Any, Mapping

from ..schema import ValidationFinding


def _literal_string_sequence(node: ast.AST) -> list[str] | None:
    if not isinstance(node, (ast.List, ast.Tuple)):
        return None
    values: list[str] = []
    for element in node.elts:
        if not isinstance(element, ast.Constant) or not isinstance(element.value, str):
            return None
        values.append(element.value)
    return values


def _numeric_mapping_loop(
    tree: ast.Module,
    *,
    sequence_name: str,
) -> tuple[str, ast.For] | None:
    """Return the exact ``mapping[column] = numeric(frame[column])`` loop."""

    matches: list[tuple[str, ast.For]] = []
    for node in ast.walk(tree):
        if not (
            isinstance(node, ast.For)
            and isinstance(node.target, ast.Name)
            and isinstance(node.iter, ast.Name)
            and node.iter.id == sequence_name
        ):
            continue
        column_name = node.target.id
        mapping_names: set[str] = set()
        for statement in node.body:
            if not (
                isinstance(statement, ast.Assign)
                and len(statement.targets) == 1
                and isinstance(statement.targets[0], ast.Subscript)
                and isinstance(statement.targets[0].value, ast.Name)
                and isinstance(statement.targets[0].slice, ast.Name)
                and statement.targets[0].slice.id == column_name
                and isinstance(statement.value, ast.Call)
                and isinstance(statement.value.func, ast.Name)
                and statement.value.func.id in {"strict_numeric", "to_numeric"}
                and statement.value.args
                and isinstance(statement.value.args[0], ast.Subscript)
                and isinstance(statement.value.args[0].slice, ast.Name)
                and statement.value.args[0].slice.id == column_name
            ):
                continue
            mapping_names.add(statement.targets[0].value.id)
        if len(mapping_names) == 1:
            matches.append((next(iter(mapping_names)), node))
    return matches[0] if len(matches) == 1 else None


def mapping_literal_key(node: ast.AST, *, mapping_name: str) -> str | None:
    """Return the string key a chained ``mapping[key]...`` expression reads.

    Single owner: the repair that rewrites these reads must recognise exactly
    the same expression shapes as the gate that rejects them, or it will
    rewrite code the gate still refuses and miss code the gate blocks.
    """

    while True:
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute):
            node = node.func.value
            continue
        if isinstance(node, ast.Attribute):
            node = node.value
            continue
        break
    if not (
        isinstance(node, ast.Subscript)
        and isinstance(node.value, ast.Name)
        and node.value.id == mapping_name
        and isinstance(node.slice, ast.Constant)
        and isinstance(node.slice.value, str)
    ):
        return None
    return node.slice.value


def _explicit_numeric_aliases(
    tree: ast.Module,
    *,
    mapping_name: str,
) -> set[frozenset[str]]:
    aliases: set[frozenset[str]] = set()
    for node in ast.walk(tree):
        if not (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and node.func.attr == "allclose"
            and len(node.args) >= 2
        ):
            continue
        left = mapping_literal_key(node.args[0], mapping_name=mapping_name)
        right = mapping_literal_key(node.args[1], mapping_name=mapping_name)
        if left is not None and right is not None and left != right:
            aliases.add(frozenset((left, right)))
    return aliases


def host_schema_numeric_alias_findings(
    tree: ast.Module,
    resolved_input_bindings: Mapping[str, Any] | None,
) -> list[ValidationFinding]:
    """Reject coercing a host-proven nonnumeric field as a numeric measure.

    Authority is deliberately narrow: one digest-bound v3 table, a literal
    numeric-column sequence, an exact conversion loop, and a unique numeric
    sibling that the candidate itself reconciles with ``allclose``.  The host
    recognizes an authored alias; it never chooses a variable by name.
    """

    if not isinstance(resolved_input_bindings, Mapping):
        return []
    bindings = [
        value
        for value in resolved_input_bindings.values()
        if isinstance(value, Mapping)
    ]
    if len(bindings) != 1:
        return []
    contract = bindings[0].get("product_contract")
    if not isinstance(contract, Mapping):
        return []
    if str(contract.get("schema_version") or "") != "easyicu.host_typed_product.v3":
        return []
    columns_raw = contract.get("columns")
    numeric_raw = contract.get("numeric_columns")
    dtypes = contract.get("column_dtypes")
    if not (
        isinstance(columns_raw, list)
        and all(isinstance(value, str) for value in columns_raw)
        and isinstance(numeric_raw, list)
        and all(isinstance(value, str) for value in numeric_raw)
        and isinstance(dtypes, Mapping)
    ):
        return []
    columns = set(columns_raw)
    host_numeric = set(numeric_raw)
    host_nonnumeric = {
        column
        for column in columns - host_numeric
        if isinstance(dtypes.get(column), str) and str(dtypes[column]).strip()
    }
    if not host_numeric or not host_nonnumeric:
        return []

    findings: list[ValidationFinding] = []
    for assignment in ast.walk(tree):
        if not (
            isinstance(assignment, ast.Assign)
            and len(assignment.targets) == 1
            and isinstance(assignment.targets[0], ast.Name)
        ):
            continue
        values = _literal_string_sequence(assignment.value)
        if values is None or not set(values) <= columns:
            continue
        sequence_name = assignment.targets[0].id
        loop_match = _numeric_mapping_loop(tree, sequence_name=sequence_name)
        if loop_match is None:
            continue
        mapping_name, _loop = loop_match
        aliases = _explicit_numeric_aliases(tree, mapping_name=mapping_name)
        occurrences: list[dict[str, object]] = []
        elements = assignment.value.elts  # type: ignore[union-attr]
        for element in elements:
            source_column = str(element.value)
            if source_column not in host_nonnumeric:
                continue
            candidates = sorted(
                next(iter(pair - {source_column}))
                for pair in aliases
                if source_column in pair and len(pair - {source_column}) == 1
            )
            candidates = [value for value in candidates if value in host_numeric]
            if (
                len(candidates) != 1
                or element.end_lineno is None
                or element.end_col_offset is None
            ):
                continue
            occurrences.append(
                {
                    "line": int(element.lineno),
                    "column": int(element.col_offset),
                    "end_line": int(element.end_lineno),
                    "end_column": int(element.end_col_offset),
                    "mapping_name": mapping_name,
                    "source_column": source_column,
                    "numeric_alias_column": candidates[0],
                }
            )
        if occurrences:
            findings.append(
                ValidationFinding(
                    validator="mechanical_code_preflight",
                    severity="error",
                    message=(
                        "A host-proven nonnumeric typed-product field is included "
                        "in an authored numeric-conversion loop even though the "
                        "candidate explicitly reconciles it to a unique numeric "
                        "field. Use that authored numeric alias for calculations."
                    ),
                    detail={
                        "reason": "host_schema_nonnumeric_numeric_alias",
                        "input_key": str(next(iter(resolved_input_bindings))),
                        "sequence_name": sequence_name,
                        "occurrences": occurrences,
                    },
                )
            )
    return findings


__all__ = ["host_schema_numeric_alias_findings"]
