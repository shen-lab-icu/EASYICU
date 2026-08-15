"""Science-neutral repairs authorized by host typed-product schemas."""

from __future__ import annotations

import ast
from typing import Sequence

from ..gates.typed_schema import mapping_literal_key
from ..schema import ValidationFinding


def patch_host_schema_numeric_alias(
    code: str,
    *,
    repair_findings: Sequence[ValidationFinding],
) -> str:
    """Use an authored numeric alias for a host-proven nonnumeric field."""

    supplied = [
        finding
        for finding in repair_findings
        if (finding.detail or {}).get("reason")
        == "host_schema_nonnumeric_numeric_alias"
    ]
    if len(supplied) != 1:
        return code
    detail = supplied[0].detail or {}
    raw_occurrences = detail.get("occurrences")
    sequence_name = detail.get("sequence_name")
    if not isinstance(raw_occurrences, list) or not raw_occurrences:
        return code
    if not isinstance(sequence_name, str) or not sequence_name.isidentifier():
        return code
    try:
        tree = ast.parse(code)
    except SyntaxError:
        return code

    def mapping_key(node: ast.AST, mapping_name: str) -> str | None:
        # Same matcher the gate uses; see gates.typed_schema.mapping_literal_key.
        return mapping_literal_key(node, mapping_name=mapping_name)

    allclose_pairs: set[tuple[str, frozenset[str]]] = set()
    for node in ast.walk(tree):
        if not (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and node.func.attr == "allclose"
            and len(node.args) >= 2
        ):
            continue
        mapping_names = {
            candidate.value.id
            for argument in node.args[:2]
            for candidate in ast.walk(argument)
            if isinstance(candidate, ast.Subscript)
            and isinstance(candidate.value, ast.Name)
        }
        for mapping_name in mapping_names:
            left = mapping_key(node.args[0], mapping_name)
            right = mapping_key(node.args[1], mapping_name)
            if left is not None and right is not None and left != right:
                allclose_pairs.add((mapping_name, frozenset((left, right))))

    replacements: dict[tuple[int, int, int, int], str] = {}
    alias_assignments: list[tuple[str, str, str]] = []
    literal_constants = {
        (
            int(node.lineno),
            int(node.col_offset),
            int(node.end_lineno),
            int(node.end_col_offset),
        ): node
        for node in ast.walk(tree)
        if isinstance(node, ast.Constant)
        and isinstance(node.value, str)
        and node.end_lineno is not None
        and node.end_col_offset is not None
    }
    for raw in raw_occurrences:
        if not isinstance(raw, dict):
            return code
        coordinates = (
            raw.get("line"),
            raw.get("column"),
            raw.get("end_line"),
            raw.get("end_column"),
        )
        mapping_name = raw.get("mapping_name")
        source_column = raw.get("source_column")
        alias_column = raw.get("numeric_alias_column")
        if not (
            all(
                isinstance(value, int) and not isinstance(value, bool) and value >= 0
                for value in coordinates
            )
            and isinstance(mapping_name, str)
            and mapping_name.isidentifier()
            and isinstance(source_column, str)
            and isinstance(alias_column, str)
            and source_column != alias_column
            and (mapping_name, frozenset((source_column, alias_column)))
            in allclose_pairs
        ):
            return code
        coordinate = tuple(int(value) for value in coordinates)
        literal = literal_constants.get(coordinate)
        if literal is None or literal.value != source_column:
            return code
        replacements[coordinate] = repr(alias_column)
        alias_assignments.append((mapping_name, source_column, alias_column))

    loops = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.For)
        and isinstance(node.iter, ast.Name)
        and node.iter.id == sequence_name
        and node.end_lineno is not None
    ]
    if len(loops) != 1:
        return code
    loop = loops[0]
    lines = code.splitlines(keepends=True)
    line_starts: list[int] = []
    offset = 0
    for line in lines:
        line_starts.append(offset)
        offset += len(line)

    def absolute_offset(lineno: int, utf8_column: int) -> int:
        line = lines[lineno - 1]
        char_column = len(line.encode("utf-8")[:utf8_column].decode("utf-8"))
        return line_starts[lineno - 1] + char_column

    edits = [
        (
            absolute_offset(coordinate[0], coordinate[1]),
            absolute_offset(coordinate[2], coordinate[3]),
            replacement,
        )
        for coordinate, replacement in replacements.items()
    ]
    insertion_offset = (
        line_starts[int(loop.end_lineno)]
        if int(loop.end_lineno) < len(line_starts)
        else len(code)
    )
    indentation = " " * int(loop.col_offset)
    insertion = "".join(
        f"{indentation}{mapping_name}[{source_column!r}] = "
        f"{mapping_name}[{alias_column!r}]\n"
        for mapping_name, source_column, alias_column in alias_assignments
    )
    edits.append((insertion_offset, insertion_offset, insertion))
    repaired = code
    for start, end, replacement in sorted(edits, reverse=True):
        repaired = repaired[:start] + replacement + repaired[end:]
    try:
        ast.parse(repaired)
    except SyntaxError:
        return code
    return repaired


__all__ = ["patch_host_schema_numeric_alias"]
