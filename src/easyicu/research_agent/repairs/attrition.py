"""Case-neutral source repairs for host-verified cohort attrition labels."""

from __future__ import annotations

import ast
import re
from typing import Any, Sequence

_ATTRITION_RULE_ID_RE = re.compile(
    r"^(?P<kind>include|exclude)_(?P<order>[0-9]{2})_[a-z0-9]+(?:_[a-z0-9]+)*$"
)


def patch_attrition_rule_id_canonicalization(
    code: str,
    *,
    expected_rule_ids: Sequence[Any],
    reported_rule_ids: Sequence[Any],
) -> str:
    """Rename proven attrition labels to their Planner-owned canonical ids.

    The integrity gate has already replayed every locked predicate and checked
    the ordered remaining/exclusion counts before it emits the finding that
    authorizes this repair. This transform therefore changes labels only: the
    boundary row must already be ``universe`` and each predicate row must retain
    the same include/exclude role and two-digit ordinal. Arbitrary aliases,
    reordered predicates, terminal rows, dynamic labels, and partial literal
    coverage all fail closed.
    """

    if not all(isinstance(value, str) for value in expected_rule_ids) or not all(
        isinstance(value, str) for value in reported_rule_ids
    ):
        return code
    expected = list(expected_rule_ids)
    reported = list(reported_rule_ids)
    if (
        len(expected) < 2
        or len(expected) != len(reported)
        or expected[0] != "universe"
        or reported[0] != "universe"
        or len(set(expected)) != len(expected)
        or len(set(reported)) != len(reported)
    ):
        return code

    replacements_by_value: dict[str, str] = {}
    for expected_id, reported_id in zip(expected[1:], reported[1:], strict=True):
        expected_match = _ATTRITION_RULE_ID_RE.fullmatch(expected_id)
        reported_match = _ATTRITION_RULE_ID_RE.fullmatch(reported_id)
        if (
            expected_match is None
            or reported_match is None
            or expected_match.group("kind") != reported_match.group("kind")
            or expected_match.group("order") != reported_match.group("order")
        ):
            return code
        if expected_id != reported_id:
            replacements_by_value[reported_id] = expected_id
    if not replacements_by_value:
        return code

    try:
        tree = ast.parse(code)
    except SyntaxError:
        return code

    docstring_nodes: set[int] = set()
    parent_by_node_id: dict[int, ast.AST] = {}
    for parent in ast.walk(tree):
        for child in ast.iter_child_nodes(parent):
            parent_by_node_id[id(child)] = parent
    for owner in ast.walk(tree):
        if not isinstance(
            owner,
            (ast.Module, ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef),
        ):
            continue
        if not owner.body:
            continue
        first = owner.body[0]
        if (
            isinstance(first, ast.Expr)
            and isinstance(first.value, ast.Constant)
            and isinstance(first.value.value, str)
        ):
            docstring_nodes.add(id(first.value))

    literal_nodes: dict[str, list[ast.Constant]] = {
        value: [] for value in replacements_by_value
    }
    for node in ast.walk(tree):
        if (
            isinstance(node, ast.Constant)
            and isinstance(node.value, str)
            and node.value in literal_nodes
            and id(node) not in docstring_nodes
        ):
            literal_nodes[node.value].append(node)
    if any(not nodes for nodes in literal_nodes.values()):
        return code

    def is_label_only_literal(node: ast.Constant) -> bool:
        parent = parent_by_node_id.get(id(node))
        if isinstance(parent, (ast.List, ast.Tuple, ast.Set)):
            return node in parent.elts
        if isinstance(parent, ast.Dict):
            return node in parent.values
        if isinstance(parent, ast.Compare):
            return node is parent.left or node in parent.comparators
        return False

    if any(
        not is_label_only_literal(node)
        for nodes in literal_nodes.values()
        for node in nodes
    ):
        return code

    lines = code.splitlines(keepends=True)
    line_starts: list[int] = []
    offset = 0
    for line in lines:
        line_starts.append(offset)
        offset += len(line)

    def absolute_offset(lineno: int, utf8_col: int) -> int:
        line = lines[lineno - 1]
        char_col = len(line.encode("utf-8")[:utf8_col].decode("utf-8"))
        return line_starts[lineno - 1] + char_col

    replacements: list[tuple[int, int, str]] = []
    for reported_id, nodes in literal_nodes.items():
        for node in nodes:
            if not all(
                isinstance(value, int)
                for value in (
                    node.lineno,
                    node.col_offset,
                    node.end_lineno,
                    node.end_col_offset,
                )
            ):
                return code
            replacements.append(
                (
                    absolute_offset(node.lineno, node.col_offset),
                    absolute_offset(node.end_lineno, node.end_col_offset),
                    repr(replacements_by_value[reported_id]),
                )
            )

    repaired = code
    for start, end, replacement in sorted(replacements, reverse=True):
        repaired = repaired[:start] + replacement + repaired[end:]
    try:
        ast.parse(repaired)
    except SyntaxError:
        return code
    return repaired if repaired != code else code
