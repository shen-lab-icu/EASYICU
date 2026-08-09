"""Representation-only repairs for generated clustering summaries."""

from __future__ import annotations

import ast
from collections.abc import Sequence
from typing import Any, Optional


_COUNT_KEYS = frozenset({"n_clusters", "cluster_count"})


def _mapping_items(node: ast.Dict) -> Optional[dict[str, ast.expr]]:
    items: dict[str, ast.expr] = {}
    for key, value in zip(node.keys, node.values, strict=True):
        if not (
            isinstance(key, ast.Constant)
            and isinstance(key.value, str)
            and key.value not in items
        ):
            return None
        items[key.value] = value
    return items


def _finding_requests_cluster_count_alias(findings: Sequence[Any]) -> bool:
    matching = 0
    for finding in findings:
        validator = getattr(finding, "validator", None)
        severity = getattr(finding, "severity", None)
        message = getattr(finding, "message", None)
        detail = getattr(finding, "detail", None)
        if isinstance(finding, dict):
            validator = finding.get("validator")
            severity = finding.get("severity")
            message = finding.get("message")
            detail = finding.get("detail")
        if not (
            validator == "step_contract"
            and severity == "error"
            and isinstance(detail, dict)
            and set(detail.get("required_keys") or ()) == _COUNT_KEYS
            and "clustering summary" in str(message or "").lower()
        ):
            continue
        matching += 1
    return matching == 1


def _dict_assignments(tree: ast.AST) -> dict[str, list[ast.Dict]]:
    assignments: dict[str, list[ast.Dict]] = {}
    for node in ast.walk(tree):
        if not (
            isinstance(node, ast.Assign)
            and len(node.targets) == 1
            and isinstance(node.targets[0], ast.Name)
            and isinstance(node.value, ast.Dict)
        ):
            continue
        assignments.setdefault(node.targets[0].id, []).append(node.value)
    return assignments


def _is_step_summary_dump(tree: ast.AST, variable: str) -> bool:
    matches = 0
    for node in ast.walk(tree):
        if not isinstance(node, ast.With):
            continue
        writes_step_summary = any(
            isinstance(child, ast.Constant)
            and child.value == "step_summary.json"
            for item in node.items
            for child in ast.walk(item.context_expr)
        )
        if not writes_step_summary:
            continue
        for child in ast.walk(node):
            if not (
                isinstance(child, ast.Call)
                and isinstance(child.func, ast.Attribute)
                and isinstance(child.func.value, ast.Name)
                and child.func.value.id == "json"
                and child.func.attr == "dump"
                and child.args
                and isinstance(child.args[0], ast.Name)
                and child.args[0].id == variable
            ):
                continue
            matches += 1
    return matches == 1


def _same_expression(left: ast.expr, right: ast.expr) -> bool:
    return ast.dump(left, include_attributes=False) == ast.dump(
        right,
        include_attributes=False,
    )


def patch_cluster_count_summary_alias(code: str, findings: Sequence[Any]) -> str:
    """Surface a cluster count already proven by three generated artefacts.

    The repair adds no calculation and makes no selection. It acts only when
    the top-level summary value, the selected-k manifest, and the registered
    ``cluster_count`` statistic all contain the same AST expression and the
    summary is uniquely written to ``step_summary.json``.
    """

    if not _finding_requests_cluster_count_alias(findings):
        return code
    try:
        tree = ast.parse(code)
    except SyntaxError:
        return code

    assignments = _dict_assignments(tree)
    summaries: list[tuple[ast.Dict, dict[str, ast.expr]]] = []
    for variable, nodes in assignments.items():
        if len(nodes) != 1 or not _is_step_summary_dump(tree, variable):
            continue
        items = _mapping_items(nodes[0])
        if items is None or "primary_selected_n_clusters" not in items:
            continue
        summaries.append((nodes[0], items))
    if len(summaries) != 1:
        return code
    summary_node, summary_items = summaries[0]
    if _COUNT_KEYS & set(summary_items):
        return code

    selected_expression = summary_items["primary_selected_n_clusters"]
    selection_reference = summary_items.get("cluster_selection")
    if not isinstance(selection_reference, ast.Name):
        return code
    selection_nodes = assignments.get(selection_reference.id, [])
    if len(selection_nodes) != 1:
        return code
    selection_items = _mapping_items(selection_nodes[0])
    if selection_items is None or not _same_expression(
        selection_items.get("selected_n_clusters", ast.Constant(None)),
        selected_expression,
    ):
        return code
    required_selection_keys = {
        "criterion",
        "selection_rule",
        "direction",
        "selected_n_clusters",
        "candidates",
    }
    if not required_selection_keys <= set(selection_items):
        return code

    statistic_matches = 0
    for nodes in assignments.values():
        for node in nodes:
            items = _mapping_items(node)
            if items is None:
                continue
            name = items.get("name")
            value = items.get("value")
            if not (
                isinstance(name, ast.Constant)
                and name.value == "cluster_count"
                and value is not None
                and _same_expression(value, selected_expression)
            ):
                continue
            statistic_matches += 1
    if statistic_matches != 1:
        return code

    output_files = summary_items.get("output_files")
    if not isinstance(output_files, ast.Dict):
        return code
    output_items = _mapping_items(output_files)
    registered_count = (
        output_items.get("statistic:cluster_count") if output_items else None
    )
    if not (
        isinstance(registered_count, ast.Constant)
        and registered_count.value == "cluster_count.json"
    ):
        return code

    expression_source = ast.get_source_segment(code, selected_expression)
    primary_index = next(
        (
            index
            for index, key in enumerate(summary_node.keys)
            if isinstance(key, ast.Constant)
            and key.value == "primary_selected_n_clusters"
        ),
        None,
    )
    if primary_index is None or not expression_source:
        return code
    following_key = (
        summary_node.keys[primary_index + 1]
        if primary_index + 1 < len(summary_node.keys)
        else None
    )
    insertion_line = (
        following_key.lineno
        if isinstance(following_key, ast.expr)
        else summary_node.end_lineno
    )
    key_node = summary_node.keys[primary_index]
    if not (
        isinstance(insertion_line, int)
        and isinstance(key_node, ast.expr)
        and isinstance(key_node.col_offset, int)
    ):
        return code
    lines = code.splitlines(keepends=True)
    if not (1 <= insertion_line <= len(lines)):
        return code
    insertion_offset = sum(len(line) for line in lines[: insertion_line - 1])
    injected = (
        " " * key_node.col_offset
        + f'"cluster_count": {expression_source},\n'
    )
    repaired = code[:insertion_offset] + injected + code[insertion_offset:]
    try:
        ast.parse(repaired)
    except SyntaxError:
        return code
    return repaired


__all__ = ["patch_cluster_count_summary_alias"]
