"""Science-neutral repairs for stable host-helper result contracts."""

from __future__ import annotations

import ast
from typing import Sequence

from ..schema import ValidationFinding

_HELPER_MODULE = "easyicu.research_agent.methods.descriptive_inputs"


def patch_closed_counts_level_column(
    code: str,
    *,
    findings: Sequence[ValidationFinding],
) -> str:
    """Replace only proven closed-counts ``table.index`` level reads."""

    lines = {
        int((finding.detail or {}).get("line") or 0)
        for finding in findings
        if (finding.detail or {}).get("reason")
        == "closed_counts_table_index_used_as_levels"
    }
    lines.discard(0)
    if not lines:
        return code
    try:
        tree = ast.parse(code)
    except SyntaxError:
        return code

    direct_names: set[str] = set()
    module_names: set[str] = set()
    for node in tree.body:
        if (
            isinstance(node, ast.ImportFrom)
            and node.level == 0
            and node.module == _HELPER_MODULE
        ):
            direct_names.update(
                alias.asname or alias.name
                for alias in node.names
                if alias.name == "closed_categorical_counts"
            )
        elif isinstance(node, ast.Import):
            module_names.update(
                alias.asname or alias.name
                for alias in node.names
                if alias.name == _HELPER_MODULE
            )

    def _call_name(node: ast.AST) -> str:
        if isinstance(node, ast.Name):
            return node.id
        if isinstance(node, ast.Attribute):
            prefix = _call_name(node.value)
            return f"{prefix}.{node.attr}" if prefix else node.attr
        return ""

    helper_calls = direct_names | {
        f"{name}.closed_categorical_counts" for name in module_names
    }
    result_names: set[str] = set()
    table_to_result: dict[str, str] = {}
    assignment_counts: dict[str, int] = {}
    for node in ast.walk(tree):
        if not isinstance(node, ast.Assign) or len(node.targets) != 1:
            continue
        target = node.targets[0]
        if not isinstance(target, ast.Name):
            continue
        assignment_counts[target.id] = assignment_counts.get(target.id, 0) + 1
        if isinstance(node.value, ast.Call) and _call_name(node.value.func) in (
            helper_calls
        ):
            result_names.add(target.id)
        elif (
            isinstance(node.value, ast.Attribute)
            and node.value.attr == "table"
            and isinstance(node.value.value, ast.Name)
        ):
            table_to_result[target.id] = node.value.value.id

    offsets = [0]
    for source_line in code.splitlines(keepends=True):
        offsets.append(offsets[-1] + len(source_line))

    replacements: list[tuple[int, int, str]] = []
    for node in ast.walk(tree):
        if not (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Name)
            and node.func.id == "list"
            and len(node.args) == 1
            and not node.keywords
            and isinstance(node.args[0], ast.Attribute)
            and node.args[0].attr == "index"
            and int(node.args[0].lineno) in lines
        ):
            continue
        index_node = node.args[0]
        table_expression = index_node.value
        if isinstance(table_expression, ast.Name):
            result_name = table_to_result.get(table_expression.id, "")
            if assignment_counts.get(table_expression.id) != 1:
                continue
        elif (
            isinstance(table_expression, ast.Attribute)
            and table_expression.attr == "table"
            and isinstance(table_expression.value, ast.Name)
        ):
            result_name = table_expression.value.id
        else:
            continue
        if result_name not in result_names or assignment_counts.get(result_name) != 1:
            continue
        source = ast.get_source_segment(code, table_expression)
        if (
            not source
            or index_node.end_lineno is None
            or index_node.end_col_offset is None
        ):
            continue
        start = offsets[index_node.lineno - 1] + index_node.col_offset
        end = offsets[index_node.end_lineno - 1] + index_node.end_col_offset
        replacements.append((start, end, f'{source}["level"]'))

    if not replacements or len(replacements) != len(lines):
        return code
    repaired = code
    for start, end, replacement in sorted(replacements, reverse=True):
        repaired = repaired[:start] + replacement + repaired[end:]
    try:
        ast.parse(repaired)
    except SyntaxError:
        return code
    return repaired


__all__ = ["patch_closed_counts_level_column"]
