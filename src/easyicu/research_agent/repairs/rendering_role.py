"""Fail-safe repair for typed analysis-role selection in figure renderers."""

from __future__ import annotations

import ast

_ERROR = (
    "Could not identify exactly one primary and one complete-case row "
    "from robustness_summary.restriction"
)
_PRIMARY_LABELS = {"primary", "full", "none", "all"}
_COMPLETE_CASE_LABELS = {"complete_case", "complete case", "complete-case"}


def _assignment_value(tree: ast.AST, name: str) -> ast.AST | None:
    values: list[ast.AST] = []
    for node in ast.walk(tree):
        if (
            isinstance(node, ast.Assign)
            and len(node.targets) == 1
            and isinstance(node.targets[0], ast.Name)
            and node.targets[0].id == name
        ):
            values.append(node.value)
        elif (
            isinstance(node, ast.AnnAssign)
            and isinstance(node.target, ast.Name)
            and node.target.id == name
            and node.value is not None
        ):
            values.append(node.value)
    return values[0] if len(values) == 1 else None


def _literal_strings(node: ast.AST | None) -> set[str] | None:
    if not isinstance(node, (ast.List, ast.Tuple, ast.Set)):
        return None
    if any(
        not isinstance(item, ast.Constant) or not isinstance(item.value, str)
        for item in node.elts
    ):
        return None
    return {item.value for item in node.elts}


def _selector_source_column(node: ast.AST) -> ast.Constant | None:
    matches = [
        item.slice
        for item in ast.walk(node)
        if isinstance(item, ast.Subscript)
        and isinstance(item.value, ast.Name)
        and item.value.id == "robustness_summary"
        and isinstance(item.slice, ast.Constant)
        and item.slice.value == "restriction"
    ]
    if len(matches) != 1:
        return None
    expected = (
        "robustness_summary['restriction']" ".astype(str).str.strip().str.lower()"
    )
    return matches[0] if ast.unparse(node) == expected else None


def _role_filter_labels(
    node: ast.AST,
    *,
    selector_name: str,
) -> ast.AST | None:
    if (
        not isinstance(node, ast.Subscript)
        or not isinstance(node.value, ast.Name)
        or node.value.id != "robustness_summary"
        or not isinstance(node.slice, ast.Call)
        or not isinstance(node.slice.func, ast.Attribute)
        or node.slice.func.attr != "isin"
        or not isinstance(node.slice.func.value, ast.Name)
        or node.slice.func.value.id != selector_name
        or len(node.slice.args) != 1
        or node.slice.keywords
        or _literal_strings(node.slice.args[0]) is None
    ):
        return None
    return node.slice.args[0]


def _offsets(code: str, node: ast.AST) -> tuple[int, int] | None:
    if not all(
        hasattr(node, attribute)
        for attribute in ("lineno", "col_offset", "end_lineno", "end_col_offset")
    ):
        return None
    lines = code.splitlines(keepends=True)

    def character_offset(line_number: int, byte_column: int) -> int:
        line = lines[line_number - 1]
        character_column = len(line.encode("utf-8")[:byte_column].decode("utf-8"))
        return sum(len(part) for part in lines[: line_number - 1]) + character_column

    return (
        character_offset(node.lineno, node.col_offset),
        character_offset(node.end_lineno, node.end_col_offset),
    )


def patch_structured_analysis_role_selection(
    code: str,
    run_log: str,
) -> str | None:
    """Use the declared role field instead of interpreting restriction prose.

    This repair is intentionally closed over one generated-renderer defect. It
    changes only the typed role selector and its two literal role vocabularies;
    source rows and all numeric values remain untouched.
    """

    if _ERROR not in (run_log or ""):
        return None
    try:
        tree = ast.parse(code)
    except SyntaxError:
        return None

    required = _literal_strings(_assignment_value(tree, "required_summary_columns"))
    if required is None or not {"analysis", "restriction"}.issubset(required):
        return None

    selector_name = "restriction_values"
    selector_value = _assignment_value(tree, selector_name)
    if selector_value is None:
        return None
    source_column = _selector_source_column(selector_value)
    if source_column is None:
        return None

    primary_value = _assignment_value(tree, "primary_rows")
    complete_value = _assignment_value(tree, "complete_rows")
    if primary_value is None or complete_value is None:
        return None
    primary_labels = _role_filter_labels(
        primary_value,
        selector_name=selector_name,
    )
    complete_labels = _role_filter_labels(
        complete_value,
        selector_name=selector_name,
    )
    if (
        primary_labels is None
        or complete_labels is None
        or _literal_strings(primary_labels) != _PRIMARY_LABELS
        or _literal_strings(complete_labels) != _COMPLETE_CASE_LABELS
    ):
        return None

    allowed_loads = {
        id(primary_value.slice.func.value),
        id(complete_value.slice.func.value),
    }
    selector_loads = {
        id(node)
        for node in ast.walk(tree)
        if isinstance(node, ast.Name)
        and isinstance(node.ctx, ast.Load)
        and node.id == selector_name
    }
    if selector_loads != allowed_loads:
        return None

    replacements: list[tuple[int, int, str]] = []
    for node, replacement in (
        (source_column, repr("analysis")),
        (primary_labels, repr({"primary"})),
        (
            complete_labels,
            repr({"complete_case", "complete_case_sensitivity"}),
        ),
    ):
        span = _offsets(code, node)
        if span is None:
            return None
        replacements.append((*span, replacement))

    repaired = code
    for start, end, replacement in sorted(replacements, reverse=True):
        repaired = repaired[:start] + replacement + repaired[end:]
    try:
        ast.parse(repaired)
    except SyntaxError:
        return None
    return repaired


__all__ = ["patch_structured_analysis_role_selection"]
