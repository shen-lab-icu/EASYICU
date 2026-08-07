"""Representation-only repairs for generated Matplotlib source-data exports."""

from __future__ import annotations

import ast
from typing import Optional


_EMPTY_PANEL_ERROR = (
    "Clustering visualization panel has no traceable plotted values"
)


def _raised_message(node: ast.If) -> Optional[str]:
    for statement in node.body:
        if not (
            isinstance(statement, ast.Raise)
            and isinstance(statement.exc, ast.Call)
            and isinstance(statement.exc.func, ast.Name)
            and statement.exc.func.id == "ValueError"
            and len(statement.exc.args) == 1
            and not statement.exc.keywords
            and isinstance(statement.exc.args[0], ast.Constant)
            and isinstance(statement.exc.args[0].value, str)
        ):
            continue
        return statement.exc.args[0].value
    return None


def _enumerated_attribute(
    loop: ast.For,
    *,
    owner_name: str,
) -> Optional[str]:
    iterator = loop.iter
    if not (
        isinstance(iterator, ast.Call)
        and isinstance(iterator.func, ast.Name)
        and iterator.func.id == "enumerate"
        and len(iterator.args) == 1
        and not iterator.keywords
        and isinstance(iterator.args[0], ast.Attribute)
        and isinstance(iterator.args[0].value, ast.Name)
        and iterator.args[0].value.id == owner_name
    ):
        return None
    return iterator.args[0].attr


def patch_matplotlib_patch_source_rows(code: str, run_log: str) -> str:
    """Decline artist-to-source-data projection.

    Matplotlib artists are a rendering output, not scientific source evidence.
    Recovering bar heights after the fact would make a figure validate without
    its table-level lineage, so this repair intentionally leaves the run at its
    fail-closed source-data gate.
    """

    return code

    if _EMPTY_PANEL_ERROR not in str(run_log or ""):
        return code
    try:
        tree = ast.parse(code)
    except SyntaxError:
        return code

    if not any(
        isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr == "bar"
        for node in ast.walk(tree)
    ):
        return code

    parents: dict[ast.AST, ast.AST] = {}
    for parent in ast.walk(tree):
        for child in ast.iter_child_nodes(parent):
            parents[child] = parent

    candidates: list[tuple[ast.If, ast.For, str, str]] = []
    for node in ast.walk(tree):
        if not (
            isinstance(node, ast.If)
            and isinstance(node.test, ast.UnaryOp)
            and isinstance(node.test.op, ast.Not)
            and isinstance(node.test.operand, ast.Name)
            and _raised_message(node) == _EMPTY_PANEL_ERROR
        ):
            continue
        panel_rows_name = node.test.operand.id
        panel_loop = parents.get(node)
        if not (
            isinstance(panel_loop, ast.For)
            and node in panel_loop.body
            and isinstance(panel_loop.target, (ast.Tuple, ast.List))
            and len(panel_loop.target.elts) == 2
            and all(isinstance(item, ast.Name) for item in panel_loop.target.elts)
        ):
            continue
        panel_ax_name = panel_loop.target.elts[1].id
        panel_iterator = panel_loop.iter
        if not (
            isinstance(panel_iterator, ast.Call)
            and isinstance(panel_iterator.func, ast.Name)
            and panel_iterator.func.id == "enumerate"
            and len(panel_iterator.args) == 1
            and not panel_iterator.keywords
            and isinstance(panel_iterator.args[0], ast.Attribute)
            and panel_iterator.args[0].attr == "axes"
        ):
            continue

        extractor_attributes = {
            attribute
            for statement in panel_loop.body
            if isinstance(statement, ast.For)
            and (
                attribute := _enumerated_attribute(
                    statement,
                    owner_name=panel_ax_name,
                )
            )
            is not None
        }
        if "patches" in extractor_attributes:
            continue
        if not {"collections", "lines"}.issubset(extractor_attributes):
            continue
        candidates.append((node, panel_loop, panel_rows_name, panel_ax_name))

    if len(candidates) != 1:
        return code
    guard, _, panel_rows_name, panel_ax_name = candidates[0]
    if not isinstance(guard.lineno, int) or not isinstance(guard.col_offset, int):
        return code

    lines = code.splitlines(keepends=True)
    if not (1 <= guard.lineno <= len(lines)):
        return code
    insertion_offset = sum(len(line) for line in lines[: guard.lineno - 1])
    indent = " " * guard.col_offset
    body_indent = indent + "    "
    mapping_indent = body_indent + "    "
    injected = (
        f"{indent}for _easyicu_patch_index, _easyicu_patch in "
        f"enumerate({panel_ax_name}.patches):\n"
        f"{body_indent}if not all(hasattr(_easyicu_patch, _easyicu_attr) "
        f"for _easyicu_attr in (\"get_x\", \"get_width\", \"get_height\")):\n"
        f"{mapping_indent}continue\n"
        f"{body_indent}{panel_rows_name}.append({{\n"
        f'{mapping_indent}"source_row_index": int(_easyicu_patch_index),\n'
        f'{mapping_indent}"artist_index": int(_easyicu_patch_index),\n'
        f'{mapping_indent}"x_coordinate": float(_easyicu_patch.get_x() + '
        f'_easyicu_patch.get_width() / 2.0),\n'
        f'{mapping_indent}"y_coordinate": float(_easyicu_patch.get_height()),\n'
        f"{body_indent}}})\n"
    )
    repaired = code[:insertion_offset] + injected + code[insertion_offset:]
    try:
        ast.parse(repaired)
    except SyntaxError:
        return code
    return repaired


__all__ = ["patch_matplotlib_patch_source_rows"]
