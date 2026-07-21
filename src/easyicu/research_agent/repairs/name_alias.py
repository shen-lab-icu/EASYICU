"""Narrow runtime repair for one proven near-match mapping alias typo."""

from __future__ import annotations

import ast
import difflib
import re

_NAME_ERROR = re.compile(
    r"NameError:\s*name\s*['\"](?P<missing>[A-Za-z_]\w*)['\"]\s*is not defined\.\s*"
    r"Did you mean:\s*['\"](?P<candidate>[A-Za-z_]\w*)['\"]\?"
)


def _module_scope_nodes(tree: ast.Module) -> list[ast.AST]:
    nodes: list[ast.AST] = []

    def visit(node: ast.AST) -> None:
        nodes.append(node)
        for child in ast.iter_child_nodes(node):
            if isinstance(
                child,
                (ast.AsyncFunctionDef, ast.ClassDef, ast.FunctionDef, ast.Lambda),
            ):
                continue
            visit(child)

    visit(tree)
    return nodes


def patch_undefined_mapping_near_match_alias(code: str, run_log: str) -> str:
    """Replace one undefined mapping name with its unique defined near-match."""

    match = _NAME_ERROR.search(run_log or "")
    if match is None:
        return code
    missing = match.group("missing")
    candidate = match.group("candidate")
    missing_suffix = missing.partition("_")[2]
    candidate_suffix = candidate.partition("_")[2]
    if not (
        missing != candidate
        and missing_suffix == candidate_suffix
        and len(missing_suffix) >= 5
        and difflib.SequenceMatcher(None, missing, candidate).ratio() >= 0.75
    ):
        return code
    try:
        tree = ast.parse(code)
    except SyntaxError:
        return code
    module_nodes = _module_scope_nodes(tree)
    missing_nodes = [
        node
        for node in module_nodes
        if isinstance(node, ast.Name)
        and isinstance(node.ctx, ast.Load)
        and node.id == missing
        and node.end_lineno is not None
    ]
    if not missing_nodes or any(
        isinstance(node, ast.Name)
        and isinstance(node.ctx, ast.Store)
        and node.id == missing
        for node in module_nodes
    ):
        return code
    assignments = [
        node
        for node in tree.body
        if isinstance(node, ast.Assign)
        and len(node.targets) == 1
        and isinstance(node.targets[0], ast.Name)
        and node.targets[0].id == candidate
        and isinstance(node.value, ast.Dict)
    ]
    if len(assignments) != 1:
        return code
    lines = code.splitlines(keepends=True)
    line_starts: list[int] = []
    offset = 0
    for source_line in lines:
        line_starts.append(offset)
        offset += len(source_line)

    def absolute_offset(lineno: int, utf8_col: int) -> int:
        source_line = lines[lineno - 1]
        char_col = len(source_line.encode("utf-8")[:utf8_col].decode("utf-8"))
        return line_starts[lineno - 1] + char_col

    replacements = [
        (
            absolute_offset(int(node.lineno), int(node.col_offset)),
            absolute_offset(int(node.end_lineno), int(node.end_col_offset)),
        )
        for node in missing_nodes
    ]
    repaired = code
    for start, end in sorted(replacements, reverse=True):
        if repaired[start:end] != missing:
            return code
        repaired = repaired[:start] + candidate + repaired[end:]
    try:
        ast.parse(repaired)
    except SyntaxError:
        return code
    return repaired


__all__ = ["patch_undefined_mapping_near_match_alias"]
