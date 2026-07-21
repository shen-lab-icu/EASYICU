"""Host/package checks used by deterministic import repairs."""

from __future__ import annotations

import ast
import importlib.util


def host_module_is_available(module_name: str) -> bool:
    """Distinguish sandbox image drift from a hallucinated EasyICU module."""

    try:
        return importlib.util.find_spec(module_name) is not None
    except (AttributeError, ImportError, ModuleNotFoundError, ValueError):
        return False


def insert_after_imports(source: str, block: str) -> str:
    """Insert ``block`` after the leading import block, never inside one."""

    parsed = ast.parse(source)
    lines = source.splitlines()
    body = list(parsed.body)
    insert_at = 0
    if (
        body
        and isinstance(body[0], ast.Expr)
        and isinstance(body[0].value, ast.Constant)
        and isinstance(body[0].value.value, str)
    ):
        insert_at = int(body.pop(0).end_lineno or 0)
    for node in body:
        if not isinstance(node, (ast.Import, ast.ImportFrom)):
            break
        insert_at = int(node.end_lineno or node.lineno)
    lines.insert(insert_at, block)
    return "\n".join(lines)


__all__ = ["host_module_is_available", "insert_after_imports"]
