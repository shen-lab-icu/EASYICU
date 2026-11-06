"""The deprecated ConceptLoader fails closed (lead-r13 delta15w).

Every ``except`` in ``src/easyicu/load_concepts.py`` must raise.  An empty
result is legitimate only for a table this database does not declare, and
that decision is made before the read, on the source ``load_table`` selects
(``ConceptLoader._declared_table_source``),
never by catching an exception.  The one handler that may finish without
raising is the documented missing-column projection retry, which reads the
same table in full instead of returning nothing
(docs/deprecation_policy.md).
"""

from __future__ import annotations

import ast
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SOURCE = ROOT / "src" / "easyicu" / "load_concepts.py"

# (owning function, caught type) -> why finishing without a raise is allowed.
ALLOWED_NON_RAISING = {
    ("_safe_load_table", "Exception"): "missing-column projection retry: full re-read, never empty",
}


def _handlers(tree: ast.AST):
    found = []

    def visit(node, owner):
        for child in ast.iter_child_nodes(node):
            name = child.name if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef)) else owner
            if isinstance(child, ast.ExceptHandler):
                found.append((name, child))
            visit(child, name)

    visit(tree, "<module>")
    return found


def _ends_in_raise(body) -> bool:
    """Every path through ``body`` ends in ``raise`` (no fall-through)."""
    if not body:
        return False
    last = body[-1]
    if isinstance(last, ast.Raise):
        return True
    if isinstance(last, ast.If):
        return _ends_in_raise(last.body) and _ends_in_raise(last.orelse)
    return False


def test_every_except_in_the_legacy_loader_raises():
    tree = ast.parse(SOURCE.read_text(encoding="utf-8"))
    offenders = []
    for owner, handler in _handlers(tree):
        caught = ast.unparse(handler.type) if handler.type is not None else "<bare>"
        if handler.type is None:
            offenders.append(f"{owner}:{handler.lineno} bare except")
            continue
        if _ends_in_raise(handler.body):
            continue
        if (owner, caught) in ALLOWED_NON_RAISING:
            assert any(isinstance(node, ast.Raise) for node in ast.walk(handler)), (
                f"{owner}:{handler.lineno} allowlisted handler must still re-raise "
                "everything outside its documented case"
            )
            continue
        offenders.append(f"{owner}:{handler.lineno} except {caught} can finish without raising")
    assert not offenders, "\n".join(offenders)


def test_declared_absence_is_decided_before_the_read():
    tree = ast.parse(SOURCE.read_text(encoding="utf-8"))
    names = {
        node.name for node in ast.walk(tree)
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
    }
    assert {"_declared_table_source", "_read_declared_table", "_table_source"} <= names
    # MissingTableSourceError is never a reason to return data: it is only
    # caught to be re-raised as the typed declared-source failure.
    for owner, handler in _handlers(tree):
        if handler.type is not None and "MissingTableSourceError" in ast.unparse(handler.type):
            assert owner == "_read_declared_table" and _ends_in_raise(handler.body)
