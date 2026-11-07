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


# --- delta20w: the loader resolves its source once (D20W-TIME-RESELECT class)

# Calls that select or build a data source, or resolve a root, by name.
_RESOLVING_CALLS = {
    "load_table", "load_id_tbl", "select_table_source", "load_src_cfg",
    "ICUDataSource", "get_source", "is_attached", "attach_src",
}
_RESOLVING_IMPORTS = _RESOLVING_CALLS | {"attach"}

# (owning function, name) -> why it may resolve.  Nothing else may.
ALLOWED_RESOLUTION = {
    ("_select_source", "select_table_source"): "the one by-name selection, at construction",
    ("_select_source", "ICUDataSource"): "a DataSourceConfig passed to the constructor",
    ("_select_source", "data_path"): "the constructor's root for that selection",
    ("__init__", "data_path"): "stores the constructor argument and passes it to _select_source",
    ("load_concepts", "data_path"): "the module entry forwards data_path to ConceptLoader",
}


def _owned_nodes(tree):
    found = []

    def visit(node, owner):
        for child in ast.iter_child_nodes(node):
            name = child.name if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef)) else owner
            found.append((name, child))
            visit(child, name)

    visit(tree, "<module>")
    return found


def test_source_resolution_happens_only_in_select_source():
    tree = ast.parse(SOURCE.read_text(encoding="utf-8"))
    offenders = []
    for owner, node in _owned_nodes(tree):
        hit = None
        if isinstance(node, ast.Call):
            func = node.func
            # ``source.load_table(...)`` on a source object reads through it;
            # a bare or module-qualified resolver call selects by name.
            if isinstance(func, ast.Name) and func.id in _RESOLVING_CALLS:
                hit = func.id
            elif (
                isinstance(func, ast.Attribute)
                and func.attr in _RESOLVING_CALLS - {"load_table"}
            ):
                hit = func.attr
            elif (
                isinstance(func, ast.Attribute) and func.attr == "load_table"
                and isinstance(func.value, ast.Name) and func.value.id in {"table", "easyicu"}
            ):
                hit = "load_table"
        elif isinstance(node, (ast.Import, ast.ImportFrom)):
            names = {alias.name.split(".")[-1] for alias in node.names}
            module = getattr(node, "module", None) or ""
            if names & _RESOLVING_IMPORTS or module.endswith("attach"):
                bad = sorted(names & _RESOLVING_IMPORTS) or [module]
                hit = "import " + ",".join(bad)
        elif isinstance(node, ast.Attribute) and node.attr == "data_path":
            hit = "data_path"
        elif isinstance(node, ast.Name) and node.id == "data_path":
            hit = "data_path"
        if hit is None:
            continue
        key = hit.replace("import ", "")
        if (owner, key) in ALLOWED_RESOLUTION:
            continue
        if owner == "<module>" and hit == "import ICUDataSource":
            continue  # the type used for isinstance and annotations
        offenders.append(f"{owner}:{getattr(node, 'lineno', '?')} {hit}")
    assert not offenders, "\n".join(sorted(set(offenders)))
