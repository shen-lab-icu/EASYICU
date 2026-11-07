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

import pytest

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


# --- delta20w/delta24w: the loader resolves its source once (D20W-TIME-RESELECT class)
#
# What this pins: direct calls and imports of the resolvers, including module
# aliases (``import easyicu.table as t``), qualified names
# (``easyicu.table.load_table``), module-level aliases of the source class
# (``DataSource = ICUDataSource``), string lookups (``getattr(m, "load_table")``)
# and dynamic imports.  It is a static check of this module's text, not a proof
# that no runtime path can resolve a source some other way.

_RESOLVERS = {
    "load_table", "load_id_tbl", "select_table_source", "load_src_cfg",
    "ICUDataSource", "get_source", "is_attached", "attach_src", "storage_source",
}
_RESOLVER_MODULES = ("table", "attach", "datasource", "config", "data_tools")
# Receivers whose ``.load_table`` is a read through the bound source object.
_BOUND_RECEIVERS = {"preload_source"}

# (Class.method or function, name) -> why it may resolve, and how many times.
ALLOWED_RESOLUTION = {
    ("ConceptLoader._select_source", "select_table_source"): ("the one by-name selection", 2),
    ("ConceptLoader._select_source", "ICUDataSource"): ("isinstance + a DataSourceConfig passed in", 3),
    ("ConceptLoader._select_source", "import select_table_source"): ("its import", 1),
    ("ConceptLoader._select_source", "data_path"): ("the constructor's root for that selection", 4),
    ("ConceptLoader.__init__", "data_path"): ("stores the argument, passes it to _select_source", 4),
    ("ConceptLoader.__init__", "ICUDataSource"): ("isinstance check of the argument", 1),
    ("load_concepts", "data_path"): ("the module entry forwards data_path", 3),
    ("<module>", "ICUDataSource"): ("the import and the DataSource type alias", 3),
}


def _owned_nodes(tree):
    found = []

    def visit(node, owner, cls):
        for child in ast.iter_child_nodes(node):
            if isinstance(child, ast.ClassDef):
                visit(child, owner, child.name)
                continue
            if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef)):
                name = f"{cls}.{child.name}" if cls and owner == "<module>" else child.name
                found.append((name, child))
                visit(child, name, None)
                continue
            found.append((owner, child))
            visit(child, owner, cls)

    visit(tree, "<module>", None)
    return found


def _aliases(tree):
    """Module aliases of resolver modules and names bound to resolvers."""
    modules, names = set(), set(_RESOLVERS)
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                if alias.name.split(".")[-1] in _RESOLVER_MODULES or alias.name == "easyicu":
                    modules.add(alias.asname or alias.name.split(".")[0])
        elif isinstance(node, ast.ImportFrom):
            for alias in node.names:
                if alias.name in _RESOLVER_MODULES:
                    modules.add(alias.asname or alias.name)
                if alias.name in _RESOLVERS and alias.asname:
                    names.add(alias.asname)
        elif isinstance(node, ast.Assign) and isinstance(node.value, ast.Name) and node.value.id in names:
            names.update(t.id for t in node.targets if isinstance(t, ast.Name))
    return modules, names


def _resolution(node, modules, names):
    if isinstance(node, ast.Call):
        func = node.func
        if isinstance(func, ast.Name):
            if func.id in names:
                return func.id if func.id in _RESOLVERS else "ICUDataSource"
            strings = [a.value for a in node.args if isinstance(a, ast.Constant) and isinstance(a.value, str)]
            if func.id == "getattr" and any(value in _RESOLVERS for value in strings):
                return "dynamic"
            if func.id == "__import__":
                return "dynamic"
        if isinstance(func, ast.Attribute):
            text = ast.unparse(func)
            if func.attr == "import_module":
                return "dynamic"
            if func.attr in _RESOLVERS:
                root = text.split(".")[0]
                if func.attr == "load_table" and isinstance(func.value, ast.Name) and func.value.id in _BOUND_RECEIVERS:
                    return None
                if func.attr == "load_table" and root not in modules and not text.startswith("easyicu."):
                    return "load_table on an unbound receiver"
                return func.attr
    elif isinstance(node, (ast.Import, ast.ImportFrom)):
        imported = {alias.name.split(".")[-1] for alias in node.names}
        hit = sorted(imported & _RESOLVERS)
        module = getattr(node, "module", None) or ""
        if hit:
            return "ICUDataSource" if hit == ["ICUDataSource"] else "import " + ",".join(hit)
        if module.split(".")[-1] == "attach" or imported & {"attach"}:
            return "import attach"
    elif isinstance(node, ast.Attribute) and node.attr == "data_path":
        return "data_path"
    elif isinstance(node, ast.Name) and node.id == "data_path":
        return "data_path"
    return None


def test_source_resolution_happens_only_in_select_source():
    tree = ast.parse(SOURCE.read_text(encoding="utf-8"))
    modules, names = _aliases(tree)
    offenders, counts = [], {}
    for owner, node in _owned_nodes(tree):
        hit = _resolution(node, modules, names)
        if hit is None:
            continue
        key = (owner, hit)
        if key in ALLOWED_RESOLUTION:
            counts[key] = counts.get(key, 0) + 1
            continue
        offenders.append(f"{owner}:{getattr(node, 'lineno', '?')} {hit}")
    for key, (why, most) in ALLOWED_RESOLUTION.items():
        if counts.get(key, 0) > most:
            offenders.append(f"{key[0]}: {counts[key]} x {key[1]} (allowed {most}: {why})")
    assert not offenders, "\n".join(sorted(set(offenders)))


_MUTATIONS = {
    "module alias": "import easyicu.table as t\ndef f(self):\n    return t.load_table('x', 'y')\n",
    "qualified name": "import easyicu\ndef f(self):\n    return easyicu.table.load_table('x', 'y')\n",
    "getattr string": "from . import table\ndef f(self):\n    return getattr(table, 'load_id_tbl')('x', 'icustay')\n",
    "fetch then call": "from .table import load_table as lt\ndef f(self):\n    g = lt\n    return g('x', 'y')\n",
    "lambda": "def f(self):\n    from .table import select_table_source\n    return (lambda: select_table_source('x'))()\n",
    "dynamic import": "import importlib\ndef f(self):\n    return importlib.import_module('easyicu.table')\n",
    "string __import__": "def f(self):\n    return __import__('easyicu.table')\n",
    "class alias construction": "from .datasource import ICUDataSource\nDataSource = ICUDataSource\ndef f(self):\n    return DataSource(None)\n",
    "unbound receiver": "def f(self, other):\n    return other.load_table('icustays')\n",
    "same-named method elsewhere": "class Other:\n    def _select_source(self):\n        from .table import select_table_source\n        return select_table_source('x')\n",
    "data_path read": "def f(self):\n    return self.data_path\n",
}


@pytest.mark.parametrize("label", sorted(_MUTATIONS))
def test_resolution_gate_catches_each_mutation(label):
    tree = ast.parse(_MUTATIONS[label])
    modules, names = _aliases(tree)
    hits = [
        (owner, _resolution(node, modules, names))
        for owner, node in _owned_nodes(tree)
        if _resolution(node, modules, names) is not None
    ]
    assert any(key not in ALLOWED_RESOLUTION for key in hits), (label, hits)
