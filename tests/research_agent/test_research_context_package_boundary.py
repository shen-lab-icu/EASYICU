"""Compatibility contracts for research-context and method-gate ownership."""

from __future__ import annotations

import ast
import importlib
import os
from pathlib import Path
import subprocess
import sys

import pytest

CONTEXT_MODULE_ALIASES = (
    (
        "easyicu.research_agent.context",
        "easyicu.research_agent.research_context.builder",
    ),
    (
        "easyicu.research_agent.coder_context",
        "easyicu.research_agent.research_context.prompt_scope",
    ),
    (
        "easyicu.research_agent.research_context_v2",
        "easyicu.research_agent.research_context.typed",
    ),
    (
        "easyicu.research_agent.method_compatibility",
        "easyicu.research_agent.gates.method_compatibility",
    ),
)


@pytest.mark.parametrize("legacy,canonical", CONTEXT_MODULE_ALIASES)
def test_context_legacy_path_is_canonical_module_object(
    legacy: str,
    canonical: str,
) -> None:
    old_module = importlib.import_module(legacy)
    new_module = importlib.import_module(canonical)
    assert old_module is new_module
    assert old_module.__file__ == new_module.__file__


@pytest.mark.parametrize("order", ("legacy_first", "canonical_first", "root_first"))
def test_context_aliases_survive_clean_import_order(order: str) -> None:
    script = f"""
import importlib
pairs = {CONTEXT_MODULE_ALIASES!r}
if {order!r} == 'root_first':
    root = importlib.import_module('easyicu.research_agent')
    getattr(root, 'build_research_context')
for legacy, canonical in pairs:
    names = (legacy, canonical) if {order!r} == 'legacy_first' else (canonical, legacy)
    assert importlib.import_module(names[0]) is importlib.import_module(names[1])
"""
    env = dict(os.environ)
    source_root = str(Path(__file__).resolve().parents[2] / "src")
    env["PYTHONPATH"] = source_root + os.pathsep + env.get("PYTHONPATH", "")
    subprocess.run([sys.executable, "-c", script], check=True, env=env)


def test_research_context_package_is_lazy() -> None:
    script = """
import importlib
import sys
package = 'easyicu.research_agent.research_context'
importlib.import_module(package)
loaded = sorted(name for name in sys.modules if name.startswith(package + '.'))
assert loaded == [], loaded
"""
    env = dict(os.environ)
    source_root = str(Path(__file__).resolve().parents[2] / "src")
    env["PYTHONPATH"] = source_root + os.pathsep + env.get("PYTHONPATH", "")
    subprocess.run([sys.executable, "-c", script], check=True, env=env)


def _resolved_imports(path: Path, module_name: str) -> set[str]:
    tree = ast.parse(path.read_text(encoding="utf-8"))
    package = module_name.rpartition(".")[0]
    imported: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            if node.level:
                imported.add(
                    importlib.util.resolve_name(
                        "." * node.level + node.module,
                        package,
                    )
                )
            else:
                imported.add(node.module)
    return imported


def test_production_modules_do_not_route_through_context_facades() -> None:
    package_root = Path(__file__).resolve().parents[2] / "src/easyicu/research_agent"
    legacy_modules = {legacy for legacy, _canonical in CONTEXT_MODULE_ALIASES}
    shim_files = {f"{name.rsplit('.', 1)[-1]}.py" for name in legacy_modules}
    offenders: dict[str, list[str]] = {}
    for path in package_root.rglob("*.py"):
        relative = path.relative_to(package_root)
        if len(relative.parts) == 1 and relative.name in shim_files:
            continue
        parts = list(relative.with_suffix("").parts)
        if parts[-1] == "__init__":
            parts.pop()
        module_name = ".".join(("easyicu", "research_agent", *parts))
        legacy_imports = sorted(
            _resolved_imports(path, module_name).intersection(legacy_modules)
        )
        if legacy_imports:
            offenders[relative.as_posix()] = legacy_imports
    assert offenders == {}


def test_legacy_monkeypatch_reaches_canonical_context_module() -> None:
    legacy = importlib.import_module("easyicu.research_agent.context")
    canonical = importlib.import_module(
        "easyicu.research_agent.research_context.builder"
    )
    marker = object()
    original = canonical._safe_get_concept_info
    try:
        legacy._safe_get_concept_info = marker
        assert canonical._safe_get_concept_info is marker
    finally:
        canonical._safe_get_concept_info = original
