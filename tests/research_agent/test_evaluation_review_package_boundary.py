"""Compatibility and ownership contracts for evaluation/review modules."""

from __future__ import annotations

import ast
import importlib
import os
from pathlib import Path
import subprocess
import sys

import pytest

MODULE_ALIASES = (
    (
        "easyicu.research_agent.cross_model_panel",
        "easyicu.research_agent.evaluation.cross_model_panel",
    ),
    (
        "easyicu.research_agent.tier2_jury",
        "easyicu.research_agent.evaluation.tier2_jury",
    ),
    (
        "easyicu.research_agent.tier2_rubric",
        "easyicu.research_agent.evaluation.tier2_rubric",
    ),
    (
        "easyicu.research_agent.causal_audit",
        "easyicu.research_agent.review.causal_audit",
    ),
    (
        "easyicu.research_agent.methodological_rigor",
        "easyicu.research_agent.review.methodological_rigor",
    ),
)


@pytest.mark.parametrize("legacy,canonical", MODULE_ALIASES)
def test_legacy_path_is_canonical_module_object(
    legacy: str,
    canonical: str,
) -> None:
    old_module = importlib.import_module(legacy)
    new_module = importlib.import_module(canonical)
    assert old_module is new_module
    assert old_module.__file__ == new_module.__file__


@pytest.mark.parametrize("order", ("legacy_first", "canonical_first", "root_first"))
def test_aliases_survive_clean_import_order(order: str) -> None:
    script = f"""
import importlib
pairs = {MODULE_ALIASES!r}
if {order!r} == 'root_first':
    root = importlib.import_module('easyicu.research_agent')
    getattr(root, 'run_causal_audit')
for legacy, canonical in pairs:
    names = (legacy, canonical) if {order!r} == 'legacy_first' else (canonical, legacy)
    assert importlib.import_module(names[0]) is importlib.import_module(names[1])
"""
    env = dict(os.environ)
    source_root = str(Path(__file__).resolve().parents[2] / "src")
    env["PYTHONPATH"] = source_root + os.pathsep + env.get("PYTHONPATH", "")
    subprocess.run([sys.executable, "-c", script], check=True, env=env)


@pytest.mark.parametrize(
    "package",
    (
        "easyicu.research_agent.evaluation",
        "easyicu.research_agent.review",
    ),
)
def test_responsibility_packages_are_lazy(package: str) -> None:
    script = f"""
import importlib
import sys
package = {package!r}
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


def test_production_modules_do_not_route_through_legacy_facades() -> None:
    package_root = Path(__file__).resolve().parents[2] / "src/easyicu/research_agent"
    legacy_modules = {legacy for legacy, _canonical in MODULE_ALIASES}
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


def test_command_line_tools_use_canonical_evaluation_paths() -> None:
    repo_root = Path(__file__).resolve().parents[2]
    expected = {
        "tools/run_cross_model_check.py": (
            "easyicu.research_agent.evaluation.cross_model_panel",
        ),
        "tools/run_tier2_jury.py": (
            "easyicu.research_agent.evaluation.tier2_jury",
            "easyicu.research_agent.evaluation.tier2_rubric",
        ),
    }
    legacy_modules = {legacy for legacy, _canonical in MODULE_ALIASES}
    for relative, required in expected.items():
        path = repo_root / relative
        imports = _resolved_imports(
            path, relative.replace("/", ".").removesuffix(".py")
        )
        assert set(required).issubset(imports)
        assert imports.isdisjoint(legacy_modules)


def test_root_public_causal_audit_uses_canonical_module() -> None:
    root = importlib.import_module("easyicu.research_agent")
    canonical = importlib.import_module("easyicu.research_agent.review.causal_audit")
    assert root.run_causal_audit is canonical.run_causal_audit
