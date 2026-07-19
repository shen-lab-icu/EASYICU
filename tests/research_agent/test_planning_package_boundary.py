"""Compatibility and dependency contracts for scientific-planning modules."""

from __future__ import annotations

import ast
import importlib
import os
from pathlib import Path
import subprocess
import sys

import pytest

PLANNING_MODULE_ALIASES = tuple(
    (
        f"easyicu.research_agent.{leaf}",
        f"easyicu.research_agent.planning.{leaf}",
    )
    for leaf in (
        "study_design",
        "study_design_playbook",
        "capability_registry",
        "analysis_method_suite",
        "figure_strategy",
        "analysis_types",
    )
)


@pytest.mark.parametrize("legacy,canonical", PLANNING_MODULE_ALIASES)
def test_planning_legacy_path_is_canonical_module_object(
    legacy: str,
    canonical: str,
) -> None:
    old_module = importlib.import_module(legacy)
    new_module = importlib.import_module(canonical)
    assert old_module is new_module
    assert old_module.__file__ == new_module.__file__
    assert "/planning/" in Path(new_module.__file__).as_posix()


@pytest.mark.parametrize("order", ("legacy_first", "canonical_first", "root_first"))
def test_planning_aliases_survive_clean_import_order(order: str) -> None:
    script = f"""
import importlib
pairs = {PLANNING_MODULE_ALIASES!r}
if {order!r} == 'root_first':
    root = importlib.import_module('easyicu.research_agent')
    getattr(root, 'infer_analysis_type')
for legacy, canonical in pairs:
    names = (legacy, canonical) if {order!r} == 'legacy_first' else (canonical, legacy)
    assert importlib.import_module(names[0]) is importlib.import_module(names[1])
"""
    env = dict(os.environ)
    source_root = str(Path(__file__).resolve().parents[2] / "src")
    env["PYTHONPATH"] = source_root + os.pathsep + env.get("PYTHONPATH", "")
    subprocess.run([sys.executable, "-c", script], check=True, env=env)


def test_planning_package_is_lazy() -> None:
    script = """
import importlib
import sys
package = 'easyicu.research_agent.planning'
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


def test_production_modules_do_not_route_through_planning_facades() -> None:
    package_root = Path(__file__).resolve().parents[2] / "src/easyicu/research_agent"
    legacy_modules = {legacy for legacy, _canonical in PLANNING_MODULE_ALIASES}
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


def test_legacy_registry_regeneration_commands_still_work() -> None:
    env = dict(os.environ)
    source_root = str(Path(__file__).resolve().parents[2] / "src")
    env["PYTHONPATH"] = source_root + os.pathsep + env.get("PYTHONPATH", "")
    commands = (
        "easyicu.research_agent.capability_registry",
        "easyicu.research_agent.analysis_method_suite",
    )
    for module_name in commands:
        result = subprocess.run(
            [sys.executable, "-m", module_name],
            check=True,
            capture_output=True,
            env=env,
            text=True,
        )
        assert result.stdout.startswith("# ")
