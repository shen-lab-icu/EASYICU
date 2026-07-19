"""Canonical package boundary for durable authority components."""

from __future__ import annotations

import ast
import importlib
import importlib.util
import os
from pathlib import Path
import subprocess
import sys


MOVED_AUTHORITY_MODULES = {
    "authority_fs": "authority.filesystem",
    "lock_authority": "authority.lock_contract",
    "step_attempt_authority": "authority.step_attempt",
    "step_authority_capsule": "authority.step_capsule",
    "step_authority_runtime": "authority.step_runtime",
}


def test_authority_package_import_is_lazy() -> None:
    script = """
import importlib
import sys
package = 'easyicu.research_agent.authority'
module = importlib.import_module(package)
assert module.__name__ == package
assert not {
    name for name in sys.modules
    if name.startswith(package + '.')
}
"""
    env = dict(os.environ)
    source_root = str(Path(__file__).resolve().parents[2] / "src")
    env["PYTHONPATH"] = source_root + os.pathsep + env.get("PYTHONPATH", "")
    subprocess.run([sys.executable, "-c", script], check=True, env=env)


def test_moved_authority_modules_have_one_physical_owner() -> None:
    root = Path(__file__).resolve().parents[2] / "src" / "easyicu" / "research_agent"
    for retired, canonical in MOVED_AUTHORITY_MODULES.items():
        assert not (root / f"{retired}.py").exists()
        module = importlib.import_module(f"easyicu.research_agent.{canonical}")
        assert Path(module.__file__).resolve().is_relative_to(root / "authority")


def test_production_code_does_not_import_retired_authority_paths() -> None:
    root = Path(__file__).resolve().parents[2] / "src" / "easyicu" / "research_agent"
    retired = {
        f"easyicu.research_agent.{name}" for name in MOVED_AUTHORITY_MODULES
    }
    for path in root.rglob("*.py"):
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom):
                module = node.module or ""
                if node.level:
                    package = ".".join(
                        ("easyicu", "research_agent", *path.relative_to(root).parts[:-1])
                    )
                    module = importlib.util.resolve_name(
                        "." * node.level + module,
                        package,
                    )
                assert module not in retired, (path, module)
            elif isinstance(node, ast.Import):
                for alias in node.names:
                    assert alias.name not in retired, (path, alias.name)
