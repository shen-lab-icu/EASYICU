"""Canonical ownership contracts for evaluation and review modules."""

from __future__ import annotations

import ast
import importlib
import importlib.util
import os
from pathlib import Path
import subprocess
import sys

import pytest


MODULES = (
    "evaluation.cross_model_panel",
    "evaluation.tier2_jury",
    "evaluation.tier2_rubric",
    "review.causal_audit",
    "review.methodological_rigor",
)


@pytest.mark.parametrize("target", MODULES)
def test_evaluation_or_review_module_has_one_canonical_home(target: str) -> None:
    module = importlib.import_module(f"easyicu.research_agent.{target}")
    assert module.__name__ == f"easyicu.research_agent.{target}"


@pytest.mark.parametrize(
    "package",
    ("easyicu.research_agent.evaluation", "easyicu.research_agent.review"),
)
def test_responsibility_package_is_lazy(package: str) -> None:
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
            imported.add(
                importlib.util.resolve_name(
                    "." * node.level + node.module, package
                )
                if node.level
                else node.module
            )
    return imported


def test_command_line_tools_use_canonical_evaluation_paths() -> None:
    repo_root = Path(__file__).resolve().parents[2]
    expected = {
        "tools/run_cross_model_check.py": {
            "easyicu.research_agent.evaluation.cross_model_panel"
        },
        "tools/run_tier2_jury.py": {
            "easyicu.research_agent.evaluation.tier2_jury",
            "easyicu.research_agent.evaluation.tier2_rubric",
        },
    }
    for relative, required in expected.items():
        imports = _resolved_imports(
            repo_root / relative, relative.replace("/", ".").removesuffix(".py")
        )
        assert required.issubset(imports)


def test_root_public_causal_audit_uses_canonical_module() -> None:
    root = importlib.import_module("easyicu.research_agent")
    canonical = importlib.import_module(
        "easyicu.research_agent.review.causal_audit"
    )
    assert root.run_causal_audit is canonical.run_causal_audit
