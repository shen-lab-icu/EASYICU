"""Canonical ownership contracts for research context and method gates."""

from __future__ import annotations

import importlib
import os
from pathlib import Path
import subprocess
import sys

import pytest


CONTEXT_MODULES = (
    "research_context.builder",
    "research_context.prompt_scope",
    "research_context.typed",
    "gates.method_compatibility",
)


@pytest.mark.parametrize("target", CONTEXT_MODULES)
def test_context_module_has_one_canonical_home(target: str) -> None:
    module = importlib.import_module(f"easyicu.research_agent.{target}")
    assert module.__name__ == f"easyicu.research_agent.{target}"


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


def test_root_context_api_resolves_to_canonical_builder() -> None:
    root = importlib.import_module("easyicu.research_agent")
    builder = importlib.import_module(
        "easyicu.research_agent.research_context.builder"
    )
    assert root.build_research_context is builder.build_research_context
