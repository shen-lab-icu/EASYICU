"""Compatibility and dependency contracts for repair-control modules."""

from __future__ import annotations

import importlib
import os
from pathlib import Path
import subprocess
import sys

import pytest

REPAIR_CONTROL_MODULE_ALIASES = (
    (
        "easyicu.research_agent.provider_budget",
        "easyicu.research_agent.authority.provider_budget",
    ),
)


@pytest.mark.parametrize("legacy,canonical", REPAIR_CONTROL_MODULE_ALIASES)
def test_repair_control_legacy_path_is_canonical_module_object(
    legacy: str,
    canonical: str,
) -> None:
    old_module = importlib.import_module(legacy)
    new_module = importlib.import_module(canonical)
    assert old_module is new_module
    assert old_module.__file__ == new_module.__file__


@pytest.mark.parametrize("order", ("legacy_first", "canonical_first", "pipeline_first"))
def test_repair_control_aliases_survive_clean_import_order(order: str) -> None:
    script = f"""
import importlib
pairs = {REPAIR_CONTROL_MODULE_ALIASES!r}
if {order!r} == 'pipeline_first':
    importlib.import_module('easyicu.research_agent.pipeline_execute')
for legacy, canonical in pairs:
    names = (legacy, canonical) if {order!r} == 'legacy_first' else (canonical, legacy)
    assert importlib.import_module(names[0]) is importlib.import_module(names[1])
"""
    env = dict(os.environ)
    source_root = str(Path(__file__).resolve().parents[2] / "src")
    env["PYTHONPATH"] = source_root + os.pathsep + env.get("PYTHONPATH", "")
    subprocess.run([sys.executable, "-c", script], check=True, env=env)


def test_provider_budget_legacy_monkeypatch_owner_is_canonical() -> None:
    legacy = importlib.import_module("easyicu.research_agent.provider_budget")
    canonical = importlib.import_module(
        "easyicu.research_agent.authority.provider_budget"
    )
    assert legacy.os is canonical.os
