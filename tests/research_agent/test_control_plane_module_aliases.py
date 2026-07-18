"""Compatibility contracts for responsibility-subpackage migrations."""

from __future__ import annotations

import importlib
import os
import subprocess
import sys
from pathlib import Path

import pytest


MODULE_ALIASES = (
    (
        "easyicu.research_agent.evidence_registration",
        "easyicu.research_agent.authority.registration",
    ),
    (
        "easyicu.research_agent.gate_evaluator",
        "easyicu.research_agent.gates.visual",
    ),
    (
        "easyicu.research_agent.contract_gate",
        "easyicu.research_agent.gates.contract",
    ),
)


@pytest.mark.parametrize("legacy,canonical", MODULE_ALIASES)
def test_legacy_path_is_the_canonical_module_object(legacy: str, canonical: str) -> None:
    assert importlib.import_module(legacy) is importlib.import_module(canonical)


@pytest.mark.parametrize("order", ("legacy_first", "canonical_first", "pipeline_first"))
def test_module_aliases_survive_clean_import_order(order: str) -> None:
    pairs = repr(MODULE_ALIASES)
    script = f"""
import importlib
pairs = {pairs}
if {order!r} == 'pipeline_first':
    importlib.import_module('easyicu.research_agent.pipeline_execute')
for legacy, canonical in pairs:
    names = (legacy, canonical) if {order!r} == 'legacy_first' else (canonical, legacy)
    first = importlib.import_module(names[0])
    second = importlib.import_module(names[1])
    assert first is second, (legacy, canonical, first, second)
"""
    env = dict(os.environ)
    source_root = str(Path(__file__).resolve().parents[2] / "src")
    env["PYTHONPATH"] = source_root + os.pathsep + env.get("PYTHONPATH", "")
    subprocess.run([sys.executable, "-c", script], check=True, env=env)
