"""Canonical package boundary for planner/coder role implementations."""

from __future__ import annotations

import ast
import importlib
import inspect
import os
from pathlib import Path
import subprocess
import sys

import easyicu.research_agent as research_agent
from easyicu.research_agent import agents
from easyicu.research_agent.agents import core

PUBLIC_CORE_AGENTS = (
    "PlannerAgent",
    "ReplannerAgent",
    "ClinicalSemanticsAgent",
    "DataExtractionAgent",
    "StatisticalAnalysisAgent",
    "VisualizationAgent",
    "ManuscriptAgent",
    "CriticAgent",
    "RuntimeSupervisor",
    "CoderAgent",
    "AnalyzerAgent",
    "WriterAgent",
)


def test_agent_package_exports_match_core_and_root_identity() -> None:
    for name in PUBLIC_CORE_AGENTS:
        assert getattr(agents, name) is getattr(core, name)
        assert getattr(research_agent, name) is getattr(core, name)


def test_agent_package_import_is_lazy() -> None:
    script = """
import importlib
import sys
package = 'easyicu.research_agent.agents'
module = importlib.import_module(package)
assert module.__name__ == package
assert package + '.core' not in sys.modules
assert package + '.agentic_coder' not in sys.modules
"""
    env = dict(os.environ)
    source_root = str(Path(__file__).resolve().parents[3] / "src")
    env["PYTHONPATH"] = source_root + os.pathsep + env.get("PYTHONPATH", "")
    subprocess.run([sys.executable, "-c", script], check=True, env=env)


def test_flat_agent_implementation_files_are_absent() -> None:
    package_root = (
        Path(__file__).resolve().parents[3] / "src" / "easyicu" / "research_agent"
    )
    assert not (package_root / "agents.py").exists()
    assert not (package_root / "agentic_coder.py").exists()
    assert importlib.util.find_spec("easyicu.research_agent.agentic_coder") is None


def test_production_pipeline_imports_agent_implementations_directly() -> None:
    for module_name in (
        "easyicu.research_agent.pipeline",
        "easyicu.research_agent.execution.phase",
        "easyicu.research_agent.reporting.write_phase",
    ):
        module = importlib.import_module(module_name)
        tree = ast.parse(inspect.getsource(module))
        imported = {
            importlib.util.resolve_name(
                "." * node.level + (node.module or ""), module.__package__
            )
            for node in ast.walk(tree)
            if isinstance(node, ast.ImportFrom) and node.level
        }
        assert "easyicu.research_agent.agents" not in imported
        assert "easyicu.research_agent.agentic_coder" not in imported
        assert "easyicu.research_agent.agents.core" in imported


def test_manuscript_facade_forwards_reader_display_labels(monkeypatch) -> None:
    from easyicu.research_agent.agents import manuscript, roles

    captured = {}

    class _Writer:
        def __init__(self, *_args, **_kwargs) -> None:
            pass

        def run(self, **kwargs):
            captured.update(kwargs)
            return "draft"

    monkeypatch.setattr(manuscript, "WriterAgent", _Writer)
    agent = roles.ManuscriptAgent(object())

    result = agent.run(
        context=object(),
        evidence_ids=("result",),
        reader_display_labels={"raw_column": "Clinical label"},
    )

    assert result == "draft"
    assert captured["reader_display_labels"] == {
        "raw_column": "Clinical label"
    }
