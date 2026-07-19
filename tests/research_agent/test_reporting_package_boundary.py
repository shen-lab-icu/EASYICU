"""Dependency contracts for canonical reporting modules."""

from __future__ import annotations

import ast
import importlib
import inspect
import os
from pathlib import Path
import subprocess
import sys

import pytest

REPORTING_MODULES = (
    "pdf_render",
    "reporting_checklist",
    "reviewer",
    "review_artifacts",
    "display_suite",
    "article_contract",
    "bibtex",
    "latex",
    "manuscript_post",
    "readiness",
    "side_findings",
    "write_phase",
    "writer_evidence",
)


@pytest.mark.parametrize("leaf", REPORTING_MODULES)
def test_reporting_module_has_one_canonical_home(leaf: str) -> None:
    module = importlib.import_module(f"easyicu.research_agent.reporting.{leaf}")
    assert module.__name__.endswith(f"reporting.{leaf}")
    assert "/reporting/" in Path(module.__file__).as_posix()


def test_reporting_package_is_lazy_and_does_not_import_pipeline_modules() -> None:
    script = """
import importlib
import sys
package = 'easyicu.research_agent.reporting'
importlib.import_module(package)
loaded = sorted(name for name in sys.modules if name.startswith(package + '.'))
assert loaded == [], loaded
"""
    env = dict(os.environ)
    source_root = str(Path(__file__).resolve().parents[2] / "src")
    env["PYTHONPATH"] = source_root + os.pathsep + env.get("PYTHONPATH", "")
    subprocess.run([sys.executable, "-c", script], check=True, env=env)

    for leaf in REPORTING_MODULES:
        tree = ast.parse(
            inspect.getsource(
                importlib.import_module(f"easyicu.research_agent.reporting.{leaf}")
            )
        )
        imported = {
            node.module or ""
            for node in ast.walk(tree)
            if isinstance(node, ast.ImportFrom)
        }
        assert imported.isdisjoint({"pipeline", "pipeline_execute"})


def test_root_reporting_api_uses_canonical_objects() -> None:
    root = importlib.import_module("easyicu.research_agent")
    pdf = importlib.import_module("easyicu.research_agent.reporting.pdf_render")
    checklist = importlib.import_module(
        "easyicu.research_agent.reporting.reporting_checklist"
    )
    reviewer = importlib.import_module("easyicu.research_agent.reporting.reviewer")
    assert root.render_pdf_for_run is pdf.render_pdf_for_run
    assert root.choose_checklist is checklist.choose_checklist
    assert root.run_reviewer_round is reviewer.run_reviewer_round


def test_pipeline_reporting_helpers_use_canonical_objects() -> None:
    pipeline = importlib.import_module("easyicu.research_agent.pipeline")
    readiness = importlib.import_module("easyicu.research_agent.reporting.readiness")
    writer = importlib.import_module("easyicu.research_agent.reporting.writer_evidence")
    for name in (
        "execution_gate_status",
        "render_report",
        "write_readiness_artifacts",
    ):
        assert getattr(pipeline, name) is getattr(readiness, name)
    for name in (
        "_preferred_writer_evidence_names",
        "_render_writer_evidence_digest",
        "_render_writer_evidence_digest_v2",
    ):
        assert getattr(pipeline, name) is getattr(writer, name)
