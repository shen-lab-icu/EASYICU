"""Compatibility and dependency contracts for reporting modules."""

from __future__ import annotations

import ast
import importlib
import inspect
import os
from pathlib import Path
import subprocess
import sys

import pytest

REPORTING_MODULE_ALIASES = tuple(
    (
        f"easyicu.research_agent.{leaf}",
        f"easyicu.research_agent.reporting.{leaf}",
    )
    for leaf in (
        "pdf_render",
        "reporting_checklist",
        "reviewer",
        "review_artifacts",
        "display_suite",
        "article_contract",
        "bibtex",
        "latex",
        "manuscript_post",
    )
)


@pytest.mark.parametrize("legacy,canonical", REPORTING_MODULE_ALIASES)
def test_reporting_legacy_path_is_canonical_module_object(
    legacy: str,
    canonical: str,
) -> None:
    old_module = importlib.import_module(legacy)
    new_module = importlib.import_module(canonical)
    assert old_module is new_module
    assert old_module.__file__ == new_module.__file__
    assert "/reporting/" in Path(new_module.__file__).as_posix()


@pytest.mark.parametrize("order", ("legacy_first", "canonical_first", "root_first"))
def test_reporting_aliases_survive_clean_import_order(order: str) -> None:
    script = f"""
import importlib
pairs = {REPORTING_MODULE_ALIASES!r}
if {order!r} == 'root_first':
    root = importlib.import_module('easyicu.research_agent')
    getattr(root, 'render_pdf_for_run')
for legacy, canonical in pairs:
    names = (legacy, canonical) if {order!r} == 'legacy_first' else (canonical, legacy)
    assert importlib.import_module(names[0]) is importlib.import_module(names[1])
"""
    env = dict(os.environ)
    source_root = str(Path(__file__).resolve().parents[2] / "src")
    env["PYTHONPATH"] = source_root + os.pathsep + env.get("PYTHONPATH", "")
    subprocess.run([sys.executable, "-c", script], check=True, env=env)


def test_reporting_package_is_lazy_and_does_not_import_pipeline_modules() -> None:
    package = importlib.import_module("easyicu.research_agent.reporting")
    package_tree = ast.parse(inspect.getsource(package))
    assert not [node for node in ast.walk(package_tree) if isinstance(node, ast.Import)]
    assert not [
        node for node in ast.walk(package_tree) if isinstance(node, ast.ImportFrom)
    ]

    for _legacy, canonical in REPORTING_MODULE_ALIASES:
        tree = ast.parse(inspect.getsource(importlib.import_module(canonical)))
        imported_modules = {
            node.module or ""
            for node in ast.walk(tree)
            if isinstance(node, ast.ImportFrom)
        }
        assert not any(name.startswith("pipeline") for name in imported_modules)


def test_root_lazy_reporting_api_uses_canonical_objects() -> None:
    root = importlib.import_module("easyicu.research_agent")
    pdf = importlib.import_module("easyicu.research_agent.reporting.pdf_render")
    checklist = importlib.import_module(
        "easyicu.research_agent.reporting.reporting_checklist"
    )
    reviewer = importlib.import_module("easyicu.research_agent.reporting.reviewer")
    latex = importlib.import_module("easyicu.research_agent.reporting.latex")
    bibtex = importlib.import_module("easyicu.research_agent.reporting.bibtex")
    assert root.render_pdf_for_run is pdf.render_pdf_for_run
    assert root.choose_checklist is checklist.choose_checklist
    assert root.run_reviewer_round is reviewer.run_reviewer_round
    assert root.scaffold_to_latex is latex.scaffold_to_latex
    assert root.render_bibtex is bibtex.render_bibtex
