"""Dependency contracts for canonical reporting modules."""

from __future__ import annotations

import ast
import importlib
import inspect
from pathlib import Path

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
)


@pytest.mark.parametrize("leaf", REPORTING_MODULES)
def test_reporting_module_has_one_canonical_home(leaf: str) -> None:
    module = importlib.import_module(f"easyicu.research_agent.reporting.{leaf}")
    assert module.__name__.endswith(f"reporting.{leaf}")
    assert "/reporting/" in Path(module.__file__).as_posix()


def test_reporting_package_is_lazy_and_does_not_import_pipeline_modules() -> None:
    package = importlib.import_module("easyicu.research_agent.reporting")
    package_tree = ast.parse(inspect.getsource(package))
    assert not [node for node in ast.walk(package_tree) if isinstance(node, ast.Import)]
    assert not [
        node for node in ast.walk(package_tree) if isinstance(node, ast.ImportFrom)
    ]
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
        assert not any(name.startswith("pipeline") for name in imported)


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
