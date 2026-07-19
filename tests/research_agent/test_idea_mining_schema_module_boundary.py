"""Boundary checks for the extracted idea-mining schema module."""

from __future__ import annotations

import ast
from pathlib import Path


def test_idea_mining_schema_entrypoints_are_importable() -> None:
    from easyicu.research_agent.idea_mining_schema import (
        IdeaMiningError,
        LiteratureIdeaCandidate,
        _sha256_text,
        _stable_idea_id,
    )

    assert issubclass(IdeaMiningError, RuntimeError)
    assert LiteratureIdeaCandidate.__name__ == "LiteratureIdeaCandidate"
    assert _sha256_text("x") == _sha256_text("x")  # deterministic
    assert _stable_idea_id({"a": 1}) == _stable_idea_id({"a": 1})


def test_idea_mining_reexports_schema_by_identity() -> None:
    """Zero-behavior-change split: idea_mining keeps exposing the same model
    and helper objects so existing references and isinstance checks hold."""
    from easyicu.research_agent import idea_mining, idea_mining_schema

    for name in (
        "LiteratureIdeaCandidate",
        "ExecutableHypothesisCandidate",
        "PriorArtAssessment",
        "IdeaMiningError",
        "NoveltyLabel",
        "_stable_idea_id",
        "DISCOVERY_REPORT_SCHEMA_VERSION",
    ):
        assert getattr(idea_mining, name) is getattr(idea_mining_schema, name)


def test_idea_mining_schema_is_a_leaf_module() -> None:
    """The schema module must not import idea_mining at module top (cycle)."""
    path = (
        Path(__file__).resolve().parents[2]
        / "src"
        / "easyicu"
        / "research_agent"
        / "discovery"
        / "idea_mining_schema.py"
    )
    tree = ast.parse(path.read_text(encoding="utf-8"))
    forbidden = {
        "idea_mining",
        "easyicu.research_agent.idea_mining",
        "idea_mining_pubmed",
    }
    assert not any(
        isinstance(node, ast.ImportFrom) and node.module in forbidden
        for node in tree.body
    )
