"""Boundary checks for the extracted PubMed/prior-art helper module."""

from __future__ import annotations

import ast
from pathlib import Path


def test_idea_mining_pubmed_entrypoints_are_importable() -> None:
    from easyicu.research_agent.discovery.idea_mining_pubmed import (
        _ordered_unique,
        _pubmed_phrase_clause,
        _prior_art_query_tokens,
    )

    assert _pubmed_phrase_clause("septic shock") == '"septic shock"[Title/Abstract]'
    assert _ordered_unique(["a", "a", "b"]) == ["a", "b"]
    assert _prior_art_query_tokens("lactate clearance") == ["lactate", "clearance"]


def test_idea_mining_reexports_pubmed_helpers_by_identity() -> None:
    """The split must be behavior-preserving: idea_mining keeps exposing the
    same function objects so existing internal references resolve unchanged."""
    from easyicu.research_agent.discovery import idea_mining, idea_mining_pubmed

    for name in (
        "_pubmed_phrase_clause",
        "_pubmed_recall_clause",
        "_prior_art_phrase_facets",
        "_ordered_unique",
        "_PRIOR_ART_QUERY_STOPWORDS",
    ):
        assert getattr(idea_mining, name) is getattr(idea_mining_pubmed, name)


def test_idea_mining_pubmed_is_a_leaf_module() -> None:
    """The helper module must not import idea_mining at module top (would be a
    cycle, since idea_mining imports it)."""
    path = (
        Path(__file__).resolve().parents[2]
        / "src"
        / "easyicu"
        / "research_agent"
        / "discovery"
        / "idea_mining_pubmed.py"
    )
    tree = ast.parse(path.read_text(encoding="utf-8"))
    forbidden = {"idea_mining", "easyicu.research_agent.discovery.idea_mining"}
    assert not any(
        isinstance(node, ast.ImportFrom) and node.module in forbidden
        for node in tree.body
    )
