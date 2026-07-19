"""Boundary checks for the extracted prior-art / discovery-report module."""

from __future__ import annotations

import ast
from pathlib import Path


def test_idea_mining_priorart_entrypoints_are_importable() -> None:
    from easyicu.research_agent.idea_mining_priorart import (
        _escape_md_cell,
        _saturation_for_novelty_label,
    )

    # Behavioral canary: pure helpers keep their exact semantics after the move.
    assert _escape_md_cell("a|b\nc") == "a\\|b<br>c"
    assert _saturation_for_novelty_label("already_done") == 0.95
    assert _saturation_for_novelty_label("apparently_gap") == 0.05


def test_idea_mining_reexports_priorart_by_identity() -> None:
    """The split must be behavior-preserving: idea_mining keeps exposing the
    same function objects so existing internal references resolve unchanged."""
    from easyicu.research_agent import idea_mining, idea_mining_priorart

    for name in (
        "build_prior_art_queries",
        "assess_prior_art_for_idea",
        "assess_prior_art_for_candidates",
        "render_discovery_report",
        "_go_no_go_decision",
        "_label_prior_art",
        "_discovery_report_counts",
        "_escape_md_cell",
    ):
        assert getattr(idea_mining, name) is getattr(idea_mining_priorart, name)


def test_idea_mining_priorart_is_a_leaf_module() -> None:
    """The helper module must not import idea_mining at module top (would be a
    cycle, since idea_mining imports it)."""
    path = (
        Path(__file__).resolve().parents[2]
        / "src"
        / "easyicu"
        / "research_agent"
        / "discovery"
        / "idea_mining_priorart.py"
    )
    tree = ast.parse(path.read_text(encoding="utf-8"))
    forbidden = {"idea_mining", "easyicu.research_agent.idea_mining"}
    assert not any(
        isinstance(node, ast.ImportFrom) and node.module in forbidden
        for node in tree.body
    )
