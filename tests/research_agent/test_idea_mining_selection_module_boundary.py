"""Boundary checks for feasibility-first Idea Mining selection."""

from __future__ import annotations

import ast
from pathlib import Path


def test_idea_mining_reexports_selection_by_identity() -> None:
    from easyicu.research_agent.discovery import idea_mining, idea_mining_selection

    assert (
        idea_mining.select_actionable_prior_art_screen
        is idea_mining_selection.select_actionable_prior_art_screen
    )


def test_idea_mining_selection_is_a_leaf_module() -> None:
    path = (
        Path(__file__).resolve().parents[2]
        / "src"
        / "easyicu"
        / "research_agent"
        / "discovery"
        / "idea_mining_selection.py"
    )
    tree = ast.parse(path.read_text(encoding="utf-8"))
    forbidden = {"idea_mining", "easyicu.research_agent.discovery.idea_mining"}
    assert not any(
        isinstance(node, ast.ImportFrom) and node.module in forbidden
        for node in tree.body
    )
