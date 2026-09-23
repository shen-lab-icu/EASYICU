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
    # The age-group rule is consumed through the public export; the consumer
    # must not reach into a private helper of the leaf module.
    assert (
        idea_mining.population_matches_age_group
        is idea_mining_selection.population_matches_age_group
    )
    assert "population_matches_age_group" in idea_mining_selection.__all__


def test_idea_mining_selection_is_a_leaf_module() -> None:
    path = (
        Path(__file__).resolve().parents[3]
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
