"""Boundary checks for the extracted deterministic-repair helper module."""

from __future__ import annotations

import ast
from pathlib import Path


def test_code_repair_helpers_entrypoints_are_importable() -> None:
    from easyicu.research_agent.code_repair_helpers import (
        _extract_required_cols_list,
        _family_allows_binary_model_repair,
    )

    # Behavioral canary: pure helpers keep their exact semantics after the move.
    assert _family_allows_binary_model_repair(None) is True
    assert _family_allows_binary_model_repair("association_study") is True
    assert _family_allows_binary_model_repair("not_a_family") is False
    assert _extract_required_cols_list('required_cols = ["a", "b"]') == ["a", "b"]
    assert _extract_required_cols_list("no list here") == []


def test_code_repair_reexports_helpers_by_identity() -> None:
    """The split must be behavior-preserving: code_repair keeps exposing the
    same objects so existing internal references (and pipeline imports) resolve
    unchanged."""
    from easyicu.research_agent import code_repair, code_repair_helpers

    for name in (
        "_primary_association_fallback_code",
        "_ordinal_primary_association_fallback_code",
        "_patch_primary_predictor_into_design_matrix",
        "_strip_columns_from_list_literals",
        "_extract_missing_index_columns",
        "_patch_json_dump_numpy_key_sanitizer",
        "_BINARY_MODEL_REPAIR_FAMILIES",
        "_KEYERROR_NOT_IN_INDEX_RE",
    ):
        assert getattr(code_repair, name) is getattr(code_repair_helpers, name)


def test_code_repair_helpers_is_a_leaf_module() -> None:
    """The helper module must not import code_repair at module top (would be a
    cycle, since code_repair imports it)."""
    path = (
        Path(__file__).resolve().parents[2]
        / "src"
        / "easyicu"
        / "research_agent"
        / "code_repair_helpers.py"
    )
    tree = ast.parse(path.read_text(encoding="utf-8"))
    forbidden = {"code_repair", "easyicu.research_agent.code_repair"}
    assert not any(
        isinstance(node, ast.ImportFrom) and node.module in forbidden
        for node in tree.body
    )
