"""Tests for the generic variable-kind × method compatibility matrix.

The matrix must classify variables by *kind* (ordinal / binary / count /
right-skewed continuous) — never by concept name. New kinds extend the
matrix; new benchmark cases do not. These tests pin that contract.
"""

from __future__ import annotations

import pytest

from easyicu.research_agent.schema import (
    AggregationRule,
    CohortDescriptor,
    ConceptDescriptor,
    ResearchContext,
    VariableRole,
)
from easyicu.research_agent.method_compatibility import (
    FORBIDDEN_METHOD_BY_KIND,
    _variable_kind,
    render_variable_constraints,
    variable_kind_constraints,
)


def _make_context(variables) -> ResearchContext:
    return ResearchContext(
        research_question="test",
        cohort=CohortDescriptor(
            cohort_name="t",
            database="miiv",
            n_patients=1,
            n_stays=1,
            id_columns=["stay_id"],
            outcome_columns=["death"],
        ),
        variables=variables,
    )


# ---------------------------------------------------------------------------
# Per-kind classification — these are the cells we promise to detect
# ---------------------------------------------------------------------------


def test_ordinal_score_role_is_ordinal_kind():
    v = ConceptDescriptor(name="gcs", dtype="int64", role=VariableRole.ORDINAL_SCORE)
    assert _variable_kind(v) == "ordinal"


def test_is_ordinal_flag_is_ordinal_kind():
    v = ConceptDescriptor(name="custom_score", dtype="int64", is_ordinal=True)
    assert _variable_kind(v) == "ordinal"


def test_bool_dtype_is_binary_kind():
    v = ConceptDescriptor(name="mech_vent", dtype="bool")
    assert _variable_kind(v) == "binary"


def test_int_0_1_range_is_binary_kind():
    v = ConceptDescriptor(name="vaso_any", dtype="int64", valid_range=[0, 1])
    assert _variable_kind(v) == "binary"


def test_sum_aggregation_is_count_kind():
    v = ConceptDescriptor(
        name="n_episodes",
        dtype="int64",
        allowed_aggregations=[AggregationRule.SUM],
    )
    assert _variable_kind(v) == "count"


def test_lab_named_lactate_is_right_skewed_kind():
    v = ConceptDescriptor(name="lactate", dtype="float64", role=VariableRole.LAB)
    assert _variable_kind(v) == "right_skewed_continuous"


def test_continuous_vital_without_matrix_entry_returns_none():
    """A normal continuous vital sign has no compatibility constraint."""
    v = ConceptDescriptor(name="hr", dtype="float64", role=VariableRole.VITAL)
    assert _variable_kind(v) is None


def test_id_column_returns_none():
    v = ConceptDescriptor(name="stay_id", dtype="int64", role=VariableRole.ID)
    assert _variable_kind(v) is None


# ---------------------------------------------------------------------------
# Matrix contract — no concept-specific patterns leak in
# ---------------------------------------------------------------------------


def test_matrix_keys_are_variable_kinds_not_concept_names():
    """The matrix must be keyed by variable *kind*, not by concept names.

    If this test starts failing because someone added e.g. ``"gcs"`` or
    ``"sofa2"`` as a top-level key, the addition is overfitting the
    agent to a specific benchmark case — see CLAUDE.md prompt-hygiene
    rule. Add a new kind instead.
    """
    expected_kinds = {"ordinal", "binary", "count", "right_skewed_continuous"}
    actual_keys = set(FORBIDDEN_METHOD_BY_KIND.keys())
    # Must be a subset of allowed kinds — extending the matrix with new
    # kinds is fine, leaking concept names is not.
    forbidden_substrings = (
        "gcs", "sofa", "kdigo", "vaso", "shock", "hepato", "renal",
        "mimic", "eicu", "hirid", "sicdb",
    )
    for key in actual_keys:
        assert not any(s in key.lower() for s in forbidden_substrings), (
            f"Matrix key '{key}' looks concept- or case-specific; "
            "add a new variable kind instead."
        )
    # And every kind we've shipped must still be present.
    assert expected_kinds.issubset(actual_keys)


def test_matrix_entries_have_required_fields():
    for kind, rule in FORBIDDEN_METHOD_BY_KIND.items():
        assert "forbidden_patterns" in rule, kind
        assert "preferred" in rule, kind
        assert "rationale" in rule, kind
        assert isinstance(rule["forbidden_patterns"], tuple)
        assert isinstance(rule["preferred"], tuple)
        assert rule["forbidden_patterns"], f"{kind} has empty forbidden list"
        assert rule["preferred"], f"{kind} has empty preferred list"


# ---------------------------------------------------------------------------
# Rendering — what reaches the LLM
# ---------------------------------------------------------------------------


def test_render_returns_empty_when_no_constraints_match():
    """A continuous-only cohort produces no constraint block."""
    ctx = _make_context([
        ConceptDescriptor(name="hr", dtype="float64", role=VariableRole.VITAL),
        ConceptDescriptor(name="map", dtype="float64", role=VariableRole.VITAL),
    ])
    assert render_variable_constraints(ctx) == ""


def test_render_lists_one_line_per_constrained_variable():
    ctx = _make_context([
        ConceptDescriptor(name="gcs", dtype="int64", role=VariableRole.ORDINAL_SCORE),
        ConceptDescriptor(name="lactate", dtype="float64", role=VariableRole.LAB),
        ConceptDescriptor(name="mech_vent", dtype="bool"),
        ConceptDescriptor(name="hr", dtype="float64", role=VariableRole.VITAL),
    ])
    block = render_variable_constraints(ctx)
    # Three constrained variables → three "  - `<name>`" lines.
    bullet_count = block.count("  - `")
    assert bullet_count == 3, block
    assert "`gcs`" in block
    assert "`lactate`" in block
    assert "`mech_vent`" in block
    # HR has no entry — must not be listed.
    assert "`hr`" not in block


def test_render_includes_machine_readable_kind_label():
    """The LLM sees the kind label so it can reason generically.

    The kind label is intentionally one of the four matrix keys — not
    a clinical concept name. A reviewer can audit by greping for these
    four strings in any envelope.
    """
    ctx = _make_context([
        ConceptDescriptor(name="kdigo_stage", dtype="int64", role=VariableRole.ORDINAL_SCORE),
    ])
    block = render_variable_constraints(ctx)
    assert "(kind: ordinal)" in block
    assert "DO NOT use" in block
    assert "PREFERRED" in block


def test_machine_readable_constraints_for_ordinal_block_kmeans():
    """The matrix MUST forbid distance-based clustering for ordinal vars.

    Locks the policy that motivated Patch B in the first place: the
    rep1/rep2 shock_discordance halt was caused by KMeans-on-ordinal.
    """
    ctx = _make_context([
        ConceptDescriptor(name="any_ordinal", dtype="int64", is_ordinal=True),
    ])
    rules = variable_kind_constraints(ctx.variables)
    assert len(rules) == 1
    patterns = [p.lower() for p in rules[0]["forbidden_patterns"]]
    assert any("kmeans" in p or "k-means" in p or "k_means" in p for p in patterns)
