"""Observed value-domain facts surfaced to the planner.

Motivating failure (E1, sepsis-3): the per-stay BINARY sepsis criterion was
exported with summary suffixes (``sep3_sofa2_max`` etc.). Seeing only the column
NAME + dtype, the planner read ``sep3_sofa2_max`` as a 0-24 SOFA score and
derived ``sepsis3 = (sep3_sofa2_max >= 2)`` -> 0 positives -> a degenerate
exposure -> self-blocked the whole association.

The fix is to give the planner the column's ACTUAL observed value domain so it
reasons from the data, not the name. These tests lock the two properties that
matters: a binary {0,1} column is flagged binary (so a ``>= 2`` cutoff is
obviously degenerate), and a genuinely continuous 0-24 score is NOT — the hint
states facts, it never prescribes a derivation.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from easyicu.research_agent.agents.core import _format_observed_domain
from easyicu.research_agent.cohort.artifact_facts import observed_domain_for_series
from easyicu.research_agent.research_context.builder import _observed_domain
from easyicu.research_agent.research_context.prompt_variables import (
    project_observed_domain,
)


def test_legacy_observed_domain_name_is_canonical_object() -> None:
    assert _observed_domain is observed_domain_for_series


def test_binary_criterion_flagged_binary():
    # sep3_sofa2_max-like: within-window max of a per-stay binary criterion.
    s = pd.Series([0, 1, 0, 1, 1, 0, 0] * 100, dtype="int64")
    d = _observed_domain(s)
    assert d["is_binary"] is True
    assert d["is_constant"] is False
    assert d["n_unique"] == 2
    hint = _format_observed_domain(d)
    assert "BINARY" in hint and "degenerate" in hint


def test_continuous_score_not_flagged_binary():
    # A genuine 0-24 SOFA score must NOT be treated as binary.
    s = pd.Series(list(range(0, 25)) * 10, dtype="int64")
    d = _observed_domain(s)
    assert d["is_binary"] is False
    assert d["min"] == 0.0 and d["max"] == 24.0
    hint = _format_observed_domain(d)
    assert "NUMERIC" in hint and "n_unique=25" in hint and "BINARY" not in hint
    assert "[0,24]" not in hint


def test_constant_column_flagged_constant():
    s = pd.Series([0] * 50, dtype="int64")
    d = _observed_domain(s)
    assert d["is_constant"] is True
    assert "CONSTANT" in _format_observed_domain(d)


def test_all_missing_returns_none():
    s = pd.Series([np.nan, np.nan, np.nan])
    assert _observed_domain(s) is None
    assert _format_observed_domain(None) == ""


def test_float_continuous_reports_range_not_binary():
    s = pd.Series([1.2, 3.4, 5.6, 2.1, 9.9] * 20)
    d = _observed_domain(s)
    assert d["is_binary"] is False
    assert d["min"] == 1.2 and d["max"] == 9.9


def test_two_level_nonzero_numeric_is_not_binary():
    # {1, 2} is two-level but NOT {0,1} — not a binary indicator; do not claim it
    # is (an agent might legitimately compare/threshold a 2-category ordinal).
    s = pd.Series([1, 2, 1, 2, 2] * 20, dtype="int64")
    d = _observed_domain(s)
    assert d["is_binary"] is False
    assert d["n_unique"] == 2


def test_categorical_two_level_is_not_numeric_binary():
    # A 2-level STRING column (e.g. sex {Male, Female}) must NOT be labelled a
    # numeric {0,1} binary — that misled the agent into coercing it to {0,1} and
    # spiralling into a "sex cleaning failure" repair loop. It keeps its labels.
    s = pd.Series(["Male", "Female", "Male", "Female", "Female"] * 20, dtype="object")
    d = _observed_domain(s)
    assert d["is_binary"] is False
    assert d["levels"] == ["Female", "Male"]
    hint = _format_observed_domain(d)
    assert "{0,1}" not in hint and "Male" not in hint
    assert "CATEGORICAL" in hint and "n_unique=2" in hint


def test_high_cardinality_categorical_omits_levels():
    s = pd.Series([f"cat{i}" for i in range(30)] * 3, dtype="object")
    d = _observed_domain(s)
    assert "levels" not in d
    assert d["is_binary"] is False


def test_opaque_tokens_require_host_bindable_levels():
    projection = project_observed_domain(
        {"is_categorical": True, "is_binary": False, "n_unique": 2}
    )
    assert projection == {"shape": "unknown", "n_unique": 2}


def test_descriptor_carries_observed_domain():
    # End-to-end: build_research_context attaches observed_domain so the binary
    # criterion is visible to the planner.
    from easyicu.research_agent.research_context.builder import build_research_context

    df = pd.DataFrame(
        {
            "stay_id": range(200),
            "age": np.random.randint(18, 90, 200).astype("float32"),
            "sep3_sofa2_max": ([0, 1] * 100),
            "death": ([0] * 180 + [1] * 20),
        }
    )
    ctx = build_research_context(
        cohort=df,
        cohort_name="t",
        research_question="q",
        target_outcome="death",
        database="miiv",
    )
    by_name = {v.name: v for v in ctx.variables}
    assert by_name["sep3_sofa2_max"].observed_domain["is_binary"] is True
    assert by_name["age"].observed_domain["is_binary"] is False
