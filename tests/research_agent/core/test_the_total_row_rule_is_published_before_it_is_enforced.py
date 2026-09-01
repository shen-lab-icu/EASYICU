"""The host refuses a total row it never told the Coder to label.

``audits/aggregate_row.py`` raises ``emitted_table_aggregate_row`` as an
``error`` when a written table holds a row equal to the sum of the others in
several independent count columns and nothing in the table says it is a total.
The refusal is right -- a consumer reading only those bytes doubles every
denominator it sums -- and its message names the exact remedy: a ``row_role``
column marking that row ``overall`` and each partition row ``exposure_level``.

The Coder prompt said nothing about any of it. Measured 2026-08-02 over every
recorded run: 16 repair triggers across 6 distinct (run, step) pairs. In
canary41 it took TWO of E3's three failing steps -- the stage-stratified figure
and the secondary length-of-stay model -- so the recurring cost is a repair
spent on relabelling instead of on the analysis, and sometimes the step.

Publishing the rule where the table is written is the whole change. The
validator, its severity and its spellings are untouched.
"""

from __future__ import annotations

import re
from pathlib import Path

from easyicu.research_agent.audits.aggregate_row import (
    AGGREGATE_ROW_ROLE_COLUMNS,
    LEVEL_ROW_ROLE,
    OVERALL_ROW_ROLE,
)

_CODER_PROMPT = (
    Path(__file__).resolve().parents[2]
    / "src"
    / "easyicu"
    / "research_agent"
    / "providers"
    / "prompts"
    / "v1"
    / "coder.txt"
)


def _prompt() -> str:
    return _CODER_PROMPT.read_text(encoding="utf-8")


def test_the_prompt_names_the_two_role_values_the_validator_accepts():
    """Anchored on the audit module, so renaming a role breaks this test."""

    text = _prompt()

    assert f"`{OVERALL_ROW_ROLE}`" in text
    assert f"`{LEVEL_ROW_ROLE}`" in text


def test_the_prompt_names_the_default_role_column():
    text = _prompt()

    assert "row_role" in text
    assert "row_role" in AGGREGATE_ROW_ROLE_COLUMNS


def test_every_accepted_role_column_is_offered():
    """A Coder that picks a legal alternative spelling must not be refused
    for having read only half the contract."""

    text = _prompt()
    window = text[text.index("TOTAL / OVERALL row") :][:900]

    for column in AGGREGATE_ROW_ROLE_COLUMNS:
        assert column in window, column


def test_it_states_the_reason_not_just_the_rule():
    """Without the consequence this reads as formatting pedantry."""

    text = _prompt()

    assert "double every denominator it sums" in text


def test_it_offers_the_simpler_way_out():
    """Labelling is one answer; not writing the row is the other, and the
    Coder should know it is allowed."""

    text = _prompt()

    assert "not write a\n  total row at all" in text


def test_the_rule_sits_with_the_other_table_writing_rules():
    """It has to be read where tables are written, not in the figure block."""

    text = _prompt()

    assert text.index("TOTAL / OVERALL row") < text.index(
        "When reporting a source-status count map"
    )


def test_the_validator_still_refuses_and_still_says_how():
    """This publishes the rule; it must not soften the gate."""

    from easyicu.research_agent.audits import aggregate_row

    source = aggregate_row.__loader__.get_source(aggregate_row.__name__)
    assert 'severity="error"' in source
    assert "row_role" in source


def test_no_case_specific_token_entered_the_rule():
    text = _prompt()
    window = text[text.index("TOTAL / OVERALL row") :][:900].casefold()
    for token in (
        "sep3",
        "kdigo",
        "aki_stage",
        "mimic",
        "sepsis",
        "lactate",
        "e1",
        "e3",
    ):
        assert not re.search(rf"\b{re.escape(token)}\b", window), token
