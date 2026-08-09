"""One citation per sentence is not enough when the numbers have two owners.

The writer rule said every result-like sentence must carry
``{evidence:<id>}`` "somewhere in the same sentence" -- singular. The binder
restricts each value to the evidence the sentence names, and refuses when no
candidate belongs to it rather than attaching another step's number to the
claim. That refusal is correct and is not being weakened here.

Three consecutive complete runs were blocked by exactly one value, and the last
two are the same shape:

  * canary39: "The source cohort comprised 94,458 ICU stays identified by
    `stay_id`." cited ``02_table_one``; the count belongs to
    ``01_cohort_definition_flow``.
  * canary40: "... ICU length of stay had 14 missing observations among 94,458
    stays." cited ``03_measurement_and_missingness_audit``, which owns the 14
    and never reported the 94,458. Its record's declared inputs are the cohort
    TABLE and the development cohort -- not the cohort step's summary, where
    ``n_universe`` lives -- so lineage does not reach it either, and it should
    not: letting any downstream step own an ancestor's number would hollow out
    the provenance guard rather than fix the sentence.

canary37 was a different shape (the owner WAS cited, just outside the sentence
window) and was fixed in the binder. This one belongs to the writer.
"""

from __future__ import annotations

import re
from pathlib import Path

_WRITER_PROMPT = (
    Path(__file__).resolve().parents[2]
    / "src"
    / "easyicu"
    / "research_agent"
    / "providers"
    / "prompts"
    / "v1"
    / "writer.txt"
)


def _prompt() -> str:
    return _WRITER_PROMPT.read_text(encoding="utf-8")


def test_the_rule_is_per_number_not_per_sentence():
    text = _prompt()

    assert "ONE ID PER NUMBER, NOT ONE PER SENTENCE" in text
    assert "must cite EVERY step that owns" in text


def test_it_gives_the_two_owner_example_the_runs_actually_produced():
    """An abstract rule is easy to satisfy in the wrong direction."""

    text = _prompt()

    assert "14 missing observations among 94,458 stays" in text
    assert "takes two ids" in text


def test_it_states_the_consequence_of_citing_only_one():
    """Without the cost, this reads as a style preference."""

    text = _prompt()

    assert "blocks the whole manuscript" in text
    assert "refuses rather than attaching another step's number" in text


def test_it_says_what_to_do_instead_of_dropping_a_citation():
    """The writer already has a rule that says 'drop the sentence'; this must
    not be read as licence to drop the citation instead."""

    text = _prompt()

    assert "split the\n    sentence -- do not drop a citation" in text


def test_the_older_one_per_sentence_rule_is_still_there():
    """The new rule refines the old one; it must not replace it."""

    text = _prompt()

    assert "MUST include `{evidence:<id>}` somewhere in the same" in text


def test_the_rule_precedes_the_section_scope_paragraph():
    """It has to be read with the audit rule it refines, not after the
    section list that follows it."""

    text = _prompt()

    assert text.index("ONE ID PER NUMBER") < text.index(
        "This rule applies to every manuscript section"
    )


def test_no_case_specific_token_entered_the_rule():
    """The example is a shape, not a benchmark: numbers only, no concepts."""

    text = _prompt()
    start = text.index("ONE ID PER NUMBER")
    window = text[start : start + 900].casefold()
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
