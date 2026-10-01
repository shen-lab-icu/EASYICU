"""A bounded repair never prints the same host claim twice in one block.

A Writer often restates a claim token's estimate in its own words on the line
next to the token.  The strict gate rejects the restatement, and the bounded
repair may choose that same claim as its replacement.  The block then held the
token twice, and the reader saw the host sentence twice.  A claim decision for
a token its block already states is now applied as a drop, with a reason code.

Generic manuscripts only; no study's values.
"""

from __future__ import annotations

from easyicu.research_agent.reporting.manuscript_post import (
    _apply_writer_evidence_repair_decisions,
)
from easyicu.research_agent.reporting.writer_repair_decision import WriterRepairDecision

REF = "primary_model.adjusted_association"
TOKEN = "{claim:" + REF + "}"
RESTATEMENT = "After adjustment the odds ratio was 1.40 in exposed stays {evidence:primary_model}."


def _apply(text: str, target: str):
    return _apply_writer_evidence_repair_decisions(
        text,
        missing_sentences=[target],
        decisions=[WriterRepairDecision.claim(0, REF)],
        allowed_claim_refs=[REF],
    )


def test_a_restatement_next_to_its_token_is_dropped_not_duplicated():
    text = (
        f"## Results\n\n### Primary association\n\n{RESTATEMENT}\n{TOKEN}\n\n"
        "## Discussion\n\nInterpretation stays cautious.\n"
    )

    repaired, [receipt] = _apply(text, RESTATEMENT)

    assert repaired.count(TOKEN) == 1
    assert RESTATEMENT not in repaired
    assert "Interpretation stays cautious." in repaired
    assert receipt["action"] == "drop"
    assert receipt["reason_code"] == "claim_already_stated_in_block"
    assert receipt["claim_ref"] == REF


def test_the_same_claim_in_another_block_does_not_count():
    text = (
        f"## Results\n\n### Primary association\n\n{RESTATEMENT}\n\n"
        f"## Discussion\n\n{TOKEN}\n\nInterpretation stays cautious.\n"
    )

    repaired, [receipt] = _apply(text, RESTATEMENT)

    association = repaired.split("### Primary association", 1)[1].split("## Discussion", 1)[0]
    assert association.strip() == TOKEN
    assert repaired.count(TOKEN) == 2
    assert receipt["action"] == "claim" and "reason_code" not in receipt


def test_the_rejected_line_s_own_token_is_replaced_not_counted():
    target = f"**Results:** The analysis cohort included 1,200 ICU stays {{evidence:cohort}}. {TOKEN}"
    text = f"## Abstract\n\n{target}\n\n**Conclusions:** Independent validation is required.\n"

    repaired, [receipt] = _apply(text, target)

    assert f"**Results:**\n\n{TOKEN}\n\n**Conclusions:**" in repaired
    assert receipt["action"] == "claim"


def test_a_labelled_restatement_keeps_its_label_when_the_block_states_the_claim():
    target = f"**Results:** {RESTATEMENT}"
    text = (
        f"## Abstract\n\n{target}\n\n{TOKEN}\n\n"
        "**Conclusions:** Independent validation is required.\n"
    )

    repaired, [receipt] = _apply(text, target)

    assert repaired.count(TOKEN) == 1
    assert f"**Results:**\n\n{TOKEN}\n\n**Conclusions:**" in repaired
    assert receipt["reason_code"] == "claim_already_stated_in_block"
