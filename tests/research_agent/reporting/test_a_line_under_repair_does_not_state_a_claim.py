"""A claim carried only by a line under repair is not yet stated in its block.

A bounded repair turns a rejected sentence into its host claim token, unless
the sentence's block already states that token; then it drops the sentence,
so the reader does not see the claim twice.  The block was searched as plain
text, so a token inside another rejected sentence counted, even when the same
repair dropped that sentence next.  Both sentences then left, and the claim
was gone from the block.  Results restores a missing claim; the Discussion and
the abstract do not.  Text a later decision removes or replaces now never
counts: a sentence to be dropped, with the dependent sentences it takes, or a
sentence to be replaced by a claim.  A sentence to be cited stays as written,
so its token still counts.

Generic manuscripts only; no study's values.
"""

from __future__ import annotations

import pytest

from easyicu.research_agent.reporting.manuscript_post import (
    _apply_writer_evidence_repair_decisions,
)
from easyicu.research_agent.reporting.writer_repair_decision import WriterRepairDecision

REF = "primary_model.adjusted_association"
TOKEN = "{claim:" + REF + "}"
EVIDENCE = "statistic_step_summary_primary_model"
RESTATEMENT = "Early therapy was associated with lower mortality in this cohort."
CARRIER = f"That difference {TOKEN} held in every subgroup the authors examined."
TAIL = "\n\n## Limitations\n\nInterpretation stays cautious.\n"


def _discussion(repaired: str) -> str:
    return repaired.split("## Discussion", 1)[1].split("## Limitations", 1)[0]


@pytest.mark.parametrize("claim_first", [True, False], ids=["claim_first", "drop_first"])
@pytest.mark.parametrize(
    ("header", "block"),
    [("## Discussion\n\n", "## Discussion"), ("## Abstract\n\n**Results:** ", "**Results:**")],
    ids=["discussion", "abstract"],
)
def test_a_claim_carried_only_by_a_dropped_line_stays_once(claim_first, header, block):
    text = f"{header}{RESTATEMENT} {CARRIER}{TAIL}"
    decisions = [WriterRepairDecision.claim(0, REF), WriterRepairDecision.drop(1)]

    repaired, receipts = _apply_writer_evidence_repair_decisions(
        text,
        missing_sentences=[RESTATEMENT, CARRIER],
        decisions=decisions if claim_first else decisions[::-1],
        allowed_claim_refs=[REF],
    )

    stated = repaired.split(block, 1)[1].split("## Limitations", 1)[0]
    assert stated.count(TOKEN) == 1
    assert CARRIER not in repaired and RESTATEMENT not in repaired
    assert {receipt["action"] for receipt in receipts} == {"claim", "drop"}


def test_a_claim_in_a_sentence_a_drop_takes_along_stays_once():
    opener = "Early therapy lowered the peak lactate concentration."
    dependent = f"This estimate {TOKEN} held in every subgroup the authors examined."
    text = f"## Discussion\n\n{opener} {dependent}\n\n{RESTATEMENT}{TAIL}"

    repaired, receipts = _apply_writer_evidence_repair_decisions(
        text,
        missing_sentences=[opener, RESTATEMENT],
        decisions=[WriterRepairDecision.claim(1, REF), WriterRepairDecision.drop(0)],
        allowed_claim_refs=[REF],
    )

    assert _discussion(repaired).count(TOKEN) == 1
    assert "This estimate" not in repaired and RESTATEMENT not in repaired
    (drop,) = [receipt for receipt in receipts if receipt["action"] == "drop"]
    assert drop["dependent_context_drops"] == [dependent]


@pytest.mark.parametrize("claim_first", [True, False], ids=["claim_first", "cite_first"])
def test_a_claim_in_a_line_to_be_cited_still_counts(claim_first):
    text = f"## Discussion\n\n{RESTATEMENT} {CARRIER}{TAIL}"
    decisions = [WriterRepairDecision.claim(0, REF), WriterRepairDecision.cite(1, [EVIDENCE])]

    repaired, receipts = _apply_writer_evidence_repair_decisions(
        text,
        missing_sentences=[RESTATEMENT, CARRIER],
        decisions=decisions if claim_first else decisions[::-1],
        allowed_evidence_ids=[EVIDENCE],
        allowed_claim_refs=[REF],
    )

    assert _discussion(repaired).count(TOKEN) == 1
    assert "{evidence:" + EVIDENCE + "}" in _discussion(repaired)
    assert RESTATEMENT not in repaired
    (claim,) = [receipt for receipt in receipts if receipt["index"] == 0]
    assert claim["reason_code"] == "claim_already_stated_in_block"


def test_a_line_outside_the_repair_still_states_the_claim():
    text = f"## Discussion\n\n{RESTATEMENT}\n{TOKEN}{TAIL}"

    repaired, [receipt] = _apply_writer_evidence_repair_decisions(
        text, missing_sentences=[RESTATEMENT], decisions=[WriterRepairDecision.claim(0, REF)],
        allowed_claim_refs=[REF],
    )

    assert repaired.count(TOKEN) == 1
    assert receipt["reason_code"] == "claim_already_stated_in_block"
