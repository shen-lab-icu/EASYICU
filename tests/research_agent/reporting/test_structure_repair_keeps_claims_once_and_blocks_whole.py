"""Reader-structure repair is idempotent and never moves prose between blocks.

The Writer's assembly and the write phase each run the structure repair, the
latter more than once.  A Conclusion holding a claim token and the validation
caveat read as "caveat only", because the audit-markup strip removes tokens,
so every call prepended the primary claim again (one token became two, then
four).  And when filtering emptied the abstract Background, the label went to
the first unlabeled paragraph anywhere in the abstract: an owner-projected
Results sentence then printed under a second Background label.

Generic manuscripts only; no study's values.
"""

from __future__ import annotations

import re

import pytest

from easyicu.research_agent.reporting.manuscript_quality import (
    repair_reader_structure_from_existing_prose,
)

CLAIM = "{claim:primary_model.adjusted_association}"
CAVEAT = "Independent validation is required."


def _manuscript(*, abstract_conclusions: str, conclusion: str) -> str:
    return (
        "# Synthetic cohort study\n\n## Abstract\n\n"
        "**Background:** Delirium is common in the ICU [@ely_2001].\n\n"
        "**Methods:** We fitted a prespecified adjusted model.\n\n"
        f"**Results:**\n\n{CLAIM}\n\n"
        f"**Conclusions:** {abstract_conclusions}\n\n"
        "## Introduction\n\nDelirium is common in the ICU [@ely_2001].\n\n"
        "## Results\n\n### Cohort characteristics\n\n"
        "The analysis cohort included 1,200 ICU stays {evidence:cohort}.\n\n"
        f"### Primary association\n\n{CLAIM}\n\n"
        f"## Conclusion\n\n{conclusion}\n"
    )


def _codes(repairs) -> set[str]:
    return {repair["code"] for repair in repairs}


def _section(text: str, heading: str) -> str:
    return text.split(f"## {heading}\n", 1)[1].split("\n## ", 1)[0]


def test_a_conclusion_that_states_its_claim_is_left_as_written():
    source = _manuscript(abstract_conclusions=f"\n\n{CLAIM}\n\n{CAVEAT}", conclusion=f"{CLAIM}\n\n{CAVEAT}")

    repaired, repairs = repair_reader_structure_from_existing_prose(source)

    assert repaired == source
    assert not _codes(repairs) & {"MANUSCRIPT_CONCLUSION_RESTORED", "MANUSCRIPT_ABSTRACT_CONCLUSIONS_RESTORED"}


@pytest.mark.parametrize(
    ("abstract_conclusions", "conclusion", "restored"),
    [
        (CAVEAT, CAVEAT, {"MANUSCRIPT_CONCLUSION_RESTORED", "MANUSCRIPT_ABSTRACT_CONCLUSIONS_RESTORED"}),
        (f"\n\n{CLAIM}\n\n{CAVEAT}", CAVEAT, {"MANUSCRIPT_CONCLUSION_RESTORED"}),
        (CAVEAT, f"{CLAIM}\n\n{CAVEAT}", {"MANUSCRIPT_ABSTRACT_CONCLUSIONS_RESTORED"}),
    ],
)
def test_a_caveat_only_conclusion_gains_its_claim_once(abstract_conclusions, conclusion, restored):
    once, repairs = repair_reader_structure_from_existing_prose(
        _manuscript(abstract_conclusions=abstract_conclusions, conclusion=conclusion)
    )
    twice, again = repair_reader_structure_from_existing_prose(once)

    assert restored <= _codes(repairs)
    assert twice == once and again == ()
    assert _section(once, "Conclusion").count(CLAIM) == 1
    abstract_conclusions_block = _section(once, "Abstract").split("**Conclusions:**", 1)[1]
    assert abstract_conclusions_block.count(CLAIM) == 1
    assert CAVEAT in _section(once, "Conclusion")


def test_a_projected_results_paragraph_keeps_its_block_when_background_is_emptied():
    projected = "The unadjusted median length of stay was 4.0 days {evidence:los}."
    source = _manuscript(abstract_conclusions=CAVEAT, conclusion=f"{CLAIM}\n\n{CAVEAT}").replace(
        "**Background:** Delirium is common in the ICU [@ely_2001].", "**Background:**",
    ).replace(f"**Results:**\n\n{CLAIM}\n\n", f"**Results:**\n\n{CLAIM}\n\n{projected}\n\n")

    repaired, repairs = repair_reader_structure_from_existing_prose(source)
    abstract = _section(repaired, "Abstract")

    assert re.findall(r"\*\*Background:\*\*", abstract) == ["**Background:**"]
    assert "**Background:** Delirium is common in the ICU [@ely_2001]." in abstract
    results_block = abstract.split("**Results:**", 1)[1].split("**Conclusions:**", 1)[0]
    assert projected in results_block
    assert "MANUSCRIPT_ABSTRACT_LABEL_RESTORED" not in _codes(repairs)
    assert repair_reader_structure_from_existing_prose(repaired)[0] == repaired


def test_a_leading_unlabeled_abstract_paragraph_is_still_the_background():
    source = _manuscript(abstract_conclusions=CAVEAT, conclusion=f"{CLAIM}\n\n{CAVEAT}").replace(
        "**Background:** Delirium", "Delirium",
    )

    repaired, repairs = repair_reader_structure_from_existing_prose(source)

    assert "**Background:** Delirium is common in the ICU [@ely_2001]." in _section(repaired, "Abstract")
    assert "MANUSCRIPT_ABSTRACT_LABEL_RESTORED" in _codes(repairs)
