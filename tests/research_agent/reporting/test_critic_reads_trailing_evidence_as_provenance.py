"""A sentence's evidence, placed right after its full stop, is its provenance.

The host renders a claim for a conclusion without its estimate, and the claim
names its contrast levels ("3 versus 1").  The evidence follows the sentence:
an ``{evidence:<id>}`` placeholder in the scaffold, a link once bound.  The
study here is a lactate tertile and ICU readmission.
"""

from __future__ import annotations

import pytest

from easyicu.research_agent import CriticAgent

_CLAIM = (
    "After adjustment for patient age and patient sex, first lactate tertile 3 "
    "versus 1 showed no clear association with ICU readmission and hospital "
    "mortality in the analysis set (adjusted odds ratio)."
)
_LINK = (
    '[statistic_step_summary_ab12](evidence/statistic_step_summary_ab12__step_summary.json '
    '"sha256=6ca69b4c")'
)


def _review(text: str):
    return CriticAgent().review_manuscript(scaffold=text, available_evidence_ids=[])


@pytest.mark.parametrize(
    "evidence",
    [_LINK, "{evidence:statistic_step_summary_ab12}", f"{_LINK} {_LINK}"],
)
def test_a_claim_followed_by_its_evidence_is_supported(evidence: str) -> None:
    critique = _review(f"## Conclusion\n\n{_CLAIM} {evidence}\n")

    assert critique.status == "pass"
    assert critique.unsupported_claims == []


def test_the_same_claim_without_evidence_is_flagged() -> None:
    critique = _review(f"## Conclusion\n\n{_CLAIM}\n")

    assert critique.status == "needs_revision"
    assert critique.unsupported_claims == [_CLAIM]


def test_evidence_supports_only_the_sentence_it_follows() -> None:
    uncited = "Hospital mortality was 12% in the cohort."
    referred = f"Hospital mortality was 14% in the cohort. See {_LINK} for the table."

    followed = _review(f"{uncited} {_CLAIM} {_LINK}\n")
    inside_the_next = _review(f"{referred}\n")

    assert followed.unsupported_claims == [uncited]
    assert inside_the_next.unsupported_claims == [
        "Hospital mortality was 14% in the cohort."
    ]
