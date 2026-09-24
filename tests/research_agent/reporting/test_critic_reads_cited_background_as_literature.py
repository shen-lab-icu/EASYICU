"""The manuscript critic tells cited background apart from unsupported results."""

from __future__ import annotations

import pytest

from easyicu.research_agent import CriticAgent


def _review(sentence: str):
    return CriticAgent().review_manuscript(scaffold=sentence, available_evidence_ids=[])


@pytest.mark.parametrize(
    "sentence",
    [
        "KDIGO provides a standardized framework for classifying AKI into ordered "
        "severity stages, supporting consistent clinical description across "
        "intensive care studies [@kdigo_aki_2012].",
        "The Sepsis-3 consensus defines sepsis as life-threatening organ "
        "dysfunction, giving a consistent basis for case ascertainment "
        "[@singer_sepsis3_2016].",
        "Lactate clearance has shown robust prognostic performance in septic "
        "shock cohorts [@nguyen_lactate_2004].",
    ],
)
def test_d_cited_background_is_not_a_result_claim(sentence: str) -> None:
    critique = _review(sentence)

    assert critique.status == "pass"
    assert critique.unsupported_claims == []


@pytest.mark.parametrize(
    "sentence",
    [
        # Cited, but about this study's own findings.
        "Our findings were consistent with earlier cohorts [@kdigo_aki_2012].",
        "Our lactate model showed robust performance [@nguyen_lactate_2004].",
        "We found consistent associations across hospitals [@strobe_2007].",
        "Discrimination was robust in this cohort [@collins_tripod_2015].",
        "These results were robust to the choice of adjustment set [@strobe_2007].",
        "The estimates were consistent across sensitivity analyses [@strobe_2007].",
        # Uncited.
        "Staging supported consistent clinical description across ICU stays.",
        "Discrimination was robust across hospitals.",
        # Cited, but quantitative.
        "Mortality was 10% in stage 3 [@kdigo_aki_2012].",
        "Mortality was 12% higher above a lactate of 4 mmol/L [@nguyen_lactate_2004].",
    ],
)
def test_d_study_findings_and_uncited_or_numeric_prose_stay_flagged(
    sentence: str,
) -> None:
    critique = _review(sentence)

    assert critique.status == "needs_revision"
    assert critique.unsupported_claims
