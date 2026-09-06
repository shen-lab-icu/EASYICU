"""A disease defining the cohort is not automatically the studied exposure."""

import pytest

from easyicu.research_agent.literature import (
    CitationRecord,
    screen_source_backed_direct_comparator,
)


def _screen(exposure: str, excerpt: str):
    return screen_source_backed_direct_comparator(
        exposure=exposure,
        outcome="hospital mortality",
        adult_required=False,
        record=CitationRecord(
            key="source_backed_candidate",
            year="2024",
            title="A biomarker model for hospital mortality in ICU patients",
            relevance="Study-design excerpt: " + excerpt,
            publication_types=["Observational Study"],
        ),
        source="pubmed",
        query=None,
    )


@pytest.mark.parametrize("exposure", ["Sepsis-3", "acute kidney injury", "heart failure"])
@pytest.mark.parametrize(
    "population",
    [
        "cohorts of {exposure} patients",
        "a cohort of adult patients with {exposure}",
        "{exposure} patients",
        "{exposure} cohorts",
    ],
)
def test_model_validation_within_disease_population_is_not_disease_comparison(
    exposure, population
):
    excerpt = (
        "We validated the biomarker model using "
        + population.format(exposure=exposure)
        + " from an ICU database. The endpoint was hospital mortality."
    )
    decision = _screen(exposure, excerpt)

    assert decision.population_match
    assert decision.outcome_match
    assert decision.design_excerpt_available
    assert not decision.exposure_match
    assert decision.disposition == "exclude"
    assert decision.evidence_role == "related_context"


@pytest.mark.parametrize("exposure", ["Sepsis-3", "acute kidney injury", "heart failure"])
@pytest.mark.parametrize(
    "comparison",
    [
        "We assessed {exposure} prevalence and hospital mortality in ICU patients.",
        "We compared patients with {exposure} and without {exposure} for hospital mortality.",
        "We compared {exposure} patients versus patients without {exposure} for hospital mortality.",
        "The biomarker was validated in {exposure} patients. Separately, we assessed the association between {exposure} and hospital mortality in all ICU patients.",
    ],
)
def test_actual_disease_comparison_is_preserved(exposure, comparison):
    decision = _screen(exposure, comparison.format(exposure=exposure))

    assert decision.exposure_match
    assert decision.disposition == "include"
    assert decision.evidence_role == "direct_comparator"
