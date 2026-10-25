"""A disease defining the cohort is not automatically the studied exposure."""

import pytest

from easyicu.research_agent.literature import (
    CitationRecord,
    screen_source_backed_direct_comparator,
)


def _screen(
    exposure: str,
    excerpt: str,
    title: str = "A biomarker model for hospital mortality in ICU patients",
):
    return screen_source_backed_direct_comparator(
        exposure=exposure,
        outcome="hospital mortality",
        adult_required=False,
        record=CitationRecord(
            key="source_backed_candidate",
            year="2024",
            title=title,
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


def test_secondary_performance_comparator_does_not_become_direct_comparator():
    record = CitationRecord(
        key="renin_versus_lactate",
        year="2022",
        title=(
            "Renin Kinetics Are Superior to Lactate Kinetics for Predicting "
            "In-Hospital Mortality in Hypotensive Critically Ill Patients"
        ),
        relevance=(
            "Study-design excerpt: This prospective observational cohort "
            "studied adult ICU patients on vasopressors. Peak lactate "
            "demonstrated moderate discrimination for in-hospital mortality."
        ),
        publication_types=["Observational Study"],
    )

    decision = screen_source_backed_direct_comparator(
        exposure="lactate",
        outcome="in hospital mortality",
        adult_required=True,
        record=record,
        source="pubmed",
        query="focused query",
    )

    assert decision.population_match is True
    assert decision.outcome_match is True
    assert decision.exposure_match is False
    assert decision.disposition == "exclude"
    assert decision.evidence_role == "related_context"


@pytest.mark.parametrize("marker", ["marker alpha", "enzyme omega"])
@pytest.mark.parametrize(
    ("studied", "difference"),
    [
        ("{marker}-to-protein beta ratio", "比值"),
        ("{marker}-protein beta ratio", "比值"),
        ("{marker}/protein beta", "比值"),
        ("protein beta-to-{marker} ratio", "比值"),
        ("ratio of {marker} to protein beta", "比值"),
        ("ratio of protein beta to {marker}", "比值"),
        ("{marker} and protein beta composite score", "复合指标"),
        ("{marker}-protein beta score", "复合指标"),
        ("composite index incorporating {marker} and protein beta", "复合指标"),
        ("composite of {marker} and protein beta", "复合指标"),
        ("{marker} and protein beta composite predictor", "复合指标"),
        ("combination of {marker} and protein beta", "复合指标"),
        ("{marker} combined with protein beta", "复合指标"),
        ("{marker} trajectory", "动态变化"),
        ("trajectories of {marker}", "动态变化"),
        ("{marker} clearance", "动态变化"),
        ("change in {marker}", "动态变化"),
        ("{marker} variability", "动态变化"),
        ("{marker} kinetics", "动态变化"),
    ],
)
def test_derived_exposure_is_related_context_for_its_component(marker, studied, difference):
    studied = studied.format(marker=marker)
    decision = _screen(
        marker,
        f"We evaluated {studied} for hospital mortality in ICU patients.",
        title=f"{studied} predicts hospital mortality in ICU patients",
    )

    assert decision.population_match and decision.outcome_match
    assert decision.design_excerpt_available and decision.publication_type_eligible
    assert not decision.exposure_match
    assert decision.disposition == "exclude"
    assert decision.evidence_role == "related_context"
    assert difference in decision.rationale
    assert "相关背景" in decision.rationale


@pytest.mark.parametrize(
    "exposure",
    [
        "marker alpha-to-protein beta ratio",
        "marker alpha and protein beta composite score",
        "marker alpha trajectory",
    ],
)
def test_plain_component_cannot_authorize_a_derived_exposure(exposure):
    decision = _screen(
        exposure,
        "We evaluated marker alpha for hospital mortality in ICU patients. "
        "Protein beta was also measured.",
        title="Marker alpha predicts hospital mortality in ICU patients",
    )
    assert not decision.exposure_match
    assert decision.evidence_role == "related_context"


@pytest.mark.parametrize(
    "exposure",
    [
        "marker alpha",
        "marker alpha-to-protein beta ratio",
        "marker alpha/protein beta",
        "marker alpha and protein beta composite score",
        "marker alpha trajectory",
    ],
)
def test_matching_studied_identity_remains_a_candidate_pending_review(exposure):
    decision = _screen(
        exposure,
        f"We evaluated {exposure} for hospital mortality in ICU patients.",
        title=f"{exposure} predicts hospital mortality in ICU patients",
    )
    assert decision.exposure_match
    assert decision.disposition == "include"
    assert decision.evidence_role == "direct_comparator"
    assert "Human review" in decision.rationale
    assert "time zero, estimand, and adjustment" in decision.rationale


@pytest.mark.parametrize(
    "studied",
    [
        "marker alpha-to-protein beta ratio",
        "marker alpha and protein beta composite score",
        "marker alpha trajectory",
    ],
)
@pytest.mark.parametrize("location", ["title", "excerpt"])
def test_identity_guard_applies_to_each_source_field(studied, location):
    statement = f"We evaluated {studied} for hospital mortality in ICU patients."
    decision = _screen(
        "marker alpha",
        statement if location == "excerpt" else "A retrospective study of ICU patients.",
        title=statement if location == "title" else "A biomarker study in ICU patients",
    )
    assert not decision.exposure_match
    assert decision.evidence_role == "related_context"


def test_ratio_direction_is_part_of_the_exposure_identity():
    decision = _screen(
        "marker alpha-to-protein beta ratio",
        "We evaluated protein beta-to-marker alpha ratio for hospital mortality in ICU patients.",
    )
    assert not decision.exposure_match
    assert decision.evidence_role == "related_context"


@pytest.mark.parametrize(
    "excerpt",
    [
        "We evaluated marker alpha for hospital mortality, adjusting for protein beta in ICU patients.",
        "Adjusted for age, marker alpha predicts hospital mortality in ICU patients.",
        "We evaluated marker alpha and reported an odds ratio for hospital mortality in ICU patients.",
        "Marker alpha was a predictor of hospital mortality; the hazard ratio was estimated in ICU patients.",
        "We evaluated the odds ratio of marker alpha for hospital mortality in ICU patients.",
        "We evaluated the hazard ratio of marker alpha for hospital mortality in ICU patients.",
        "We evaluated the risk ratio of marker alpha for hospital mortality in ICU patients.",
        "We assessed marker alpha for hospital mortality and a composite outcome in ICU patients.",
        "We evaluated marker alpha for hospital mortality in ICU patients with a trajectory of organ failure.",
        "We evaluated marker alpha-to-protein beta ratio; separately we evaluated marker alpha for hospital mortality in ICU patients.",
        "We evaluated marker alpha-to-protein beta ratio and independently evaluated marker alpha for hospital mortality in ICU patients.",
        "We evaluated marker alpha kinetics and separately evaluated marker alpha for hospital mortality in ICU patients.",
    ],
)
def test_separate_plain_marker_analysis_is_not_hidden_by_other_terms(excerpt):
    decision = _screen("marker alpha", excerpt)
    assert decision.exposure_match
    assert decision.evidence_role == "direct_comparator"


@pytest.mark.parametrize(
    "adjustment",
    [
        "after adjusting for marker alpha",
        "after adjustment for age and marker alpha",
        "with adjustment for marker alpha and age",
        "after controlling for marker alpha",
        "with marker alpha as a covariate",
        "with marker alpha included as an adjustment variable",
        "with marker alpha used only as a covariate",
        "where covariates included age and marker alpha",
        "using marker alpha-adjusted models",
    ],
)
def test_adjustment_only_marker_mention_stays_related_context(adjustment):
    decision = _screen(
        "marker alpha",
        f"We evaluated protein beta {adjustment} for hospital mortality in ICU patients.",
        title="Protein beta predicts hospital mortality in ICU patients",
    )
    assert decision.population_match and decision.outcome_match
    assert not decision.exposure_match
    assert decision.disposition == "exclude"
    assert decision.evidence_role == "related_context"
    assert "调整变量" in decision.rationale


def test_adjustment_covariate_does_not_become_studied_exposure():
    record = CitationRecord(
        key="urine_output_adjusted_for_aki",
        year="2023",
        title="Urine output for predicting in-hospital mortality in ICU patients",
        relevance=(
            "Study-design excerpt: We evaluated urine output as a predictor of "
            "in-hospital mortality in ICU patients. After adjusting for "
            "confounding factors including diuretic use and acute kidney injury, "
            "urine output remained an independent risk factor."
        ),
        publication_types=["Observational Study"],
    )

    decision = screen_source_backed_direct_comparator(
        exposure="acute kidney injury",
        outcome="in-hospital mortality",
        adult_required=False,
        record=record,
        source="pubmed",
        query="focused query",
    )

    assert decision.population_match
    assert decision.outcome_match
    assert not decision.exposure_match
    assert decision.evidence_role == "related_context"


def test_inhospital_mortality_spelling_matches_declared_outcome():
    record = CitationRecord(
        key="inhospital_spelling",
        year="2022",
        title=(
            "Serial Lactate Predicts Inhospital Mortality in Critically Ill "
            "Adult Patients"
        ),
        relevance=(
            "Study-design excerpt: Adult ICU patients had serial lactate "
            "measurements during the first 24 hours."
        ),
        publication_types=["Observational Study"],
    )

    decision = screen_source_backed_direct_comparator(
        exposure="lactate",
        outcome="in hospital mortality",
        adult_required=True,
        record=record,
        source="pubmed",
        query="focused query",
    )

    assert decision.population_match is True
    assert decision.exposure_match is True
    assert decision.outcome_match is True
