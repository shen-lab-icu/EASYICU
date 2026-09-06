"""A comparator's age population must match the actual bound cohort."""

import pytest

from easyicu.research_agent.literature import (
    CitationRecord,
    _adult_population_required,
    _screening_decision_for_record,
    screen_source_backed_direct_comparator,
)
from easyicu.research_agent.schema import (
    CohortDescriptor,
    ConceptDescriptor,
    ResearchContext,
)


def _context(*, minimum=18, missing=0, rows=4):
    return ResearchContext(
        research_question="Describe AKI and hospital mortality in ICU stays.",
        cohort=CohortDescriptor(cohort_name="local cohort", database="miiv", n_stays=rows),
        variables=[
            ConceptDescriptor(
                name="baseline_age", source_concept="age", dtype="float64", unit="years",
                observed_domain={"min": minimum, "max": 80, "n_unique": 4},
                missingness={"fraction_missing": missing / 4, "n_missing": missing, "n_total": 4},
            ),
            ConceptDescriptor(name="aki", dtype="int64"),
            ConceptDescriptor(name="death", dtype="int64"),
        ],
        primary_exposure="aki", target_outcome="death",
    )


def _pediatric_record(background=""):
    return CitationRecord(
        key="pediatric_aki", year="2022",
        title="Acute kidney injury and in-hospital mortality in a paediatric intensive care unit",
        relevance="Study-design excerpt: Patients aged 1 month to 16 years admitted to intensive care were studied for the association between acute kidney injury and hospital mortality. " + background,
        publication_types=["Observational Study"],
    )


def test_fully_observed_adult_cohort_excludes_pediatric_comparator():
    context = _context()
    assert _adult_population_required(context)
    decision = _screening_decision_for_record(
        context=context, record=_pediatric_record(), source="pubmed", query=None,
    )
    assert not decision.population_match
    assert decision.disposition == "exclude"


@pytest.mark.parametrize("changes", [{"minimum": 5}, {"missing": 1}, {"rows": 0}, {"rows": 20}])
def test_partial_or_mixed_age_observations_do_not_assert_adult_scope(changes):
    assert not _adult_population_required(_context(**changes))


def test_chinese_declared_adult_scope_is_retained():
    context = _context(rows=0)
    context.cohort.inclusion_criteria = ["纳入成年 ICU 患者"]
    assert _adult_population_required(context)


def test_adult_background_does_not_relabel_pediatric_study_population():
    decision = screen_source_backed_direct_comparator(
        exposure="acute kidney injury", outcome="hospital mortality", adult_required=True,
        record=_pediatric_record("Background: Adult ICU patients often develop AKI."),
        source="pubmed", query=None,
    )
    assert not decision.population_match
    assert decision.disposition == "exclude"
