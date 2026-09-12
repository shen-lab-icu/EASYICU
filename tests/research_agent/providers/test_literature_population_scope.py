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


@pytest.mark.parametrize("database", ["MIMIC-III", "MIMIC-IV", "eICU"])
@pytest.mark.parametrize(
    ("population", "expected_match"),
    [
        ("", False),
        ("We studied neonates aged under 28 days admitted to the ICU.", False),
        ("We studied adult ICU patients.", True),
        ("Background: AKI also occurs in children. We studied adult ICU patients.", True),
        ("We studied adult ICU patients, excluding children.", True),
        ("Methods: We studied adult ICU patients and excluded children.", True),
        ("Methods: Adult ICU patients were enrolled; children were excluded.", True),
        ("We studied adult patients and children in the ICU.", False),
        ("Background: Adult patients often develop AKI. We studied neonates in the ICU.", False),
    ],
)
def test_database_name_cannot_replace_study_population_evidence(
    database, population, expected_match,
):
    decision = screen_source_backed_direct_comparator(
        exposure="lactate",
        outcome="in hospital mortality",
        adult_required=True,
        record=CitationRecord(
            key="mimic_lactate",
            year="2020",
            title=(
                "Lactate indices as predictors of in-hospital mortality after "
                "admission to an intensive care unit in unselected critically "
                "ill patients"
            ),
            relevance=(
                f"Study-design excerpt: The analysis used the {database} database. "
                + population
            ),
            publication_types=["Observational Study"],
        ),
        source="pubmed",
        query=None,
    )

    assert decision.population_match is expected_match
    assert decision.exposure_match is True
    assert decision.outcome_match is True
    assert decision.disposition == ("include" if expected_match else "exclude")
    assert decision.evidence_role == (
        "direct_comparator" if expected_match else "related_context"
    )


@pytest.mark.parametrize(
    ("excerpt", "adult_title", "expected_match"),
    [
        ("Background: Adult ICU patients often develop AKI. Methods: We studied ICU admissions.", False, False),
        ("AKI is common in adult ICU patients. We studied ICU admissions.", False, False),
        ("Previous studies enrolled adult ICU patients. We studied ICU admissions.", False, False),
        ("In a previous study, we enrolled adult ICU patients. We now studied ICU admissions.", False, False),
        ("Adult ICU patients had frequent acute kidney injury in previous studies. Methods: We studied ICU admissions.", False, False),
        ("Background: Adult patients develop AKI. The study population comprised children in the ICU.", False, False),
        ("Background: Adult patients develop AKI. Neonates admitted to the ICU were included.", False, False),
        ("Methods: The cohort included adults and children in the ICU.", False, False),
        ("Methods: Patients of all ages admitted to the ICU were enrolled.", True, False),
        ("Methods: Patients aged 12 to 85 years admitted to the ICU were enrolled.", True, False),
        ("Methods: Patients under 18 years admitted to the ICU were enrolled.", True, False),
        ("Methods: We enrolled ICU patients aged <18 years.", True, False),
        ("Methods: Children admitted to the ICU were included.", True, False),
        ("Methods: Adults were excluded. We studied ICU admissions.", False, False),
        ("Methods: Adults were excluded. We studied ICU admissions.", True, False),
        ("Methods: We excluded adults and studied children admitted to the ICU.", True, False),
        ("Methods: Adult ICU patients were excluded. We studied ICU admissions.", True, False),
        ("Methods: We excluded adult ICU patients. We studied ICU admissions.", True, False),
        ("Methods: The adult severity score was calculated in ICU patients.", False, False),
        ("The analysis used the MIMIC-IV database.", False, False),
        ("The analysis used the MIMIC-IV database.", True, True),
        ("Background: AKI occurs in children. Methods: We included adult ICU patients.", False, True),
        ("AKI occurs in children. Adult ICU patients were included in this cohort.", False, True),
        ("Methods: We included adult ICU patients and excluded children.", False, True),
        ("Methods: We included adult ICU patients, and children were excluded.", False, True),
        ("Methods: Patients under 18 years were excluded. We studied adult ICU patients.", False, True),
        ("Methods: Adults with chronic kidney disease were excluded. We studied adult ICU patients.", False, True),
        ("Methods: We excluded adults with chronic kidney disease. We studied adult ICU patients.", False, True),
        ("A retrospective adult ICU cohort was evaluated for AKI and hospital mortality.", False, True),
        ("In adult ICU patients, AKI was associated with hospital mortality.", False, True),
        ("Methods: We enrolled ICU patients aged 18 years or older.", False, True),
        ("Methods: We enrolled ICU patients aged >=18 years.", False, True),
        ("Methods: Patients aged 18 to 85 years admitted to the ICU were enrolled.", False, True),
    ],
)
def test_adult_evidence_describes_the_enrolled_population(
    excerpt, adult_title, expected_match,
):
    decision = screen_source_backed_direct_comparator(
        exposure="acute kidney injury", outcome="hospital mortality", adult_required=True,
        record=CitationRecord(
            key="population_scope", year="2024",
            title=(
                "Association between acute kidney injury and hospital mortality in "
                + ("adult " if adult_title else "")
                + "ICU patients"
            ),
            relevance="Study-design excerpt: " + excerpt,
            publication_types=["Observational Study"],
        ),
        source="pubmed", query=None,
    )
    assert decision.exposure_match and decision.outcome_match
    assert decision.population_match is expected_match
    assert decision.disposition == ("include" if expected_match else "exclude")
    assert decision.evidence_role == (
        "direct_comparator" if expected_match else "related_context"
    )


def test_design_analogue_cannot_use_background_adults_as_population_authority():
    context = _context().model_copy(update={
        "primary_exposure": None,
        "research_question": "Develop an AKI prediction model in adult ICU patients.",
    })
    decision = _screening_decision_for_record(
        context=context,
        record=CitationRecord(
            key="population_design", year="2024",
            title="A prediction model for acute kidney injury in ICU patients",
            relevance=(
                "Study-design excerpt: Background: Adult patients often develop AKI. "
                "Methods: We developed a prediction model using ICU admissions."
            ),
            publication_types=["Observational Study"],
        ),
        source="pubmed", query=None,
    )
    assert not decision.population_match
    assert decision.disposition == "exclude"
    assert decision.evidence_role == "related_context"
