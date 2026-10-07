"""An adult study is screened against adult analogues wherever it says so.

Literature screening requires an adult comparison population when the study
declares one. It read the declaration only from the cohort's inclusion
criteria, which now list only what the input rows already meet: the Web host
sends the study's own cohort wording in ``data_constraints.cohort`` instead,
because nothing executes prose. Before planning the catalog has no rows, so a
study that said "adults" only in its question or wording was screened as if
any age would do, and a pediatric study could pass as its design analogue.
A preset that names adults declares the scope too: Data Extraction applies
the adult age floor with it whether or not an age bound is set.
"""

from __future__ import annotations

import json

import pytest

from easyicu.research_agent.literature import (
    CitationRecord,
    _adult_population_required,
    _screening_decision_for_record,
)
from easyicu.research_agent.schema import (
    CohortDescriptor,
    ConceptDescriptor,
    ResearchContext,
    UserPreferences,
)

_NEUTRAL = "Describe condition-x and hospital mortality in ICU stays."


def _context(question: str = _NEUTRAL, *, stated: dict | None = None, raw: str | None = None):
    constraints = raw if raw is not None else (json.dumps({"cohort": stated}) if stated is not None else None)
    return ResearchContext(
        research_question=question,
        cohort=CohortDescriptor(cohort_name="local cohort", database="miiv", n_stays=0),
        variables=[
            ConceptDescriptor(name="condition_x", dtype="int64"),
            ConceptDescriptor(name="death", dtype="int64"),
        ],
        primary_exposure="condition_x",
        target_outcome="death",
        user_preferences=UserPreferences(data_constraints=constraints) if constraints is not None else None,
    )


@pytest.mark.parametrize(
    "question",
    [
        "Among adults with condition-x, is it associated with hospital mortality?",
        "在成人 ICU 患者中，condition-x 与住院死亡有何关联？",
    ],
)
def test_a_question_that_names_adults_declares_the_scope(question: str) -> None:
    assert _adult_population_required(_context(question))


@pytest.mark.parametrize(
    "stated",
    [
        {"label": "Adults with condition-x"},
        {"review": "纳入成人 ICU 患者；具体资格标准由研究计划提出"},
        {"age_min": 18},
        {"age_min": 65.0},
        {"preset": "adult_first"},
        {"preset": " Adult_All "},
    ],
)
def test_the_study_cohort_declares_the_scope_in_its_wording_age_floor_or_preset(stated: dict) -> None:
    assert _adult_population_required(_context(stated=stated))


@pytest.mark.parametrize(
    "stated",
    [
        {},
        {"label": "ICU stays with condition-x"},
        {"age_min": 16},
        {"age_min": True},
        {"age_min": "18"},
        {"preset": "all_icu"},
        {"preset": "sepsis3"},
        {"exclusion_statement": "adults transferred from another ICU"},
    ],
)
def test_no_adult_scope_is_read_where_none_is_stated(stated: dict) -> None:
    assert not _adult_population_required(_context(stated=stated))


@pytest.mark.parametrize(
    "question",
    [
        "Among adult ICU patients, excluding children, is condition-x associated with hospital mortality?",
        "在成人 ICU 患者中（排除儿童），condition-x 与住院死亡有何关联？",
        "Among ICU patients aged 18 years or older, is condition-x associated with hospital mortality?",
        "在年满18岁的 ICU 患者中，condition-x 与住院死亡有何关联？",
    ],
)
def test_an_adult_floor_or_an_excluded_child_still_declares_the_scope(question: str) -> None:
    assert _adult_population_required(_context(question))


def test_a_background_sentence_about_children_leaves_the_adult_scope() -> None:
    # Each statement is read on its own: children named in another sentence
    # are not a group this study includes.
    assert _adult_population_required(_context(
        "Among adult ICU patients, is condition-x associated with hospital mortality? "
        "Condition-x is also common in children."
    ))


@pytest.mark.parametrize(
    "question",
    [
        # A minor or a non-adult group is not an adult one.
        "在未成年 ICU 患者中，condition-x 与住院死亡有何关联？",
        "在非成人 ICU 患者中，condition-x 与住院死亡有何关联？",
        "Among non-adult ICU patients, is condition-x associated with hospital mortality?",
        # An age bound below 18 names children.
        "Among ICU patients under age 18, is condition-x associated with hospital mortality?",
        "Among ICU patients aged 18 or younger, is condition-x associated with hospital mortality?",
        # A study that includes children is not adult-only.
        "Among adult and pediatric ICU patients, is condition-x associated with hospital mortality?",
        "比较成人和儿童 ICU 患者中 condition-x 与住院死亡的关联。",
    ],
)
def test_a_minor_or_a_mixed_population_declares_no_adult_only_scope(question: str) -> None:
    assert not _adult_population_required(_context(question))


@pytest.mark.parametrize("raw", ["", "   ", "not json", "[]", json.dumps({"cohort": "adults"})])
def test_unreadable_constraints_declare_nothing(raw: str) -> None:
    assert not _adult_population_required(_context(raw=raw))


def test_a_stated_adult_scope_excludes_a_pediatric_analogue() -> None:
    record = CitationRecord(
        key="pediatric_condition_x",
        year="2022",
        title="Condition-x and in-hospital mortality in a paediatric intensive care unit",
        relevance=(
            "Study-design excerpt: Patients aged 1 month to 16 years admitted to intensive "
            "care were studied for the association between condition-x and hospital mortality."
        ),
        publication_types=["Observational Study"],
    )

    neutral = _screening_decision_for_record(
        context=_context(), record=record, source="pubmed", query=None
    )
    adult = _screening_decision_for_record(
        context=_context(stated={"label": "Adults with condition-x"}),
        record=record,
        source="pubmed",
        query=None,
    )

    assert neutral.population_match
    assert not adult.population_match
    assert adult.disposition == "exclude"
