"""The Planner cannot relabel a recorded study restriction as its own proposal."""

import json

import pytest

from easyicu.research_agent.planning.progressive_compiler import _stated_population_spec
from easyicu.research_agent.planning.progressive_contract import ProgressiveCohortIntent
from easyicu.research_agent.schema import UserPreferences

from .progressive_planner_fixtures import _context, _payload


def _read(*, source="outline", quote="Adults", kind="age_years", bounds=None,
          question="Describe outcomes in Adults.", settings=None, inclusion=()):
    context = _context().model_copy(update={"research_question": question})
    context = context.model_copy(update={
        "cohort": context.cohort.model_copy(update={"inclusion_criteria": list(inclusion)}),
    })
    if settings is not None:
        context = context.model_copy(update={
            "user_preferences": (context.user_preferences or UserPreferences()).model_copy(update={
                "data_constraints": json.dumps({"cohort": settings}),
            }),
        })
    cohort = _payload()["cohort"]
    cohort["population_spec"] = {"criteria": [{
        "id": "c1", "source": source, "quote": quote, "kind": kind, "role": "include",
        **(bounds if bounds is not None else {"min_years": 18, "max_years": None}),
    }]}
    return _stated_population_spec(ProgressiveCohortIntent.model_validate(cohort), context=context)


@pytest.mark.parametrize("source", ["outline", "preset"])
def test_verbatim_study_words_take_priority_over_a_planner_source_label(source):
    criterion = _read(source=source).criteria[0]
    assert criterion.source == "question"
    assert criterion.quote == "Adults"


def test_an_exact_typed_study_minimum_uses_the_study_label_without_rewriting_the_question():
    label = "Adults whose intensive care stay lasted at least 36 hours."
    criterion = _read(quote="ICU stay at least thirty-six hours", kind="icu_stay_hours",
                      bounds={"min_hours": 36, "max_hours": None},
                      settings={"min_icu_los_hours": 36, "label": label}).criteria[0]
    assert criterion.source == "study_wording"
    assert criterion.quote == label
    assert criterion.min_hours == 36


def test_typed_age_bounds_are_study_settings_even_when_the_planner_paraphrases_the_label():
    criterion = _read(quote="older adults", bounds={"min_years": 40, "max_years": 80},
                      settings={"age_min": 40, "age_max": 80,
                                "label": "Patients aged 40 to 80 years."}).criteria[0]
    assert criterion.source == "study_wording"
    assert criterion.quote == "Patients aged 40 to 80 years."


def test_an_existing_study_statement_is_not_mislabeled_as_question_text():
    criterion = _read(question="Describe the registered cohort.", inclusion=("Adults",)).criteria[0]
    assert criterion.source == "study_wording"
    assert criterion.quote == "Adults"


def test_a_boolean_setting_cannot_masquerade_as_a_numeric_one_hour_bound():
    criterion = _read(quote="At least one hour", kind="icu_stay_hours",
                      bounds={"min_hours": 1, "max_hours": None},
                      settings={"min_icu_los_hours": True, "label": "One hour."}).criteria[0]
    assert criterion.source == "outline"


@pytest.mark.parametrize("settings", [
    {"min_icu_los_hours": 48, "label": "At least 48 hours."},
    {"min_icu_los_hours": 36},
    {"min_icu_los_hours": "36", "label": "At least 36 hours."},
    {"min_icu_los_hours": True, "label": "At least 36 hours."},
])
def test_unmatched_or_untyped_settings_do_not_authorize_a_source_change(settings):
    criterion = _read(quote="ICU stay at least thirty-six hours", kind="icu_stay_hours",
                      bounds={"min_hours": 36, "max_hours": None}, settings=settings).criteria[0]
    assert criterion.source == "outline"


def test_a_new_upper_bound_remains_a_system_proposal():
    criterion = _read(quote="Stay between 36 and 72 hours", kind="icu_stay_hours",
                      bounds={"min_hours": 36, "max_hours": 72},
                      settings={"min_icu_los_hours": 36, "label": "At least 36 hours."}).criteria[0]
    assert criterion.source == "outline"


def test_a_claim_to_quote_the_question_still_rejects_paraphrasing():
    with pytest.raises(ValueError, match="not written there"):
        _read(source="question", quote="Adult patients",
              settings={"age_min": 18, "label": "Adult patients"})


def test_host_bound_study_wording_survives_recompilation_without_lending_a_new_bound():
    settings = {"min_icu_los_hours": 36, "label": "Minimum stay is 36 hours."}
    criterion = _read(quote="ICU stay at least thirty-six hours", kind="icu_stay_hours",
                      bounds={"min_hours": 36, "max_hours": None}, settings=settings).criteria[0]
    restored = _read(source=criterion.source, quote=criterion.quote, kind="icu_stay_hours",
                     bounds={"min_hours": 36, "max_hours": None}, settings=settings).criteria[0]
    assert restored == criterion
    with pytest.raises(ValueError, match="not written there"):
        _read(source=criterion.source, quote=criterion.quote, kind="icu_stay_hours",
              bounds={"min_hours": 48, "max_hours": None}, settings=settings)
