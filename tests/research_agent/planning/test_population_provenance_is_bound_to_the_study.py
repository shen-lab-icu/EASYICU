"""Host attribution never turns overlapping words or display labels into intent."""

import json

import pytest

from easyicu.research_agent.orchestration.progressive_planning import population_proposals_finding
from easyicu.research_agent.planning.population_compile import compile_population
from easyicu.research_agent.planning.progressive_compiler import _stated_population_spec
from easyicu.research_agent.planning.progressive_contract import ProgressiveCohortIntent
from easyicu.research_agent.schema import UserPreferences

from .progressive_planner_fixtures import _context, _payload


@pytest.mark.parametrize("source", ["outline", "preset"])
@pytest.mark.parametrize("question,quote,kind,bounds", [
    (
        "Describe maximum lactate during the first 24 hours of the ICU stay.",
        "first 24 hours", "icu_stay_hours", {"min_hours": 24, "max_hours": None},
    ),
    (
        "Describe outcomes in adults.",
        "adults", "age_years", {"min_years": 18, "max_years": 89},
    ),
    (
        "Describe ICU outcomes.",
        "IC", "icu_stay_hours", {"min_hours": 24, "max_hours": None},
    ),
])
def test_system_quote_overlap_is_not_question_attribution(source, question, quote, kind, bounds):
    criterion = _read(source=source, question=question, quote=quote, kind=kind, bounds=bounds).criteria[0]
    assert criterion.source == source
    assert criterion.quote == quote
    for field, value in bounds.items():
        assert getattr(criterion, field) == value


def _study_context(question="Describe outcomes in Adults.", settings=None, inclusion=()):
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
    return context


def _read(*, source="outline", quote="Adults", kind="age_years", bounds=None,
          question="Describe outcomes in Adults.", settings=None, inclusion=()):
    context = _study_context(question, settings, inclusion)
    cohort = _payload()["cohort"]
    cohort["population_spec"] = {"criteria": [{
        "id": "c1", "source": source, "quote": quote, "kind": kind, "role": "include",
        **(bounds if bounds is not None else {"min_years": 18, "max_years": None}),
    }]}
    return _stated_population_spec(ProgressiveCohortIntent.model_validate(cohort), context=context)


def _compiled(spec, *, question="Describe outcomes in Adults.", settings=None):
    return compile_population(spec, _study_context(question, settings))


@pytest.mark.parametrize("source", ["outline", "preset"])
def test_matching_words_are_recorded_separately_from_the_planner_source(source):
    spec = _read(source=source, question="Describe outcomes in ＡＤＵＬＴＳ.", quote="Ad ults")
    population = _compiled(spec, question="Describe outcomes in ＡＤＵＬＴＳ.")
    row = population.record()["criteria"][0]
    assert row["criterion"]["source"] == source
    assert row["provenance"] == {
        "attribution": source, "quote_found_in_question": True,
        "stated_by_study": False, "typed_study_setting": None,
    }
    finding = population_proposals_finding(population)
    assert finding.detail["reason_code"] == "population_criteria_proposed_by_system"
    assert finding.detail["criteria"][0]["stated_by_study"] is False
    assert finding.detail["criteria"][0]["quote_found_in_question"] is True


@pytest.mark.parametrize("label_field", ["label", "review", None])
def test_exact_typed_stay_settings_use_fields_and_values_never_display_labels(label_field):
    settings = {"min_icu_los_hours": 24}
    if label_field:
        settings[label_field] = "Adults with sepsis"
    spec = _read(quote="Observe the exposure window", kind="icu_stay_hours",
                 bounds={"min_hours": 24, "max_hours": None}, settings=settings)
    population = _compiled(spec, settings=settings)
    row = population.record()["criteria"][0]
    assert row["criterion"]["source"] == "outline"
    assert row["criterion"]["quote"] == "Observe the exposure window"
    assert row["criterion"]["min_hours"] == 24
    assert row["provenance"] == {
        "attribution": "typed_study_setting", "stated_by_study": False,
        "quote_found_in_question": False,
        "typed_study_setting": {"cohort.min_icu_los_hours": 24},
    }
    assert population_proposals_finding(population).detail["criteria"][0]["stated_by_study"] is False


def test_typed_age_bounds_are_study_settings_even_when_the_planner_paraphrases_the_label():
    settings = {"age_min": 40, "age_max": 80, "label": "Adults with sepsis"}
    spec = _read(quote="older adults", bounds={"min_years": 40, "max_years": 80}, settings=settings)
    row = _compiled(spec, settings=settings).record()["criteria"][0]
    assert row["criterion"]["source"] == "outline"
    assert row["criterion"]["quote"] == "older adults"
    assert row["provenance"]["typed_study_setting"] == {"cohort.age_min": 40, "cohort.age_max": 80}
    assert row["provenance"]["stated_by_study"] is False


@pytest.mark.parametrize("bounds,settings,expected", [
    ({"min_years": 18, "max_years": None}, {"age_min": 18}, {"cohort.age_min": 18}),
    ({"min_years": None, "max_years": 80}, {"age_max": 80}, {"cohort.age_max": 80}),
    ({"min_years": 40, "max_years": 80}, {"age_min": 18, "age_max": 80}, None),
    ({"min_years": 40, "max_years": 80}, {"age_min": 40, "age_max": 89}, None),
    ({"min_years": 40, "max_years": 80}, {"age_min": 40}, None),
])
def test_typed_age_attribution_requires_both_bounds_to_match(bounds, settings, expected):
    spec = _read(quote="Proposed age range", bounds=bounds, settings=settings)
    row = _compiled(spec, settings=settings).record()["criteria"][0]
    assert row["provenance"]["typed_study_setting"] == expected
    assert row["provenance"]["stated_by_study"] is False
    assert {field: row["criterion"][field] for field in bounds} == bounds


def test_overlapping_study_wording_cannot_upgrade_a_system_interpretation_either():
    criterion = _read(question="Describe the registered cohort.", inclusion=("Adults",)).criteria[0]
    assert criterion.source == "outline"
    assert criterion.quote == "Adults"


@pytest.mark.parametrize("source", ["question", "study_wording"])
def test_a_genuinely_stated_restriction_still_counts_as_stated(source):
    question = "Describe outcomes in patients aged 18 years or older."
    spec = _read(source=source, question=question, quote="patients aged 18 years or older")
    population = _compiled(spec, question=question, settings={"age_min": 18})
    assert population.criteria[0].stated_by_study is True
    assert population_proposals_finding(population) is None


def test_a_boolean_setting_cannot_masquerade_as_a_numeric_one_hour_bound():
    settings = {"min_icu_los_hours": True, "label": "One hour."}
    spec = _read(quote="At least one hour", kind="icu_stay_hours",
                 bounds={"min_hours": 1, "max_hours": None}, settings=settings)
    row = _compiled(spec, settings=settings).record()["criteria"][0]
    assert row["criterion"]["source"] == "outline"
    assert row["provenance"]["typed_study_setting"] is None


@pytest.mark.parametrize("settings", [
    {"min_icu_los_hours": 48, "label": "At least 48 hours."},
    {"min_icu_los_hours": "36", "label": "At least 36 hours."},
    {"min_icu_los_hours": True, "label": "At least 36 hours."},
    {"min_icu_los_hours": float("inf")},
])
def test_unmatched_or_untyped_settings_do_not_authorize_a_source_change(settings):
    criterion = _read(quote="ICU stay at least thirty-six hours", kind="icu_stay_hours",
                      bounds={"min_hours": 36, "max_hours": None}, settings=settings).criteria[0]
    assert criterion.source == "outline"
    row = _compiled(_read(quote=criterion.quote, kind=criterion.kind,
                          bounds={"min_hours": 36, "max_hours": None}, settings=settings),
                    settings=settings).record()["criteria"][0]
    assert row["provenance"]["typed_study_setting"] is None


def test_a_new_upper_bound_remains_a_system_proposal():
    settings = {"min_icu_los_hours": 36, "label": "At least 36 hours."}
    spec = _read(quote="Stay between 36 and 72 hours", kind="icu_stay_hours",
                 bounds={"min_hours": 36, "max_hours": 72}, settings=settings)
    row = _compiled(spec, settings=settings).record()["criteria"][0]
    assert row["criterion"]["source"] == "outline"
    assert row["provenance"]["typed_study_setting"] is None


def test_a_claim_to_quote_the_question_still_rejects_paraphrasing():
    with pytest.raises(ValueError, match="not written there"):
        _read(source="question", quote="Adult patients",
              settings={"age_min": 18, "label": "Adult patients"})


def test_host_attribution_survives_recompilation_without_lending_a_new_bound():
    settings = {"min_icu_los_hours": 36, "label": "Minimum stay is 36 hours."}
    criterion = _read(quote="ICU stay at least thirty-six hours", kind="icu_stay_hours",
                      bounds={"min_hours": 36, "max_hours": None}, settings=settings).criteria[0]
    restored = _read(source=criterion.source, quote=criterion.quote, kind="icu_stay_hours",
                     bounds={"min_hours": 36, "max_hours": None}, settings=settings).criteria[0]
    assert restored == criterion
    changed = _read(source=criterion.source, quote=criterion.quote, kind="icu_stay_hours",
                    bounds={"min_hours": 48, "max_hours": None}, settings=settings)
    assert _compiled(changed, settings=settings).record()["criteria"][0]["provenance"]["typed_study_setting"] is None


def test_a_typed_match_does_not_validate_a_borrowed_label_as_study_words():
    with pytest.raises(ValueError, match="not written there"):
        _read(source="study_wording", quote="Adults with sepsis", kind="icu_stay_hours",
              bounds={"min_hours": 24, "max_hours": None},
              settings={"min_icu_los_hours": 24, "label": "Adults with sepsis"})


@pytest.mark.parametrize("field", ["review", "exclusion_statement"])
def test_question_attribution_cannot_borrow_explicit_study_review_words(field):
    with pytest.raises(ValueError, match="not written there"):
        _read(source="question", quote="Adult patients", settings={field: "Adult patients"})


@pytest.mark.parametrize("field", ["review", "exclusion_statement"])
def test_explicit_study_wording_keeps_its_own_source(field):
    criterion = _read(source="study_wording", quote="Adult patients", settings={field: "Adult patients"}).criteria[0]
    assert criterion.source == "study_wording" and criterion.quote == "Adult patients"
