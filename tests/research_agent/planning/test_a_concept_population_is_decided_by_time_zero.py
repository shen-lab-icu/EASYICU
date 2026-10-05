"""A concept-derived population must be decided by the plan's time zero.

Data Extraction admits a stay into a concept-derived population (Sepsis-3,
ventilation ...) on a positive concept row at or before the end of the study's
cohort window.  A window ending after the plan's time zero (a landmark, or a
prediction's feature window) lets who enters depend on what happens after time
zero.  The host states the window it extracts by in
``data_constraints.concept_cohort_window``, and the family request refuses such
a plan before any Planner call.  A window ending at time zero is planned.

Synthetic study records and contexts only.
"""

from __future__ import annotations

import json

import pytest

from easyicu.research_agent.agents.family_spec_planner import FAMILY_SPEC_STRATEGY
from easyicu.research_agent.agents.progressive_planner import ProgressivePlannerAgent
from easyicu.research_agent.orchestration.scientific_runtime import ScientificRuntimeAuthorities
from easyicu.research_agent.planning.family_spec import FamilySpecError
from easyicu.research_agent.providers.mocks import ScriptedMockLLMClient
from easyicu.research_agent.research_context.concept_population import (
    ConceptCohortWindow,
    ConceptCohortWindowError,
    concept_cohort_window,
    context_data_constraints,
)
from easyicu.research_agent.schema import ConceptDescriptor, ResearchContext, VariableRole
from easyicu.webserver import dataio, primary_cohort
from easyicu.webserver.agent_pipeline_runs import _research_user_preferences
from easyicu.webserver.pi_copilot.extraction_handoff import compile_study_cohort
from tests.support.survival_proposal import survival_context, survival_request
from tests.support.survival_sealed import sealed_request, sealed_survival

from .family_spec_fixtures import (
    ALLOWED_CITATIONS,
    DIRECT_COMPARATORS,
    _context,
    _phenotyping_context,
    _prediction_context,
    _request,
)


def _with_concept_window(context: ResearchContext, window: object) -> ResearchContext:
    constraints = json.loads(context.user_preferences.data_constraints or "{}")
    constraints["concept_cohort_window"] = window
    return context.model_copy(
        update={
            "user_preferences": context.user_preferences.model_copy(
                update={"data_constraints": json.dumps(constraints)}
            )
        }
    )


def _host_window(study: dict) -> object:
    compiled = _research_user_preferences(study)
    return json.loads(compiled.get("data_constraints") or "{}").get("concept_cohort_window")


def test_a_concept_window_past_the_landmark_is_refused_before_the_planner_call() -> None:
    """Sepsis-3 found up to 720 h cannot define a cohort followed from 24 h."""

    context = _with_concept_window(_context(), {"definition": "sepsis3", "window_end_hours": 720})

    with pytest.raises(FamilySpecError) as caught:
        _request(context)
    assert caught.value.reason_code == "family_spec_cohort_eligibility_after_time_zero"
    assert caught.value.path == "cohort"
    assert all(part in str(caught.value) for part in ("sepsis3", "720 h", "24 h"))

    llm = ScriptedMockLLMClient([])
    with pytest.raises(Exception, match="family_spec_cohort_eligibility_after_time_zero"):
        ProgressivePlannerAgent(llm).run_attempt(
            context,
            planner_strategy=FAMILY_SPEC_STRATEGY,
            allowed_literature_citation_keys=ALLOWED_CITATIONS,
            direct_comparator_literature_keys=DIRECT_COMPARATORS,
            comparison_literature_keys=DIRECT_COMPARATORS,
            enforce_article_contract=True,
            article_contract_context=context,
            planning_contract_context="",
            required_primary_cohort_selection_mode="predicate_filtered",
        )
    assert llm.calls == []


@pytest.mark.parametrize("hours", [24, 12])
def test_a_concept_window_ending_by_the_landmark_is_planned(hours: int) -> None:
    request = _request(
        _with_concept_window(_context(), {"definition": "sepsis3", "window_end_hours": hours})
    )

    assert request.cohort_time_zero_hours == 24.0
    assert request.concept_cohort_definition == "sepsis3"
    assert request.concept_cohort_window_end_hours == float(hours)


def test_a_sealed_survival_suite_is_held_to_its_landmark(tmp_path) -> None:
    """The suite's own landmark is the time zero its population must be decided by."""

    context, authority = sealed_survival(tmp_path)
    authorities = ScientificRuntimeAuthorities(trajectory=None, current_case=authority)
    landmark = sealed_request(context, authorities).sealed_suite.landmark_hours

    late = _with_concept_window(context, {"definition": "sepsis3", "window_end_hours": 720})
    with pytest.raises(FamilySpecError) as caught:
        sealed_request(late, authorities)
    assert caught.value.reason_code == "family_spec_cohort_eligibility_after_time_zero"

    on_time = _with_concept_window(context, {"definition": "sepsis3", "window_end_hours": landmark})
    request = sealed_request(on_time, authorities)
    assert request.cohort_time_zero_hours == landmark
    assert request.concept_cohort_window_end_hours == landmark


def test_a_proposed_survival_suite_is_held_to_its_landmark() -> None:
    context = survival_context()
    landmark = survival_request(context).proposed_suite.landmark_hours

    with pytest.raises(FamilySpecError, match="family_spec_cohort_eligibility_after_time_zero"):
        survival_request(
            _with_concept_window(context, {"definition": "sepsis3", "window_end_hours": 720})
        )
    request = survival_request(
        _with_concept_window(context, {"definition": "sepsis3", "window_end_hours": landmark})
    )
    assert request.cohort_time_zero_hours == landmark


def test_a_survival_suites_minimum_stay_must_end_by_its_landmark(tmp_path) -> None:
    """Staying in the ICU beyond the landmark selects on survival after it."""

    context, authority = sealed_survival(tmp_path)
    authorities = ScientificRuntimeAuthorities(trajectory=None, current_case=authority)
    landmark = sealed_request(context, authorities).sealed_suite.landmark_hours

    def staying(hours: float) -> ResearchContext:
        return context.model_copy(
            update={
                "variables": [
                    *context.variables,
                    ConceptDescriptor(
                        name="los_icu", description="ICU length of stay",
                        role=VariableRole.OUTCOME, dtype="float64", unit="days",
                        source_concept="los_icu",
                    ),
                ],
                "user_preferences": context.user_preferences.model_copy(
                    update={"data_constraints": json.dumps({"cohort": {"min_icu_los_hours": hours}})}
                ),
            }
        )

    with pytest.raises(FamilySpecError) as caught:
        sealed_request(staying(landmark * 2), authorities)
    assert caught.value.reason_code == "family_spec_cohort_eligibility_after_time_zero"
    assert "minimum ICU stay" in str(caught.value)
    assert sealed_request(staying(landmark), authorities).minimum_icu_hours == landmark


@pytest.mark.parametrize(
    "build",
    [
        lambda: _request(_context()),
        lambda: _request(_phenotyping_context(), cohort_mode=None),
        lambda: _request(_prediction_context(), cohort_mode=None),
    ],
    ids=["landmark_categorical", "phenotyping", "prediction"],
)
def test_a_family_without_a_survival_suite_keeps_its_time_zero(build) -> None:
    """The suite's landmark is read only by survival requests; the rest are unchanged."""

    request = build()
    assert request.sealed_suite is None and request.proposed_suite is None
    assert request.cohort_time_zero_hours == (
        request.landmark_hours or request.observation_window_hours
    )
    assert request.cohort_time_zero_hours == 24.0


def test_a_prediction_is_held_to_the_end_of_its_feature_window() -> None:
    """A first-24-hour prediction on a population found up to 72 h would read the future."""

    late = _with_concept_window(
        _prediction_context(), {"definition": "ventilation", "window_end_hours": 72}
    )
    with pytest.raises(FamilySpecError) as caught:
        _request(late, cohort_mode=None)
    assert caught.value.reason_code == "family_spec_cohort_eligibility_after_time_zero"

    on_time = _with_concept_window(
        _prediction_context(), {"definition": "ventilation", "window_end_hours": 24}
    )
    assert _request(on_time, cohort_mode=None).concept_cohort_window_end_hours == 24.0


def test_without_a_concept_window_the_request_keeps_its_digest() -> None:
    request = _request(_context())

    assert request.concept_cohort_definition is None
    assert request.concept_cohort_window_end_hours is None
    dumped = request.model_dump(mode="json")
    assert "concept_cohort_definition" not in dumped
    assert "concept_cohort_window_end_hours" not in dumped


_UNREADABLE_WINDOWS = [
    {},
    {"definition": "sepsis3"},
    {"window_end_hours": 24},
    {"definition": "", "window_end_hours": 24},
    {"definition": "s" * 65, "window_end_hours": 24},
    {"definition": 3, "window_end_hours": 24},
    {"definition": "sepsis3", "window_end_hours": "24"},
    {"definition": "sepsis3", "window_end_hours": True},
    {"definition": "sepsis3", "window_end_hours": 0},
    {"definition": "sepsis3", "window_end_hours": -24},
    {"definition": "sepsis3", "window_end_hours": float("nan")},
    {"definition": "sepsis3", "window_end_hours": float("inf")},
    "sepsis3 within 24 h",
    None,
]


@pytest.mark.parametrize("window", _UNREADABLE_WINDOWS)
def test_a_concept_window_that_cannot_be_read_fails_closed(window: object) -> None:
    with pytest.raises(FamilySpecError) as caught:
        _request(_with_concept_window(_context(), window))
    assert caught.value.reason_code == "family_spec_concept_cohort_window_invalid"
    assert caught.value.path == "cohort"


def test_one_typed_reader_states_the_window_for_every_owner() -> None:
    """Planning and reporting read the same record through one reader."""

    context = _with_concept_window(
        _context(), {"definition": " sepsis3 ", "window_end_hours": 24}
    )

    assert concept_cohort_window(context) == ConceptCohortWindow("sepsis3", 24.0)
    assert _request(context).concept_cohort_definition == "sepsis3"
    assert concept_cohort_window(_context()) is None


@pytest.mark.parametrize("window", _UNREADABLE_WINDOWS)
def test_the_typed_reader_refuses_a_window_it_cannot_read(window: object) -> None:
    with pytest.raises(ConceptCohortWindowError):
        concept_cohort_window(_with_concept_window(_context(), window))


@pytest.mark.parametrize("raw", [None, "", "  ", "not json", "[1, 2]", '"text"'])
def test_constraints_that_are_not_one_json_object_state_no_window(raw: object) -> None:
    context = _context()
    context = context.model_copy(
        update={
            "user_preferences": context.user_preferences.model_copy(
                update={"data_constraints": raw}
            )
        }
    )

    assert context_data_constraints(context) == {}
    assert concept_cohort_window(context) is None


@pytest.mark.parametrize(
    ("study", "expected"),
    [
        (
            {"cohort": {"preset": "sepsis3"}, "time_window": {"hours": 48}},
            {"definition": "sepsis3", "window_end_hours": 48},
        ),
        (
            {"cohort": {"preset": "ventilation", "observation_window_hours": 24},
             "time_window": {"hours": 72}},
            {"definition": "ventilation", "window_end_hours": 24},
        ),
        ({"cohort": {"preset": "aki"}}, {"definition": "aki", "window_end_hours": 720}),
    ],
)
def test_the_host_states_the_window_its_export_is_selected_by(study: dict, expected: dict) -> None:
    assert _host_window(study) == expected
    recorded = dataio.export_cohort_execution(compile_study_cohort(study))
    assert recorded["concept_cohort_window"] == {
        **expected,
        "positive_rows": primary_cohort.CONCEPT_POSITIVE_ROWS,
    }


@pytest.mark.parametrize(
    "cohort",
    [
        {"preset": "all_icu"},
        {"preset": "adult_first"},
        {"label": "Adults with septic shock in MIMIC-IV"},
        # Extraction refuses an ICD cohort without codes with its own reason.
        {"preset": "icd"},
    ],
)
def test_a_population_not_derived_from_a_concept_states_no_window(cohort: dict) -> None:
    assert _host_window({"cohort": cohort, "time_window": {"hours": 24}}) is None


def test_the_window_the_host_states_is_the_one_planning_reads() -> None:
    for hours, refused in ((720, True), (24, False)):
        window = _host_window({"cohort": {"preset": "sepsis3"}, "time_window": {"hours": hours}})
        context = _with_concept_window(_context(), window)
        if refused:
            with pytest.raises(FamilySpecError, match="family_spec_cohort_eligibility_after_time_zero"):
                _request(context)
        else:
            assert _request(context).concept_cohort_window_end_hours == 24.0
