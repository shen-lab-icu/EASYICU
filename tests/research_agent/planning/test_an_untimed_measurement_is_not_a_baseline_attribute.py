"""Synthetic measurements need a window before claiming baseline availability."""

import json

import pytest

from easyicu.research_agent.agents.progressive_planner import (
    FAMILY_SPEC_STRATEGY,
    ProgressivePlannerAgent,
)
from easyicu.research_agent.planning.association_scope import (
    baseline_association_context,
    post_baseline_exposure,
)
from easyicu.research_agent.planning.family_spec import (
    build_family_spec_request,
    family_template_id_for_context,
)
from easyicu.research_agent.planning.progressive_contract import ProgressivePlanCompileError
from easyicu.research_agent.planning.sensitivity_authority import PrespecifiedSensitivitySpec
from easyicu.research_agent.providers.mocks import ScriptedMockLLMClient
from easyicu.research_agent.schema import (
    CohortDescriptor,
    ConceptDescriptor,
    ResearchContext,
    UserPreferences,
)
from easyicu.webserver.agent_pipeline_runs import _progressive_compile_failure_message

_WINDOW_REQUIRED = "progressive_exposure_window_required"


def _context(*, role="lab", window=None, host_hours=None, landmark=False):
    constraints = {} if host_hours is None else {"materialization_window": {
        "role": "outer_observation_window", "anchor": "icu_admission", "hours": host_hours,
    }}
    return ResearchContext(
        research_question="Association of signal_q with outcome_r.",
        cohort=CohortDescriptor(
            cohort_name="synthetic_admissions", database="synthetic", n_stays=0,
            id_columns=["stay_key"], outcome_columns=["outcome_r"],
        ),
        variables=[
            ConceptDescriptor(name="stay_key", role="id", dtype="int64"),
            ConceptDescriptor(name="signal_q", role=role, dtype="float64", analysis_window=window),
            ConceptDescriptor(name="outcome_r", role="outcome", dtype="bool",
                              observed_domain={"levels": [False, True], "n_unique": 2, "is_binary": True}),
        ],
        primary_exposure="signal_q", target_outcome="outcome_r",
        user_preferences=UserPreferences(
            inferred_analysis_family="association_study", data_constraints=json.dumps(constraints),
            sensitivity_specs=[PrespecifiedSensitivitySpec(
                spec_id="landmark_24h", axis="timing", strategy="landmark", landmark_hours=24,
                require_alive_at_landmark=True, exclude_negative_event_times=True,
                event_time_variable="event_hours", observation_duration_variable="followup_hours",
                observation_duration_unit="hours",
            )] if landmark else [],
        ),
    )


@pytest.mark.parametrize("role", ["lab", "vital"])
def test_an_untimed_measurement_stops_at_the_timing_owner(role):
    context = _context(role=role)
    for classify in (post_baseline_exposure, baseline_association_context):
        with pytest.raises(ProgressivePlanCompileError) as stopped:
            classify(context)
        assert stopped.value.reason_code == _WINDOW_REQUIRED
        assert stopped.value.path == "variables.signal_q.analysis_window"
        assert stopped.value.easyicu_safe_diagnostic["question_required"] is True
        assert "timing relative to time zero is unknown" in str(stopped.value)


def test_a_family_request_cannot_assume_an_untimed_measurement_is_baseline():
    context = _context()
    with pytest.raises(ProgressivePlanCompileError, match=_WINDOW_REQUIRED):
        build_family_spec_request(
            context, analysis_types=("association_study",),
            variable_roster=[item.name for item in context.variables],
            allowed_literature_citation_keys=(),
        )


def test_the_host_displays_the_typed_window_request():
    with pytest.raises(ProgressivePlanCompileError) as stopped:
        post_baseline_exposure(_context())
    sentence = _progressive_compile_failure_message(stopped.value)
    assert sentence.startswith("The exposure is a measurement with no time window")
    assert "no analysis was run" in sentence and "TODO" not in sentence


@pytest.mark.parametrize("strategy", [FAMILY_SPEC_STRATEGY, "progressive_v2"])
def test_the_planner_stops_before_writing_the_baseline_admission_claim(strategy):
    llm = ScriptedMockLLMClient([])
    with pytest.raises(ProgressivePlanCompileError) as stopped:
        ProgressivePlannerAgent(llm).run_attempt(_context(), planner_strategy=strategy)
    assert stopped.value.reason_code == _WINDOW_REQUIRED
    assert "available at admission" not in str(stopped.value)
    assert not llm.calls


@pytest.mark.parametrize("role", ["lab", "vital"])
def test_a_host_window_preserves_the_measurements_post_baseline_classification(role):
    context = _context(role=role, host_hours=24)
    assert post_baseline_exposure(context) == (True, "outer_materialization:icu_admission[0,24]h")
    assert not baseline_association_context(context)
    assert family_template_id_for_context(context, analysis_types=("association_study",)) is None


@pytest.mark.parametrize("host_hours", [None, 24])
def test_an_owner_declared_static_attribute_stays_baseline(host_hours):
    context = _context(role="demographic", host_hours=host_hours)
    assert post_baseline_exposure(context) == (False, None)
    assert baseline_association_context(context)
    assert family_template_id_for_context(context, analysis_types=("association_study",)) == "baseline_adjusted_association"


@pytest.mark.parametrize("role", ["lab", "vital"])
@pytest.mark.parametrize("host_hours", [None, 48])
def test_a_windowed_measurement_keeps_the_landmark_route(role, host_hours):
    context = _context(role=role, window="icu_admission[0,12]h", host_hours=host_hours, landmark=True)
    assert post_baseline_exposure(context) == (True, "icu_admission[0,12]h")
    assert family_template_id_for_context(context, analysis_types=("association_study",)) == "landmark_spline_association"


@pytest.mark.parametrize("role", ["lab", "vital"])
def test_an_explicit_baseline_window_is_still_authoritative(role):
    context = _context(role=role, window="icu_admission[-6,0]h", host_hours=24)
    assert post_baseline_exposure(context) == (False, "icu_admission[-6,0]h")
    assert baseline_association_context(context)
