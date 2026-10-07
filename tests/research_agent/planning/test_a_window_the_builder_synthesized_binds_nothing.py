"""A window the context builder synthesized binds no landmark, prediction time or time zero.

Without the host's materialization record (the CLI, a benchmark), a context's
windows are the ones its caller declared or, when it declared none, the
builder's own: windows inferred from the question's wording, else a default
roster whose ``full_stay`` ends at 720 h.  The feature-window end was the widest
window the context carried, whatever its origin.  So a CLI context proposed a
30-day landmark for 90-day mortality, and a prediction model predicted at 720 h
(only stays longer than 30 days) or at the 72 h of "death within 72 hours",
which excludes every death the question asks about.  The materialization
owner now answers: the host's record, else the caller's windows, else nothing.
Synthetic contexts; no rows.
"""

from __future__ import annotations

import json

import pytest

from easyicu.research_agent.icu_rules import default_time_windows
from easyicu.research_agent.planning.adjustment_authority import (
    host_outer_feature_window_end_hours,
    host_proven_temporal_roles,
)
from easyicu.research_agent.planning.family_spec.request import (
    proposed_survival_suite_coordinates,
)
from easyicu.research_agent.planning.scientific_review import plan_time_zero_hours
from easyicu.research_agent.research_context.materialization_window import (
    bound_feature_window_end_hours,
    caller_declared_time_windows,
)
from easyicu.research_agent.research_context.temporal_semantics import (
    TemporalAlignmentEngine,
)
from easyicu.research_agent.schema import (
    ConceptDescriptor,
    ResearchContext,
    TimeWindow,
    UserPreferences,
    VariableRole,
)
from tests.support.survival_proposal import survival_context

from .family_spec_fixtures import _prediction_context, _request

WITHIN_72_HOURS = (
    "Among adult ICU stays, how well do vitals and labs predict in-hospital death "
    "within 72 hours after ICU admission?"
)
FIRST_DAY = (
    "Among adult ICU stays, is the lactate of the first 24 hours associated with "
    "death within 90 days?"
)


def _inferred(question: str) -> list[TimeWindow]:
    windows, _constraints = TemporalAlignmentEngine().infer(research_question=question)
    assert windows, question
    return list(windows)


def _with_inferred_windows(context: ResearchContext, question: str) -> ResearchContext:
    """The context as the builder makes it when its caller declares no window."""

    return context.model_copy(
        update={"research_question": question, "time_windows": _inferred(question)}
    )


def _with_host_record(context: ResearchContext, hours: float) -> ResearchContext:
    constraints = json.loads(context.user_preferences.data_constraints or "{}")
    constraints["materialization_window"] = {
        "role": "outer_observation_window", "anchor": "icu_admission", "hours": hours,
    }
    return context.model_copy(
        update={
            "user_preferences": context.user_preferences.model_copy(
                update={"data_constraints": json.dumps(constraints)}
            )
        }
    )


def test_the_default_roster_proposes_no_landmark_and_states_no_time_zero() -> None:
    declared = survival_context()
    assert host_outer_feature_window_end_hours(declared) == 24.0
    assert proposed_survival_suite_coordinates(declared).landmark_hours == 24.0

    defaulted = survival_context(time_windows=default_time_windows())
    assert max(window.end_hours for window in defaulted.time_windows) == 720.0
    assert caller_declared_time_windows(defaulted) == ()
    assert host_outer_feature_window_end_hours(defaulted) is None
    assert plan_time_zero_hours(defaulted, None, None) is None
    # It proposed a 720 h (30-day) landmark for 90-day mortality.
    assert proposed_survival_suite_coordinates(defaulted) is None


def test_a_window_inferred_from_the_wording_binds_nothing() -> None:
    for question in (WITHIN_72_HOURS, FIRST_DAY):
        context = _with_inferred_windows(survival_context(), question)
        assert caller_declared_time_windows(context) == (), question
        assert bound_feature_window_end_hours(context) is None, question
        assert plan_time_zero_hours(context, None, None) is None, question
        assert proposed_survival_suite_coordinates(context) is None, question


def _without_host_record(context: ResearchContext) -> ResearchContext:
    """The context as the CLI or a benchmark builds it: no materialization record."""

    constraints = json.loads(context.user_preferences.data_constraints or "{}")
    constraints.pop("materialization_window", None)
    return context.model_copy(
        update={
            "user_preferences": context.user_preferences.model_copy(
                update={"data_constraints": json.dumps(constraints) if constraints else None}
            )
        }
    )


def test_a_prediction_without_a_bound_window_states_no_prediction_time() -> None:
    # The CLI binds no population mode either.
    cli = _without_host_record(_prediction_context())
    assert _request(cli, cohort_mode=None).prediction_time_hours == 24.0  # the caller's window

    for context in (
        cli.model_copy(update={"time_windows": default_time_windows()}),
        _with_inferred_windows(cli, WITHIN_72_HOURS),
    ):
        request = _request(context, cohort_mode=None)
        # It predicted at 720 h, or at the 72 h horizon of the outcome itself.
        assert request.prediction_time_hours is None
        assert request.observation_window_hours is None


def test_the_callers_windows_bind_the_run_and_another_anchor_is_ignored() -> None:
    context = survival_context(
        time_windows=[
            TimeWindow(name="first_36h", anchor="icu_admission", start_hours=0.0, end_hours=36.0),
            TimeWindow(name="first_48h_hospital", anchor="hospital_admission", start_hours=0.0,
                       end_hours=48.0),
        ]
    )
    assert len(caller_declared_time_windows(context)) == 2
    assert host_outer_feature_window_end_hours(context) == 36.0
    assert plan_time_zero_hours(context, None, None) == 36.0


def test_a_caller_window_that_repeats_one_default_still_binds() -> None:
    first_day = [window for window in default_time_windows() if window.name == "first_24h"]
    context = survival_context(time_windows=first_day)
    assert host_outer_feature_window_end_hours(context) == 24.0


def test_the_hosts_record_binds_the_run_whatever_windows_it_carries() -> None:
    for windows in (default_time_windows(), _inferred(WITHIN_72_HOURS)):
        context = _with_host_record(survival_context(time_windows=windows), 24.0)
        assert host_outer_feature_window_end_hours(context) == 24.0
        assert proposed_survival_suite_coordinates(context).landmark_hours == 24.0


def test_a_stored_context_reads_the_same_after_a_round_trip() -> None:
    for context in (
        survival_context(time_windows=default_time_windows()),
        _with_inferred_windows(survival_context(), WITHIN_72_HOURS),
    ):
        stored = ResearchContext.model_validate_json(context.model_dump_json())
        assert host_outer_feature_window_end_hours(stored) is None


@pytest.mark.parametrize("bound", [False, True])
def test_a_synthesized_window_proves_no_covariate_timing(bound: bool) -> None:
    lactate = ConceptDescriptor(
        name="lactate_max", description="maximum lactate", role=VariableRole.LAB,
        dtype="float64", unit="mmol/L",
    )
    context = _with_inferred_windows(survival_context(), FIRST_DAY)
    context = context.model_copy(
        update={
            "variables": [*context.variables, lactate],
            "user_preferences": UserPreferences(
                inferred_analysis_family="survival", landmark_hours=24.0
            ),
        }
    )
    if bound:
        context = _with_host_record(context, 24.0)
    roles = host_proven_temporal_roles(context)
    assert roles.get("age") == "baseline_static"
    assert (roles.get("lactate_max") == "at_or_before_time_zero") is bound
