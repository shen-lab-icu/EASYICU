"""A template must carry every requested outcome before it spends a draft.

Synthetic contexts only; routing is exercised with the offline scripted client.
"""

from __future__ import annotations

import pytest

from easyicu.research_agent.agents import family_spec_planner, progressive_planner
from easyicu.research_agent.agents.family_spec_planner import (
    FAMILY_SPEC_STRATEGY,
    family_spec_fallback_reason,
)
from easyicu.research_agent.agents.progressive_planner import ProgressivePlannerAgent
from easyicu.research_agent.planning.family_spec import (
    FamilySpecError,
    build_family_spec_request,
    family_template_id_for_context,
)
from easyicu.research_agent.planning.progressive_contract import ProgressivePlanCompileError
from easyicu.research_agent.providers.mocks import ScriptedMockLLMClient
from easyicu.research_agent.providers.structured_retry import StructuredResponseFailure
from easyicu.research_agent.schema import ConceptDescriptor, VariableRole
from tests.research_agent.planning.family_spec_fixtures import (
    _context,
    _descriptive_context,
    _prediction_context,
)
from tests.support.survival_proposal import survival_context, survival_request

FALLBACK = "family_template_requested_outcomes_not_supported"


def _requested(context, names):
    return context.model_copy(update={
        "cohort": context.cohort.model_copy(update={"requested_outcome_columns": names}),
    })


def _grouped_descriptive():
    context = _descriptive_context()
    return _requested(context, ["death", "support_days", "recovery_days"]).model_copy(update={
        "research_question": (
            "Describe mortality, days receiving support and recovery duration "
            "by the first-day phenotype group among adult ICU stays."
        ),
        "variables": [
            *context.variables,
            *[
                ConceptDescriptor(
                    name=name, role=VariableRole.OUTCOME, dtype="float64", unit="days"
                )
                for name in ("support_days", "recovery_days")
            ],
        ],
    })


def _reason(context, family):
    return family_spec_fallback_reason(
        context, analysis_types=(family,), resume_checkpoint=None, stop_after_outline=False,
    )


@pytest.mark.parametrize(
    ("build", "family", "template"),
    [
        (_grouped_descriptive, "descriptive_epidemiology", "descriptive_exposure_outcome"),
        (
            lambda: _requested(_prediction_context(), ["death", "los_icu"]),
            "prediction_model", "static_prediction_model",
        ),
        (
            lambda: _requested(_context(), ["death", "los_icu", "recovery_days"]),
            "association_study", "landmark_categorical_association",
        ),
    ],
)
def test_an_uncovered_outcome_yields_to_the_general_planner(build, family, template):
    context = build()
    assert family_template_id_for_context(context, analysis_types=(family,)) == template
    assert _reason(context, family) == FALLBACK


@pytest.mark.parametrize(
    ("build", "family", "requested"),
    [
        (_grouped_descriptive, "descriptive_epidemiology", []),
        (_grouped_descriptive, "descriptive_epidemiology", ["death"]),
        (_prediction_context, "prediction_model", ["death"]),
        (_context, "association_study", ["death"]),
        (_context, "association_study", ["death", "los_icu"]),
    ],
)
def test_a_subset_of_carried_outcomes_keeps_the_template(build, family, requested):
    context = _requested(build(), requested)
    assert family_template_id_for_context(context, analysis_types=(family,)) is not None
    assert _reason(context, family) is None


def test_the_grouped_descriptive_question_reaches_the_v2_outline(monkeypatch):
    monkeypatch.setattr(progressive_planner, "_MAX_INITIAL_PARSE_RETRIES", 0)
    monkeypatch.setattr(progressive_planner, "_MAX_STEP_PARSE_RETRIES", 0)
    events = []
    llm = ScriptedMockLLMClient(["{}"])
    with pytest.raises(StructuredResponseFailure):
        ProgressivePlannerAgent(llm).run_attempt(
            _grouped_descriptive(), planner_strategy=FAMILY_SPEC_STRATEGY,
            progress_callback=events.append,
        )
    assert len(llm.calls) == 1
    assert events[0].role == "progressive_planner_outline"


@pytest.mark.parametrize("family", ["descriptive_epidemiology", "prediction_model"])
def test_an_invalid_request_keeps_the_owners_typed_refusal(monkeypatch, family):
    context = _grouped_descriptive() if family == "descriptive_epidemiology" else _prediction_context()
    refusal = FamilySpecError("family_spec_synthetic_refusal", "Private study details")
    calls = []

    def refuse(*args, **kwargs):
        calls.append(kwargs)
        raise refusal

    monkeypatch.setattr(family_spec_planner, "build_family_spec_request", refuse)
    assert _reason(context, family) is None
    assert len(calls) == 1
    llm = ScriptedMockLLMClient([])
    with pytest.raises(ProgressivePlanCompileError) as stopped:
        ProgressivePlannerAgent(llm).run_attempt(context, planner_strategy=FAMILY_SPEC_STRATEGY)
    assert stopped.value.__cause__ is refusal
    assert stopped.value.reason_code == "progressive_family_spec_synthetic_refusal"
    assert llm.calls == []


@pytest.mark.parametrize(
    "suite_field",
    ["sealed_suite", "proposed_suite", "sealed_continuous_suite", "proposed_continuous_suite"],
)
def test_the_request_owner_counts_survival_outcome_coordinates(suite_field):
    request = survival_request(survival_context())
    suite = request.survival_suite
    if "continuous" in suite_field:
        from easyicu.research_agent.planning.family_spec.contract import (
            SealedContinuousSuiteCoordinates,
        )

        suite = SealedContinuousSuiteCoordinates(
            primary_owner="signed_landmark_continuous_survival_suite",
            exposure_column="marker_first", exposure_window_summary="first",
            event_column=suite.event_column, followup_time_column=suite.followup_time_column,
            landmark_hours=suite.landmark_hours, endpoint_horizon_days=suite.endpoint_horizon_days,
            plan_outputs=suite.plan_outputs,
        )
    request = request.model_copy(update={"proposed_suite": None, suite_field: suite})
    assert set(request.covered_outcome_columns) == {"mort_90d", "followup_days_90d"}
    assert "rrt" not in request.covered_outcome_columns
    assert "covered_outcome_columns" not in request.model_dump()
    assert "covered_outcome_columns" not in request.model_json_schema()["properties"]


def test_a_survival_template_keeps_its_followup_and_stops_if_no_fallback_owner_exists():
    context = _requested(survival_context(), ["mort_90d", "followup_days_90d"])
    assert _reason(context, "survival") is None
    context = _requested(context, ["mort_90d", "recovery_days"])
    assert _reason(context, "survival") == FALLBACK
    llm = ScriptedMockLLMClient([])
    with pytest.raises(ProgressivePlanCompileError) as stopped:
        ProgressivePlannerAgent(llm).run_attempt(context, planner_strategy=FAMILY_SPEC_STRATEGY)
    assert stopped.value.reason_code == "progressive_family_result_contract_unwritable"
    assert llm.calls == []


def test_outcome_coverage_is_owned_by_the_typed_request(monkeypatch):
    context = _grouped_descriptive()
    request = build_family_spec_request(
        context, analysis_types=("descriptive_epidemiology",),
        variable_roster=[v.name for v in context.variables], allowed_literature_citation_keys=(),
    )
    monkeypatch.setattr(type(request), "covered_outcome_columns", property(
        lambda self: tuple(context.cohort.requested_outcome_columns)
    ))
    assert _reason(context, "descriptive_epidemiology") is None
