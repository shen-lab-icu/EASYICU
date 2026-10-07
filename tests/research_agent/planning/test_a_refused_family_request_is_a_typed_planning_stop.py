"""A family request the host refuses is a typed planning stop.

The family-spec planner refuses a request it cannot seal before any Planner
call: a cohort decided after the plan's time zero, an accepted Table 1 the
template cannot describe, and so on.  A template projection the gates reject
already stopped planning with its own reason code.  A refused request did not:
its error left the planner untyped, so the Web run recorded
``research_pipeline_execution_failed`` with no cause, and the conversation read
it as an analysis step that failed while running.  The refusal now stops
planning with the request's reason code, and states that the Planner made no
Provider call, so the host can say what stopped and what to change.  A
prediction refused for its risk set (no ICU length of stay, every input row
bound, an unread unit) names the study change that lifts it.

Synthetic contexts only.
"""

from __future__ import annotations

import json

import pytest

from easyicu.research_agent.agents.family_spec_planner import FAMILY_SPEC_STRATEGY
from easyicu.research_agent.agents.progressive_planner import ProgressivePlannerAgent
from easyicu.research_agent.planning.baseline_requirements import bind_baseline_requirements
from easyicu.research_agent.planning.family_spec import FamilySpecError
from easyicu.research_agent.planning.progressive_contract import ProgressivePlanCompileError
from easyicu.research_agent.providers.mocks import ScriptedMockLLMClient
from easyicu.research_agent.schema import ConceptDescriptor, ResearchContext, VariableRole
from easyicu.webserver import agent_pipeline_runs
from easyicu.webserver.pi_copilot.workflow import gate_detail_projection

from .family_spec_fixtures import (
    ALLOWED_CITATIONS,
    DIRECT_COMPARATORS,
    _context,
    _prediction_context,
)

LATE_COHORT = "family_spec_cohort_eligibility_after_time_zero"
UNGROUPABLE = "family_spec_accepted_baseline_grouping_unsupported"


def _late_cohort() -> ResearchContext:
    """Sepsis-3 found up to 720 h cannot define a cohort followed from 24 h."""

    context = _context()
    constraints = json.loads(context.user_preferences.data_constraints or "{}")
    constraints["concept_cohort_window"] = {"definition": "sepsis3", "window_end_hours": 720}
    return context.model_copy(
        update={
            "user_preferences": context.user_preferences.model_copy(
                update={"data_constraints": json.dumps(constraints)}
            )
        }
    )


def _ungroupable() -> ResearchContext:
    """An accepted Table 1 grouped by the outcome, which the template cannot group by."""

    return bind_baseline_requirements(
        _context(exact=True),
        {
            "schema_version": "easyicu.accepted_baseline_requirements/1",
            "source_plan_sha256": "c" * 64,
            "tables": [
                {
                    "source_step_id": "baseline_context",
                    "group_by": {"name": "death", "source_concept": None},
                    "variables": [{"name": "age", "source_concept": None}],
                }
            ],
        },
    )


REFUSALS = {LATE_COHORT: _late_cohort, UNGROUPABLE: _ungroupable}


def _planning_stop(
    context: ResearchContext, mode: str | None = "predicate_filtered"
) -> ProgressivePlanCompileError:
    llm = ScriptedMockLLMClient([])
    with pytest.raises(ProgressivePlanCompileError) as stopped:
        ProgressivePlannerAgent(llm).run_attempt(
            context,
            planner_strategy=FAMILY_SPEC_STRATEGY,
            allowed_literature_citation_keys=ALLOWED_CITATIONS,
            direct_comparator_literature_keys=DIRECT_COMPARATORS,
            comparison_literature_keys=DIRECT_COMPARATORS,
            enforce_article_contract=True,
            article_contract_context=context,
            planning_contract_context="",
            required_primary_cohort_selection_mode=mode,
        )
    assert llm.calls == []
    return stopped.value


@pytest.mark.parametrize("code", sorted(REFUSALS))
def test_a_refused_request_stops_planning_with_its_own_reason(code: str) -> None:
    stop = _planning_stop(REFUSALS[code]())

    assert stop.reason_code == f"progressive_{code}"
    assert isinstance(stop.__cause__, FamilySpecError)
    assert stop.__cause__.reason_code == code
    assert stop.path == (stop.__cause__.path or "family_spec")
    assert stop.easyicu_safe_diagnostic["metrics"] == {"planner_provider_calls": 0}


@pytest.mark.parametrize("code", sorted(REFUSALS))
def test_the_run_records_a_planning_stop_not_a_failed_analysis_step(code: str, tmp_path) -> None:
    stop = _planning_stop(REFUSALS[code]())
    # The untyped refusal is what the run recorded before.
    assert agent_pipeline_runs._pipeline_failure_code(stop.__cause__) == (
        "research_pipeline_execution_failed"
    )

    failure_code = agent_pipeline_runs._pipeline_failure_code(stop)
    assert failure_code == "research_pipeline_progressive_compile_failed"
    agent_pipeline_runs._record_pipeline_failure(
        wrapper_dir=tmp_path,
        study={"id": "study_synthetic"},
        provider={},
        exc=stop,
        code=failure_code,
        execution_retry_id=None,
    )

    gate = json.loads((tmp_path / "quality_gate.json").read_text())["gate"]
    assert gate["reason"] == failure_code
    assert gate["detail"] == {"reason_code": f"progressive_{code}"}
    assert gate_detail_projection(gate["detail"])["gate_detail_code"] == f"progressive_{code}"
    diagnostic = json.loads(
        (tmp_path / "diagnostics" / "research_pipeline_failure.json").read_text()
    )
    assert diagnostic["typed_failure"]["reason_code"] == f"progressive_{code}"
    assert diagnostic["typed_failure"]["metrics"] == {"planner_provider_calls": 0}
    # The request's own wording names study values; it stays out of the record.
    recorded = json.dumps(gate) + json.dumps(diagnostic)
    assert str(stop.__cause__) not in recorded
    assert "sepsis3" not in recorded


def test_the_host_says_what_stopped_planning_and_what_to_change() -> None:
    late = agent_pipeline_runs._progressive_compile_failure_message(
        _planning_stop(_late_cohort())
    )
    assert "before the Planner was called" in late
    assert "move time zero later" in late
    assert "replay artifact" not in late

    refused = agent_pipeline_runs._progressive_compile_failure_message(
        _planning_stop(_ungroupable())
    )
    assert "refused this study's planning request before the Planner was called" in refused
    assert "replay artifact" not in refused

    # A template projection the gates reject after the Planner's answer keeps the
    # compiler's sentence: only the recorded call count says no call was made.
    projected = ProgressivePlanCompileError(
        f"progressive_{UNGROUPABLE}", "synthetic projection stop", path="family_spec"
    )
    assert "replay artifact was preserved" in (
        agent_pipeline_runs._progressive_compile_failure_message(projected)
    )


def _with_icu_stay(context: ResearchContext, unit: str | None) -> ResearchContext:
    """The context with its stay-level ICU length of stay in ``unit``, or without it."""

    variables = [item for item in context.variables if item.name != "los_icu"]
    if unit is not None:
        variables.append(
            ConceptDescriptor(
                name="los_icu", description="ICU length of stay", role=VariableRole.OUTCOME,
                dtype="float64", unit=unit, source_concept="los_icu",
            )
        )
    return context.model_copy(update={"variables": variables})


PREDICTION_STOPS = [
    pytest.param(
        lambda: _with_icu_stay(_prediction_context(), None), None,
        "family_spec_prediction_risk_set_unavailable",
        "Declare the study's analysis as a prediction model",
        id="no_icu_stay",
    ),
    pytest.param(
        lambda: _with_icu_stay(_prediction_context(), "weeks"), None,
        "family_spec_icu_stay_unit_unread",
        "Prepare the export again with the unit recorded",
        id="unread_unit",
    ),
]


@pytest.mark.parametrize(("build", "mode", "code", "remedy"), PREDICTION_STOPS)
def test_a_prediction_stop_names_the_study_change_that_lifts_it(
    build, mode: str | None, code: str, remedy: str
) -> None:
    stop = _planning_stop(build(), mode)

    assert stop.reason_code == f"progressive_{code}"
    assert stop.easyicu_safe_diagnostic["metrics"] == {"planner_provider_calls": 0}
    message = agent_pipeline_runs._progressive_compile_failure_message(stop)
    assert remedy in message
    assert "before the Planner was called" in message
