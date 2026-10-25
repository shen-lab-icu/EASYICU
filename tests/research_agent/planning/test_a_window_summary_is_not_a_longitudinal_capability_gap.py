"""A scalar window exposure needs time alignment, not a trajectory representation."""

import json

import pytest

from easyicu.research_agent.agents.progressive_planner import ProgressivePlannerAgent
from easyicu.research_agent.agents.progressive_prompt_contracts import outline_study_anchors
from easyicu.research_agent.planning.capability_gap import capability_gap_stop, check_capability_gap
from easyicu.research_agent.planning.progressive_contract import (
    ProgressiveCapabilityGap, ProgressivePlanCompileError, ProgressivePlanOutline,
)
from easyicu.research_agent.providers.mocks import ScriptedMockLLMClient
from easyicu.research_agent.providers.structured_retry import StructuredResponseFailure
from easyicu.research_agent.schema import ConceptDescriptor, EndpointSpec, UserPreferences

from tests.research_agent.planning.progressive_planner_fixtures import _context, _outline_payload


GAP = "longitudinal_representation_unavailable"


def _study(database="eicu_demo", *, prepared=False):
    context = _context()
    exposure = ConceptDescriptor(
        name="trial_signal_min", source_concept="trial_signal", role="lab", dtype="float64",
        **({"unit_normalization": "window_numeric_min", "analysis_window": "icu_admission[0,24]h"}
           if prepared else {}),
    )
    return context.model_copy(update={
        "research_question": "Relate the first day's minimum trial signal to hospital death within each recorded sex.",
        "cohort": context.cohort.model_copy(update={
            "database": database, "n_stays": 0,
            "provenance": {"evidence_stage": "prepared" if prepared else "metadata_only_planning",
                           "patient_rows_read": False, "patient_grouping_status": "available_unbound"},
        }),
        "variables": [exposure, *context.variables[1:]],
        "primary_exposure": exposure.name,
        "endpoint": EndpointSpec(name="outcome_flag", kind="binary", levels=[0, 1], absence_semantics="no_absent_rows"),
        "user_preferences": UserPreferences(data_constraints=json.dumps({
            "materialization_window": {"anchor": "icu_admission", "hours": 24, "role": "outer_observation_window"},
            "event_time_semantics": {"death_time": "recorded_deathtime" if database == "miiv" else "structurally_unavailable"},
            "stated_subgroups": [{"variable": "recorded sex", "words": "within each recorded sex",
                                  "concept": "sex_code", "source_status": "available"}],
        })),
    })


def _claim():
    return {**_outline_payload(), "steps": [], "capability_gap": GAP,
            "rationale": "The post-admission exposure requires repeated measurements that the roster lacks."}


@pytest.mark.parametrize("database", ["eicu_demo", "miiv"])
@pytest.mark.parametrize("prepared", [False, True])
def test_the_host_contradicts_a_longitudinal_gap_for_a_fixed_window_summary(database, prepared):
    context = _study(database, prepared=prepared)
    before = context.model_dump(mode="json")
    check = check_capability_gap(ProgressiveCapabilityGap(requirement=GAP), context=context)
    assert check.verification == "unverified"
    for word in ("association_study", "trial_signal_min", "24", "landmark", "deaths before"):
        assert word in check.fact
    assert "Verify landmark eligibility separately before execution" in check.fact
    assert context.model_dump(mode="json") == before


def test_the_first_prompt_projects_the_summary_and_the_separate_eligibility_requirement():
    facts = json.loads(outline_study_anchors(_study()))["fixed_window_association"]
    assert facts["analysis_family"] == "association_study"
    assert facts["exposure"] == "trial_signal_min"
    assert facts["summary"] == "min"
    assert facts["window_hours"] == [0, 24]
    assert facts["proposed_time_zero_hours"] == 24
    assert facts["death_time_semantics"] == "structurally_unavailable"
    assert facts["patient_grouping_status"] == "available_unbound"
    assert facts["requires_longitudinal_representation"] is False
    assert "deaths before" in facts["planning_rule"]
    assert "eligibility" in facts["planning_rule"]


def test_a_contradicted_gap_uses_the_existing_typed_outline_retry_budget():
    llm = ScriptedMockLLMClient([json.dumps(_claim())] * 4)
    llm.supports_strict_json_schema = True
    with pytest.raises(StructuredResponseFailure) as caught:
        ProgressivePlannerAgent(llm).run(_study(), stop_after_outline=True)
    assert caught.value.__cause__.reason_code == "progressive_capability_gap_claim_unverified"
    assert len(llm.calls) == 4
    feedback = llm.calls[1][0][-1].content
    for word in ("association_study", "trial_signal_min", "24", "landmark", "deaths before", "eligibility",
                 "structurally_unavailable", "available_unbound"):
        assert word in feedback


def test_a_true_repeated_measure_question_still_stops_with_the_existing_cause():
    context = _study().model_copy(update={
        "research_question": "Relate repeated measurements of trial signal over time to hospital death within each sex.",
    })
    llm = ScriptedMockLLMClient([json.dumps(_claim())])
    llm.supports_strict_json_schema = True
    checkpoints = []
    with pytest.raises(ProgressivePlanCompileError) as caught:
        ProgressivePlannerAgent(llm).run(context, checkpoint_callback=checkpoints.append, stop_after_outline=True)
    assert caught.value.reason_code == "progressive_capability_gap"
    assert caught.value.easyicu_safe_diagnostic["cause_code"] == GAP
    assert checkpoints[0].prompt_metrics["capability_gap"]["verification"] == "verified"
    assert len(llm.calls) == 1


def test_a_saved_gap_cannot_become_a_terminal_receipt_after_the_host_contradicts_it():
    outline = ProgressivePlanOutline.model_validate(_claim())
    check = check_capability_gap(outline.declared_capability_gap(), context=_study())
    with pytest.raises(ProgressivePlanCompileError) as caught:
        capability_gap_stop(outline, check)
    assert caught.value.reason_code == "progressive_capability_gap_claim_unverified"
    assert caught.value.path == "capability_gap"
    assert not hasattr(caught.value, "easyicu_planning_capability_gap")


@pytest.mark.parametrize("missing", [
    "association", "exposure", "outcome", "identity", "binary_endpoint", "source", "source_values",
    "window", "icu_anchor", "ordered_window", "summary", "materialized_transform",
])
def test_a_summary_proposal_requires_host_evidence(missing):
    context = _study(prepared=True)
    if missing == "association":
        context = context.model_copy(update={"research_question": "Describe the distribution of trial signal among ICU stays."})
    elif missing == "exposure":
        context = context.model_copy(update={"primary_exposure": "absent_column"})
    elif missing == "outcome":
        context = context.model_copy(update={"target_outcome": "absent_column"})
    elif missing == "identity":
        context.cohort.id_columns = []
    elif missing == "binary_endpoint":
        context = context.model_copy(update={"endpoint": None})
        context.variables[1].observed_domain = None
    elif missing == "source":
        context.variables[1].source_concept = "mort_28d"
    elif missing == "source_values":
        context.cohort.provenance["concepts_without_values"] = ["trial_signal"]
    elif missing == "window":
        context.variables[0].analysis_window = None
        context.user_preferences.data_constraints = "{}"
    elif missing == "icu_anchor":
        context.variables[0].analysis_window = "event_onset[0,24]h"
    elif missing == "ordered_window":
        context.variables[0].analysis_window = "icu_admission[24,0]h"
    elif missing == "summary":
        context.variables[0].name = "trial_signal"
        context = context.model_copy(update={"primary_exposure": "trial_signal"})
        context.variables[0].unit_normalization = None
    elif missing == "materialized_transform":
        # A column suffix without a materialization receipt is not proof.
        context.variables[0].unit_normalization = None
    assert "fixed_window_association" not in json.loads(outline_study_anchors(context))


def test_the_column_window_overrides_the_outer_window_without_binding_a_landmark():
    context = _study("miiv", prepared=True)
    context.variables[0].analysis_window = "icu_admission[6,36]h"
    facts = json.loads(outline_study_anchors(context))["fixed_window_association"]
    assert facts["window_hours"] == [6, 36]
    assert facts["proposed_time_zero_hours"] == 36
    assert facts["death_time_semantics"] == "recorded_deathtime"
    assert context.user_preferences.landmark_hours is None


def test_the_retry_can_return_an_explicit_landmark_design_without_changing_the_question():
    context = _study("miiv")
    outline = json.loads(json.dumps(_outline_payload()).replace("exposure_flag", "trial_signal_min"))
    outline["steps"][2].update(module_id="absolute_risk_context", population_scope="analysis_cohort")
    selected = outline["design_selection"]["candidates"][0]
    selected.update(
        time_zero="24 h after ICU admission; a proposed fixed landmark.",
        observation_window="Hospital follow-up after the landmark; deaths before 24 h excluded.",
        estimand="Association of the first-day minimum trial signal with hospital death within each sex.",
    )
    selected["reviewable_plan"][1] = "Use the first-day minimum trial signal, fixed at the 24 h landmark."
    selected["reviewable_plan"][2] = "Exclude deaths before 24 h and verify continuing observation at the landmark."
    llm = ScriptedMockLLMClient([json.dumps(_claim()), json.dumps(outline)])
    llm.supports_strict_json_schema = True
    result = ProgressivePlannerAgent(llm).run(context, stop_after_outline=True)
    assert len(llm.calls) == 2
    assert result.capability_gap is None
    candidate = next(item for item in result.design_selection.candidates if item.disposition == "selected")
    assert candidate.time_zero == selected["time_zero"]
    assert candidate.observation_window == selected["observation_window"]
    assert "within each sex" in candidate.estimand
    assert context.user_preferences.landmark_hours is None
