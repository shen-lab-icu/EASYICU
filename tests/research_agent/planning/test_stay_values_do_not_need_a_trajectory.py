"""A gap is about the question's required representation, across study families."""

import json

import pytest

from easyicu.research_agent.agents.progressive_planner import ProgressivePlannerAgent
from easyicu.research_agent.planning.capability_gap import (
    capability_gap_confirmation_facts, capability_gap_stop, check_capability_gap,
)
from easyicu.research_agent.planning.progressive_contract import (
    ProgressiveCapabilityGap, ProgressivePlanCompileError, ProgressivePlanOutline,
)
from easyicu.research_agent.providers.mocks import ScriptedMockLLMClient
from easyicu.research_agent.providers.structured_retry import StructuredResponseFailure
from easyicu.research_agent.schema import ConceptDescriptor, UserPreferences

from tests.research_agent.planning.progressive_planner_fixtures import _context, _outline_payload


GAP = "longitudinal_representation_unavailable"


def _study(kind="description", *, prepared=False, database="eicu_demo"):
    base = _context()
    group = ConceptDescriptor(name="ward_category", role="demographic", dtype="string")
    duration = ConceptDescriptor(name="care_duration", source_concept="los_icu", role="outcome", dtype="float64")

    def summary(concept, statistic):
        return ConceptDescriptor(
            name=f"{concept}_{statistic}", source_concept=concept, role="lab", dtype="float64",
            **({"unit_normalization": f"window_numeric_{statistic}",
                "analysis_window": "icu_admission[0,18]h"} if prepared else {}),
        )

    signal = summary("trial_signal", "min")
    companion = summary("companion_signal", "max")
    family, question, exposure, outcome, variables = {
        "description": (
            "descriptive_epidemiology", "Describe care duration by ward category.",
            group.name, duration.name, [group, duration],
        ),
        "window_description": (
            "descriptive_epidemiology", "Describe the first window's minimum trial signal by ward category.",
            group.name, signal.name, [group, signal],
        ),
        "association": (
            "association_study", "Relate ward category to care duration.",
            group.name, duration.name, [group, duration],
        ),
        "prediction": (
            "prediction_model", "Predict care duration from the minimum trial signal and maximum companion signal.",
            signal.name, duration.name, [group, duration, signal, companion],
        ),
    }[kind]
    if kind == "window_description":
        # The outcome is already supplied as a summary, even in metadata.
        signal.unit_normalization = "window_numeric_min"
        signal.analysis_window = "icu_admission[0,18]h"
    preferences = UserPreferences(
        inferred_analysis_family=family,
        covariates=[companion.name] if kind == "prediction" else [],
        covariate_selection="exact" if kind == "prediction" else "planner_selectable",
        data_constraints=json.dumps({
            "materialization_window": {"anchor": "icu_admission", "hours": 18, "role": "outer_observation_window"},
            "stated_subgroups": [{"variable": group.name, "words": "by ward category",
                                  "concept": group.name, "source_status": "available"}],
        }),
    )
    return base.model_copy(update={
        "research_question": question, "variables": variables,
        "primary_exposure": exposure, "target_outcome": outcome,
        "cohort": base.cohort.model_copy(update={
            "database": database, "n_stays": 0, "outcome_columns": [outcome],
            "provenance": {"evidence_stage": "prepared" if prepared else "metadata_only_planning",
                           "patient_rows_read": False},
        }),
        "user_preferences": preferences,
    })


def _claim(context):
    return {**_outline_payload(), "analysis_type": context.user_preferences.inferred_analysis_family,
            "steps": [], "capability_gap": GAP,
            "rationale": "Time alignment and patient dependence require an unavailable repeated-measure representation."}


@pytest.mark.parametrize("kind", ["description", "window_description", "association", "prediction"])
@pytest.mark.parametrize("prepared", [False, True])
def test_stay_values_and_available_window_summaries_contradict_the_gap(kind, prepared):
    context = _study(kind, prepared=prepared)
    before = context.model_dump(mode="json")
    check = check_capability_gap(ProgressiveCapabilityGap(requirement=GAP), context=context)
    assert check.verification == "unverified"
    assert context.user_preferences.inferred_analysis_family in check.fact
    assert context.primary_exposure in check.fact
    assert context.target_outcome in check.fact
    assert "repeated measurements" in check.fact
    feedback = capability_gap_confirmation_facts(context, offered_families=())
    facts = json.loads(feedback.split("\n", 1)[1])["stay_level_representation"]
    assert facts["analysis_family"] == context.user_preferences.inferred_analysis_family
    assert facts["requires_longitudinal_representation"] is False
    if kind in {"description", "association"}:
        assert facts["stay_level_variables"] == ["care_duration", "ward_category"]
        assert facts["window_summary_variables"] == []
    else:
        assert {item["column"] for item in facts["window_summary_variables"]} == (
            {"trial_signal_min", "companion_signal_max"} if kind == "prediction" else {"trial_signal_min"}
        )
        assert all(item["window_hours"] == [0, 18] for item in facts["window_summary_variables"])
        assert {item["summary"] for item in facts["window_summary_variables"]} == (
            {"min", "max"} if kind == "prediction" else {"min"}
        )
    assert context.model_dump(mode="json") == before


def test_the_same_stay_level_rule_applies_to_a_mimic_iv_fixture():
    context = _study(database="miiv")
    assert check_capability_gap(ProgressiveCapabilityGap(requirement=GAP), context=context).verification == "unverified"


@pytest.mark.parametrize("kind", ["description", "window_description", "prediction"])
def test_false_gap_feedback_uses_the_existing_typed_retry_for_each_family(kind):
    context = _study(kind)
    llm = ScriptedMockLLMClient([json.dumps(_claim(context))] * 4)
    llm.supports_strict_json_schema = True
    with pytest.raises(StructuredResponseFailure) as caught:
        ProgressivePlannerAgent(llm).run(context, stop_after_outline=True)
    cause = caught.value.__cause__
    assert cause.reason_code == "progressive_capability_gap_claim_unverified"
    assert cause.path == "capability_gap"
    assert cause.easyicu_safe_diagnostic["cause_code"] == GAP
    assert len(llm.calls) == 4
    feedback = llm.calls[1][0][-1].content
    for required in (context.user_preferences.inferred_analysis_family, context.primary_exposure, context.target_outcome):
        assert required in feedback
    if kind == "prediction":
        assert "companion_signal_max" in feedback


@pytest.mark.parametrize("question", [
    "Describe repeated measurements of trial signal within each ICU stay by ward category.",
    "Describe the trial signal trajectory over time within each ICU stay.",
    "描述每次入住期间试验指标随时间变化的轨迹。",
])
def test_a_true_trajectory_question_still_stops_without_retry(question):
    context = _study("window_description").model_copy(update={"research_question": question})
    llm = ScriptedMockLLMClient([json.dumps(_claim(context))])
    llm.supports_strict_json_schema = True
    checkpoints = []
    with pytest.raises(ProgressivePlanCompileError) as caught:
        ProgressivePlannerAgent(llm).run(context, checkpoint_callback=checkpoints.append, stop_after_outline=True)
    assert caught.value.reason_code == "progressive_capability_gap"
    assert caught.value.easyicu_safe_diagnostic["cause_code"] == GAP
    assert checkpoints[0].prompt_metrics["capability_gap"]["verification"] == "verified"
    assert len(llm.calls) == 1


@pytest.mark.parametrize("missing", [
    "exposure", "outcome", "identity", "source", "source_values", "derived_source_values",
    "window", "ordered_window", "anchor", "summary_receipt", "predictor", "subgroup", "adjustment",
    "metadata_summary", "nonfinite_window",
])
def test_incomplete_scalar_evidence_never_verifies_a_longitudinal_requirement(missing):
    context = _study("prediction", prepared=missing != "metadata_summary")
    signal = context.variable("trial_signal_min")
    constraints = json.loads(context.user_preferences.data_constraints)
    if missing in {"exposure", "outcome"}:
        context = context.model_copy(update={"primary_exposure" if missing == "exposure" else "target_outcome": "absent"})
    elif missing == "identity":
        context.cohort.id_columns = []
    elif missing == "source":
        context.variable("care_duration").source_concept = "mort_28d"
    elif missing in {"source_values", "derived_source_values"}:
        if missing == "derived_source_values":
            signal.derived_from_concepts = ["unavailable_signal"]
        context.cohort.provenance["concepts_without_values"] = [
            "trial_signal" if missing == "source_values" else "unavailable_signal",
        ]
    elif missing == "window":
        signal.analysis_window = None
        constraints.pop("materialization_window")
    elif missing == "ordered_window":
        signal.analysis_window = "icu_admission[18,0]h"
    elif missing == "nonfinite_window":
        signal.analysis_window = "icu_admission[0," + "9" * 310 + "]h"
    elif missing == "anchor":
        signal.analysis_window = "unreadable_window"
    elif missing == "summary_receipt":
        signal.unit_normalization = None
    elif missing == "predictor":
        context.variable("companion_signal_max").unit_normalization = None
    elif missing == "metadata_summary":
        # A suffix and an analysis label do not offer materialization.
        signal.analysis_window = "icu_admission[0,18]h"
    elif missing in {"subgroup", "adjustment"}:
        constraints["stated_subgroups" if missing == "subgroup" else "stated_adjustments"] = [
            {"variable": "unbound_marker", "words": "by an additional marker",
             "concept": "unbound_marker", "source_status": "available"},
        ]
    context.user_preferences.data_constraints = json.dumps(constraints)
    check = check_capability_gap(ProgressiveCapabilityGap(requirement=GAP), context=context)
    assert check.verification == "unverifiable"


def test_an_optional_roster_measure_does_not_add_a_question_requirement():
    context = _study()
    context.variables.append(ConceptDescriptor(name="optional_signal", role="lab", dtype="float64"))
    assert check_capability_gap(ProgressiveCapabilityGap(requirement=GAP), context=context).verification == "unverified"


def test_a_provided_summary_on_another_anchor_is_still_a_scalar():
    context = _study("window_description", prepared=True)
    context.variable("trial_signal_min").analysis_window = "event_onset[-6,18]h"
    check = check_capability_gap(ProgressiveCapabilityGap(requirement=GAP), context=context)
    assert check.verification == "unverified"
    assert "event_onset" in check.fact
    assert "-6-18 h" in check.fact


@pytest.mark.parametrize("status", ["unavailable", "unresolved", "unbound"])
def test_a_roster_column_does_not_override_a_required_variables_source_receipt(status):
    context = _study()
    constraints = json.loads(context.user_preferences.data_constraints)
    constraints["stated_subgroups"][0]["source_status"] = status
    context.user_preferences.data_constraints = json.dumps(constraints)
    assert check_capability_gap(ProgressiveCapabilityGap(requirement=GAP), context=context).verification == "unverifiable"


def test_a_materialized_stay_value_does_not_need_a_dictionary_entry_or_window():
    context = _study("association")
    group = context.variable("ward_category")
    group.role = "other"
    group.unit_normalization = "stay_level_unique_value"
    assert check_capability_gap(ProgressiveCapabilityGap(requirement=GAP), context=context).verification == "unverified"


def test_an_unbound_question_has_no_proof_that_all_its_variables_are_scalar():
    context = _study().model_copy(update={"primary_exposure": None, "target_outcome": None})
    assert check_capability_gap(ProgressiveCapabilityGap(requirement=GAP), context=context).verification == "unverifiable"


@pytest.mark.parametrize("representation", ["stay_outcome", "aliased_stay_value"])
def test_required_groups_use_their_source_binding_without_imposing_covariate_eligibility(representation):
    context = _study()
    group = context.variable("ward_category")
    if representation == "stay_outcome":
        group.role = "outcome"
    else:
        group.role = "other"
        group.source_concept = "source_ward_category"
        group.unit_normalization = "stay_level_unique_value"
        constraints = json.loads(context.user_preferences.data_constraints)
        constraints["stated_subgroups"][0]["concept"] = group.source_concept
        context.user_preferences.data_constraints = json.dumps(constraints)
    check = check_capability_gap(ProgressiveCapabilityGap(requirement=GAP), context=context)
    assert check.verification == "unverified"
    assert group.name in check.fact


def test_a_saved_scalar_gap_cannot_issue_a_terminal_capability_receipt():
    context = _study()
    outline = ProgressivePlanOutline.model_validate(_claim(context))
    check = check_capability_gap(outline.declared_capability_gap(), context=context)
    with pytest.raises(ProgressivePlanCompileError) as caught:
        capability_gap_stop(outline, check)
    assert caught.value.reason_code == "progressive_capability_gap_claim_unverified"
    assert not hasattr(caught.value, "easyicu_planning_capability_gap")
