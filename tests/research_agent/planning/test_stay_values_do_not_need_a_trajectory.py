"""A gap is about the question's required representation, across study families."""

import json

import pytest

from easyicu.research_agent.agents.progressive_planner import ProgressivePlannerAgent
from easyicu.research_agent.agents.progressive_prompt_contracts import outline_study_anchors
from easyicu.research_agent.contracts.association_execution import ASSOCIATION_HOST_ESTIMATORS
from easyicu.research_agent.planning.capability_gap import (
    capability_gap_confirmation_facts, capability_gap_stop, check_capability_gap,
    metadata_continuous_association_options,
)
from easyicu.research_agent.planning.progressive_contract import (
    ProgressiveCapabilityGap, ProgressivePlanCompileError, ProgressivePlanOutline,
)
from easyicu.research_agent.providers.mocks import ScriptedMockLLMClient
from easyicu.research_agent.providers.structured_retry import StructuredResponseFailure
from easyicu.research_agent.schema import AggregationRule, ConceptDescriptor, EndpointSpec, UserPreferences

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


def _unbound_window_study():
    """A metadata-only first-day question, with no actual summary receipt."""

    context = _study("association", database="sic")
    exposure = ConceptDescriptor(
        name="lact", source_concept="lact", role="lab", dtype="float64",
        aggregation_default=AggregationRule.MEDIAN_ONLY,
    )
    outcome = context.variable("care_duration")
    preferences = context.user_preferences.model_copy(update={
        "data_constraints": json.dumps({
            "materialization_window": {
                "anchor": "icu_admission", "hours": 24, "role": "outer_observation_window",
            },
        }),
    })
    return context.model_copy(update={
        "variables": [exposure, outcome], "primary_exposure": exposure.name,
        "research_question": (
            "Relate the first ICU day's lactate level to ICU length of stay in adults."
        ),
        "user_preferences": preferences,
    })


def _continuous_metadata_study():
    return _unbound_window_study().model_copy(update={
        "endpoint": EndpointSpec(name="care_duration", kind="continuous", absence_semantics="absent_row_is_unmeasured"),
        "notes": "PRIVATE_PATIENT_SENTINEL",
        "cohort_parquet": "PRIVATE_PATH_SENTINEL",
    })


def test_metadata_continuous_association_components_remain_unselected_proposals():
    context = _continuous_metadata_study()
    before = context.model_dump(mode="json")
    options = metadata_continuous_association_options(context)
    assert options is not None
    assert options["status"] == "proposal_only_not_selected_or_materialized"
    assert options["exposure_concept"] == "lact"
    assert options["outcome"] == "care_duration"
    assert options["continuous_estimator_component"] == ASSOCIATION_HOST_ESTIMATORS["continuous"]
    [window] = options["summary_options"]
    assert window["window_hours_after_icu_admission"] == [0, 24]
    assert {item["proposed_column"] for item in window["options"]} == {
        "lact_first", "lact_max", "lact_mean", "lact_min",
    }
    assert all(item["summary"] != "median" for item in window["options"])
    rule = options["planning_rule"]
    for text in ("not an approved choice", "not the first, maximum", "risk set",
                 "source-specific outcome semantics", "changes the estimand", "actual offered contracts"):
        assert text in rule
    first_prompt = outline_study_anchors(context)
    confirmation = capability_gap_confirmation_facts(context, offered_families=("association_study",))
    assert json.loads(first_prompt)["metadata_continuous_association_options"] == options
    assert json.loads(confirmation.split("\n", 1)[1])["metadata_continuous_association_options"] == options
    assert "PRIVATE_" not in first_prompt and "PRIVATE_" not in confirmation
    assert context.model_dump(mode="json") == before
    assert check_capability_gap(ProgressiveCapabilityGap(requirement=GAP), context=context).verification == "unverifiable"


@pytest.mark.parametrize("missing", ["not_metadata", "rows_read", "outcome", "endpoint", "exposure", "window", "trajectory"])
def test_metadata_options_never_override_missing_or_changed_authority(missing):
    context = _continuous_metadata_study()
    provenance = dict(context.cohort.provenance)
    if missing in {"not_metadata", "rows_read"}:
        provenance["evidence_stage" if missing == "not_metadata" else "patient_rows_read"] = (
            "prepared" if missing == "not_metadata" else True
        )
        context = context.model_copy(update={"cohort": context.cohort.model_copy(update={"provenance": provenance})})
    elif missing == "outcome":
        context = context.model_copy(update={"target_outcome": "missing"})
    elif missing == "endpoint":
        context = context.model_copy(update={"endpoint": None})
    elif missing == "exposure":
        context = context.model_copy(update={"primary_exposure": "missing"})
    elif missing == "window":
        context = context.model_copy(update={"user_preferences": context.user_preferences.model_copy(update={"data_constraints": "{}"})})
    else:
        context = context.model_copy(update={"research_question": "Study repeated lactate measurements over time within each ICU stay."})
    assert metadata_continuous_association_options(context) is None


@pytest.mark.parametrize("earlier_rejections", [0, 1, 2, 3])
def test_an_unverifiable_longitudinal_gap_gets_one_bounded_confirmation(earlier_rejections):
    context = _unbound_window_study()
    claim = json.dumps(_claim(context))
    llm = ScriptedMockLLMClient(["{}"] * earlier_rejections + [claim, claim])
    llm.supports_strict_json_schema = True
    checkpoints, progress = [], []

    assert check_capability_gap(ProgressiveCapabilityGap(requirement=GAP), context=context).verification == "unverifiable"
    with pytest.raises(ProgressivePlanCompileError) as caught:
        ProgressivePlannerAgent(llm).run(
            context, checkpoint_callback=checkpoints.append,
            progress_callback=progress.append, stop_after_outline=True,
        )

    assert caught.value.reason_code == "progressive_capability_gap"
    assert caught.value.easyicu_safe_diagnostic["cause_code"] == GAP
    assert caught.value.details["findings"][0]["verification"] == "unverifiable"
    assert "question_required" not in caught.value.easyicu_safe_diagnostic
    assert "question_required" not in caught.value.details
    assert "whose necessity the host cannot verify" in str(caught.value)
    assert len(llm.calls) == min(earlier_rejections + 2, 4)
    confirmations = [event for event in progress if event.reason_code == "progressive_capability_gap_confirmation_required"]
    assert len(confirmations) == (1 if earlier_rejections < 3 else 0)
    assert checkpoints[0].prompt_metrics.get("capability_gap_confirmation_count", 0) == len(confirmations)
    if confirmations:
        feedback = llm.calls[earlier_rejections + 1][0][-1].content
        for phrase in (
            "cannot establish", "outer window", "allowed aggregation", "not patient data",
            "eligibility", "exposure alignment", "time origin", "keep an appropriate capability gap",
            '"stay_level_representation":null', '"fixed_window_association":null',
        ):
            assert phrase in feedback


def test_confirmation_does_not_hide_a_different_unsupported_design():
    context = _unbound_window_study()
    revised = {**_claim(context), "capability_gap": "design_element_unsupported",
               "rationale": "The proposed time alignment cannot be established from the declared source facts."}
    llm = ScriptedMockLLMClient([json.dumps(_claim(context)), json.dumps(revised)])
    llm.supports_strict_json_schema = True
    with pytest.raises(ProgressivePlanCompileError) as caught:
        ProgressivePlannerAgent(llm).run(context, stop_after_outline=True)
    assert len(llm.calls) == 2
    assert caught.value.reason_code == "progressive_capability_gap"
    assert caught.value.easyicu_safe_diagnostic["cause_code"] == "design_element_unsupported"
    assert caught.value.details["findings"][0]["verification"] == "unverifiable"


def test_the_planner_can_revise_an_unverifiable_gap_to_a_valid_outline():
    context = _context()
    outline = _outline_payload()
    claim = {**outline, "steps": [], "capability_gap": GAP,
             "rationale": "The proposed association requires an unavailable longitudinal representation."}
    assert check_capability_gap(ProgressiveCapabilityGap(requirement=GAP), context=context).verification == "unverifiable"
    llm = ScriptedMockLLMClient([json.dumps(claim), json.dumps(outline)])
    llm.supports_strict_json_schema = True
    checkpoints = []

    corrected = ProgressivePlannerAgent(llm).run(
        context, checkpoint_callback=checkpoints.append, stop_after_outline=True,
    )

    assert len(llm.calls) == 2
    assert corrected.capability_gap is None
    assert corrected.steps
    assert checkpoints[0].prompt_metrics["capability_gap_confirmation_count"] == 1


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
