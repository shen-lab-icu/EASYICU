"""A Planner that cannot express what a question needs says so, and the host checks it.

When a design element the question needs fits no offered family, the outline
declares a capability gap and drafts no steps, instead of steps that quietly
answer another question.  The host checks the claim against the study's typed
context first: a claim the context contradicts goes back to the Planner as a
repair; a verified claim, or one the host has no evidence about, stops planning
with the requirement as its cause and records the stopped outline checkpoint.
"""

from __future__ import annotations

import json
from typing import get_args

import pandas as pd
import pytest

from easyicu.research_agent.agents.progressive_payload import (
    progressive_outline_structured_output_request,
)
from easyicu.research_agent.agents.progressive_planner import ProgressivePlannerAgent
from easyicu.research_agent.canonical_json import canonical_sha256
from easyicu.research_agent.planning.capability_gap import (
    capability_gap_confirmation_facts, check_capability_gap,
)
from easyicu.research_agent.planning.family_spec.request import (
    SEALED_TRAJECTORY_SUITE_MARKER,
)
from easyicu.research_agent.planning.progressive_contract import (
    CapabilityGapRequirement,
    ProgressiveCapabilityGap,
    ProgressivePlanCompileError,
    ProgressivePlanOutline,
)
from easyicu.research_agent.providers.mocks import ScriptedMockLLMClient
from easyicu.research_agent.providers.structured_retry import StructuredResponseFailure
from easyicu.research_agent.schema import (
    AggregationRule, ConceptDescriptor, EndpointSpec, TemporalConstraint, TimeWindow,
    UserPreferences, VariableRole,
)
from easyicu.research_agent.trajectory.contract import (
    infer_fixed_window_trajectory_metadata,
)
from tests.research_agent.planning.progressive_planner_fixtures import (
    _context,
    _foundation_payload,
    _materialization_payloads,
    _outline_payload,
    _payload,
)


def _claim(requirement: str, concept: str | None = None) -> ProgressiveCapabilityGap:
    return ProgressiveCapabilityGap(requirement=requirement, concept=concept)


def _gap_outline(requirement: str, concept: str | None = None) -> dict:
    return {
        **_outline_payload(),
        "steps": [],
        "capability_gap": requirement,
        "capability_gap_concept": concept,
        "rationale": "The question groups the exposure by clinical thresholds.",
    }


def _run(responses: list[dict], checkpoints: list | None = None):
    llm = ScriptedMockLLMClient([json.dumps(item) for item in responses])
    llm.supports_strict_json_schema = True
    agent = ProgressivePlannerAgent(llm)
    return llm, agent.run(
        _context(),
        checkpoint_callback=(checkpoints.append if checkpoints is not None else None),
    )


def _typed_association():
    """Synthetic maximum, outer window, endpoint and stated adjustment receipts."""
    context = _context()
    exposure = context.variables[0].model_copy(update={
        "role": VariableRole.LAB, "dtype": "float64", "observed_domain": None,
        "source_concept": "synthetic_signal", "aggregation_default": AggregationRule.NONE,
        "allowed_aggregations": [AggregationRule.NONE], "unit_normalization": "window_numeric_max",
    })
    return context.model_copy(update={
        "research_question": (
            "With recorded sex as a covariate, is the first ICU day's maximum "
            "synthetic signal associated with in-hospital death?"
        ),
        "variables": [exposure, *context.variables[1:]],
        "time_windows": [TimeWindow(name="outer_window", anchor="icu_admission", start_hours=0, end_hours=24)],
        "temporal_constraints": [TemporalConstraint(
            raw_text="first ICU day's maximum synthetic signal", relation="first_window",
            target_concept="synthetic_signal", aggregation_hint="max",
            start_hours=0, end_hours=24, executable_repr="synthetic_signal:max:icu_admission:0:24",
        )],
        "endpoint": EndpointSpec(name="outcome_flag", kind="binary", levels=[0, 1], absence_semantics="no_absent_rows"),
        "user_preferences": UserPreferences(
            inferred_analysis_family="association_study",
            data_constraints=json.dumps({"stated_adjustments": [{
                "variable": "recorded sex", "words": "recorded sex as a covariate",
                "concept": "sex_code", "source_status": "available",
            }]}),
        ),
    })


def _typed_association_responses():
    source = _payload()
    source["steps"][1].update(
        table_one_group_by="outcome_flag", raw_inputs=["outcome_flag", "age_years", "sex_code"],
    )
    source["steps"][2].update(
        module_id="absolute_risk_context", reference_exposure_level_index=None,
        comparison_exposure_level_index=None, event_level_index=None, denominator_policy=None,
        missing_exposure_policy=None, missing_outcome_policy=None, confidence_level=None,
        population_scope="analysis_cohort",
    )
    source["steps"][4]["model_terms"][0].update(coding="continuous", reference_level_index=None)
    source["steps"][6]["product_inputs"][0]["product_id"] = "table:absolute_risk_context"
    source["display_labels"] = [item for item in source["display_labels"] if "=" not in item["key"]]
    outline = _outline_payload(source)
    outline["steps"][2]["population_scope"] = "analysis_cohort"
    parsed = ProgressivePlanOutline.model_validate(outline)
    foundation = _foundation_payload(source)
    foundation["outline_sha256"] = canonical_sha256(parsed.model_dump(mode="json"))
    materializations = _materialization_payloads(source)
    for step, materialization in zip(parsed.steps, materializations, strict=True):
        materialization["outline_step_sha256"] = canonical_sha256(step.model_dump(mode="json"))
    return [outline, foundation, *materializations]


def test_the_synthetic_typed_association_compiles_without_a_gap():
    llm = ScriptedMockLLMClient([json.dumps(item) for item in _typed_association_responses()])
    llm.supports_strict_json_schema = True
    assert ProgressivePlannerAgent(llm).run(_typed_association()).steps


def test_an_unnamed_design_gap_gets_one_host_facts_confirmation_then_compiles():
    """HEAD refuses on call one even though all the stated elements are typed."""
    responses = [_gap_outline("design_element_unsupported"), *_typed_association_responses()]
    llm = ScriptedMockLLMClient([json.dumps(item) for item in responses])
    llm.supports_strict_json_schema = True
    checkpoints = []

    plan = ProgressivePlannerAgent(llm).run(_typed_association(), checkpoint_callback=checkpoints.append)

    assert plan.steps
    assert len(llm.calls) == len(responses)
    assert all("capability_gap" not in checkpoint.prompt_metrics for checkpoint in checkpoints)
    primary = next(step for step in plan.steps if step.planned_analysis_role == "primary")
    assert "sex_code" in primary.model_requirements[0].covariates
    confirmation = llm.calls[1][0][-1].content
    for fact in ("association_study", "exposure_flag", "max", "icu_admission", "24",
                 "outcome_flag", "binary", "sex_code", "available", "offered_families"):
        assert fact in confirmation
    assert "otherwise keep the gap" in confirmation
    assert "what the question asks" in confirmation


@pytest.mark.parametrize("needed", ["mediation", "instrumental-variable identification", "competing risks"])
def test_an_unexpressible_design_kept_after_confirmation_still_stops(needed):
    outline = _gap_outline("design_element_unsupported")
    outline["rationale"] = f"The question requires {needed}, which none of the offered families expresses."
    llm = ScriptedMockLLMClient([json.dumps(outline)] * 2)
    llm.supports_strict_json_schema = True
    context = _typed_association().model_copy(update={
        "research_question": f"Estimate an effect using {needed} for the synthetic exposure and outcome.",
    })
    with pytest.raises(ProgressivePlanCompileError) as caught:
        ProgressivePlannerAgent(llm).run(context)

    assert len(llm.calls) == 2
    assert caught.value.reason_code == "progressive_capability_gap"
    assert caught.value.path == "capability_gap"
    assert caught.value.easyicu_safe_diagnostic["cause_code"] == "design_element_unsupported"
    assert caught.value.details["findings"][0]["verification"] == "unverifiable"


@pytest.mark.parametrize("earlier_rejections", [0, 1, 2, 3])
def test_confirmation_uses_only_the_remaining_outline_retry_budget(earlier_rejections):
    claim = json.dumps(_gap_outline("design_element_unsupported"))
    llm = ScriptedMockLLMClient(["{}"] * earlier_rejections + [claim, claim])
    llm.supports_strict_json_schema = True
    progress = []
    checkpoints = []
    with pytest.raises(ProgressivePlanCompileError) as caught:
        ProgressivePlannerAgent(llm).run(
            _typed_association(), progress_callback=progress.append, checkpoint_callback=checkpoints.append,
        )

    assert caught.value.reason_code == "progressive_capability_gap"
    assert len(llm.calls) == min(earlier_rejections + 2, 4)
    confirmations = [event for event in progress if event.reason_code == "progressive_capability_gap_confirmation_required"]
    assert len(confirmations) == (1 if earlier_rejections < 3 else 0)
    assert checkpoints[0].prompt_metrics.get("capability_gap_confirmation_count", 0) == len(confirmations)


def test_a_named_unverifiable_gap_is_not_an_unnamed_design_confirmation():
    context = _typed_association()
    context = context.model_copy(update={"user_preferences": context.user_preferences.model_copy(update={
        "data_constraints": json.dumps({"stated_subgroups": [{
            "variable": "sex", "words": "within each sex", "concept": "sex_code", "source_status": "unbound",
        }]}),
    })})
    llm = ScriptedMockLLMClient([json.dumps(_gap_outline("design_element_unsupported", "sex_code"))])
    llm.supports_strict_json_schema = True
    with pytest.raises(ProgressivePlanCompileError) as caught:
        ProgressivePlannerAgent(llm).run(context)
    assert len(llm.calls) == 1
    assert caught.value.details["findings"][0]["verification"] == "unverifiable"


def test_confirmation_projects_metadata_and_outer_windows_without_patient_summaries():
    context = _typed_association()
    context = context.model_copy(update={
        "notes": "PRIVATE_NOTES_SENTINEL", "cohort_parquet": "PRIVATE_PATH_SENTINEL",
        "variables": [context.variables[0].model_copy(update={
            "observed_domain": {"levels": ["PRIVATE_OBSERVATION_SENTINEL"]},
            "description": "PRIVATE_DESCRIPTION_SENTINEL", "source_files": ["PRIVATE_SOURCE_SENTINEL"],
        }), *context.variables[1:]],
    })
    text = capability_gap_confirmation_facts(context, offered_families=("association_study", "descriptive_epidemiology"))
    facts = json.loads(text.split("\n", 1)[1])
    assert "PRIVATE_" not in text
    assert facts["family"] == "association_study"
    assert facts["offered_families"] == ["association_study", "descriptive_epidemiology"]
    assert facts["exposure"]["column"] == "exposure_flag"
    assert facts["outcome"]["column"] == "outcome_flag"
    assert facts["exposure"]["metadata"]["analysis_window"] is None
    assert facts["exposure"]["metadata"]["unit_normalization"] == "window_numeric_max"
    assert facts["exposure"]["metadata"]["aggregation_default"] == "none"
    assert facts["time_windows"] == [{"name": "outer_window", "anchor": "icu_admission", "start_hours": 0.0, "end_hours": 24.0}]
    assert facts["temporal_constraints"][0]["aggregation_hint"] == "max"
    assert facts["endpoint"]["kind"] == "binary"
    assert facts["stated_adjustments"][0]["source_status"] == "available"
    assert facts["stated_subgroups"] == []


def test_long_stated_requirements_are_not_clipped_out_of_confirmation():
    context = _typed_association()
    requirements = [{"variable": f"stratum_{index}", "words": "synthetic stratum " + "x" * 200,
                     "concept": None, "source_status": "unresolved"} for index in range(16)]
    context = context.model_copy(update={"user_preferences": context.user_preferences.model_copy(update={
        "data_constraints": json.dumps({"stated_subgroups": requirements}),
    })})
    claim = json.dumps(_gap_outline("design_element_unsupported"))
    llm = ScriptedMockLLMClient([claim, claim])
    llm.supports_strict_json_schema = True
    with pytest.raises(ProgressivePlanCompileError):
        ProgressivePlannerAgent(llm).run(context)
    confirmation = llm.calls[1][0][-1].content
    facts = json.loads(confirmation.split("Host facts for capability-gap confirmation:\n", 1)[1])
    assert facts["stated_subgroups"] == requirements


def test_an_outline_drafts_steps_or_declares_a_gap_never_both() -> None:
    outline = _outline_payload()

    with pytest.raises(ValueError, match=r"set capability_gap=null.*steps=\[\]"):
        ProgressivePlanOutline.model_validate(
            {**outline, "capability_gap": "estimand_unsupported"}
        )
    with pytest.raises(ValueError, match="provide non-empty steps"):
        ProgressivePlanOutline.model_validate({**outline, "steps": []})
    with pytest.raises(ValueError, match="otherwise set capability_gap_concept=null"):
        ProgressivePlanOutline.model_validate(
            {**outline, "capability_gap_concept": "age_years"}
        )
    # A thresholds gap names the variable to group; a design-element gap may
    # name the stated adjustment it concerns; no other gap names one.
    for requirement, concept in (
        ("estimand_unsupported", "age_years"),
        ("levels_from_thresholds_unavailable", None),
    ):
        with pytest.raises(ValueError, match="set capability_gap_concept to the non-empty variable name"):
            ProgressivePlanOutline.model_validate(_gap_outline(requirement, concept))
        with pytest.raises(ValueError, match="no other requirement names one"):
            _claim(requirement, concept)


def test_an_outline_without_a_gap_keeps_its_digest() -> None:
    dumped = ProgressivePlanOutline.model_validate(_outline_payload()).model_dump(
        mode="json"
    )

    assert "capability_gap" not in dumped
    assert "capability_gap_concept" not in dumped


@pytest.mark.parametrize(
    ("requirement", "concept", "context_update", "verification"),
    [
        # A continuous variable with no closed levels cannot be compared by group.
        ("levels_from_thresholds_unavailable", "age_years", {}, "verified"),
        # A binary exposure already has two levels to compare.
        ("levels_from_thresholds_unavailable", "exposure_flag", {}, "unverified"),
        ("levels_from_thresholds_unavailable", "glucose_max", {}, "unverified"),
        ("multiple_sources_required", None, {}, "verified"),
        (
            "multiple_sources_required",
            None,
            {"cross_database_validation": ["eicu"]},
            "unverified",
        ),
        ("longitudinal_representation_unavailable", None, {}, "verified"),
        ("estimand_unsupported", None, {}, "unverifiable"),
        ("design_element_unsupported", None, {}, "unverifiable"),
    ],
)
def test_the_host_checks_each_claim_against_the_study(
    requirement, concept, context_update, verification
) -> None:
    context = _context().model_copy(update=context_update)

    check = check_capability_gap(_claim(requirement, concept), context=context)

    assert check.verification == verification
    assert check.fact


def test_a_longitudinal_claim_does_not_hold_for_a_study_with_a_trajectory() -> None:
    gap = _claim("longitudinal_representation_unavailable")
    context = _context()
    window = ConceptDescriptor(
        name="severity_state_h0_6",
        role=VariableRole.ORDINAL_SCORE,
        dtype="int64",
        is_ordinal=True,
        fixed_window_trajectory=infer_fixed_window_trajectory_metadata(
            column_name="severity_state_h0_6",
            values=pd.Series([0, 1, 2], dtype="int64"),
            source_scale="ordinal",
        ),
    )
    with_windows = context.model_copy(
        update={"variables": [*context.variables, window]}
    )
    sealed_suite = SEALED_TRAJECTORY_SUITE_MARKER + json.dumps(
        {
            "sealed_representation_owner": "signed_trajectory_representation",
            "sealed_candidate_owner": "signed_trajectory_candidates",
            "coordinate_concepts": ["sofa_cardio", "sofa_resp"],
            "window_hours": [0, 72],
            "grid_width_hours": 24,
            "candidate_cluster_counts": [2, 3],
            "representation_outputs": ["trajectory_classes"],
        }
    )

    assert check_capability_gap(gap, context=with_windows).verification == "unverified"
    assert (
        check_capability_gap(
            gap, context=context, planning_contract_context=sealed_suite
        ).verification
        == "unverified"
    )


def test_a_checked_gap_stops_planning_with_its_cause_and_outline_checkpoint() -> None:
    checkpoints: list = []

    with pytest.raises(ProgressivePlanCompileError) as caught:
        _run(
            [_gap_outline("levels_from_thresholds_unavailable", "age_years")],
            checkpoints,
        )

    stop = caught.value
    assert stop.reason_code == "progressive_capability_gap"
    assert (
        stop.easyicu_safe_diagnostic["cause_code"]
        == "levels_from_thresholds_unavailable"
    )
    assert stop.path == "capability_gap"
    assert stop.easyicu_safe_diagnostic["element"] == "levels_from_thresholds_unavailable"
    assert stop.easyicu_safe_diagnostic["question_required"] is True
    (finding,) = stop.details["findings"]
    assert (finding["requirement"], finding["concept"], finding["verification"]) == (
        "levels_from_thresholds_unavailable",
        "age_years",
        "verified",
    )
    assert len(checkpoints) == 1
    assert checkpoints[0].stage == "outline"
    assert checkpoints[0].prompt_metrics["capability_gap"]["verification"] == "verified"


def test_a_gap_the_host_cannot_check_stops_as_unverifiable() -> None:
    with pytest.raises(ProgressivePlanCompileError) as caught:
        _run([_gap_outline("estimand_unsupported")])

    assert caught.value.details["findings"][0]["verification"] == "unverifiable"


def test_a_claim_the_study_contradicts_goes_back_to_the_planner() -> None:
    llm, plan = _run(
        [
            _gap_outline("levels_from_thresholds_unavailable", "exposure_flag"),
            _outline_payload(),
            _foundation_payload(),
            *_materialization_payloads(),
        ]
    )

    assert plan.steps
    repair = llm.calls[1][0][-1].content
    assert "progressive_capability_gap_claim_unverified" in repair or (
        "the declared capability gap does not hold" in repair
    )


def test_a_contradicted_claim_on_every_attempt_exhausts_the_repairs() -> None:
    claim = _gap_outline("levels_from_thresholds_unavailable", "exposure_flag")

    with pytest.raises(StructuredResponseFailure) as caught:
        _run([claim] * 4)

    assert getattr(caught.value.__cause__, "reason_code", None) == (
        "progressive_capability_gap_claim_unverified"
    )


def test_the_strict_transport_offers_two_small_nullable_gap_fields() -> None:
    request = progressive_outline_structured_output_request(
        analysis_types=("association_study", "descriptive_epidemiology"),
        variable_names=("age_years", "outcome_flag"),
        scientific_action_ids=(),
    )
    schema = json.loads(request.schema_json)
    gap = schema["properties"]["capability_gap"]["anyOf"]
    concept = schema["properties"]["capability_gap_concept"]["anyOf"]

    assert {"capability_gap", "capability_gap_concept"} <= set(schema["required"])
    assert {"type": "null"} in gap
    assert {"type": "null"} in concept
    assert [item["enum"] for item in gap if "enum" in item] == [
        list(get_args(CapabilityGapRequirement))
    ]
    # The host checks the concept against the study, so the transport repeats
    # no variable list for it.
    assert all("enum" not in item for item in concept)
    assert "ProgressiveCapabilityGap" not in schema.get("$defs", {})
    assert "minItems" not in schema["properties"]["steps"]


def test_the_outline_shape_lists_the_gap_and_when_to_use_it() -> None:
    prompt = ProgressivePlannerAgent.request_messages(_context())[1].content

    assert '"capability_gap":null,"capability_gap_concept":null' in prompt
    assert (
        "capability_gap and capability_gap_concept stay null unless the question "
        "needs a design element" in prompt
    )
    assert (
        "do not change the analysis type, population or exposure to work around it"
        in prompt
    )
