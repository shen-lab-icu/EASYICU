"""A survival question whose study declares no survival design gets a reviewable proposal.

The landmark survival template used to route only when the host had already
sealed the suite, and the Progressive v2 compiler cannot write a survival
result contract.  A survival study without a declared design therefore had no
reviewable candidate at all, so review could never compile its design into
the study configuration.  The host now proposes the suite it could seal --
every coordinate from a host vocabulary -- and the Planner selects only the
adjustment roster.  Synthetic contexts only (renal replacement therapy and
90-day mortality); zero patient rows.
"""

from __future__ import annotations

import json

import pytest

from easyicu.research_agent.agents.family_spec_planner import (
    FAMILY_SPEC_GUIDE,
    family_spec_user_prompt,
)
from easyicu.research_agent.agents.progressive_planner import candidate_analysis_types
from easyicu.research_agent.planning.family_spec import (
    LANDMARK_SURVIVAL_FAMILY_ID,
    FamilySpecError,
    family_template_id_for_context,
    validate_family_plan_spec,
)
from easyicu.research_agent.planning.family_spec.contract import (
    SealedSuiteCoordinates,
    spec_from_mapping,
)
from easyicu.research_agent.planning.family_spec.request import (
    SEALED_SURVIVAL_SUITE_MARKER,
    proposed_survival_suite_coordinates,
)
from easyicu.research_agent.planning.primary_result_contract import (
    validate_required_primary_result,
)
from easyicu.research_agent.schema import ResearchContext, TimeWindow, UserPreferences
from tests.support.survival_proposal import (
    AGE,
    SEX,
    proposed_survival_plan as _plan,
    survival_context as _context,
    survival_request as _request,
    survival_spec as _spec,
)


def test_closed_host_coordinates_propose_the_suite_with_an_open_roster():
    context = _context()
    types = candidate_analysis_types(context)
    assert types[0] == "survival"
    assert family_template_id_for_context(context, analysis_types=types) == LANDMARK_SURVIVAL_FAMILY_ID

    proposed = proposed_survival_suite_coordinates(context)
    assert proposed is not None
    assert proposed.primary_owner == "signed_landmark_survival_suite"
    assert (proposed.exposure_status_column, proposed.exposure_onset_column) == ("rrt", "rrt_first_time")
    assert (proposed.event_column, proposed.followup_time_column) == ("mort_90d", "followup_days_90d")
    assert (proposed.landmark_hours, proposed.endpoint_horizon_days) == (24.0, 90.0)
    assert proposed.adjustment_columns == []

    request = _request(context)
    assert request.sealed_suite is None and request.proposed_suite == proposed
    assert request.adjustment_selection == "planner_selectable"
    selectable = {item.name: item.host_temporal_role for item in request.adjustment_candidates if item.selectable}
    assert selectable == {"age": "baseline_static", "sex": "baseline_static"}
    assert request.required_reader_label_keys == ["rrt", "mort_90d", "followup_days_90d"]


def test_the_planner_selects_the_roster_and_the_plan_names_the_suite_owner():
    context = _context()
    plan, llm = _plan(context, _spec(_request(context), [AGE, SEX]))

    assert len(llm.calls) == 1
    assert plan.analysis_type == "survival"
    primary = next(step for step in plan.steps if step.planned_analysis_role == "primary")
    assert primary.method == "signed_landmark_survival_suite"
    # The onset companion is materialized only once the design is declared.
    assert "rrt_first_time" not in primary.inputs
    assert {"rrt", "mort_90d", "followup_days_90d", "age", "sex"} <= set(primary.inputs)
    assert "rrt_first_time" not in plan.design_selection.selected.required_variables
    proposal = plan.adjustment_proposal
    assert proposal is not None
    assert proposal.covariates == ["age", "sex"]
    assert proposal.covariate_rationales == {"age": AGE["clinical_rationale"], "sex": SEX["clinical_rationale"]}
    assert proposal.covariate_temporal_roles == {"age": "baseline_static", "sex": "baseline_static"}
    # Final acceptance names the owner; execution fails closed until it is sealed.
    validate_required_primary_result(plan=plan, context=context)


def test_an_exact_user_roster_is_kept_with_its_own_decisions():
    preferences = UserPreferences(
        inferred_analysis_family="survival", covariate_selection="exact", covariates=["age"],
        covariate_authority="user",
        covariate_rationales={"age": "User-reviewed: age confounds renal support and death."},
        covariate_temporal_roles={"age": "baseline_static"},
    )
    context = _context(user_preferences=preferences)
    request = _request(context)
    assert request.adjustment_selection == "exact" and request.exact_roster == ["age"]
    plan, _llm = _plan(context, _spec(request, [AGE]))
    assert plan.adjustment_proposal.covariates == ["age"]
    assert plan.adjustment_proposal.covariate_rationales == {
        "age": "User-reviewed: age confounds renal support and death."
    }


def _without(context: ResearchContext, name: str) -> ResearchContext:
    return context.model_copy(update={"variables": [item for item in context.variables if item.name != name]})


@pytest.mark.parametrize(
    "case",
    ["unpaired_endpoint", "no_followup", "three_level_exposure", "no_host_window", "landmark_past_horizon", "sealed"],
)
def test_coordinates_the_host_cannot_close_propose_nothing(case):
    context = _context()
    disclosure = ""
    if case == "unpaired_endpoint":
        # In-hospital death has no fixed horizon or paired follow-up concept.
        variables = [
            item.model_copy(update={"name": "death"}) if item.name == "mort_90d" else item
            for item in context.variables
        ]
        context = context.model_copy(update={"variables": variables, "target_outcome": "death"})
    elif case == "no_followup":
        context = _without(context, "followup_days_90d")
    elif case == "three_level_exposure":
        variables = [
            item.model_copy(update={"observed_domain": {"n_unique": 3, "levels": [0, 1, 2]}})
            if item.name == "rrt" else item
            for item in context.variables
        ]
        context = context.model_copy(update={"variables": variables})
    elif case == "no_host_window":
        context = context.model_copy(update={"time_windows": []})
    elif case == "landmark_past_horizon":
        context = context.model_copy(update={"time_windows": [
            TimeWindow(name="icu_admission_0_2400h", anchor="icu_admission", start_hours=0.0, end_hours=2400.0)
        ]})
    else:
        disclosure = SEALED_SURVIVAL_SUITE_MARKER + "\n" + json.dumps({
            "sealed_primary_owner": "signed_landmark_survival_suite",
            "exposure_status_column": "another_exposure", "exposure_onset_column": "another_exposure_first_time",
            "event_column": "mort_90d", "followup_time_column": "followup_days_90d",
            "landmark_hours": 24, "endpoint_horizon_days": 90, "plan_outputs": ["table:x"],
        })

    assert proposed_survival_suite_coordinates(context, planning_contract_context=disclosure) is None
    assert family_template_id_for_context(
        context, analysis_types=candidate_analysis_types(context), planning_contract_context=disclosure
    ) is None


def test_the_spec_selects_only_host_timed_candidates_and_labels_them():
    context = _context()
    request = _request(context)
    with pytest.raises(FamilySpecError) as caught:
        validate_family_plan_spec(
            spec_from_mapping(_spec(request, [{**AGE, "name": "followup_days_90d"}])), request
        )
    assert caught.value.reason_code == "family_spec_covariate_unavailable"
    unlabeled = _spec(request, [AGE])
    unlabeled["reader_display_labels"] = [
        item for item in unlabeled["reader_display_labels"] if item["key"] != "age"
    ]
    with pytest.raises(FamilySpecError) as caught:
        validate_family_plan_spec(spec_from_mapping(unlabeled), request)
    assert caught.value.reason_code == "family_spec_reader_label_missing"
    described = _spec(request, [AGE])
    described["baseline_variables"] = ["sex"]
    with pytest.raises(FamilySpecError) as caught:
        validate_family_plan_spec(spec_from_mapping(described), request)
    assert caught.value.reason_code == "family_spec_baseline_variables_not_applicable"


def test_a_request_is_sealed_or_proposed_and_a_proposal_seals_no_roster():
    request = _request(_context())
    proposed = request.proposed_suite
    with pytest.raises(ValueError, match="either sealed or proposed"):
        type(request).model_validate(
            {**request.model_dump(mode="json"), "sealed_suite": proposed.model_dump(mode="json")}
        )
    with pytest.raises(ValueError, match="seals no adjustment roster"):
        type(request).model_validate({
            **request.model_dump(mode="json"),
            "proposed_suite": {**proposed.model_dump(mode="json"), "adjustment_columns": ["age"]},
        })
    assert isinstance(proposed, SealedSuiteCoordinates)


def test_only_a_proposal_adds_its_coordinates_to_the_planner_prompt():
    context = _context()
    request = _request(context)
    prompt = family_spec_user_prompt(request, variable_descriptions={})
    assert '"proposed_suite"' in prompt and "rrt_first_time" in prompt
    assert "proposed_suite" in FAMILY_SPEC_GUIDE
    sealed_shape = request.model_copy(update={"proposed_suite": None})
    assert '"proposed_suite"' not in family_spec_user_prompt(sealed_shape, variable_descriptions={})
