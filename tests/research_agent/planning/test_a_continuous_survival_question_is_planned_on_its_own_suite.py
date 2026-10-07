"""A survival question about a continuous exposure is planned on its own suite.

The family-spec planner templated survival only on the binary landmark suite,
so a question about a laboratory value or a vital sign had no template and the
Progressive v2 compiler cannot write a survival result contract: the planner
stopped before any Provider call.  The host now proposes the continuous
suite it could seal -- the exposure's window summary from ICU admission to the
landmark, modelled per unit -- and, once sealed, plans it like the binary
suite: the Planner labels the columns and its roster is the authority's.
Synthetic contexts only; zero patient rows.
"""

from __future__ import annotations

import json
from types import SimpleNamespace

import pytest

from easyicu.research_agent import pipeline as _pipeline
from easyicu.research_agent.agents.family_spec_planner import (
    FAMILY_SPEC_GUIDE,
    family_spec_user_prompt,
)
from easyicu.research_agent.agents.progressive_planner import (
    FAMILY_SPEC_STRATEGY,
    ProgressivePlannerAgent,
    candidate_analysis_types,
    select_progressive_variables,
)
from easyicu.research_agent.authority.current_case_scientific_runtime import (
    build_current_case_scientific_runtime_authority,
)
from easyicu.research_agent.execution.runners.selection import select_standard_executor
from easyicu.research_agent.orchestration.scientific_runtime import (
    ScientificRuntimeAuthorities,
)
from easyicu.research_agent.planning import figure_plan_shaping as _figure_plan
from easyicu.research_agent.planning import final_plan_shape as _final_plan
from easyicu.research_agent.planning.dependence_authority import (
    bind_context_dependence_authority,
)
from easyicu.research_agent.planning.family_spec import (
    LANDMARK_CONTINUOUS_SURVIVAL_FAMILY_ID,
    FamilySpecError,
    build_family_spec_request,
    family_template_id_for_context,
    sealed_continuous_survival_suite_coordinates,
    validate_family_plan_spec,
)
from easyicu.research_agent.planning.family_spec.contract import (
    SealedContinuousSuiteCoordinates,
    spec_from_mapping,
)
from easyicu.research_agent.planning.family_spec.request import (
    SEALED_CONTINUOUS_SURVIVAL_SUITE_MARKER,
    SEALED_SURVIVAL_SUITE_MARKER,
    _require_sealed_table_one,
    proposed_continuous_survival_suite_coordinates,
    proposed_survival_suite_coordinates,
)
from easyicu.research_agent.planning.figure_strategy import build_article_figure_strategy
from easyicu.research_agent.planning.primary_result_contract import (
    validate_required_primary_result,
)
from easyicu.research_agent.planning.scientific_review import build_plan_scientific_review
from easyicu.research_agent.providers.mocks import ScriptedMockLLMClient
from easyicu.research_agent.schema import (
    AnalysisPlan,
    CohortDescriptor,
    ConceptDescriptor,
    ResearchContext,
    TimeWindow,
    UserPreferences,
    VariableRole,
)
from tests.support.continuous_survival import (
    AGE,
    ALLOWED,
    BINARY,
    SEX,
    continuous_authority_body,
    continuous_survival_context as _context,
    continuous_survival_request as _request,
    continuous_survival_spec as _spec,
    proposed_continuous_survival_plan as _plan,
)
from tests.support.survival_proposal import survival_context, survival_request


def _primary(plan):
    return next(step for step in plan.steps if step.planned_analysis_role == "primary")


def test_a_continuous_exposure_gets_a_proposed_continuous_suite():
    context = _context()
    types = candidate_analysis_types(context)
    assert types[0] == "survival"
    # The binary suite has nothing to contrast; the continuous suite is proposed.
    assert proposed_survival_suite_coordinates(context) is None
    assert (
        family_template_id_for_context(context, analysis_types=types)
        == LANDMARK_CONTINUOUS_SURVIVAL_FAMILY_ID
    )

    proposed = proposed_continuous_survival_suite_coordinates(context)
    assert proposed is not None
    assert proposed.primary_owner == "signed_landmark_continuous_survival_suite"
    assert (proposed.exposure_column, proposed.exposure_window_summary) == ("bili_max", "max")
    assert proposed.exposure_unit == "mg/dL"
    assert (proposed.event_column, proposed.followup_time_column) == ("mort_90d", "followup_days_90d")
    assert (proposed.landmark_hours, proposed.endpoint_horizon_days) == (24.0, 90.0)
    assert proposed.adjustment_columns == []
    assert proposed.source_columns == ("bili_max", "mort_90d", "followup_days_90d")

    request = _request(context)
    assert request.family_id == LANDMARK_CONTINUOUS_SURVIVAL_FAMILY_ID
    assert request.proposed_continuous_suite == proposed
    assert request.sealed_continuous_suite is None and request.proposed_suite is None
    assert request.exposure_kind == "continuous" and request.exposure_levels == []
    assert request.adjustment_selection == "planner_selectable"
    selectable = {
        item.name: item.host_temporal_role
        for item in request.adjustment_candidates
        if item.selectable
    }
    # Another summary of the exposure's own measurement is never a confounder.
    assert selectable == {"age": "baseline_static", "sex": "baseline_static"}
    assert "bili_mean" not in {item.name for item in request.adjustment_candidates}
    assert request.required_reader_label_keys == ["bili_max", "mort_90d", "followup_days_90d"]
    assert request.level_label_keys == []
    assert request.cohort_time_zero_hours == 24.0


def test_the_planner_selects_the_roster_and_the_plan_names_the_continuous_owner():
    context = _context()
    plan, llm = _plan(context, _spec(_request(context), [AGE, SEX]))

    assert len(llm.calls) == 1
    assert plan.analysis_type == "survival"
    primary = _primary(plan)
    assert primary.method == "signed_landmark_continuous_survival_suite"
    assert {"bili_max", "mort_90d", "followup_days_90d", "age", "sex"} <= set(primary.inputs)
    assert "bili_mean" not in primary.inputs
    assert "table:landmark_continuous_cox_summary" in primary.expected_outputs
    selected = plan.design_selection.selected
    assert selected.design_id == "fixed_landmark_continuous_survival_suite"
    assert "per 1 mg/dL increase in" in selected.estimand
    assert "(its highest value from ICU admission to the 24 h landmark)" in selected.estimand
    rejected = next(
        item for item in plan.design_selection.candidates if item.disposition == "rejected"
    )
    assert rejected.design_id == "tertile_cox_contrast"
    proposal = plan.adjustment_proposal
    assert proposal is not None
    assert proposal.source_requirement_id == "proposed_landmark_continuous_survival_suite"
    assert proposal.covariates == ["age", "sex"]
    assert proposal.covariate_temporal_roles == {"age": "baseline_static", "sex": "baseline_static"}
    # Final acceptance names the owner; execution fails closed until it is sealed.
    validate_required_primary_result(plan=plan, context=context)


def test_the_review_hands_a_closing_continuous_proposal_to_the_host():
    context = _context()
    plan, _llm = _plan(context, _spec(_request(context), [AGE, SEX]))

    review = build_plan_scientific_review(context=context, plan=plan)

    blockers = {
        item.code: item.remediation_route for item in review.findings if item.severity == "blocker"
    }
    assert blockers["SURVIVAL_LANDMARK_OWNER_NOT_SEALED"] == "runtime_capability"
    facts = review.facts["landmark_survival_suite"]
    assert facts["sealed"] is False and facts["executable"] is True
    assert (facts["exposure"], facts["event_column"], facts["followup_column"]) == (
        "bili_max", "mort_90d", "followup_days_90d",
    )
    assert facts["covariates"] == ["age", "sex"]


def _without(context: ResearchContext, name: str) -> ResearchContext:
    return context.model_copy(update={"variables": [item for item in context.variables if item.name != name]})


def _replace(context: ResearchContext, column: str, **update) -> ResearchContext:
    return context.model_copy(update={"variables": [
        item.model_copy(update=update) if item.name == column else item for item in context.variables
    ]})


@pytest.mark.parametrize(
    "case",
    [
        "not_a_window_summary",
        "recorded_once_per_stay",
        "ordinal_exposure",
        "unpaired_endpoint",
        "no_followup",
        "no_host_window",
        "no_interval_inside_followup",
        "sealed_binary_suite",
    ],
)
def test_coordinates_the_host_cannot_close_propose_no_continuous_suite(case):
    context = _context()
    disclosure = ""
    if case == "not_a_window_summary":
        # A count of the measurements is not a summary of their values.
        context = _replace(context, "bili_max", name="bili_n").model_copy(
            update={"primary_exposure": "bili_n"}
        )
    elif case == "recorded_once_per_stay":
        # Named like a window summary, but its source has one value per stay.
        context = _replace(context, "bili_max", name="age_max", source_concept="age").model_copy(
            update={"primary_exposure": "age_max"}
        )
    elif case == "ordinal_exposure":
        context = _replace(context, "bili_max", is_ordinal=True)
    elif case == "unpaired_endpoint":
        # In-hospital death has no fixed horizon or paired follow-up concept.
        context = _replace(context, "mort_90d", name="death").model_copy(
            update={"target_outcome": "death"}
        )
    elif case == "no_followup":
        context = _without(context, "followup_days_90d")
    elif case == "no_host_window":
        context = context.model_copy(update={"time_windows": []})
    elif case == "no_interval_inside_followup":
        # Follow-up from day 85 to day 90 holds none of the interval cutpoints.
        context = context.model_copy(update={"time_windows": [
            TimeWindow(name="icu_admission_0_2040h", anchor="icu_admission", start_hours=0.0, end_hours=2040.0)
        ]})
    else:
        disclosure = SEALED_SURVIVAL_SUITE_MARKER + "\n" + json.dumps({
            "sealed_primary_owner": "signed_landmark_survival_suite",
            "exposure_status_column": "another_exposure", "exposure_onset_column": "another_exposure_onset_time",
            "event_column": "mort_90d", "followup_time_column": "followup_days_90d",
            "landmark_hours": 24, "endpoint_horizon_days": 90, "plan_outputs": ["table:x"],
        })

    assert proposed_continuous_survival_suite_coordinates(
        context, planning_contract_context=disclosure
    ) is None
    assert family_template_id_for_context(
        context, analysis_types=["survival"], planning_contract_context=disclosure
    ) is None


def test_a_continuous_suite_sealed_for_another_exposure_keeps_planning_open():
    # The router offers no template for a suite sealed on another exposure,
    # but the suite's owner is accepted by its primary method, so the planner
    # is not stopped before its first Provider call.
    disclosure = SEALED_CONTINUOUS_SURVIVAL_SUITE_MARKER + "\n" + json.dumps({
        "sealed_primary_owner": "signed_landmark_continuous_survival_suite",
        "exposure_column": "another_lab_max", "exposure_window_summary": "max",
        "exposure_window_hours": [0.0, 24.0], "exposure_increment": 1.0, "exposure_unit": None,
        "event_column": "mort_90d", "followup_time_column": "followup_days_90d",
        "landmark_hours": 24.0, "endpoint_horizon_days": 90.0,
        "adjustment_columns": [], "plan_outputs": ["table:landmark_continuous_cox_summary"],
    })
    context = _context()
    assert sealed_continuous_survival_suite_coordinates(disclosure) is not None
    assert family_template_id_for_context(
        context, analysis_types=["survival"], planning_contract_context=disclosure
    ) is None
    llm = ScriptedMockLLMClient([])
    stopped = None
    try:
        ProgressivePlannerAgent(llm).run_attempt(
            context, planner_strategy=FAMILY_SPEC_STRATEGY,
            allowed_literature_citation_keys=ALLOWED, direct_comparator_literature_keys=[],
            enforce_article_contract=True, article_contract_context=context,
            planning_contract_context=disclosure, required_primary_cohort_selection_mode="all_input_rows",
        )
    except Exception as exc:  # noqa: BLE001 - the test reads which stop it was
        stopped = exc
    assert getattr(stopped, "reason_code", None) != "progressive_family_result_contract_unwritable"
    assert llm.calls


def test_the_spec_selects_only_host_timed_candidates_and_labels_them():
    request = _request(_context())
    with pytest.raises(FamilySpecError) as caught:
        validate_family_plan_spec(
            spec_from_mapping(_spec(request, [{**AGE, "name": "bili_mean"}])), request
        )
    assert caught.value.reason_code == "family_spec_covariate_unavailable"
    unlabeled = _spec(request, [AGE])
    unlabeled["reader_display_labels"] = [
        item for item in unlabeled["reader_display_labels"] if item["key"] != "age"
    ]
    with pytest.raises(FamilySpecError) as caught:
        validate_family_plan_spec(spec_from_mapping(unlabeled), request)
    assert caught.value.reason_code == "family_spec_reader_label_missing"
    empty = _spec(request, [])
    with pytest.raises(FamilySpecError) as caught:
        validate_family_plan_spec(spec_from_mapping(empty), request)
    assert caught.value.reason_code == "family_spec_adjustment_set_empty"


def test_a_request_is_sealed_or_proposed_and_its_coordinates_belong_to_its_family():
    request = _request(_context())
    proposed = request.proposed_continuous_suite
    payload = request.model_dump(mode="json")
    with pytest.raises(ValueError, match="either sealed or proposed"):
        type(request).model_validate(
            {**payload, "sealed_continuous_suite": proposed.model_dump(mode="json")}
        )
    with pytest.raises(ValueError, match="seals no adjustment roster"):
        type(request).model_validate({
            **payload,
            "proposed_continuous_suite": {**proposed.model_dump(mode="json"), "adjustment_columns": ["age"]},
        })
    with pytest.raises(ValueError, match="models one continuous exposure"):
        type(request).model_validate({
            **payload, "exposure_kind": "categorical", "exposure_levels": ["0", "1"],
            "primary_contrast_level_index": 1,
        })
    binary = survival_request(survival_context()).model_dump(mode="json")
    with pytest.raises(ValueError, match="belong to the continuous survival family"):
        type(request).model_validate(
            {**binary, "proposed_continuous_suite": proposed.model_dump(mode="json")}
        )
    with pytest.raises(ValueError, match="precede its endpoint horizon"):
        SealedContinuousSuiteCoordinates.model_validate(
            {**proposed.model_dump(mode="json"), "landmark_hours": 90.0 * 24}
        )
    # Requests of every other family keep their digest: the new fields are absent.
    assert "sealed_continuous_suite" not in payload
    assert "proposed_suite" not in payload


def test_only_a_continuous_proposal_adds_its_coordinates_to_the_planner_prompt():
    request = _request(_context())
    prompt = family_spec_user_prompt(request, variable_descriptions={})
    assert '"proposed_continuous_suite"' in prompt and '"exposure_window_summary": "max"' in prompt
    assert "proposed_continuous_suite" in FAMILY_SPEC_GUIDE
    assert "sealed_continuous_suite" in FAMILY_SPEC_GUIDE
    sealed_shape = request.model_copy(update={"proposed_continuous_suite": None})
    assert '"proposed_continuous_suite"' not in family_spec_user_prompt(
        sealed_shape, variable_descriptions={}
    )


def test_a_sealed_continuous_suite_keeps_only_an_ungrouped_roster_of_its_columns():
    authority = build_current_case_scientific_runtime_authority(continuous_authority_body())
    sealed = sealed_continuous_survival_suite_coordinates(authority.planning_contract_context())
    request = SimpleNamespace(sealed_survival_suite=sealed)

    def projection(group, *columns):
        return {"tables": [{
            "group_by": {"required": group, "available_columns": [group] if group else []},
            "variables": [{"required": name, "available_columns": [name]} for name in columns],
        }]}

    _require_sealed_table_one(request, projection(None, "age", "sex"))
    with pytest.raises(FamilySpecError) as caught:
        _require_sealed_table_one(request, projection("lab_max", "age"))
    assert caught.value.reason_code == "family_spec_accepted_baseline_grouping_unsupported"
    assert "exposure tertile" in str(caught.value)
    with pytest.raises(FamilySpecError) as caught:
        _require_sealed_table_one(request, projection(None, "age", "bmi"))
    assert caught.value.reason_code == "family_spec_accepted_baseline_unsatisfiable"


def _sealed_context(authority) -> ResearchContext:
    return ResearchContext(
        research_question=(
            "Among adult ICU stays alive at 24 h, is the highest laboratory value in the "
            "first 24 h associated with 28-day mortality?"
        ),
        cohort=CohortDescriptor(
            cohort_name="continuous_synthetic", database="miiv", n_stays=0, id_columns=["stay_id"],
            outcome_columns=["mort_28d"], requested_outcome_columns=["mort_28d"],
            provenance={"analysis_unit": "icu_stay", "patient_identity_available": False,
                        "stay_id_columns": ["stay_id"], "patient_id_columns": []},
        ),
        variables=[
            ConceptDescriptor(name="stay_id", role=VariableRole.ID, dtype="int64"),
            ConceptDescriptor(name="lab_max", description="highest laboratory value by 24 h",
                              role=VariableRole.LAB, dtype="float64", unit="mmol/L"),
            ConceptDescriptor(name="mort_28d", description="Death by day 28 from ICU admission",
                              role=VariableRole.OUTCOME, dtype="float64", observed_domain=BINARY),
            ConceptDescriptor(name="followup_days_28d", description="28-day event or censoring time",
                              role=VariableRole.OUTCOME, dtype="float64", unit="days"),
            ConceptDescriptor(name="age", description="age", role=VariableRole.DEMOGRAPHIC,
                              dtype="float64", unit="years"),
            ConceptDescriptor(name="sex", description="sex", role=VariableRole.DEMOGRAPHIC,
                              dtype="float64", observed_domain=BINARY),
        ],
        target_outcome="mort_28d",
        primary_exposure="lab_max",
        endpoint=authority.research_context_endpoint(),
        user_preferences=UserPreferences(
            inferred_analysis_family="survival", covariate_selection="exact",
            covariates=["age", "sex"],
        ),
    )


def _plan_through_host(context, authorities) -> tuple[AnalysisPlan, ScriptedMockLLMClient]:
    planning_context = authorities.planning_contract_context()
    request = build_family_spec_request(
        context, analysis_types=candidate_analysis_types(context),
        variable_roster=select_progressive_variables(context),
        allowed_literature_citation_keys=ALLOWED,
        required_primary_cohort_selection_mode="all_input_rows",
        planning_contract_context=planning_context,
    )
    labels = {
        "schema_version": "easyicu.family_plan_spec/1",
        "family_id": request.family_id,
        "request_sha256": request.request_sha256,
        "adjustment_set": [],
        "reader_display_labels": [
            {"key": key, "value": f"Reader label for {key}"}
            for key in request.required_reader_label_keys
        ],
        "comparator_applications": [],
        "roster_decision_note": "Sealed suite: the host owns every coordinate; labels only.",
    }
    llm = ScriptedMockLLMClient([json.dumps(labels)])
    plan = ProgressivePlannerAgent(llm).run_attempt(
        context, planner_strategy=FAMILY_SPEC_STRATEGY,
        allowed_literature_citation_keys=ALLOWED, direct_comparator_literature_keys=[],
        enforce_article_contract=True, article_contract_context=context,
        planning_contract_context=planning_context,
        required_primary_cohort_selection_mode="all_input_rows",
    ).output
    fake = SimpleNamespace(
        _scientific_runtime_authorities=authorities,
        _enable_publication_figure_skill=True,
        _max_total_steps=24,
    )
    plan = _pipeline._shape_fresh_plan(
        pipeline=fake, plan=plan, context=context, agent_context=context,
        long_trajectory_bound=False, findings=[],
    )
    plan = bind_context_dependence_authority(plan=plan, context=context)
    bound, _bind_findings = authorities.bind_plan(plan)
    bound = _figure_plan.apply_runtime_bound_figure_contracts(bound, [])
    authorities.validate_plan(bound)
    _final_plan.validate_final_plan_shape(bound)
    return bound, llm


def test_the_sealed_continuous_suite_is_planned_in_one_call_and_bound_to_the_signed_owner():
    authority = build_current_case_scientific_runtime_authority(continuous_authority_body())
    context = _sealed_context(authority)
    authorities = ScientificRuntimeAuthorities(trajectory=None, current_case=authority)
    disclosure = authorities.planning_contract_context()
    sealed = sealed_continuous_survival_suite_coordinates(disclosure)
    assert sealed is not None
    assert SEALED_CONTINUOUS_SURVIVAL_SUITE_MARKER in disclosure
    assert sealed.primary_owner == authority.plan_method
    assert sealed.adjustment_columns == list(authority.adjustment_columns)
    assert sealed.plan_outputs == list(authority.plan_outputs)
    assert (sealed.exposure_window_summary, sealed.exposure_unit) == ("max", "mmol/L")
    types = candidate_analysis_types(context)
    assert (
        family_template_id_for_context(context, analysis_types=types, planning_contract_context=disclosure)
        == LANDMARK_CONTINUOUS_SURVIVAL_FAMILY_ID
    )
    # A disclosure of another exposure window is not this suite's.
    shifted = disclosure.replace('"exposure_window_hours": [0.0, 24.0]', '"exposure_window_hours": [0.0, 12.0]')
    assert shifted != disclosure
    assert sealed_continuous_survival_suite_coordinates(shifted) is None

    bound, llm = _plan_through_host(context, authorities)

    assert len(llm.calls) == 1
    assert [step.method for step in bound.steps] == [
        "host_materialized_locked_cohort",
        "signed_landmark_continuous_survival_suite",
        "signed_landmark_continuous_survival_figure",
    ]
    primary = bound.steps[1]
    assert primary.runtime_outcome_contract.outcomes == ("mort_28d",)
    assert authority.plan_rule_ref in primary.icu_rule_refs
    validate_required_primary_result(plan=bound, context=context)
    owned = [
        select_standard_executor(
            step, plan=bound,
            current_case_scientific_runtime_authority=authority,
            scientific_runtime_projection_sha256="b" * 64,
        )
        for step in bound.steps
    ]
    assert [item.analysis_kind for item in owned if item is not None] == [
        "host_bound_analysis_cohort",
        "signed_landmark_continuous_survival_suite",
        "signed_landmark_continuous_survival_figure",
    ]
    review = build_plan_scientific_review(
        context=context, plan=bound, literature=None,
        figure_strategy=build_article_figure_strategy(context), runtime_authority=authority,
    )
    codes = {finding.code for finding in review.findings}
    assert review.approval_allowed, codes
