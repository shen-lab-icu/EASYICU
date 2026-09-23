"""Family-spec planning for the sealed host suites (landmark survival, fixed-window trajectory,
fail-closed source feasibility).

All families are exercised with the real Canonical9 protocol authorities and
synthetic ResearchContexts: zero Provider calls, zero patient rows.  The Planner
mock returns only labels; every scientific coordinate is the authority's, and
the host's ``bind_plan`` replaces the draft with the signed owners.
"""

from __future__ import annotations

import json
from types import SimpleNamespace

import pytest

from benchmarks.figure2_canonical9.case_scientific_protocol import (
    build_runtime_scientific_projection,
    load_default_case_protocol,
)
from easyicu.research_agent import pipeline as _pipeline
from easyicu.research_agent.agents.progressive_planner import (
    FAMILY_SPEC_STRATEGY,
    ProgressivePlannerAgent,
    candidate_analysis_types,
    select_progressive_variables,
)
from easyicu.research_agent.authority.current_case_scientific_runtime import (
    LandmarkSurvivalRuntimeAuthority,
    SourceFeasibilityRuntimeAuthority,
    load_current_case_scientific_runtime_authority,
)
from easyicu.research_agent.contracts.endpoint import EndpointSpec
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
    FIXED_WINDOW_TRAJECTORY_FAMILY_ID,
    LANDMARK_SURVIVAL_FAMILY_ID,
    SOURCE_FEASIBILITY_FAMILY_ID,
    FamilySpecError,
    build_family_spec_request,
    family_template_id_for_context,
    sealed_feasibility_coordinates,
    sealed_survival_suite_coordinates,
    sealed_trajectory_suite_coordinates,
    validate_family_plan_spec,
)
from easyicu.research_agent.planning.family_spec.contract import spec_from_mapping
from easyicu.research_agent.planning.figure_strategy import build_article_figure_strategy
from easyicu.research_agent.planning.primary_result_contract import (
    validate_required_primary_result,
)
from easyicu.research_agent.planning.publication_readiness import (
    build_publication_readiness_facts,
)
from easyicu.research_agent.reporting.article_contract import (
    build_article_analysis_contract,
    roles_covered_by_plan,
)
from easyicu.research_agent.planning.scientific_review import build_plan_scientific_review
from easyicu.research_agent.providers.mocks import ScriptedMockLLMClient
from easyicu.research_agent.schema import (
    AnalysisPlan,
    CohortDescriptor,
    ConceptDescriptor,
    ResearchContext,
    UserPreferences,
    VariableRole,
)
from easyicu.research_agent.trajectory.scientific_runtime_authority import (
    load_trajectory_scientific_runtime_authority,
)

ALLOWED = ["strobe_2007", "record_2015"]


def _h1_authority() -> tuple[object, LandmarkSurvivalRuntimeAuthority]:
    projection = build_runtime_scientific_projection(
        load_default_case_protocol("h1_ventilation_survival")
    )
    authority = load_current_case_scientific_runtime_authority(
        projection.deterministic_execution_contract
    )
    assert isinstance(authority, LandmarkSurvivalRuntimeAuthority)
    return projection, authority


def _h1_context(authority: LandmarkSurvivalRuntimeAuthority) -> ResearchContext:
    binary = {"n_unique": 2, "is_binary": True, "levels": [0, 1]}
    return ResearchContext(
        research_question=(
            "Among adult ICU stays alive at 24 h, is incident invasive ventilation by "
            "24 h associated with 28-day mortality?"
        ),
        cohort=CohortDescriptor(
            cohort_name="h1_synthetic", database="miiv", n_stays=0, id_columns=["stay_id"],
            outcome_columns=["mort_28d"], requested_outcome_columns=["mort_28d"],
            provenance={"analysis_unit": "icu_stay", "patient_identity_available": False,
                        "stay_id_columns": ["stay_id"], "patient_id_columns": []},
        ),
        variables=[
            ConceptDescriptor(name="stay_id", role=VariableRole.ID, dtype="int64"),
            ConceptDescriptor(name="mech_vent_max", description="invasive ventilation by 24 h",
                              role=VariableRole.INTERVENTION, dtype="float64", observed_domain=binary),
            ConceptDescriptor(name="mech_vent_first_time", description="first ventilation time",
                              role=VariableRole.OTHER, dtype="float64", unit="hours"),
            ConceptDescriptor(name="mort_28d", description="Death by day 28 from ICU admission",
                              role=VariableRole.OUTCOME, dtype="float64", observed_domain=binary),
            ConceptDescriptor(name="followup_days_28d", description="28-day event or censoring time",
                              role=VariableRole.OUTCOME, dtype="float64", unit="days"),
            ConceptDescriptor(name="age", description="age", role=VariableRole.DEMOGRAPHIC,
                              dtype="float64", unit="years"),
            ConceptDescriptor(name="sex", description="sex", role=VariableRole.DEMOGRAPHIC,
                              dtype="float64", observed_domain=binary),
            ConceptDescriptor(name="charlson_first", description="Charlson index",
                              role=VariableRole.OTHER, dtype="float64"),
            ConceptDescriptor(name="sofa2_max", description="SOFA-2 maximum",
                              role=VariableRole.ORDINAL_SCORE, dtype="float64"),
        ],
        target_outcome="mort_28d",
        primary_exposure="mech_vent_max",
        endpoint=authority.research_context_endpoint(),
        user_preferences=UserPreferences(
            inferred_analysis_family="survival", covariate_selection="exact",
            covariates=["age", "sex", "charlson_first", "sofa2_max"],
        ),
    )


H1_LABELS = {
    "mech_vent_max": "Incident invasive ventilation by 24 h",
    "mech_vent_max=0": "No incident ventilation by 24 h",
    "mech_vent_max=1": "Incident ventilation by 24 h",
    "mech_vent_first_time": "Time of first invasive ventilation (h after ICU admission)",
    "mort_28d": "Death by day 28",
    "followup_days_28d": "Days to death or censoring within 28 days",
    "age": "Age (years)",
    "sex": "Patient sex",
    "charlson_first": "Charlson comorbidity index",
    "sofa2_max": "Maximum SOFA-2 in the first 24 h",
}


def _labels_payload(request, labels: dict[str, str]) -> dict:
    return {
        "schema_version": "easyicu.family_plan_spec/1",
        "family_id": request.family_id,
        "request_sha256": request.request_sha256,
        "adjustment_set": [],
        "reader_display_labels": [
            {"key": key, "value": labels.get(key, key.replace("_", " ") + " (label)")}
            for key in dict.fromkeys([*request.required_reader_label_keys, *request.level_label_keys])
        ],
        "comparator_applications": [],
        "roster_decision_note": "Sealed suite: the host owns every coordinate; labels only.",
    }


def _plan_through_host(
    context: ResearchContext,
    authorities: ScientificRuntimeAuthorities,
    labels: dict[str, str],
) -> tuple[AnalysisPlan, ScriptedMockLLMClient]:
    planning_context = authorities.planning_contract_context()
    types = candidate_analysis_types(context)
    variables = select_progressive_variables(context)
    request = build_family_spec_request(
        context, analysis_types=types, variable_roster=variables,
        allowed_literature_citation_keys=ALLOWED,
        required_primary_cohort_selection_mode="all_input_rows",
        planning_contract_context=planning_context,
    )
    llm = ScriptedMockLLMClient([json.dumps(_labels_payload(request, labels))])
    result = ProgressivePlannerAgent(llm).run_attempt(
        context, planner_strategy=FAMILY_SPEC_STRATEGY,
        allowed_literature_citation_keys=ALLOWED, direct_comparator_literature_keys=[],
        enforce_article_contract=True, article_contract_context=context,
        planning_contract_context=planning_context,
        required_primary_cohort_selection_mode="all_input_rows",
    )
    plan = result.output
    fake = SimpleNamespace(
        _scientific_runtime_authorities=authorities,
        _enable_publication_figure_skill=True,
        _max_total_steps=24,
    )
    findings: list = []
    plan = _pipeline._shape_fresh_plan(
        pipeline=fake, plan=plan, context=context, agent_context=context,
        long_trajectory_bound=False, findings=findings,
    )
    plan = bind_context_dependence_authority(plan=plan, context=context)
    bound, _bind_findings = authorities.bind_plan(plan)
    bound = _figure_plan.apply_runtime_bound_figure_contracts(bound, findings)
    authorities.validate_plan(bound)
    _final_plan.validate_final_plan_shape(bound)
    return bound, llm


def test_sealed_survival_suite_is_planned_in_one_call_and_bound_to_the_signed_owner() -> None:
    projection, authority = _h1_authority()
    context = _h1_context(authority)
    authorities = ScientificRuntimeAuthorities(trajectory=None, current_case=authority)
    disclosure = authorities.planning_contract_context()
    sealed = sealed_survival_suite_coordinates(disclosure)
    assert sealed is not None
    assert sealed.primary_owner == authority.plan_method
    assert sealed.adjustment_columns == list(authority.adjustment_columns)
    assert sealed_survival_suite_coordinates("no disclosure here") is None
    types = candidate_analysis_types(context)
    assert types[0] == "survival"
    # Without the sealed disclosure the survival family has no template.
    assert family_template_id_for_context(context, analysis_types=types) is None
    assert (
        family_template_id_for_context(
            context, analysis_types=types, planning_contract_context=disclosure
        )
        == LANDMARK_SURVIVAL_FAMILY_ID
    )

    bound, llm = _plan_through_host(context, authorities, H1_LABELS)
    assert len(llm.calls) == 1
    assert [step.method for step in bound.steps] == [
        "host_materialized_locked_cohort",
        authority.plan_method,
        "signed_landmark_survival_figure",
    ]
    primary = bound.steps[1]
    assert primary.planned_analysis_role == "primary"
    assert primary.runtime_outcome_contract is not None
    assert primary.runtime_outcome_contract.outcomes == (authority.event_column,)
    assert authority.plan_rule_ref in primary.icu_rule_refs
    assert primary.family_primary_result_requirement is None
    validate_required_primary_result(plan=bound, context=context)
    assert [
        panel.article_role for panel in bound.steps[2].figure_panels
    ] == ["temporal_absolute_risk", "survival_effect", "cohort_accounting", "diagnostics"]
    owned = [
        select_standard_executor(
            step, plan=bound,
            current_case_scientific_runtime_authority=authority,
            scientific_runtime_projection_sha256=projection.runtime_projection_sha256,
        )
        for step in bound.steps
    ]
    assert [item.analysis_kind for item in owned if item is not None] == [
        "host_bound_analysis_cohort",
        "signed_landmark_survival_suite",
        "signed_landmark_survival_figure",
    ]
    review = build_plan_scientific_review(
        context=context, plan=bound, literature=None,
        figure_strategy=build_article_figure_strategy(context), runtime_authority=authority,
    )
    codes = {finding.code for finding in review.findings}
    assert review.approval_allowed, codes
    assert not codes & {
        "REQUESTED_OUTCOME_COVERAGE_INCOMPLETE",
        "FIGURE_ROLE_COVERAGE_INCOMPLETE",
        "ARTICLE_CONTENT_ROLES_INCOMPLETE",
    }


def test_sealed_survival_spec_cannot_carry_a_roster() -> None:
    _projection, authority = _h1_authority()
    context = _h1_context(authority)
    disclosure = ScientificRuntimeAuthorities(
        trajectory=None, current_case=authority
    ).planning_contract_context()
    types = candidate_analysis_types(context)
    request = build_family_spec_request(
        context, analysis_types=types, variable_roster=select_progressive_variables(context),
        allowed_literature_citation_keys=ALLOWED,
        required_primary_cohort_selection_mode="all_input_rows",
        planning_contract_context=disclosure,
    )
    assert request.adjustment_selection == "exact"
    assert request.exact_roster == list(authority.adjustment_columns)
    assert not request.selectable_candidates
    payload = _labels_payload(request, H1_LABELS)
    payload["adjustment_set"] = [
        {
            "name": "age", "coding": "continuous", "reference_level_index": None,
            "clinical_rationale": "Age precedes admission and predicts both ventilation and death.",
        }
    ]
    with pytest.raises(FamilySpecError) as caught:
        validate_family_plan_spec(spec_from_mapping(payload), request)
    assert caught.value.reason_code == "family_spec_adjustment_not_applicable"


def test_unsealed_survival_draft_still_needs_a_planner_owned_contract() -> None:
    _projection, authority = _h1_authority()
    context = _h1_context(authority)
    draft = AnalysisPlan.model_validate(
        {
            "research_question": context.research_question,
            "analysis_type": "survival",
            "steps": [
                {
                    "step_id": "01_primary", "planned_analysis_role": "primary",
                    "intent": "Fit a Cox model.", "inputs": ["dataset:analysis_cohort"],
                    "expected_outputs": ["table:cox_summary"], "method": "cox_proportional_hazards",
                }
            ],
        }
    )
    with pytest.raises(ValueError, match="family_primary_result_requirement"):
        validate_required_primary_result(plan=draft, context=context)
    sealed_draft = draft.model_copy(
        update={"steps": [draft.steps[0].model_copy(update={"method": authority.plan_method})]}
    )
    validate_required_primary_result(plan=sealed_draft, context=context)


def _h3_context(traj) -> ResearchContext:
    concepts = [*traj.coordinate_concepts, *traj.descriptive_only_concepts]
    return ResearchContext(
        research_question=(
            "Among adult ICU stays, which organ-dysfunction trajectory classes emerge over "
            "the first 72 h from SOFA-2 components and lactate, and how does in-hospital "
            "mortality differ by class?"
        ),
        cohort=CohortDescriptor(
            cohort_name="h3_synthetic", database="miiv", n_stays=0, id_columns=["stay_id"],
            outcome_columns=["death"], requested_outcome_columns=["death"],
            provenance={"analysis_unit": "icu_stay", "patient_identity_available": False,
                        "stay_id_columns": ["stay_id"], "patient_id_columns": []},
        ),
        variables=[
            ConceptDescriptor(name="stay_id", role=VariableRole.ID, dtype="int64"),
            *[
                ConceptDescriptor(
                    name=concept, description=f"{concept} coordinate",
                    role=VariableRole.ORDINAL_SCORE if concept.startswith("sofa2") else VariableRole.LAB,
                    dtype="float64", analysis_window="icu_admission[0,72]h",
                )
                for concept in concepts
            ],
            ConceptDescriptor(name="death", description="in-hospital mortality", role=VariableRole.OUTCOME,
                              dtype="float64", observed_domain={"n_unique": 2, "is_binary": True, "levels": [0, 1]}),
        ],
        target_outcome="death",
        endpoint=EndpointSpec(name="death", kind="binary", absence_semantics="no_absent_rows", levels=[0, 1]),
        user_preferences=UserPreferences(
            inferred_analysis_family="trajectory_clustering", covariate_selection="planner_selectable"
        ),
    )


H3_LABELS = {
    "death": "In-hospital death",
    "sofa2_resp": "SOFA-2 respiratory score",
    "sofa2_coag": "SOFA-2 coagulation score",
    "sofa2_liver": "SOFA-2 liver score",
    "sofa2_cardio": "SOFA-2 cardiovascular score",
    "sofa2_cns": "SOFA-2 neurological score",
    "sofa2_renal": "SOFA-2 renal score",
    "lact": "Lactate (mmol/L)",
    "sofa2": "Total SOFA-2 score",
}


def test_sealed_trajectory_suite_is_planned_in_one_call_and_bound_to_the_four_signed_owners() -> None:
    projection = build_runtime_scientific_projection(
        load_default_case_protocol("h3_trajectory_clustering")
    )
    traj = load_trajectory_scientific_runtime_authority(projection.deterministic_execution_contract)
    context = _h3_context(traj)
    authorities = ScientificRuntimeAuthorities(trajectory=traj, current_case=None)
    disclosure = authorities.planning_contract_context()
    sealed = sealed_trajectory_suite_coordinates(disclosure)
    assert sealed is not None
    assert sealed.representation_owner == traj.representation_plan_method
    assert sealed.coordinate_concepts == list(traj.coordinate_concepts)
    types = candidate_analysis_types(context)
    assert types[0] == "trajectory_clustering"
    assert (
        family_template_id_for_context(
            context, analysis_types=types, planning_contract_context=disclosure
        )
        == FIXED_WINDOW_TRAJECTORY_FAMILY_ID
    )

    bound, llm = _plan_through_host(context, authorities, H3_LABELS)
    assert len(llm.calls) == 1
    signed = [step for step in bound.steps if traj.plan_rule_ref in step.icu_rule_refs]
    assert [step.method for step in signed] == [
        traj.representation_plan_method,
        "observed_data_diagonal_gaussian_mixture_candidate_selection",
        "trajectory_cluster_stability_characterization",
        "signed_trajectory_selection_diagnostic_figure",
    ]
    # The suite renders a selection diagnostic and an availability matrix. It
    # does not render phenotype structure, so it must not promise that role.
    assert [panel.article_role for panel in signed[3].figure_panels] == [
        "cluster_selection",
        "data_quality",
        "phenotype_profile",
        "phenotype_structure",
        "stability",
    ]
    owned = [
        select_standard_executor(
            step, plan=bound, trajectory_scientific_runtime_authority=traj.model_dump(mode="json")
        )
        for step in signed
    ]
    assert [item.analysis_kind for item in owned] == [
        "trajectory_signed_representation",
        "trajectory_signed_candidate_selection",
        "trajectory_cluster_stability",
        "trajectory_selection_diagnostic_figure",
    ]
    review = build_plan_scientific_review(
        context=context, plan=bound, literature=None,
        figure_strategy=build_article_figure_strategy(context), runtime_authority=None,
    )
    codes = {finding.code for finding in review.findings}
    assert review.approval_allowed, codes
    assert "ARTICLE_CONTENT_ROLES_INCOMPLETE" not in codes
    # The sealed candidate grid and stability owner are this study's
    # prespecified robustness; strip the seal and the reviewer is owed the
    # finding again.
    assert "ROBUSTNESS_AUTHORITY_NOT_PRESPECIFIED" not in codes
    unsealed = bound.model_copy(
        update={
            "steps": [
                step.model_copy(update={"icu_rule_refs": []}) for step in bound.steps
            ]
        }
    )
    unsealed_codes = {
        finding.code
        for finding in build_plan_scientific_review(
            context=context, plan=unsealed, literature=None,
            figure_strategy=build_article_figure_strategy(context),
            runtime_authority=None,
        ).findings
    }
    assert "ROBUSTNESS_AUTHORITY_NOT_PRESPECIFIED" in unsealed_codes


def test_trajectory_draft_that_only_names_the_owners_is_replaced_not_trusted() -> None:
    projection = build_runtime_scientific_projection(
        load_default_case_protocol("h3_trajectory_clustering")
    )
    traj = load_trajectory_scientific_runtime_authority(projection.deterministic_execution_contract)
    authorities = ScientificRuntimeAuthorities(trajectory=traj, current_case=None)
    draft = AnalysisPlan.model_validate(
        {
            "research_question": "Which trajectory classes emerge?",
            "analysis_type": "trajectory_clustering",
            "steps": [
                {
                    "step_id": "rep", "planned_analysis_role": "auxiliary",
                    "intent": "Represent trajectories with a wrong grid.",
                    "inputs": ["artifact:analysis_cohort"],
                    "expected_outputs": ["artifact:trajectory_representation"],
                    "method": traj.representation_plan_method,
                },
                {
                    "step_id": "cand", "planned_analysis_role": "primary",
                    "intent": "Select candidates with a wrong grid.",
                    "inputs": ["artifact:trajectory_representation"],
                    "expected_outputs": ["table:trajectory_candidate_selection"],
                    "method": "observed_data_diagonal_gaussian_mixture_candidate_selection",
                },
            ],
        }
    )
    assert traj.names_signed_owners(draft)
    assert not traj.is_development_execution_only_plan(draft)
    bound, findings = authorities.bind_plan(draft)
    assert [step.step_id for step in bound.steps] == list(traj.development_execution_step_ids)
    assert findings[0].detail["reason_code"] == (
        "trajectory_development_execution_only_authority_compiled"
    )
    reordered = draft.model_copy(update={"steps": list(reversed(draft.steps))})
    assert not traj.names_signed_owners(reordered)
    untouched, _ = authorities.bind_plan(reordered)
    assert untouched is reordered


# ---------------------------------------------------------------------------
# H2: sealed fail-closed source-feasibility decision
# ---------------------------------------------------------------------------


def _h2_authority() -> tuple[object, SourceFeasibilityRuntimeAuthority]:
    projection = build_runtime_scientific_projection(
        load_default_case_protocol("h2_vasopressor_causal")
    )
    authority = load_current_case_scientific_runtime_authority(
        projection.deterministic_execution_contract
    )
    assert isinstance(authority, SourceFeasibilityRuntimeAuthority)
    return projection, authority


def _h2_context(
    authorities: ScientificRuntimeAuthorities, *, contrast_declared: bool
) -> ResearchContext:
    """A bench-like context (no contrast) or a Web-like one naming exposure/outcome."""

    binary = {"n_unique": 2, "is_binary": True, "levels": [0, 1]}
    _endpoint, _exposure, preferences = authorities.bind_run_inputs(
        endpoint=None,
        primary_exposure=None,
        user_preferences={"inferred_analysis_family": "causal_inference"},
    )
    return ResearchContext(
        research_question=(
            "Does vasopressor initiation within the first 24 ICU hours causally change "
            "28-day mortality compared with no vasopressor use?"
        ),
        cohort=CohortDescriptor(
            cohort_name="h2_synthetic", database="miiv", n_stays=0, id_columns=["stay_id"],
            outcome_columns=["death"],
            provenance={"analysis_unit": "icu_stay", "patient_identity_available": False,
                        "stay_id_columns": ["stay_id"], "patient_id_columns": []},
        ),
        variables=[
            ConceptDescriptor(name="stay_id", role=VariableRole.ID, dtype="int64"),
            ConceptDescriptor(name="vaso_ind_max", description="recorded vasopressor administration",
                              role=VariableRole.INTERVENTION, dtype="float64", observed_domain=binary),
            ConceptDescriptor(name="death", description="in-hospital death",
                              role=VariableRole.OUTCOME, dtype="float64", observed_domain=binary),
        ],
        target_outcome="death" if contrast_declared else None,
        primary_exposure="vaso_ind_max" if contrast_declared else None,
        user_preferences=UserPreferences(**preferences),
    )


H2_LABELS = {
    "vaso_ind_max": "Recorded vasopressor administration in the first 24 h",
    "death": "In-hospital death",
}


@pytest.mark.parametrize("contrast_declared", [False, True])
def test_sealed_feasibility_decision_is_planned_in_one_call_and_bound_to_the_signed_owner(
    contrast_declared: bool,
) -> None:
    projection, authority = _h2_authority()
    authorities = ScientificRuntimeAuthorities(trajectory=None, current_case=authority)
    context = _h2_context(authorities, contrast_declared=contrast_declared)
    assert context.user_preferences.formal_result_scope == "source_feasibility_fail_closed"
    disclosure = authorities.planning_contract_context()
    sealed = sealed_feasibility_coordinates(disclosure)
    assert sealed is not None
    assert sealed.sealed_owner == authority.plan_method
    assert sealed.reason_code == authority.reason_code
    assert sealed.plan_outputs == list(authority.plan_outputs)
    assert sealed_feasibility_coordinates("no disclosure here") is None
    types = candidate_analysis_types(context)
    assert types[0] == "causal_inference"
    assert family_template_id_for_context(context, analysis_types=types) is None
    assert (
        family_template_id_for_context(
            context, analysis_types=types, planning_contract_context=disclosure
        )
        == SOURCE_FEASIBILITY_FAMILY_ID
    )

    bound, llm = _plan_through_host(context, authorities, H2_LABELS)
    assert len(llm.calls) == 1
    assert [step.method for step in bound.steps] == [authority.plan_method]
    (owner,) = bound.steps
    assert owner.planned_analysis_role == "auxiliary"
    assert owner.inputs == [] and list(owner.expected_outputs) == list(authority.plan_outputs)
    assert authority.plan_rule_ref in owner.icu_rule_refs
    validate_required_primary_result(plan=bound, context=context)
    selected = select_standard_executor(
        owner, plan=bound,
        current_case_scientific_runtime_authority=authority,
        scientific_runtime_projection_sha256=projection.runtime_projection_sha256,
    )
    assert selected is not None and selected.analysis_kind == authority.plan_method

    # The article, figure and robustness contracts narrow to the sealed decision.
    contract = build_article_analysis_contract(context, analysis_type=bound.analysis_type)
    assert contract.required_roles == ["feasibility_decision"]
    assert contract.planner_owned_result_roles == []
    assert roles_covered_by_plan(bound, contract) == {"feasibility_decision"}
    strategy = build_article_figure_strategy(context)
    assert strategy.role_strategies == [] and strategy.hero_role == "feasibility_decision"
    readiness = build_publication_readiness_facts(
        context=context, plan=bound, figure_strategy=strategy,
        sensitivity={"typed_executable": [], "missing_spec_ids": [], "protocol_only": []},
    )
    assert readiness["robustness"]["status"] == "not_applicable"
    assert readiness["content_roles"]["missing_roles"] == []
    review = build_plan_scientific_review(
        context=context, plan=bound, literature=None,
        figure_strategy=strategy, runtime_authority=authority,
    )
    codes = {finding.code for finding in review.findings}
    assert review.approval_allowed, codes
    assert not codes & {
        "OUTCOME_DEFINITION_UNRESOLVED",
        "ROBUSTNESS_AUTHORITY_NOT_PRESPECIFIED",
        "FIGURE_ROLE_COVERAGE_INCOMPLETE",
        "ARTICLE_CONTENT_ROLES_INCOMPLETE",
    }


def test_feasibility_scope_still_refuses_a_drafted_effect_step() -> None:
    _projection, authority = _h2_authority()
    authorities = ScientificRuntimeAuthorities(trajectory=None, current_case=authority)
    context = _h2_context(authorities, contrast_declared=True)
    effect_plan = AnalysisPlan.model_validate(
        {
            "research_question": context.research_question,
            "analysis_type": "causal_inference",
            "steps": [
                {
                    "step_id": "01_iptw",
                    "planned_analysis_role": "primary",
                    "intent": "Estimate the weighted effect of vasopressors on death.",
                    "inputs": ["vaso_ind_max", "death"],
                    "expected_outputs": ["table:causal_contrast"],
                    "method": "inverse_probability_weighting",
                }
            ],
        }
    )
    with pytest.raises(ValueError, match="forbids a primary effect step"):
        validate_required_primary_result(plan=effect_plan, context=context)
    # Without the sealed authority the same context keeps the ordinary causal gate.
    plain = context.model_copy(
        update={"user_preferences": UserPreferences(inferred_analysis_family="causal_inference")}
    )
    assert build_article_analysis_contract(plain, analysis_type="causal_inference").required_roles != [
        "feasibility_decision"
    ]
    assert build_article_figure_strategy(plain).role_strategies
    with pytest.raises(ValueError, match="family_primary_result_requirement"):
        validate_required_primary_result(plan=effect_plan, context=plain)


def test_feasibility_scope_conflict_and_spec_shape_fail_closed() -> None:
    _projection, authority = _h2_authority()
    authorities = ScientificRuntimeAuthorities(trajectory=None, current_case=authority)
    with pytest.raises(ValueError, match="formal result scope conflicts"):
        authorities.bind_run_inputs(
            endpoint=None, primary_exposure=None,
            user_preferences={"formal_result_scope": "something_else"},
        )
    context = _h2_context(authorities, contrast_declared=False)
    request = build_family_spec_request(
        context, analysis_types=candidate_analysis_types(context),
        variable_roster=select_progressive_variables(context),
        allowed_literature_citation_keys=ALLOWED,
        required_primary_cohort_selection_mode="all_input_rows",
        planning_contract_context=authorities.planning_contract_context(),
    )
    assert request.family_id == SOURCE_FEASIBILITY_FAMILY_ID
    assert request.outcome == "" and request.primary_exposure == ""
    payload = _labels_payload(request, H2_LABELS)
    payload["adjustment_set"] = [
        {"name": "age", "coding": "continuous", "clinical_rationale": "Age confounds vasopressor use."}
    ]
    with pytest.raises(FamilySpecError, match="family_spec_adjustment_not_applicable"):
        validate_family_plan_spec(spec_from_mapping(payload), request)


def test_a_sealed_suite_prespecifies_its_own_robustness_but_a_draft_does_not() -> None:
    """The seal, not the method name, is what makes the axes prespecified."""

    from easyicu.research_agent.contracts.sealed_suite_robustness import (
        SEALED_SUITE_ROBUSTNESS_AXES,
        sealed_suite_prespecified_axes,
    )
    from easyicu.research_agent.planning.sensitivity_authority import (
        PrespecifiedSensitivitySpec,
    )

    sealed_ref = f"scientific_runtime_contract:{'a' * 64}"
    for method, axes in SEALED_SUITE_ROBUSTNESS_AXES.items():
        assert sealed_suite_prespecified_axes(
            method=method, rule_refs=[sealed_ref]
        ) == tuple(axes)
        # An unsigned draft may spell the same method; it earns nothing.
        assert sealed_suite_prespecified_axes(method=method, rule_refs=[]) == ()
        assert (
            sealed_suite_prespecified_axes(
                method=method, rule_refs=["scientific_runtime_contract:not-a-digest"]
            )
            == ()
        )
    assert (
        sealed_suite_prespecified_axes(
            method="adjusted_association_models", rule_refs=[sealed_ref]
        )
        == ()
    )
    # The axis names a sealed suite contributes are review-only: a user cannot
    # declare them on a StudyContext sensitivity spec.
    declarable = set(PrespecifiedSensitivitySpec.model_fields["axis"].annotation.__args__)
    contributed = {
        axis for axes in SEALED_SUITE_ROBUSTNESS_AXES.values() for axis in axes
    }
    assert not (contributed - {"timing"}) & declarable


def test_sealed_suite_recommendations_follow_a_chinese_question() -> None:
    """The six review items follow the question; the signed science does not move."""

    import re

    han = re.compile(r"[一-鿿]")
    _projection, survival = _h1_authority()
    survival_authorities = ScientificRuntimeAuthorities(trajectory=None, current_case=survival)
    trajectory = load_trajectory_scientific_runtime_authority(
        build_runtime_scientific_projection(
            load_default_case_protocol("h3_trajectory_clustering")
        ).deterministic_execution_contract
    )
    trajectory_authorities = ScientificRuntimeAuthorities(trajectory=trajectory, current_case=None)
    _projection, feasibility = _h2_authority()
    feasibility_authorities = ScientificRuntimeAuthorities(trajectory=None, current_case=feasibility)
    cases = [
        (_h1_context(survival), survival_authorities, H1_LABELS),
        (_h3_context(trajectory), trajectory_authorities, H3_LABELS),
        (
            _h2_context(feasibility_authorities, contrast_declared=True),
            feasibility_authorities,
            H2_LABELS,
        ),
    ]
    def draft(context, authorities, labels):
        planning_context = authorities.planning_contract_context()
        request = build_family_spec_request(
            context,
            analysis_types=candidate_analysis_types(context),
            variable_roster=select_progressive_variables(context),
            allowed_literature_citation_keys=ALLOWED,
            required_primary_cohort_selection_mode="all_input_rows",
            planning_contract_context=planning_context,
        )
        llm = ScriptedMockLLMClient([json.dumps(_labels_payload(request, labels))])
        return ProgressivePlannerAgent(llm).run_attempt(
            context, planner_strategy=FAMILY_SPEC_STRATEGY,
            allowed_literature_citation_keys=ALLOWED, direct_comparator_literature_keys=[],
            enforce_article_contract=True, article_contract_context=context,
            planning_contract_context=planning_context,
            required_primary_cohort_selection_mode="all_input_rows",
        ).output

    for context, authorities, labels in cases:
        english = draft(context, authorities, labels)
        chinese_context = context.model_copy(
            update={"research_question": "这个签名研究问题在本数据源上能回答到什么程度？"}
        )
        chinese = draft(chinese_context, authorities, labels)
        english_items = english.design_selection.selected.reviewable_plan
        chinese_items = chinese.design_selection.selected.reviewable_plan
        assert len(english_items) == len(chinese_items) == 6
        assert all(not han.search(item) for item in english_items)
        assert all(han.search(item) for item in chinese_items)
        assert chinese_items[0].startswith("研究队列")
        for item in [*english_items, *chinese_items]:
            assert not re.match(r"^[A-Za-z /-]{3,40}:\s", item), item
            assert not re.match(r"^[\u4e00-\u9fff]{2,12}：", item), item
        # Readers label the six positions (population, exposure/timing,
        # outcome, model, missing data, sensitivity): the trajectory
        # candidate grid is the model item, not the outcome item.
        for items in (english_items, chinese_items):
            model_items = [index for index, item in enumerate(items) if "BIC" in item]
            assert model_items in ([], [3]), items
        assert [step.method for step in english.steps] == [step.method for step in chinese.steps]
        cohort_name = context.cohort.cohort_name
        assert all(cohort_name not in item for item in [*english_items, *chinese_items])
