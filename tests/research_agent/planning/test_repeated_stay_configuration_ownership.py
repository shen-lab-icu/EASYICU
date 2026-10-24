"""Host-fixed covariance is repaired by its owner, with synthetic metadata only."""

from __future__ import annotations

import json

import pytest

from easyicu.research_agent.planning.dependence_authority import (
    bind_context_dependence_authority,
    repeat_units_possible,
)
from easyicu.research_agent.planning.scientific_review import build_plan_scientific_review
from easyicu.research_agent.schema import (
    AnalysisPlan, AnalysisStep, CohortDescriptor, ConceptDescriptor,
    PlannedModelRequirement, ResearchContext, UserPreferences, VariableRole,
)
from easyicu.webserver.pi_copilot.plan_decisions import (
    PlanDecisionError,
    REPEATED_STAYS_SYSTEM_DEFAULT,
    agent_plan_configuration_available,
    compile_agent_plan_configuration,
)

REPEATED = "REPEATED_STAY_METHOD_NOT_DECLARED"


def _context(*, metadata: bool = False, clustered: bool = False) -> ResearchContext:
    design = {
        "analysis_family": "association_study", "analysis_unit": "icu_stay",
        "variance_estimator": "cluster_robust" if clustered else "model_based",
    }
    if clustered:
        design["cluster_unit"] = "patient"
    return ResearchContext(
        research_question="Is a baseline marker associated with the binary endpoint?",
        primary_exposure="marker", target_outcome="endpoint",
        cohort=CohortDescriptor(
            cohort_name="synthetic", database="synthetic",
            n_stays=0 if metadata else 5, n_patients=None if metadata else 3,
            id_columns=["person_id", "stay_id"],
            provenance={
                "analysis_unit": "icu_stay", "patient_id_columns": ["person_id"],
                **({"evidence_stage": "metadata_only_planning", "patient_rows_read": False}
                   if metadata else {}),
            },
        ),
        variables=[ConceptDescriptor(name=name, role=VariableRole.OTHER, dtype="float64")
                   for name in ("marker", "endpoint", "baseline_value")],
        user_preferences=UserPreferences(data_constraints=json.dumps({"analysis_design": design})),
    )


def _plan(*, signed: bool = False) -> AnalysisPlan:
    requirement = PlannedModelRequirement(
        requirement_id="primary", outcome="endpoint", outcome_type="binary",
        method_family="statsmodels_logit_mle", exposure_source="marker",
        analysis_role="primary", analysis_set="source_aware", covariates=["baseline_value"],
        model_terms=[
            {"name": "marker", "role": "exposure", "coding": "continuous", "transform": "identity"},
            {"name": "baseline_value", "role": "covariate", "coding": "continuous", "transform": "identity"},
        ],
    )
    return AnalysisPlan(
        research_question=_context().research_question, analysis_type="association_study",
        steps=[AnalysisStep(
            step_id="primary_model", planned_analysis_role="primary",
            intent="Estimate the marker association, adjusted for the baseline value.",
            method="signed_landmark_restricted_cubic_spline" if signed else "adjusted_association_models",
            inputs=["marker", "endpoint", "baseline_value"],
            expected_outputs=["table:adjusted_association_estimates"],
            model_requirements=[] if signed else [requirement],
            icu_rule_refs=["scientific_runtime_contract:" + "a" * 64] if signed else [],
        )],
    )


@pytest.mark.parametrize("patients", [None, 0])
def test_empty_catalog_with_patient_grouping_does_not_prove_independent_rows(patients) -> None:
    context = _context(metadata=True)
    context.cohort.n_patients = patients
    assert repeat_units_possible(context)


def test_prepared_one_stay_counts_still_establish_independent_rows() -> None:
    context = _context()
    context.cohort.n_patients = context.cohort.n_stays
    assert not repeat_units_possible(context)


@pytest.mark.parametrize("metadata", [False, True])
@pytest.mark.parametrize("signed", [False, True])
def test_host_fixed_variance_is_not_sent_back_to_the_planner(metadata, signed) -> None:
    review = build_plan_scientific_review(context=_context(metadata=metadata), plan=_plan(signed=signed))
    finding = next(item for item in review.findings if item.code == REPEATED)
    assert finding.remediation_route == "runtime_capability"
    assert "analysis_design" in finding.remediation
    assert review.facts["patient_cluster_configuration_required"] is True
    assert REPEATED not in review.facts["remediation_buckets"]["agent_plan_revision"]


def test_a_missing_binding_on_an_already_clustered_design_remains_a_plan_repair() -> None:
    review = build_plan_scientific_review(context=_context(clustered=True), plan=_plan())
    finding = next(item for item in review.findings if item.code == REPEATED)
    assert finding.remediation_route == "agent_plan_revision"
    assert review.facts["patient_cluster_configuration_required"] is False


def test_another_family_is_not_given_the_association_configuration() -> None:
    plan = _plan().model_copy(update={"analysis_type": "causal_inference"})
    review = build_plan_scientific_review(context=_context(), plan=plan)
    assert review.facts["patient_cluster_configuration_required"] is False


def _study() -> dict:
    return {
        "question": _context().research_question,
        "cohort": {"preset": "all_icu", "exclude_readmissions": False},
        "analysis_design": {"analysis_family": "association_study", "analysis_unit": "icu_stay", "variance_estimator": "model_based"},
        "confirmations": {"feature_time_window": True},
    }


@pytest.mark.parametrize("signed", [False, True])
def test_host_compiles_grouped_variance_without_narrowing_the_population(signed) -> None:
    context, plan = _context(), _plan(signed=signed)
    review = build_plan_scientific_review(context=context, plan=plan)
    study = _study()
    original = json.dumps(study, sort_keys=True)
    compiled = compile_agent_plan_configuration(
        study=study, agent_plan=plan.model_dump(mode="json"), runtime_finding_codes=(REPEATED,),
        review_facts=review.facts, patient_cluster_available=True,
    )
    assert compiled.runtime_finding_codes == (REPEATED,)
    assert compiled.patch["analysis_design"] == {
        "analysis_family": "association_study", "analysis_unit": "icu_stay",
        "variance_estimator": "cluster_robust", "cluster_unit": "patient",
    }
    assert compiled.patch["confirmations"][REPEATED_STAYS_SYSTEM_DEFAULT] is True
    assert compiled.patch["confirmations"]["plan_repeated_stays_clustered"] is True
    assert compiled.patch["confirmations"]["feature_time_window"] is True
    assert "cohort" not in compiled.patch and "question" not in compiled.patch
    assert json.dumps(study, sort_keys=True) == original
    assert agent_plan_configuration_available(
        study=study, agent_plan=plan.model_dump(mode="json"),
        runtime_finding_codes=(REPEATED,), review_facts=review.facts,
    )
    if not signed:
        # The compiler's exact patch gives the plan-binding owner what it
        # needs; the same review then clears this finding without a waiver.
        context.user_preferences.data_constraints = json.dumps({"analysis_design": compiled.patch["analysis_design"]})
        bound = bind_context_dependence_authority(context=context, plan=plan)
        final = build_plan_scientific_review(context=context, plan=bound)
        assert REPEATED not in {item.code for item in final.findings}
        assert final.facts["repeated_unit_design_executable"] is True


@pytest.mark.parametrize("available,family,code", [
    (False, "association_study", "agent_plan_patient_grouping_unavailable"),
    (True, "trajectory_clustering", "agent_plan_family_clustering_unsupported"),
])
def test_grouped_variance_requires_source_and_family_authority(available, family, code) -> None:
    study = _study()
    study["analysis_design"]["analysis_family"] = family
    review = build_plan_scientific_review(context=_context(), plan=_plan())
    with pytest.raises(PlanDecisionError) as raised:
        compile_agent_plan_configuration(
            study=study, agent_plan=_plan().model_dump(mode="json"),
            runtime_finding_codes=(REPEATED,), review_facts=review.facts,
            patient_cluster_available=available,
        )
    assert raised.value.code == code


def test_clustering_and_timing_compile_together_without_losing_either_setting() -> None:
    plan = _plan().model_dump(mode="json")
    requirement = plan["steps"][0]["model_requirements"][0]
    requirement.update(
        exposure_source="marker_max", outcome="death",
        covariate_rationales={"baseline_value": "The value precedes exposure and predicts both exposure and outcome."},
        covariate_temporal_roles={"baseline_value": "baseline_static"},
    )
    plan["design_selection"] = {"candidates": [{
        "design_id": "landmark_adjusted_association", "disposition": "selected",
        "required_variables": ["marker_max", "death", "baseline_value"],
    }]}
    study = {**_study(), "time_window": {"hours": 24, "anchor": "icu_admission"}}
    codes = (REPEATED, "POST_BASELINE_EXPOSURE_TIMING_NOT_CLOSED", "PRIMARY_POPULATION_EXECUTION_OWNER_MISSING")
    compiled = compile_agent_plan_configuration(
        study=study, agent_plan=plan, runtime_finding_codes=codes,
        review_facts={"patient_cluster_configuration_required": True}, patient_cluster_available=True,
    )
    assert set(compiled.runtime_finding_codes) == set(codes)
    assert compiled.patch["analysis_design"]["variance_estimator"] == "cluster_robust"
    assert compiled.patch["confirmations"]["plan_timing_landmark_24h"] is True
    assert compiled.patch["confirmations"]["plan_repeated_stays_clustered"] is True
    assert compiled.patch["sensitivity_specs"][0]["strategy"] == "landmark"


def test_a_stale_clustering_finding_cannot_reopen_a_first_stay_population() -> None:
    study = _study()
    study["cohort"]["exclude_readmissions"] = True
    with pytest.raises(PlanDecisionError) as raised:
        compile_agent_plan_configuration(
            study=study, agent_plan=_plan().model_dump(mode="json"),
            runtime_finding_codes=(REPEATED,),
            review_facts={"patient_cluster_configuration_required": True},
            patient_cluster_available=True, first_stay_coordinate_available=True,
        )
    assert raised.value.code == "agent_plan_runtime_finding_unsupported"


@pytest.mark.parametrize("previous", [None, False, True])
def test_recompiling_preserves_who_confirmed_the_clustering(previous) -> None:
    study = _study()
    study["confirmations"]["plan_repeated_stays_clustered"] = True
    if previous is not None:
        study["confirmations"][REPEATED_STAYS_SYSTEM_DEFAULT] = previous
    compiled = compile_agent_plan_configuration(
        study=study, agent_plan=_plan().model_dump(mode="json"), runtime_finding_codes=(REPEATED,),
        review_facts={"patient_cluster_configuration_required": True}, patient_cluster_available=True,
    )
    assert compiled.patch["confirmations"].get(REPEATED_STAYS_SYSTEM_DEFAULT) is previous
