"""A reviewed survival design replans on the signed suite with its accepted Table 1.

After review compiles a proposed survival design, the next candidate is planned
on the signed landmark survival suite and keeps the reviewed candidate's
Table 1 rows.  Two host owners still read the signed suite as if it had no
Table 1 and no design of its own:
- the family-spec request refused any accepted roster for a sealed suite
  before the Provider call, although the suite describes its adjustment
  columns by exposure status;
- the plan review looked for those rows only in a plan Table 1 step, and read
  the compiled landmark design the suite executes as a user-required
  sensitivity analysis left protocol-only.
Synthetic study only (renal replacement therapy and 90-day mortality); zero
patient rows.
"""

from __future__ import annotations

import json
from types import SimpleNamespace

import pandas as pd
import pytest

from easyicu.research_agent import pipeline as _pipeline
from easyicu.research_agent.agents.progressive_planner import (
    FAMILY_SPEC_STRATEGY,
    ProgressivePlannerAgent,
    candidate_analysis_types,
    select_progressive_variables,
)
from easyicu.research_agent.authority.current_case_scientific_runtime import (
    LandmarkSurvivalRuntimeAuthority,
    load_current_case_scientific_runtime_authority,
)
from easyicu.research_agent.orchestration.scientific_runtime import ScientificRuntimeAuthorities
from easyicu.research_agent.planning import figure_plan_shaping as _figure_plan
from easyicu.research_agent.planning import final_plan_shape as _final_plan
from easyicu.research_agent.planning.baseline_requirements import bind_baseline_requirements
from easyicu.research_agent.planning.dependence_authority import bind_context_dependence_authority
from easyicu.research_agent.planning.family_spec import FamilySpecError, build_family_spec_request
from easyicu.research_agent.planning.scientific_review import build_plan_scientific_review
from easyicu.research_agent.planning.sensitivity_authority import normalize_prespecified_sensitivities
from easyicu.research_agent.providers.mocks import ScriptedMockLLMClient
from easyicu.research_agent.schema import ConceptDescriptor, UserPreferences, VariableRole
from easyicu.webserver.landmark_survival_runtime_projection import (
    compile_landmark_survival_runtime_projection,
    survival_exposure_onset_column,
)
from easyicu.webserver.pi_copilot.plan_decisions import compile_agent_plan_configuration
from easyicu.webserver.research_launch_scientific import (
    _metadata_planning_operationalized_columns,
)
from tests.support.survival_proposal import (
    AGE,
    ALLOWED,
    SEX,
    proposed_survival_plan,
    survival_context,
    survival_request,
    survival_spec,
)

BASELINE = "ACCEPTED_BASELINE_CONTENT_MISSING"
PROTOCOL_ONLY = "REQUIRED_SENSITIVITY_IS_PROTOCOL_ONLY"
_STUDY = {
    "question": "Is renal replacement therapy associated with 90-day mortality?",
    "cohort": {"preset": "all_icu"},
    "confirmations": {"feature_time_window": True},
    "sensitivity_specs": [],
}


def _reviewed_patch() -> dict:
    """Review the proposed suite and compile its coordinates as the web does."""

    context = survival_context()
    variables = [
        item.model_copy(update={"analysis_window": "icu_admission[0,24]h"}) if item.name == "rrt" else item
        for item in context.variables
    ]
    context = context.model_copy(update={"variables": variables})
    plan, _llm = proposed_survival_plan(context, survival_spec(survival_request(context), [AGE, SEX]))
    facts = dict(build_plan_scientific_review(context=context, plan=plan).facts)
    return compile_agent_plan_configuration(
        study=dict(_STUDY), agent_plan={"steps": []},
        runtime_finding_codes=("SURVIVAL_LANDMARK_OWNER_NOT_SEALED", "POST_BASELINE_EXPOSURE_TIMING_NOT_CLOSED"),
        patient_cluster_available=False, review_facts=facts,
    ).patch


def _sealed(tmp_path, *, review_only_specs: tuple[dict, ...] = ()):
    """The signed authority and the sealed replan context.

    ``review_only_specs`` reach the review context but not the signed
    projection, which itself refuses a second landmark declaration.
    """

    study = {**_STUDY, **_reviewed_patch()}
    specs = normalize_prespecified_sensitivities(study["sensitivity_specs"])
    onset = survival_exposure_onset_column(study, sensitivity_specs=specs, primary_exposure_source="rrt")
    columns = _metadata_planning_operationalized_columns(
        primary_exposure_source="rrt", primary_exposure_aggregation=None,
        covariates=study["covariates"], covariate_selection="exact",
        covariate_operationalizations=study["covariate_operationalizations"],
        sensitivity_specs=specs, exposure_onset_column=onset,
    )
    catalog = tmp_path / "planner_catalog.parquet"
    pd.DataFrame(
        {name: pd.Series(dtype="float64") for name in ("stay_id", "rrt", "mort_90d", "weight", *columns)}
    ).to_parquet(catalog, index=False)
    projection = compile_landmark_survival_runtime_projection(
        study=study, sensitivity_specs=specs, primary_exposure="rrt", primary_exposure_source="rrt",
        target_outcome="mort_90d", declared_covariates=study["covariates"],
        covariate_operationalizations=study["covariate_operationalizations"],
        target_is_event_status=True, universe_path=catalog, scientific_configuration_sha256="d" * 64,
    )
    authority = load_current_case_scientific_runtime_authority(projection.authority)
    assert isinstance(authority, LandmarkSurvivalRuntimeAuthority)
    base = survival_context()
    context = base.model_copy(update={
        "variables": [
            *base.variables,
            ConceptDescriptor(name="rrt_first_time", description="first recorded therapy time",
                              role=VariableRole.OTHER, dtype="float64", unit="hours"),
            ConceptDescriptor(name="weight", description="admission weight",
                              role=VariableRole.DEMOGRAPHIC, dtype="float64", unit="kg"),
        ],
        "endpoint": authority.research_context_endpoint(),
        "user_preferences": UserPreferences(
            inferred_analysis_family="survival", covariate_selection="exact",
            covariates=list(study["covariates"]), covariate_authority="agent_plan",
            covariate_rationales=dict(study["covariate_rationales"]),
            covariate_temporal_roles=dict(study["covariate_temporal_roles"]),
            sensitivity_specs=list(
                normalize_prespecified_sensitivities([*study["sensitivity_specs"], *review_only_specs])
            ),
        ),
    })
    return context, authority


def _accepting(context, rows: list[str], group: str | None = "rrt"):
    return bind_baseline_requirements(context, {
        "schema_version": "easyicu.accepted_baseline_requirements/2",
        "source_plan_sha256": "c" * 64,
        "tables": [{
            "source_step_id": "baseline_context",
            "group_by": {"name": group, "source_concept": None} if group else None,
            "variables": [{"name": name, "source_concept": None} for name in rows],
        }],
    })


def _request(context, authorities):
    return build_family_spec_request(
        context, analysis_types=candidate_analysis_types(context),
        variable_roster=select_progressive_variables(context),
        allowed_literature_citation_keys=ALLOWED, required_primary_cohort_selection_mode="all_input_rows",
        planning_contract_context=authorities.planning_contract_context(),
    )


def _bound_plan(context, authorities):
    request = _request(context, authorities)
    spec = survival_spec(request, [])
    spec["roster_decision_note"] = "Sealed suite: the host owns every coordinate; labels only."
    llm = ScriptedMockLLMClient([json.dumps(spec)])
    plan = ProgressivePlannerAgent(llm).run_attempt(
        context, planner_strategy=FAMILY_SPEC_STRATEGY,
        allowed_literature_citation_keys=ALLOWED, direct_comparator_literature_keys=[],
        enforce_article_contract=True, article_contract_context=context,
        planning_contract_context=authorities.planning_contract_context(),
        required_primary_cohort_selection_mode="all_input_rows",
    ).output
    findings: list = []
    fake = SimpleNamespace(
        _scientific_runtime_authorities=authorities, _enable_publication_figure_skill=True, _max_total_steps=24,
    )
    plan = _pipeline._shape_fresh_plan(
        pipeline=fake, plan=plan, context=context, agent_context=context,
        long_trajectory_bound=False, findings=findings,
    )
    plan = bind_context_dependence_authority(plan=plan, context=context)
    bound, _ = authorities.bind_plan(plan)
    bound = _figure_plan.apply_runtime_bound_figure_contracts(bound, findings)
    authorities.validate_plan(bound)
    _final_plan.validate_final_plan_shape(bound)
    return bound


def _codes(context, plan, authority):
    review = build_plan_scientific_review(
        context=context, plan=plan, require_reportable_capability=True, runtime_authority=authority,
    )
    return review, {finding.code for finding in review.findings if finding.severity == "blocker"}


def test_the_reviewed_table_one_survives_the_sealed_replan_to_an_approvable_plan(tmp_path):
    context, authority = _sealed(tmp_path)
    assert authority.table_one_columns == ("age", "sex")
    authorities = ScientificRuntimeAuthorities(trajectory=None, current_case=authority)
    accepting = _accepting(context, ["age", "sex"])

    request = _request(accepting, authorities)
    assert request.sealed_suite is not None and request.sealed_suite.adjustment_columns == ["age", "sex"]
    plan = _bound_plan(accepting, authorities)
    suite = next(step for step in plan.steps if step.method == "signed_landmark_survival_suite")
    assert authority.table_one_product in suite.expected_outputs
    assert not any(step.table_one_spec is not None for step in plan.steps)

    review, blockers = _codes(accepting, plan, authority)
    assert BASELINE not in blockers and PROTOCOL_ONLY not in blockers
    assert review.approval_allowed is True


def test_a_row_or_grouping_the_sealed_table_does_not_describe_is_refused_before_the_provider(tmp_path):
    context, authority = _sealed(tmp_path)
    authorities = ScientificRuntimeAuthorities(trajectory=None, current_case=authority)

    with pytest.raises(FamilySpecError) as caught:
        _request(_accepting(context, ["age", "weight"]), authorities)
    assert caught.value.reason_code == "family_spec_accepted_baseline_unsatisfiable"
    assert "'weight'" in str(caught.value) and "'age'" not in str(caught.value).split("not ")[-1]

    with pytest.raises(FamilySpecError) as caught:
        _request(_accepting(context, ["age"], group="sex"), authorities)
    assert caught.value.reason_code == "family_spec_accepted_baseline_grouping_unsupported"
    # An ungrouped roster of sealed columns is kept.
    assert _request(_accepting(context, ["sex"], group=None), authorities).sealed_suite is not None


def test_the_review_credits_only_the_design_and_table_the_signed_suite_owns(tmp_path):
    other = {
        "spec_id": "user_landmark_48h", "axis": "timing", "strategy": "landmark", "landmark_hours": 48.0,
        "require_alive_at_landmark": True, "exclude_negative_event_times": True,
        "observation_duration_variable": "followup_days_90d", "observation_duration_unit": "days",
    }
    context, authority = _sealed(tmp_path, review_only_specs=(other,))
    authorities = ScientificRuntimeAuthorities(trajectory=None, current_case=authority)
    accepting = _accepting(context, ["age", "sex"])
    plan = _bound_plan(accepting, authorities)

    review, blockers = _codes(accepting, plan, authority)
    # A second landmark the suite does not run is still a missing sensitivity.
    protocol_only = next(finding for finding in review.findings if finding.code == PROTOCOL_ONLY)
    assert "user_landmark_48h" in protocol_only.message
    assert "agent_plan_survival_landmark_24h" not in protocol_only.message
    # Without the signed authority nothing proves the suite's Table 1 roster.
    _review, unsigned = _codes(accepting, plan, None)
    assert BASELINE in unsigned
