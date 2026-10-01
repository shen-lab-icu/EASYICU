"""The synthetic survival study after review, sealed on the signed suite.

Test modules may not import one another (``tests/governance/test_test_organization.py``).
These helpers follow the Web path from the reviewed proposal of
``tests.support.survival_proposal`` (renal replacement therapy and 90-day
mortality): compile the reviewed design, sign the suite on a zero-row catalog,
build the sealed replan context, and plan it through the host binding.  Zero
patient rows.
"""

from __future__ import annotations

import json
from types import SimpleNamespace

import numpy as np
import pandas as pd

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
from easyicu.research_agent.execution.runners.landmark_survival_executor import (
    run_landmark_survival_suite,
)
from easyicu.research_agent.orchestration.scientific_runtime import ScientificRuntimeAuthorities
from easyicu.research_agent.planning import figure_plan_shaping as _figure_plan
from easyicu.research_agent.planning import final_plan_shape as _final_plan
from easyicu.research_agent.planning.baseline_requirements import bind_baseline_requirements
from easyicu.research_agent.planning.dependence_authority import bind_context_dependence_authority
from easyicu.research_agent.planning.family_spec import build_family_spec_request
from easyicu.research_agent.planning.scientific_review import build_plan_scientific_review
from easyicu.research_agent.planning.sensitivity_authority import normalize_prespecified_sensitivities
from easyicu.research_agent.providers.mocks import ScriptedMockLLMClient
from easyicu.research_agent.schema import (
    ConceptDescriptor,
    MissingnessProfile,
    UserPreferences,
    VariableRole,
)
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

STUDY = {
    "question": "Is renal replacement therapy associated with 90-day mortality?",
    "cohort": {"preset": "all_icu"},
    "confirmations": {"feature_time_window": True},
    "sensitivity_specs": [],
}


def reviewed_patch() -> dict:
    """Review the proposed suite and compile its coordinates as the Web does."""

    context = survival_context()
    variables = [
        item.model_copy(update={"analysis_window": "icu_admission[0,24]h"}) if item.name == "rrt" else item
        for item in context.variables
    ]
    context = context.model_copy(update={"variables": variables})
    plan, _llm = proposed_survival_plan(context, survival_spec(survival_request(context), [AGE, SEX]))
    facts = dict(build_plan_scientific_review(context=context, plan=plan).facts)
    return compile_agent_plan_configuration(
        study=dict(STUDY), agent_plan={"steps": []},
        runtime_finding_codes=("SURVIVAL_LANDMARK_OWNER_NOT_SEALED", "POST_BASELINE_EXPOSURE_TIMING_NOT_CLOSED"),
        patient_cluster_available=False, review_facts=facts,
    ).patch


def sealed_survival(
    tmp_path,
    *,
    review_only_specs: tuple[dict, ...] = (),
    unevenly_measured: str | None = None,
):
    """The signed authority and the sealed replan context.

    ``review_only_specs`` reach the review context but not the signed
    projection, which itself refuses a second landmark declaration.
    ``unevenly_measured`` gives that variable the medium missingness a prepared
    extract reports, which asks the article for data-quality evidence.
    """

    study = {**STUDY, **reviewed_patch()}
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
    variables = [
        *base.variables,
        ConceptDescriptor(name="rrt_first_time", description="first recorded therapy time",
                          role=VariableRole.OTHER, dtype="float64", unit="hours"),
        ConceptDescriptor(name="weight", description="admission weight",
                          role=VariableRole.DEMOGRAPHIC, dtype="float64", unit="kg"),
    ]
    if unevenly_measured is not None:
        variables = [
            item.model_copy(update={"missingness": MissingnessProfile(
                fraction_missing=0.2, n_missing=0, n_total=0, missingness_severity="medium",
            )}) if item.name == unevenly_measured else item
            for item in variables
        ]
    context = base.model_copy(update={
        "variables": variables,
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


def accepting(context, rows: list[str], group: str | None = "rrt"):
    """The context with the reviewed candidate's Table 1 rows bound."""

    return bind_baseline_requirements(context, {
        "schema_version": "easyicu.accepted_baseline_requirements/2",
        "source_plan_sha256": "c" * 64,
        "tables": [{
            "source_step_id": "baseline_context",
            "group_by": {"name": group, "source_concept": None} if group else None,
            "variables": [{"name": name, "source_concept": None} for name in rows],
        }],
    })


def sealed_request(context, authorities: ScientificRuntimeAuthorities):
    return build_family_spec_request(
        context, analysis_types=candidate_analysis_types(context),
        variable_roster=select_progressive_variables(context),
        allowed_literature_citation_keys=ALLOWED, required_primary_cohort_selection_mode="all_input_rows",
        planning_contract_context=authorities.planning_contract_context(),
    )


def sealed_draft(context, authorities: ScientificRuntimeAuthorities):
    """The Planner's accepted draft on the sealed suite (one scripted call)."""

    request = sealed_request(context, authorities)
    spec = survival_spec(request, [])
    spec["roster_decision_note"] = "Sealed suite: the host owns every coordinate; labels only."
    llm = ScriptedMockLLMClient([json.dumps(spec)])
    return ProgressivePlannerAgent(llm).run_attempt(
        context, planner_strategy=FAMILY_SPEC_STRATEGY,
        allowed_literature_citation_keys=ALLOWED, direct_comparator_literature_keys=[],
        enforce_article_contract=True, article_contract_context=context,
        planning_contract_context=authorities.planning_contract_context(),
        required_primary_cohort_selection_mode="all_input_rows",
    ).output


def bound_survival_plan(context, authorities: ScientificRuntimeAuthorities):
    """The accepted draft through the host's shaping and signed binding."""

    plan = sealed_draft(context, authorities)
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


def synthetic_survival_rows(n: int = 600, *, sex_missing: int = 37) -> pd.DataFrame:
    """Seeded synthetic rows for the sealed study's suite; no patient data."""

    rng = np.random.default_rng(20261001)
    treated = rng.binomial(1, 0.4, size=n)
    onset = np.where(
        treated == 1,
        rng.choice([-2.0, 3.0, 9.0, 18.0, 30.0], size=n, p=[0.05, 0.4, 0.3, 0.2, 0.05]),
        np.nan,
    )
    died = rng.binomial(1, 1.0 / (1.0 + np.exp(-(-1.2 + 0.7 * treated))), size=n)
    sex = rng.binomial(1, 0.5, size=n).astype(float)
    sex[rng.choice(n, size=sex_missing, replace=False)] = np.nan
    return pd.DataFrame({
        "rrt": treated.astype("int64"),
        "rrt_first_time": onset,
        "mort_90d": died.astype("int64"),
        "followup_days_90d": np.where(died == 1, rng.uniform(1.5, 89.0, size=n), 90.0),
        "age": rng.normal(64.0, 13.0, size=n),
        "sex": sex,
    })


def run_signed_suite(authority, frame: pd.DataFrame, out_dir):
    """Execute the signed suite on synthetic rows as the host runner does."""

    return run_landmark_survival_suite(
        frame=frame, authority=authority, runtime_projection_sha256="e" * 64, out_dir=out_dir,
        input_product="table:analysis_cohort", input_evidence_id="cohort", input_sha256="b" * 64,
    )
