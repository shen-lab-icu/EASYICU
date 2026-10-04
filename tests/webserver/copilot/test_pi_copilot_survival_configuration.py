"""The Host compiles a reviewed landmark survival proposal into the study's design.

A survival question whose study declares no survival design is planned as a
proposal of the landmark survival suite.  When its coordinates close, the
review reports ``SURVIVAL_LANDMARK_OWNER_NOT_SEALED`` for the runtime owner,
and "apply execution settings" declares exactly the published coordinates:
the survival family, the landmark paired with the endpoint's follow-up, and the
plan's reviewed roster.  The next candidate then binds the signed suite on its
zero-row catalog.  Synthetic study (renal replacement therapy, 90-day
mortality); zero patient rows.
"""

from __future__ import annotations

import pandas as pd
import pytest

from easyicu.research_agent.authority.current_case_scientific_runtime import (
    LandmarkSurvivalRuntimeAuthority,
    load_current_case_scientific_runtime_authority,
)
from easyicu.research_agent.planning.scientific_review import build_plan_scientific_review
from easyicu.research_agent.planning.sensitivity_authority import (
    normalize_prespecified_sensitivities,
)
from easyicu.webserver.landmark_survival_runtime_projection import (
    compile_landmark_survival_runtime_projection,
    survival_exposure_onset_column,
    validate_landmark_survival_declaration,
)
from easyicu.webserver.pi_copilot.plan_decisions import (
    PlanDecisionError,
    agent_plan_configuration_available,
    compile_agent_plan_configuration,
)
from easyicu.webserver.research_launch_scientific import (
    _metadata_planning_operationalized_columns,
)
from tests.support.survival_proposal import (
    AGE,
    SEX,
    proposed_survival_plan,
    survival_context,
    survival_request,
    survival_spec,
)

_OWNER = "SURVIVAL_LANDMARK_OWNER_NOT_SEALED"
_TIMING = "POST_BASELINE_EXPOSURE_TIMING_NOT_CLOSED"


def _reviewed_facts() -> dict:
    context = survival_context()
    variables = [
        item.model_copy(update={"analysis_window": "icu_admission[0,24]h"})
        if item.name == "rrt" else item
        for item in context.variables
    ]
    context = context.model_copy(update={"variables": variables})
    plan, _llm = proposed_survival_plan(context, survival_spec(survival_request(context), [AGE, SEX]))
    return dict(build_plan_scientific_review(context=context, plan=plan).facts)


_FACTS = _reviewed_facts()


def _study(**update) -> dict:
    study = {
        "question": "Is renal replacement therapy associated with 90-day mortality?",
        "cohort": {"preset": "all_icu"},
        "confirmations": {"feature_time_window": True},
        "sensitivity_specs": [],
    }
    study.update(update)
    return study


def _suite_facts(**update) -> dict:
    return {**_FACTS, "landmark_survival_suite": {**_FACTS["landmark_survival_suite"], **update}}


def _compile(study=None, codes=(_OWNER, _TIMING), facts=_FACTS):
    return compile_agent_plan_configuration(
        study=study if study is not None else _study(),
        agent_plan={"steps": []},
        runtime_finding_codes=codes,
        patient_cluster_available=False,
        review_facts=facts,
    )


def test_the_reviewed_coordinates_become_the_study_survival_design() -> None:
    compiled = _compile(codes=(_OWNER, _TIMING, "REPEATED_STAY_IDENTITY_UNAVAILABLE"))
    patch = compiled.patch

    assert patch["analysis_design"] == {
        "analysis_family": "survival",
        "analysis_unit": "icu_stay",
        "variance_estimator": "model_based",
    }
    [landmark] = patch["sensitivity_specs"]
    assert landmark == {
        "spec_id": "agent_plan_survival_landmark_24h",
        "axis": "timing",
        "strategy": "landmark",
        "landmark_hours": 24.0,
        "require_alive_at_landmark": True,
        "exclude_negative_event_times": True,
        "observation_duration_variable": "followup_days_90d",
        "observation_duration_unit": "days",
    }
    assert patch["execution_concepts"] == {
        "outcome": "mort_90d", "primary_exposure": "rrt", "covariates": ["age", "sex"],
    }
    assert (patch["covariates"], patch["covariate_selection"], patch["covariate_authority"]) == (
        ["age", "sex"], "exact", "agent_plan",
    )
    assert patch["covariate_rationales"] == {
        "age": AGE["clinical_rationale"], "sex": SEX["clinical_rationale"],
    }
    assert patch["covariate_temporal_roles"] == {"age": "baseline_static", "sex": "baseline_static"}
    # A study planned on the standing window declares it with its landmark.
    assert patch["time_window"]["hours"] == 24.0
    assert patch["confirmations"]["agent_plan_configuration_compiled"] is True
    assert patch["confirmations"]["plan_adjustment_set_confirmed"] is False
    # The suite fits one model-based row per stay: the dependence finding
    # stays a disclosed limitation instead of compiling clustering.
    assert compiled.runtime_finding_codes == (_TIMING, _OWNER)
    assert "cohort" not in patch
    # The suite owner accepts the declaration it will sign.
    assert validate_landmark_survival_declaration({**_study(), **patch}) == 24.0


def test_the_next_candidate_binds_the_signed_suite_on_its_zero_row_catalog(tmp_path) -> None:
    study = {**_study(), **_compile().patch}
    specs = normalize_prespecified_sensitivities(study["sensitivity_specs"])
    onset = survival_exposure_onset_column(study, sensitivity_specs=specs, primary_exposure_source="rrt")
    assert onset == "rrt_first_time"
    columns = _metadata_planning_operationalized_columns(
        primary_exposure_source="rrt",
        primary_exposure_aggregation=None,
        covariates=study["covariates"],
        covariate_selection="exact",
        covariate_operationalizations=study["covariate_operationalizations"],
        sensitivity_specs=specs,
        exposure_onset_column=onset,
    )
    assert {"rrt_first_time", "followup_days_90d", "age", "sex"} <= set(columns)
    catalog = tmp_path / "planner_catalog.parquet"
    pd.DataFrame(
        {name: pd.Series(dtype="float64") for name in ("stay_id", "rrt", "mort_90d", *columns)}
    ).to_parquet(catalog, index=False)

    projection = compile_landmark_survival_runtime_projection(
        study=study,
        sensitivity_specs=specs,
        primary_exposure="rrt",
        primary_exposure_source="rrt",
        target_outcome="mort_90d",
        declared_covariates=study["covariates"],
        covariate_operationalizations=study["covariate_operationalizations"],
        target_is_event_status=True,
        universe_path=catalog,
        scientific_configuration_sha256="d" * 64,
    )

    assert projection is not None
    authority = load_current_case_scientific_runtime_authority(projection.authority)
    assert isinstance(authority, LandmarkSurvivalRuntimeAuthority)
    assert (authority.exposure_status_column, authority.exposure_onset_column) == ("rrt", "rrt_first_time")
    assert (authority.event_column, authority.followup_time_column) == ("mort_90d", "followup_days_90d")
    assert (authority.landmark_hours, authority.endpoint_horizon_days) == (24.0, 90.0)
    assert authority.adjustment_columns == ("age", "sex")


def test_only_a_declared_survival_landmark_names_an_onset_column() -> None:
    study = {**_study(), **_compile().patch}
    specs = normalize_prespecified_sensitivities(study["sensitivity_specs"])
    assert survival_exposure_onset_column(_study(), sensitivity_specs=specs, primary_exposure_source="rrt") is None
    assert survival_exposure_onset_column(study, sensitivity_specs=(), primary_exposure_source="rrt") is None
    assert survival_exposure_onset_column(study, sensitivity_specs=specs, primary_exposure_source="") is None
    assert "rrt_first_time" not in _metadata_planning_operationalized_columns(
        primary_exposure_source="rrt", primary_exposure_aggregation=None, covariates=(),
        covariate_selection="planner_selectable", covariate_operationalizations={},
        sensitivity_specs=specs,
    )


def test_availability_follows_the_published_review_facts() -> None:
    def available(facts):
        return agent_plan_configuration_available(
            study=_study(), agent_plan={"steps": []},
            runtime_finding_codes=(_OWNER, _TIMING), review_facts=facts,
        )

    assert available(_FACTS)
    assert not available(None)
    assert not available(_suite_facts(executable=False))


@pytest.mark.parametrize(
    "facts",
    [
        None,
        {},
        _suite_facts(executable=False),
        {**_FACTS, "landmark_survival_suite": {"sealed": True}},
        # A tampered or stale fact still has to close from the host vocabulary.
        _suite_facts(event_column="death"),
        _suite_facts(followup_column="followup_days_28d"),
        _suite_facts(covariate_rationales={"age": AGE["clinical_rationale"]}),
        _suite_facts(covariate_temporal_roles={}),
        _suite_facts(covariates=[], covariate_rationales={}, covariate_temporal_roles={}),
    ],
    ids=["none", "absent", "not_executable", "sealed", "no_horizon", "unpaired", "rationale", "timing", "empty"],
)
def test_coordinates_the_review_did_not_publish_are_never_guessed(facts) -> None:
    with pytest.raises(PlanDecisionError) as raised:
        _compile(facts=facts)
    assert raised.value.code == "agent_plan_survival_coordinates_unavailable"


def test_the_landmark_must_end_the_window_the_plan_was_made_on() -> None:
    with pytest.raises(PlanDecisionError) as raised:
        _compile(study=_study(time_window={"hours": 48, "anchor": "icu_admission"}))
    assert raised.value.code == "agent_plan_landmark_not_compilable"
    assert raised.value.details == {"landmark_hours": 24.0, "time_window_hours": 48.0}

    declared = _compile(study=_study(time_window={"hours": 24, "anchor": "icu_admission"}))
    assert "time_window" not in declared.patch


def test_a_declared_design_is_not_replaced_and_other_runtime_gaps_are_not_absorbed() -> None:
    declared = _study(analysis_design={
        "analysis_family": "survival", "analysis_unit": "icu_stay", "variance_estimator": "model_based",
    })
    with pytest.raises(PlanDecisionError) as raised:
        _compile(study=declared)
    assert raised.value.code == "agent_plan_survival_design_already_declared"

    with pytest.raises(PlanDecisionError) as raised:
        _compile(codes=(_OWNER, "PRIMARY_POPULATION_EXECUTION_OWNER_MISSING"))
    assert raised.value.code == "agent_plan_runtime_finding_unsupported"
    assert raised.value.details == {"finding_codes": ["PRIMARY_POPULATION_EXECUTION_OWNER_MISSING"]}


_CLUSTERED = {"analysis_unit": "icu_stay", "variance_estimator": "cluster_robust", "cluster_unit": "patient"}


@pytest.mark.parametrize(
    "commitment",
    [
        {"analysis_design": dict(_CLUSTERED)},
        {"confirmations": {"feature_time_window": True, "plan_repeated_stays_clustered": True}},
        {
            "analysis_design": dict(_CLUSTERED),
            "confirmations": {"feature_time_window": True, "plan_repeated_stays_clustered": True},
        },
    ],
    ids=["design", "confirmation", "both"],
)
def test_a_patient_clustered_commitment_is_refused_not_downgraded(commitment) -> None:
    """The suite fits one model-based row per stay; it must not overwrite clustering.

    The compile used to write a model-based design over the study's
    patient-clustered one and leave its confirmation behind.
    """

    with pytest.raises(PlanDecisionError) as raised:
        _compile(
            study=_study(**commitment),
            codes=(_OWNER, _TIMING, "REPEATED_STAY_IDENTITY_UNAVAILABLE"),
        )

    assert raised.value.code == "agent_plan_survival_clustering_unexecutable"


def test_a_study_without_a_clustering_commitment_still_compiles() -> None:
    unclustered = _study(confirmations={"feature_time_window": True, "plan_repeated_stays_clustered": False})

    patch = _compile(study=unclustered).patch

    assert patch["analysis_design"]["variance_estimator"] == "model_based"


def test_a_first_stay_study_keeps_its_population() -> None:
    from easyicu.webserver import primary_cohort

    cohort = {"preset": "all_icu", "exclude_readmissions": True}
    if not primary_cohort.first_icu_stay_only(cohort):
        pytest.skip("the cohort owner reads first-stay restriction from another field")
    compiled = compile_agent_plan_configuration(
        study=_study(cohort=cohort), agent_plan={"steps": []},
        runtime_finding_codes=(_OWNER, _TIMING, "REPEATED_STAY_IDENTITY_UNAVAILABLE"),
        patient_cluster_available=False, first_stay_coordinate_available=True,
        review_facts=_FACTS,
    )
    assert compiled.patch["confirmations"]["plan_repeated_stays_first"] is True
    assert "cohort" not in compiled.patch


def test_the_workflow_offers_the_compile_for_a_closing_proposal() -> None:
    from easyicu.webserver import study_contexts as study_context_owner
    from easyicu.webserver.pi_copilot import workflow as workflow_owner
    from easyicu.webserver.pi_copilot.workflow import build_research_workflow_snapshot
    from tests.webserver.copilot.research_workflow_fixtures import complete_study

    def enriched(facts):
        study = complete_study()
        for key in ("analysis_design", "sensitivity_specs"):
            study.pop(key, None)
        digest = study_context_owner.scientific_configuration_sha256(study)
        review = {
            "status": "changes_required",
            "approval_allowed": False,
            "findings": [
                {
                    "code": code, "severity": "blocker", "remediation_route": "runtime_capability",
                    "message": "The survival plan names an unsealed suite.", "remediation": "Compile.",
                }
                for code in (_OWNER, _TIMING)
            ],
            "facts": {**facts, "remediation_buckets": {"runtime_capability": [_OWNER, _TIMING]}},
        }
        snapshot = build_research_workflow_snapshot(
            study=study, active_export_present=True, active_job=None,
            latest_run={
                "run_type": "full", "run_id": "run-survival-candidate", "study_id": study["id"],
                "engine": "easyicu.research_agent.pipeline", "gate_status": "blocked",
                "gate_reason": "human_plan_review_required", "run_status": "human_review_pending",
                "pending_review_reason_codes": ["plan_scientific_changes_required"],
                "scientific_configuration_sha256": digest,
                "artifact_names": ["agent_plan.json", "scientific_plan_review.json", "source_run_manifest.json"],
            },
            plan_review_authority={
                "run_id": "run-survival-candidate", "resumable_here": True,
                "scientific_configuration_sha256": digest, "budget_mode": "planner_canary",
                "research_input_state": "metadata_only", "plan_approval_allowed": False,
                "scientific_plan_review": review,
            },
        )
        return workflow_owner._enrich_plan_review(
            snapshot, study=study,
            review={"artifact_payloads": {"agent_plan.json": {"steps": []}, "scientific_plan_review.json": review}},
        ).next_action_code

    assert enriched(_FACTS) == "agent_plan_configuration_required"
    assert enriched(_suite_facts(executable=False)) == "plan_scientific_changes_required"
