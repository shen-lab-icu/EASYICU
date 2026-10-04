"""A proposed survival design offers the covariates measured before its landmark.

A survival question with no declared design reaches the signed suite through a
host proposal.  The proposal's candidate roster was timed against the study's
declared landmark, which does not exist before review, so only the owner-
declared demographics were ever selectable: every auto-compiled suite adjusted
for age, sex and admission type alone, whatever severity measures the cohort
carried.  The proposal now times its candidates against the landmark it
proposes, with the same window rule a sealed suite uses.

Synthetic study (renal replacement therapy, 90-day mortality); zero patient rows.
"""

from __future__ import annotations

import pandas as pd

from easyicu.research_agent.authority.current_case_scientific_runtime import (
    load_current_case_scientific_runtime_authority,
)
from easyicu.research_agent.planning.adjustment_authority import host_proven_temporal_roles
from easyicu.research_agent.planning.scientific_review import build_plan_scientific_review
from easyicu.research_agent.planning.sensitivity_authority import normalize_prespecified_sensitivities
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
    SEX,
    proposed_survival_plan,
    survival_context,
    survival_request,
    survival_spec,
)

LACTATE = {
    "name": "lactate_first_day", "coding": "continuous", "reference_level_index": None,
    "clinical_rationale": "Lactate over the first ICU day reflects shock severity, which drives both "
                          "therapy and death.",
}


def _with_measurements(**preferences):
    base = survival_context()
    variables = [
        *base.variables,
        ConceptDescriptor(name="lactate_first_day", description="lactate over the first ICU day",
                          role=VariableRole.LAB, dtype="float64", unit="mmol/L",
                          analysis_window="icu_admission[0,24]h"),
        ConceptDescriptor(name="severity_score_first_day", description="organ failure score, first ICU day",
                          role=VariableRole.COMPOSITE_SCORE, dtype="float64",
                          analysis_window="icu_admission[0,24]h"),
        ConceptDescriptor(name="creatinine_two_days", description="creatinine over the first two ICU days",
                          role=VariableRole.LAB, dtype="float64", unit="mg/dL",
                          analysis_window="icu_admission[0,48]h"),
    ]
    update = {"variables": variables}
    if preferences:
        update["user_preferences"] = UserPreferences(inferred_analysis_family="survival", **preferences)
    return base.model_copy(update=update)


def _candidates(request):
    return {item.name: item for item in request.adjustment_candidates}


def test_a_measurement_closed_by_the_proposed_landmark_is_a_selectable_baseline_covariate():
    context = _with_measurements()
    # The study declares no landmark: the declared-landmark timing proves nothing dynamic.
    assert "lactate_first_day" not in host_proven_temporal_roles(context)

    request = survival_request(context)
    assert request.proposed_suite is not None
    assert request.proposed_suite.landmark_hours == 24.0
    candidates = _candidates(request)

    for name in ("lactate_first_day", "severity_score_first_day"):
        assert candidates[name].selectable is True
        assert candidates[name].host_temporal_role == "at_or_before_time_zero"
    for name in ("age", "sex"):
        assert candidates[name].selectable is True
        assert candidates[name].host_temporal_role == "baseline_static"


def test_a_measurement_whose_window_outlasts_the_landmark_is_still_not_selectable():
    candidates = _candidates(survival_request(_with_measurements()))

    late = candidates["creatinine_two_days"]
    assert late.selectable is False
    assert late.host_temporal_role is None
    assert late.boundary.startswith("pre-time-zero availability is not provable by the host")


def test_the_proposed_landmark_is_used_only_for_the_proposal():
    """The explicit reference does not leak into the study's own timing authority."""

    context = _with_measurements()
    survival_request(context)

    assert host_proven_temporal_roles(context) == {"age": "baseline_static", "sex": "baseline_static"}
    assert host_proven_temporal_roles(context, reference_hours=24.0)["lactate_first_day"] == (
        "at_or_before_time_zero"
    )


def test_a_selected_measurement_reaches_the_signed_suite_roster(tmp_path):
    """Review, the Web compile and the signed projection keep the Planner's pick."""

    context = _with_measurements()
    exposure_window = [
        item.model_copy(update={"analysis_window": "icu_admission[0,24]h"}) if item.name == "rrt" else item
        for item in context.variables
    ]
    context = context.model_copy(update={"variables": exposure_window})
    plan, _llm = proposed_survival_plan(context, survival_spec(survival_request(context), [AGE, SEX, LACTATE]))
    facts = dict(build_plan_scientific_review(context=context, plan=plan).facts)
    suite = facts["landmark_survival_suite"]
    assert suite["covariates"] == ["age", "sex", "lactate_first_day"]
    assert suite["covariate_temporal_roles"]["lactate_first_day"] == "at_or_before_time_zero"

    study = {
        "question": "Is renal replacement therapy associated with 90-day mortality?",
        "cohort": {"preset": "all_icu"},
        "confirmations": {"feature_time_window": True},
        "sensitivity_specs": [],
    }
    study = {**study, **compile_agent_plan_configuration(
        study=dict(study), agent_plan={"steps": []},
        runtime_finding_codes=("SURVIVAL_LANDMARK_OWNER_NOT_SEALED", "POST_BASELINE_EXPOSURE_TIMING_NOT_CLOSED"),
        patient_cluster_available=False, review_facts=facts,
    ).patch}
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
        {name: pd.Series(dtype="float64") for name in ("stay_id", "rrt", "mort_90d", *columns)}
    ).to_parquet(catalog, index=False)
    projection = compile_landmark_survival_runtime_projection(
        study=study, sensitivity_specs=specs, primary_exposure="rrt", primary_exposure_source="rrt",
        target_outcome="mort_90d", declared_covariates=study["covariates"],
        covariate_operationalizations=study["covariate_operationalizations"],
        target_is_event_status=True, universe_path=catalog, scientific_configuration_sha256="d" * 64,
    )

    assert projection is not None
    authority = load_current_case_scientific_runtime_authority(projection.authority)
    assert authority.adjustment_columns == ("age", "sex", "lactate_first_day")
