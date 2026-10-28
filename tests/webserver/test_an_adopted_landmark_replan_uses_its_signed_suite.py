"""An adopted landmark survives the raw-source catalog and replan boundary.

Zero-row synthetic catalogs only. Unlike a prepared context with a manually
declared stay_id, these contexts obtain their identity from the source owner.
"""

import json

import pytest

from easyicu.research_agent.agents.family_spec_planner import family_spec_fallback_reason
from easyicu.research_agent.agents.progressive_planner import (
    FAMILY_SPEC_STRATEGY,
    ProgressivePlannerAgent,
    candidate_analysis_types,
)
from easyicu.research_agent.authority.current_case_scientific_runtime import (
    load_current_case_scientific_runtime_authority,
)
from easyicu.research_agent.planning.family_spec import LANDMARK_SURVIVAL_FAMILY_ID
from easyicu.research_agent.planning.family_spec.request import SEALED_CONTINUOUS_SURVIVAL_SUITE_MARKER
from easyicu.research_agent.planning.progressive_contract import ProgressivePlanCompileError
from easyicu.research_agent.providers.mocks import ScriptedMockLLMClient
from easyicu.research_agent.research_context.builder import build_research_context
from easyicu.research_agent.schema import TimeWindow, UserPreferences
from easyicu.webserver import agent_pipeline_runs
from easyicu.webserver.pi_copilot.survival_plan_decision import (
    OPTION_ID,
    compile_survival_landmark_choice,
    propose_survival_landmark,
)
from easyicu.webserver.scientific_runtime_projection import compile_web_scientific_runtime_projection
from tests.support.survival_proposal import ALLOWED, survival_request
from tests.support.survival_sealed import bound_survival_plan


def _adopted_raw_context(tmp_path, database):
    study = {
        "id": "synthetic-renal-support", "revision": 1,
        "question": "Does early renal support predict 90-day survival, adjusted for age and sex?",
        "data_source": {"database": database},
        "analysis_design": {
            "analysis_family": "survival", "analysis_unit": "icu_stay", "variance_estimator": "model_based",
        },
        "time_window": {"hours": 24, "anchor": "icu_admission"},
        "execution_concepts": {
            "primary_exposure": "rrt", "outcome": "mort_90d", "covariates": ["age", "sex"],
        },
        "covariates": ["age", "sex"], "covariate_selection": "exact", "covariate_authority": "user",
        "sensitivity_specs": [],
    }
    card = propose_survival_landmark(study)
    assert card is not None
    choice = compile_survival_landmark_choice(
        study=study, option_id=OPTION_ID, decision_card=card.model_dump(mode="json"), card_sha256=card.card_sha256,
    )
    assert choice.next_action == "replan"
    adopted = {**study, **choice.patch}
    preferences = UserPreferences.model_validate(agent_pipeline_runs._research_user_preferences(adopted))
    acquisition = agent_pipeline_runs._metadata_only_planning_acquisition(
        database=database, question=study["question"],
        llm=ScriptedMockLLMClient([json.dumps({"selected_concepts": ["rrt", "mort_90d", "age", "sex"]})]),
        output_dir=tmp_path / "catalog", target_outcome="mort_90d",
        required_concepts=("rrt", "mort_90d", "followup_days_90d", "age", "sex"),
        operationalized_columns=("rrt_onset_time",),
        required_coordinates=("rrt", "mort_90d"),
    )
    assert not acquisition.blocked
    projection = compile_web_scientific_runtime_projection(
        study=adopted, sensitivity_specs=preferences.sensitivity_specs,
        primary_exposure="rrt", primary_exposure_source="rrt", target_outcome="mort_90d",
        declared_covariates=adopted["covariates"],
        covariate_operationalizations=adopted["covariate_operationalizations"], target_is_event_status=True,
        universe_path=acquisition.universe_path, scientific_configuration_sha256="e" * 64,
    )
    authority = load_current_case_scientific_runtime_authority(projection.authority)
    context = build_research_context(
        research_question=study["question"], cohort=acquisition.universe_path,
        cohort_name="synthetic-raw-catalog", database=database,
        primary_exposure="rrt", target_outcome="mort_90d", endpoint=authority.research_context_endpoint(),
        time_windows=[TimeWindow(name="first_day", anchor="icu_admission", start_hours=0, end_hours=24)],
        user_preferences=preferences,
    )
    return context, authority


@pytest.mark.parametrize("database,identity", [("sic", "CaseID"), ("miiv", "stay_id")])
def test_adopting_the_card_binds_the_signed_replan_from_a_raw_catalog(tmp_path, database, identity):
    from easyicu.research_agent.orchestration.scientific_runtime import ScientificRuntimeAuthorities

    context, authority = _adopted_raw_context(tmp_path, database)
    assert context.cohort.n_stays == 0
    assert context.cohort.id_columns == [identity]
    assert context.variable(identity).role.value == "id"
    assert context.cohort.provenance["analysis_unit"] == "icu_stay"
    assert context.cohort.provenance["stay_id_columns"] == [identity]
    disclosure = authority.planning_contract_context()
    request = survival_request(context, disclosure)
    assert request.family_id == LANDMARK_SURVIVAL_FAMILY_ID
    assert request.identity_column == identity
    assert request.proposed_suite is None
    assert request.sealed_suite.primary_owner == "signed_landmark_survival_suite"
    assert request.exact_roster == ["age", "sex"]
    plan = bound_survival_plan(context, ScientificRuntimeAuthorities(trajectory=None, current_case=authority))
    assert next(step for step in plan.steps if step.planned_analysis_role == "primary").method == authority.plan_method


@pytest.mark.parametrize("case,path", [
    ("no_identity", "cohort.id_columns"), ("two_identities", "cohort.id_columns"),
    ("identity_not_published", "cohort.id_columns"),
    ("no_followup", "variables.followup_days_90d"), ("no_onset", "variables.rrt_onset_time"),
    ("no_covariate", "variables.age"), ("wrong_exposure", "primary_exposure"),
    ("wrong_endpoint", "target_outcome"), ("no_signature", "planning_contract_context.sealed_suite"),
    ("two_signatures", "planning_contract_context.sealed_suite"), ("nonbinary_event", "variables"),
])
def test_an_unresolved_adopted_coordinate_stops_typed_before_the_provider(tmp_path, case, path):
    context, authority = _adopted_raw_context(tmp_path, "miiv")
    disclosure = authority.planning_contract_context()
    if case in {"no_identity", "two_identities"}:
        context = context.model_copy(update={"cohort": context.cohort.model_copy(update={
            "id_columns": [] if case == "no_identity" else ["stay_id", "age"],
        })})
    elif case in {"identity_not_published", "no_followup", "no_onset", "no_covariate"}:
        missing = {"identity_not_published": "stay_id", "no_followup": "followup_days_90d",
                   "no_onset": "rrt_onset_time", "no_covariate": "age"}[case]
        context = context.model_copy(update={"variables": [v for v in context.variables if v.name != missing]})
    elif case == "wrong_exposure":
        context = context.model_copy(update={"primary_exposure": "sex"})
    elif case == "wrong_endpoint":
        context = context.model_copy(update={"target_outcome": "rrt"})
    elif case == "nonbinary_event":
        context = context.model_copy(update={"variables": [
            v.model_copy(update={"observed_domain": {"n_unique": 3, "levels": [0, 1, 2]}})
            if v.name == "mort_90d" else v for v in context.variables
        ]})
    elif case == "two_signatures":
        disclosure += "\n" + SEALED_CONTINUOUS_SURVIVAL_SUITE_MARKER + "\n" + json.dumps({
            "sealed_primary_owner": "signed_landmark_continuous_survival_suite",
            "exposure_column": "bili_max", "exposure_window_summary": "max",
            "exposure_window_hours": [0, 24], "exposure_increment_rule": "largest_round_step_within_interquartile_range",
            "event_column": "mort_90d", "followup_time_column": "followup_days_90d",
            "landmark_hours": 24, "endpoint_horizon_days": 90,
            "plan_outputs": ["table:landmark_continuous_cox_summary"],
        })
    else:
        disclosure = ""
    llm = ScriptedMockLLMClient([])
    with pytest.raises(ProgressivePlanCompileError) as caught:
        ProgressivePlannerAgent(llm).run_attempt(
            context, planner_strategy=FAMILY_SPEC_STRATEGY, planning_contract_context=disclosure,
            allowed_literature_citation_keys=ALLOWED,
        )
    assert caught.value.reason_code == (
        "progressive_identity_column_ambiguous" if "identit" in case
        else "progressive_family_spec_survival_coordinate_unavailable"
    )
    assert caught.value.path == path
    assert caught.value.easyicu_safe_diagnostic["metrics"]["planner_provider_calls"] == 0
    assert llm.calls == []


def test_a_signed_context_with_no_identifier_cannot_yield_to_an_outline(tmp_path):
    context, authority = _adopted_raw_context(tmp_path, "miiv")
    context = context.model_copy(update={"cohort": context.cohort.model_copy(update={"id_columns": []})})
    with pytest.raises(ProgressivePlanCompileError, match="identity"):
        family_spec_fallback_reason(
            context, analysis_types=candidate_analysis_types(context), resume_checkpoint=None,
            stop_after_outline=False, planning_contract_context=authority.planning_contract_context(),
        )


def test_an_outline_strategy_stops_before_recomposing_the_adopted_signed_suite(tmp_path):
    context, authority = _adopted_raw_context(tmp_path, "miiv")
    llm = ScriptedMockLLMClient([])
    with pytest.raises(ProgressivePlanCompileError) as caught:
        ProgressivePlannerAgent(llm).run_attempt(
            context, planner_strategy="progressive_v2", planning_contract_context=authority.planning_contract_context(),
        )
    assert caught.value.reason_code == "progressive_family_template_required"
    assert llm.calls == []
