"""A pre-plan survival stop offers a click, without inventing a reviewed plan.

Synthetic studies and zero-row catalogs only; no source data are opened.
"""

import copy
import json

import pandas as pd
import pytest

from easyicu.research_agent.agents.progressive_planner import ProgressivePlannerAgent
from easyicu.research_agent.planning.progressive_contract import ProgressivePlanCompileError
from easyicu.research_agent.providers.mocks import ScriptedMockLLMClient
from easyicu.webserver import agent_pipeline_runs
from easyicu.webserver.pi_copilot import plan_decisions
from easyicu.webserver.pi_copilot.survival_plan_decision import (
    DECISION_CODE,
    OPTION_ID,
    family_stop_result,
    pending_survival_decision,
    propose_survival_landmark,
)
from tests.support.survival_proposal import survival_context


def _study():
    return {
        "id": "synthetic-survival",
        "revision": 1,
        "question": "Does first-day renal support predict 90-day survival, adjusted for age and sex?",
        "data_source": {"database": "miiv"},
        "analysis_design": {
            "analysis_family": "survival", "analysis_unit": "icu_stay",
            "variance_estimator": "model_based",
        },
        "time_window": {"hours": 24, "anchor": "icu_admission"},
        "execution_concepts": {
            "primary_exposure": "rrt", "outcome": "mort_90d", "covariates": ["age", "sex"],
        },
        "covariates": ["age", "sex"],
        "covariate_selection": "exact",
        "covariate_authority": "user",
        "sensitivity_specs": [],
    }


def _stop():
    context = survival_context()
    # This is a pre-plan catalog without the paired follow-up. No template
    # can write the family contract, and the first Provider call is refused.
    context = context.model_copy(update={
        "variables": [v for v in context.variables if v.name != "followup_days_90d"],
    })
    llm = ScriptedMockLLMClient([])
    try:
        ProgressivePlannerAgent(llm).run_attempt(context, planner_strategy="progressive_v2")
    except ProgressivePlanCompileError as exc:
        assert exc.reason_code == "progressive_family_result_contract_unwritable"
        assert llm.calls == []
        return exc
    raise AssertionError("An unsealed family must stop before its Provider call")


def test_the_real_pre_plan_stop_persists_a_decision_without_adopting_it(tmp_path):
    study = _study()
    before = copy.deepcopy(study)
    exc = _stop()
    agent_pipeline_runs._record_pipeline_failure(
        wrapper_dir=tmp_path, study=study, provider={}, exc=exc,
        code=agent_pipeline_runs._pipeline_failure_code(exc), execution_retry_id=None,
    )
    manifest = json.loads((tmp_path / "source_run_manifest.json").read_text())
    assert manifest["status"] == "plan_decision_pending"
    decision = manifest["plan_decision"]
    assert decision["decision_code"] == "SURVIVAL_LANDMARK_OWNER_NOT_SEALED"
    assert decision["landmark_hours"] == 24
    assert decision["endpoint"] == "mort_90d"
    assert decision["horizon_days"] == 90
    assert decision["covariates"] == ["age", "sex"]
    assert len(decision["card_sha256"]) == 64
    assert study == before
    assert not (tmp_path / "agent_plan.json").exists()
    assert not (tmp_path / "scientific_plan_review.json").exists()
    assert family_stop_result(tmp_path)["status"] == "plan_decision_pending"


def _choose(study, card=None, **kwargs):
    card = card or propose_survival_landmark(study).model_dump(mode="json")
    return plan_decisions.compile_plan_decision(
        decision_code=DECISION_CODE, option_id=kwargs.pop("option_id", OPTION_ID),
        study=study, agent_plan={}, decision_card=card,
        card_sha256=kwargs.pop("card_sha256", card["card_sha256"]), **kwargs,
    )


@pytest.mark.parametrize("exposure,horizon,hours", [("rrt", 90, 24), ("abx", 28, 12), ("rrt", 365, 36)])
@pytest.mark.parametrize("rmst", [False, True])
def test_the_click_preserves_the_saved_roster_and_the_next_plan_binds_its_suite(
    tmp_path, exposure, horizon, hours, rmst,
):
    from easyicu.webserver.landmark_survival_runtime_projection import validate_landmark_survival_declaration
    from easyicu.webserver.scientific_runtime_projection import compile_web_scientific_runtime_projection
    from easyicu.research_agent.planning.sensitivity_authority import normalize_prespecified_sensitivities

    study = _study()
    study["question"] = f"Does {exposure} in the first {hours} hours predict {horizon}-day survival?"
    study["execution_concepts"].update(primary_exposure=exposure, outcome=f"mort_{horizon}d")
    study["time_window"]["hours"] = hours
    if rmst:
        study["question"] += f" Report the {horizon}-day RMST difference as a sensitivity analysis."
    study["covariate_rationales"] = {"age": "Saved age rationale"}
    study["covariate_temporal_roles"] = {"age": "baseline_static"}
    card = propose_survival_landmark(study)
    assert card is not None
    assert (card.exposure, card.landmark_hours, card.horizon_days) == (exposure, hours, horizon)
    assert plan_decisions.plan_decision_context({}, DECISION_CODE, study) == card.model_dump(mode="json")
    compiled = _choose(study)
    adopted = {**study, **compiled.patch}
    assert compiled.next_action == "replan"
    assert adopted["question"] == study["question"]
    assert adopted["covariate_authority"] == "user"
    assert adopted["covariates"] == ["age", "sex"]
    assert adopted["covariate_rationales"] == study["covariate_rationales"]
    assert adopted["covariate_temporal_roles"] == study["covariate_temporal_roles"]
    assert validate_landmark_survival_declaration(adopted) == hours
    assert plan_decisions.decision_is_resolved(adopted, DECISION_CODE)
    specs = normalize_prespecified_sensitivities(adopted["sensitivity_specs"])
    assert [s.strategy for s in specs] == ["landmark", *(["restricted_mean_survival"] if rmst else [])]
    columns = ["stay_id", exposure, f"{exposure}_onset_time", f"mort_{horizon}d", f"followup_days_{horizon}d", "age", "sex"]
    catalog = tmp_path / "zero_rows.parquet"
    pd.DataFrame({name: pd.Series(dtype="float64") for name in columns}).to_parquet(catalog)
    projection = compile_web_scientific_runtime_projection(
        study=adopted, sensitivity_specs=specs, primary_exposure=exposure,
        primary_exposure_source=exposure, target_outcome=f"mort_{horizon}d",
        declared_covariates=adopted["covariates"],
        covariate_operationalizations=adopted["covariate_operationalizations"],
        target_is_event_status=True, universe_path=catalog,
        scientific_configuration_sha256="d" * 64,
    )
    assert projection.authority["plan_method"] == "signed_landmark_survival_suite"
    assert projection.authority["adjustment_columns"] == ["age", "sex"]
    assert projection.authority["landmark_hours"] == hours
    assert projection.authority["endpoint_horizon_days"] == horizon
    if rmst:
        alternative = next(s.rmst_execution for s in specs if s.rmst_execution)
        assert alternative.tau == horizon - hours / 24
        assert alternative.time_column == projection.authority["derived_time_column"]
        assert projection.authority["rmst_product"] == "table:landmark_rmst_summary"
        if (exposure, horizon, hours) == ("rrt", 90, 24):
            from easyicu.research_agent.authority.current_case_scientific_runtime import load_current_case_scientific_runtime_authority
            from easyicu.research_agent.orchestration.scientific_runtime import ScientificRuntimeAuthorities
            from easyicu.research_agent.planning.primary_result_contract import validate_required_primary_result
            from easyicu.research_agent.schema import ConceptDescriptor, VariableRole
            from tests.support.survival_sealed import bound_survival_plan

            authority = load_current_case_scientific_runtime_authority(projection.authority)
            base = survival_context()
            context = base.model_copy(update={
                "research_question": adopted["question"],
                "variables": [*base.variables, ConceptDescriptor(
                    name="rrt_onset_time", role=VariableRole.OTHER, dtype="float64", unit="hours",
                )],
                "endpoint": authority.research_context_endpoint(),
                "user_preferences": base.user_preferences.model_copy(update={
                    "covariate_selection": "exact", "covariates": adopted["covariates"],
                    "covariate_authority": "user", "sensitivity_specs": list(specs),
                }),
            })
            plan = bound_survival_plan(context, ScientificRuntimeAuthorities(trajectory=None, current_case=authority))
            assert [step.method for step in plan.steps] == [
                "host_materialized_locked_cohort", "signed_landmark_survival_suite", "signed_landmark_survival_figure",
            ]
            validate_required_primary_result(plan=plan, context=context)


@pytest.mark.parametrize("change", [
    "no_horizon", "no_exposure", "no_window", "no_anchor", "wrong_anchor",
    "at_horizon", "no_roster", "selectable_roster", "roster_mismatch", "coordinate_collision",
    "continuous", "once_per_stay", "unsupported_unit", "clustered", "other_family", "already_declared",
])
def test_an_ineligible_study_keeps_the_original_stop(tmp_path, change):
    study = _study()
    if change == "no_horizon":
        study["execution_concepts"]["outcome"] = "death"
    elif change == "no_exposure":
        study["execution_concepts"].pop("primary_exposure")
    elif change == "no_window":
        study.pop("time_window")
    elif change == "no_anchor":
        study["time_window"].pop("anchor")
    elif change == "wrong_anchor":
        study["time_window"]["anchor"] = "hospital_discharge"
    elif change == "at_horizon":
        study["time_window"]["hours"] = 90 * 24
    elif change == "no_roster":
        study["execution_concepts"]["covariates"] = study["covariates"] = []
    elif change == "selectable_roster":
        study["covariate_selection"] = "planner_selectable"
    elif change == "roster_mismatch":
        study["execution_concepts"]["covariates"] = ["age"]
    elif change == "coordinate_collision":
        study["execution_concepts"]["covariates"] = study["covariates"] = ["age", "rrt"]
    elif change in {"continuous", "once_per_stay"}:
        study["execution_concepts"]["primary_exposure"] = "lact" if change == "continuous" else "sex"
    elif change == "unsupported_unit":
        study["analysis_design"]["analysis_unit"] = "patient"
    elif change == "clustered":
        study["analysis_design"]["variance_estimator"] = "cluster_robust"
        study["analysis_design"]["cluster_unit"] = "patient"
    elif change == "other_family":
        study["analysis_design"]["analysis_family"] = "causal_inference"
    else:
        study["sensitivity_specs"] = [{
            "spec_id": "saved_landmark", "axis": "timing", "strategy": "landmark", "landmark_hours": 12,
        }]
    assert propose_survival_landmark(study) is None
    assert not plan_decisions.decision_is_resolved(study, DECISION_CODE)
    exc = _stop()
    agent_pipeline_runs._record_pipeline_failure(
        wrapper_dir=tmp_path, study=study, provider={}, exc=exc,
        code=agent_pipeline_runs._pipeline_failure_code(exc), execution_retry_id=None,
    )
    manifest = json.loads((tmp_path / "source_run_manifest.json").read_text())
    gate = json.loads((tmp_path / "quality_gate.json").read_text())["gate"]
    assert manifest["status"] == "failed"
    assert gate["reason"] == "research_pipeline_progressive_compile_failed"
    assert gate["detail"]["reason_code"] == "progressive_family_result_contract_unwritable"
    assert "plan_decision" not in manifest
    assert family_stop_result(tmp_path) is None


@pytest.mark.parametrize("window", [
    {"hours": 24, "anchor": "icu_admission", "start_hours": 6},
    {"hours": float("nan"), "anchor": "icu_admission"},
])
def test_malformed_unsaved_windows_offer_no_decision(window):
    assert propose_survival_landmark({**_study(), "time_window": window}) is None


@pytest.mark.parametrize("clause", [
    " Estimate the 90-day RMST difference as the primary result.",
    " Report the 28-day RMST difference as a sensitivity analysis.",
])
def test_an_rmst_question_outside_the_suites_alternative_cannot_be_closed(clause):
    study = _study()
    study["question"] += clause
    assert propose_survival_landmark(study) is None


@pytest.mark.parametrize("change", ["question", "roster", "window", "digest", "missing_digest", "card", "option"])
def test_a_stale_or_tampered_click_cannot_change_the_study(change):
    study = _study()
    card = propose_survival_landmark(study).model_dump(mode="json")
    kwargs = {}
    if change == "question":
        study["question"] += " Changed research intent."
    elif change == "roster":
        study["covariates"] = study["execution_concepts"]["covariates"] = ["age"]
    elif change == "window":
        study["time_window"]["hours"] = 12
    elif change == "digest":
        kwargs["card_sha256"] = "0" * 64
    elif change == "missing_digest":
        kwargs["card_sha256"] = None
    elif change == "card":
        card["endpoint"] = "mort_28d"
    else:
        kwargs["option_id"] = "automatic_landmark"
    before = copy.deepcopy(study)
    with pytest.raises(plan_decisions.PlanDecisionError):
        _choose(study, card, **kwargs)
    assert study == before


def test_the_host_projects_the_pending_card_and_withdraws_it_after_other_edits(tmp_path):
    from easyicu.webserver import study_contexts
    from easyicu.webserver.pi_copilot.workflow import _enrich_plan_review, build_research_workflow_snapshot

    study = _study()
    exc = _stop()
    agent_pipeline_runs._record_pipeline_failure(
        wrapper_dir=tmp_path, study=study, provider={}, exc=exc,
        code=agent_pipeline_runs._pipeline_failure_code(exc), execution_retry_id=None,
    )
    payloads = {"source_run_manifest.json": json.loads((tmp_path / "source_run_manifest.json").read_text())}
    run = {
        "run_id": "synthetic-run", "run_type": "full", "engine": "easyicu.research_agent.pipeline",
        "run_status": "plan_decision_pending", "gate_status": "blocked",
        "gate_reason": "plan_scientific_changes_required",
        "scientific_configuration_sha256": study_contexts.scientific_configuration_sha256(study),
        "artifact_names": ["source_run_manifest.json", "quality_gate.json"],
    }
    snapshot = build_research_workflow_snapshot(
        study=study, active_export_present=True, active_job=None, latest_run=run,
    )
    enriched = _enrich_plan_review(snapshot, study=study, review={"artifact_payloads": payloads})
    assert enriched.next_action_code == "plan_scientific_changes_required"
    assert enriched.plan_decision is not None
    assert enriched.plan_execution_ready is False
    assert enriched.next_actions == []
    [question] = enriched.plan_review_summary["authorization_questions"]
    assert question["decision_context"] == enriched.plan_decision.model_dump(mode="json")
    assert pending_survival_decision({**study, "question": "A different study"}, payloads) is None
    running = snapshot.model_copy(update={"next_action_code": "research_planning_running"})
    assert _enrich_plan_review(running, study=study, review={"artifact_payloads": payloads}).plan_decision is None


def test_the_researchers_click_writes_through_the_real_study_owner(tmp_path, monkeypatch):
    from easyicu.webserver import study_contexts
    from easyicu.webserver.pi_copilot import service as service_module
    from easyicu.webserver.pi_copilot.contracts import AuthorityBinding, PiSessionRecord, PiCopilotError

    monkeypatch.setenv("EASYICU_HOME", str(tmp_path))
    study = _study()
    study.pop("revision")
    study["question"] += " 并以90天限制性平均生存时间差作为敏感性分析。"
    saved = study_contexts.upsert_context(study)
    card = propose_survival_landmark(saved)
    assert card is not None and card.rmst_alternative is not None
    run = {"run_id": "synthetic-run", "project_dir": str(tmp_path / "run"),
           "scientific_configuration_sha256": study_contexts.scientific_configuration_sha256(saved)}
    payloads = {"source_run_manifest.json": {
        "status": "plan_decision_pending", "plan_decision": card.model_dump(mode="json"),
    }}
    service = service_module.PiCopilotService.__new__(service_module.PiCopilotService)
    record = PiSessionRecord(session_id="synthetic-session", project_id="synthetic-project",
        binding=AuthorityBinding(study_context_id=saved["id"], study_revision=saved["revision"], run_id=run["run_id"]))
    monkeypatch.setattr(service, "_scoped_record", lambda *a, **kw: record)
    monkeypatch.setattr(service, "_stale_details", lambda *a: {})
    monkeypatch.setattr(service, "_save_record", lambda *a: None)
    monkeypatch.setattr(service_module, "list_bound_run_history", lambda **kw: [run])
    monkeypatch.setattr(service_module, "research_pipeline_project_root", lambda *a: tmp_path)
    monkeypatch.setattr(service_module.agent_runs, "read_run_review", lambda *a: {"artifact_payloads": payloads})
    arguments = dict(project_id=record.project_id, decision_code=DECISION_CODE, option_id=OPTION_ID,
        expected_revision=saved["revision"], run_id=run["run_id"], card_sha256=card.card_sha256)
    with pytest.raises(PiCopilotError):
        service.confirm_plan_decision(record.session_id, **{**arguments, "card_sha256": "0" * 64})
    assert study_contexts.get_context(saved["id"])["sensitivity_specs"] == []
    result = service.confirm_plan_decision(record.session_id, **arguments)
    assert result["next_action"] == "replan"
    updated = study_contexts.get_context(saved["id"])
    assert updated["revision"] == saved["revision"] + 1
    assert [s["strategy"] for s in updated["sensitivity_specs"]] == ["landmark", "restricted_mean_survival"]
    assert updated["covariate_authority"] == "user"
    assert updated["execution_concepts"]["covariates"] == ["age", "sex"]
    with pytest.raises(PiCopilotError):
        service.confirm_plan_decision(record.session_id, **arguments)


def test_observation_hours_is_a_typed_window_and_is_never_replaced_with_a_default():
    study = {**_study(), "time_window": {"observation_hours": 36, "anchor": "icu_admission"}}
    assert propose_survival_landmark(study).landmark_hours == 36
    assert _choose(study).patch["sensitivity_specs"][0]["landmark_hours"] == 36


@pytest.mark.parametrize("code,owner,reason", [
    ("research_pipeline_provider_timeout", "easyicu.planning.progressive_compiler_v1", "progressive_family_result_contract_unwritable"),
    ("research_pipeline_progressive_compile_failed", "untrusted", "progressive_family_result_contract_unwritable"),
    ("research_pipeline_progressive_compile_failed", "easyicu.planning.progressive_compiler_v1", "progressive_family_template_required"),
])
def test_only_the_typed_unwritable_stop_offers_recovery(code, owner, reason):
    from easyicu.webserver.pi_copilot.survival_plan_decision import family_stop_decision

    assert family_stop_decision(_study(), code=code, typed_failure={"owner": owner, "reason_code": reason}) is None


@pytest.mark.parametrize("exposure", ["rrt", "lact"])
def test_another_rmst_estimand_is_never_silently_absorbed(tmp_path, exposure):
    from easyicu.research_agent.planning.sensitivity_authority import normalize_prespecified_sensitivities
    from easyicu.webserver.scientific_runtime_projection import compile_web_scientific_runtime_projection, WebScientificRuntimeProjectionError

    study = _study()
    study["question"] += " Use RMST as a sensitivity analysis."
    adopted = {**study, **_choose(study).patch}
    adopted["sensitivity_specs"][1]["rmst_execution"]["tau"] = 40
    adopted["execution_concepts"]["primary_exposure"] = exposure
    catalog = tmp_path / "zero_rows.parquet"
    pd.DataFrame({name: pd.Series(dtype="float64") for name in (
        "stay_id", exposure, f"{exposure}_onset_time", "mort_90d", "followup_days_90d", "age", "sex",
    )}).to_parquet(catalog)
    with pytest.raises(WebScientificRuntimeProjectionError) as stopped:
        compile_web_scientific_runtime_projection(
            study=adopted, sensitivity_specs=normalize_prespecified_sensitivities(adopted["sensitivity_specs"]),
            primary_exposure=exposure, primary_exposure_source=exposure, target_outcome="mort_90d",
            declared_covariates=adopted["covariates"], covariate_operationalizations=adopted["covariate_operationalizations"],
            target_is_event_status=True, universe_path=catalog, scientific_configuration_sha256="d" * 64,
        )
    assert stopped.value.code == "web_rmst_study_binding_mismatch"


def test_the_typed_route_carries_the_displayed_card_digest(monkeypatch):
    from types import SimpleNamespace
    from pydantic import ValidationError
    from easyicu.webserver.routes import pi_copilot as route

    payload = dict(project_id="synthetic-project", decision_code=DECISION_CODE, option_id=OPTION_ID,
        expected_revision=1, run_id="synthetic-run", card_sha256="d" * 64)
    monkeypatch.setattr(route, "get_pi_copilot_service", lambda: SimpleNamespace(
        confirm_plan_decision=lambda session_id, **kwargs: {"session_id": session_id, **kwargs},
    ))
    body = route.PiPlanDecisionSelectionRequest.model_validate(payload)
    assert route.post_pi_copilot_plan_decision_selection("synthetic-session", body) == {
        "session_id": "synthetic-session", **payload,
    }
    with pytest.raises(ValidationError):
        route.PiPlanDecisionSelectionRequest.model_validate({**payload, "card_sha256": "invalid"})


def test_the_job_runner_returns_the_decision_instead_of_raising_a_failure(tmp_path, monkeypatch):
    from types import SimpleNamespace
    from easyicu.webserver import research_job_recovery
    from easyicu.webserver.research_pipeline_run_preparation import (
        PreparedResearchPipelineRun, PreparedScientificLaunch, PreparedLaunchAuthority, PreparedLaunchExecution,
    )

    monkeypatch.setenv("EASYICU_HOME", str(tmp_path))
    study = _study()
    prepared = PreparedResearchPipelineRun(
        scientific=PreparedScientificLaunch(
            study=study, question=study["question"], database="miiv", materialization_study=study,
            configured_primary_exposure="rrt", target="mort_90d", primary_exposure="rrt",
            covariates=("age", "sex"), covariate_selection="exact", sensitivity_specs=(),
            cohort_window=(0, 24), validated_analysis_design=study["analysis_design"], patient_grouping=None,
            metadata_only_planning=True, metadata_planning_coordinates={}, execution_concepts=study["execution_concepts"],
            planning_exposure_source="rrt", metadata_operationalized_columns=(), prepared_package_binding=None,
            foundation_profile={},
        ),
        authority=PreparedLaunchAuthority(provider={}, provider_environment={}, credential_source="pi_verified",
            literature_search_authorized=False, publication_skill_flags={}, user_extension_activation=None),
        execution=PreparedLaunchExecution(export_path=str(tmp_path / "synthetic-source"), project_root=str(tmp_path / "projects"),
            budget_mode="planner_canary", development_resume_binding=None, development_resume_acquisition=None,
            development_resume_literature=None, plan_revision_source_run_id="", execution_resume_source_run_id="", runner_image=""),
    )
    monkeypatch.setattr(agent_pipeline_runs, "prepare_research_pipeline_run", lambda _request: prepared)
    monkeypatch.setattr(agent_pipeline_runs.provider_adapter, "build_research_agent_provider_client", lambda *a, **kw: (object(), {}))
    monkeypatch.setattr(research_job_recovery, "wrap_pipeline_runner", lambda runner, _request: runner)
    failure = _stop()

    def stop_before_provider(**kwargs):
        raise failure

    monkeypatch.setattr(agent_pipeline_runs, "_start_web_provider_hard_stop", stop_before_provider)
    runner = agent_pipeline_runs.make_research_pipeline_run_runner(
        export_path=prepared.execution.export_path, study_context=study,
        project_root=prepared.execution.project_root, provider={},
    )
    result = runner(SimpleNamespace(id="synthetic-job", cancel_requested=False, emit=lambda _event: None))
    assert result["status"] == "plan_decision_pending"
    assert result["analysis_started"] is False
    assert result["plan_decision"]["decision_code"] == DECISION_CODE


@pytest.mark.parametrize("changes,covered", [
    ({}, True), ({"tau": 40}, False), ({"time_column": "other_time"}, False),
    ({"event_column": "other_event"}, False), ({"group_column": "other_group"}, False),
    ({"time_unit": "hours"}, False), ({"event_code": 2}, False), ({"group_levels": ["0", "1"]}, False),
])
def test_review_credits_only_the_rmst_contrast_the_signed_suite_executes(tmp_path, changes, covered):
    from easyicu.outcome_availability import fixed_horizon_mortality_endpoint
    from easyicu.webserver.landmark_survival_runtime_projection import landmark_survival_rmst_spec
    from easyicu.research_agent.orchestration.scientific_runtime import ScientificRuntimeAuthorities
    from easyicu.research_agent.planning.scientific_review import build_plan_scientific_review
    from tests.support.survival_sealed import sealed_survival, bound_survival_plan

    rmst = landmark_survival_rmst_spec(exposure="rrt", landmark_hours=24, endpoint=fixed_horizon_mortality_endpoint("mort_90d"))
    context, authority = sealed_survival(tmp_path, review_only_specs=({
        "spec_id": "landmark_survival_rmst", "axis": "estimand", "strategy": "restricted_mean_survival",
        "rmst_execution": {**rmst.model_dump(mode="json"), **changes},
    },))
    plan = bound_survival_plan(context, ScientificRuntimeAuthorities(trajectory=None, current_case=authority))
    review = build_plan_scientific_review(context=context, plan=plan, runtime_authority=authority)
    missing = review.facts["sensitivity"]["missing_spec_ids"]
    assert ("landmark_survival_rmst" not in missing) is covered
