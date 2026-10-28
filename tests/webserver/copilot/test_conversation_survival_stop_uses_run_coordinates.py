"""Conversation studies offer a landmark only from host-resolved run metadata."""

import copy
import hashlib
import json

import pandas as pd
import pytest

from easyicu.research_agent.planning.progressive_contract import ProgressivePlanCompileError
from easyicu.webserver import agent_pipeline_runs
from easyicu.webserver.pi_copilot import plan_decisions
from easyicu.webserver.pi_copilot.survival_decision_source import read_survival_decision_source
from easyicu.webserver.pi_copilot.survival_plan_decision import DECISION_CODE, OPTION_ID, pending_survival_decision
from tests.support.survival_proposal import survival_context


def _conversation_study():
    return {
        "id": "synthetic-conversation-survival", "revision": 1,
        "question": (
            "Does first-day renal support predict 90-day survival, adjusted for age and sex? "
            "Report the 90-day RMST difference as a sensitivity analysis."
        ),
        "data_source": {"database": "miiv"},
        "primary_exposure": "first-day renal support", "outcome": "90-day survival",
        "covariates": [], "covariate_selection": "planner_selectable",
        "stated_adjustments": [
            {"variable": name, "words": "adjusted for age and sex", "concept": name, "source_status": "unbound"}
            for name in ("age", "sex")
        ],
        "time_window": {}, "execution_concepts": {}, "sensitivity_specs": [],
        "analysis_design": {
            "analysis_family": "survival", "analysis_unit": "icu_stay", "variance_estimator": "model_based",
        },
        "question_family_check": {
            "model_family": "survival", "host_family": "survival", "words": "survival", "status": "agreed",
        },
    }


def _run_context(study):
    context = survival_context(research_question=study["question"])
    return context.model_copy(update={"user_preferences": context.user_preferences.model_copy(update={
        "data_constraints": json.dumps({"materialization_window": {
            "anchor": "icu_admission", "hours": 24, "role": "outer_observation_window",
        }}),
    })})


def _write_context(wrapper, context, run_id="run_synthetic"):
    path = wrapper / "pipeline" / run_id / "research_context.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(context.model_dump_json())
    return path


def _record_stop(wrapper, study):
    exc = ProgressivePlanCompileError(
        "progressive_family_result_contract_unwritable", "No sealed family result owner.",
    )
    agent_pipeline_runs._record_pipeline_failure(
        wrapper_dir=wrapper, study=study, provider={}, exc=exc,
        code=agent_pipeline_runs._pipeline_failure_code(exc), execution_retry_id=None,
    )
    return json.loads((wrapper / "source_run_manifest.json").read_text())


def test_a_conversation_study_offers_the_host_resolved_run_coordinates(tmp_path):
    study = _conversation_study()
    before = copy.deepcopy(study)
    path = _write_context(tmp_path, _run_context(study))
    manifest = _record_stop(tmp_path, study)
    assert manifest["status"] == "plan_decision_pending"
    card = manifest["plan_decision"]
    assert (card["exposure"], card["endpoint"], card["landmark_hours"]) == ("rrt", "mort_90d", 24)
    assert card["followup_column"] == "followup_days_90d"
    assert card["covariates"] == ["age", "sex"]
    assert card["rmst_alternative"]["tau"] == 89
    assert card["source_run_id"] == path.parent.name
    assert card["research_context_sha256"] == hashlib.sha256(path.read_bytes()).hexdigest()
    assert study == before


def test_the_pipeline_lock_directory_beside_the_run_is_not_a_second_run(tmp_path):
    # A real pipeline directory holds its run and the hidden lock directory.
    study = _conversation_study()
    _write_context(tmp_path, _run_context(study))
    (tmp_path / "pipeline" / ".run_locks").mkdir()
    manifest = _record_stop(tmp_path, study)
    assert manifest["status"] == "plan_decision_pending"
    # Two real runs stay ambiguous and keep the stop.
    _write_context(tmp_path, _run_context(study), run_id="run_other")
    assert read_survival_decision_source(tmp_path) is None


def test_the_run_source_uses_the_shared_database_alias_owner(tmp_path):
    study = _conversation_study()
    context = _run_context(study)
    study["data_source"]["database"] = "MIMIC-IV"
    _write_context(tmp_path, context)
    assert _record_stop(tmp_path, study)["status"] == "plan_decision_pending"


@pytest.mark.parametrize("exposure,horizon,hours", [("rrt", 90, 24), ("abx", 365, 12)])
@pytest.mark.parametrize("rmst", [True, False])
def test_the_rendered_conversation_card_adopts_through_the_owner_and_binds_the_next_suite(
    tmp_path, monkeypatch, exposure, horizon, hours, rmst,
):
    from easyicu.webserver import agent_runs, study_contexts
    from easyicu.webserver.landmark_survival_runtime_projection import validate_landmark_survival_declaration
    from easyicu.webserver.pi_copilot import service as service_module
    from easyicu.webserver.pi_copilot.contracts import AuthorityBinding, PiSessionRecord, PiCopilotError
    from easyicu.webserver.pi_copilot.workflow import _enrich_plan_review, build_research_workflow_snapshot
    from easyicu.webserver.scientific_runtime_projection import compile_web_scientific_runtime_projection
    from easyicu.research_agent.planning.sensitivity_authority import normalize_prespecified_sensitivities

    monkeypatch.setenv("EASYICU_HOME", str(tmp_path))
    monkeypatch.setattr(study_contexts, "_CONFIG_PATH", tmp_path / "studies.json")
    study = _conversation_study()
    study.pop("revision")
    study["question"] = f"Does early support predict {horizon}-day survival, adjusted for age and sex?"
    if rmst:
        study["question"] += f" Report the {horizon}-day RMST difference as a sensitivity analysis."
    saved = study_contexts.upsert_context(
        study, _server_question_wording_write=True, _server_study_family_reading_write=True,
    )
    wrapper = tmp_path / "run"
    context = _run_context(saved)
    names = {"rrt": exposure, "mort_90d": f"mort_{horizon}d", "followup_days_90d": f"followup_days_{horizon}d"}
    context = context.model_copy(update={
        "primary_exposure": exposure, "target_outcome": f"mort_{horizon}d",
        "variables": [variable.model_copy(update={"name": names.get(variable.name, variable.name)}) for variable in context.variables],
        "user_preferences": context.user_preferences.model_copy(update={"data_constraints": json.dumps({
            "materialization_window": {"anchor": "icu_admission", "hours": hours, "role": "outer_observation_window"},
        })}),
    })
    _write_context(wrapper, context)
    manifest = _record_stop(wrapper, saved)
    assert manifest["status"] == "plan_decision_pending"
    review = agent_runs.read_run_review(str(wrapper))
    assert review["ok"]
    run = {
        "run_id": wrapper.name, "project_dir": str(wrapper),
        "scientific_configuration_sha256": study_contexts.scientific_configuration_sha256(saved),
        "run_type": "full", "run_status": "plan_decision_pending", "engine": "easyicu.research_agent.pipeline",
        "gate_status": "blocked", "gate_reason": "plan_scientific_changes_required",
        "artifact_names": ["source_run_manifest.json", "quality_gate.json"],
    }
    snapshot = build_research_workflow_snapshot(
        study=saved, active_export_present=True, active_job=None, latest_run=run,
    )
    rendered = _enrich_plan_review(snapshot, study=saved, review=review)
    assert rendered.plan_decision is not None
    card = rendered.plan_decision
    assert (card.exposure, card.endpoint, card.landmark_hours) == (exposure, f"mort_{horizon}d", hours)
    assert card.rmst_alternative is not None if rmst else card.rmst_alternative is None
    assert study_contexts.get_context(saved["id"]) == saved
    service = service_module.PiCopilotService.__new__(service_module.PiCopilotService)
    record = PiSessionRecord(session_id="synthetic-session", project_id="synthetic-project", binding=AuthorityBinding(
        study_context_id=saved["id"], study_revision=saved["revision"], run_id=run["run_id"],
    ))
    monkeypatch.setattr(service, "_scoped_record", lambda *a, **kw: record)
    monkeypatch.setattr(service, "_stale_details", lambda *a: {})
    monkeypatch.setattr(service, "_save_record", lambda *a: None)
    monkeypatch.setattr(service_module, "list_bound_run_history", lambda **kw: [run])
    monkeypatch.setattr(service_module, "research_pipeline_project_root", lambda *a: tmp_path)
    arguments = dict(project_id=record.project_id, decision_code=DECISION_CODE, option_id=OPTION_ID,
                     expected_revision=saved["revision"], run_id=run["run_id"], card_sha256=card.card_sha256)
    with pytest.raises(PiCopilotError) as rejected:
        service.confirm_plan_decision(record.session_id, **{**arguments, "card_sha256": "0" * 64})
    assert rejected.value.code == "plan_decision_source_superseded"
    assert study_contexts.get_context(saved["id"]) == saved
    path = wrapper / "pipeline" / card.source_run_id / "research_context.json"
    original = path.read_bytes()
    path.write_bytes(original + b"\n")
    assert _enrich_plan_review(snapshot, study=saved, review=review).plan_decision is None
    with pytest.raises(PiCopilotError) as rejected:
        service.confirm_plan_decision(record.session_id, **arguments)
    assert rejected.value.code == "plan_decision_source_superseded"
    assert study_contexts.get_context(saved["id"]) == saved
    path.write_bytes(original)
    result = service.confirm_plan_decision(record.session_id, **arguments)
    assert result["next_action"] == "replan"
    adopted = study_contexts.get_context(saved["id"])
    assert adopted["execution_concepts"] == {
        "primary_exposure": exposure, "outcome": f"mort_{horizon}d", "covariates": ["age", "sex"],
    }
    assert adopted["time_window"] == {"anchor": "icu_admission", "hours": hours}
    assert adopted["covariate_authority"] == "researcher_stated"
    assert adopted["covariate_selection"] == "exact"
    from easyicu.webserver.study_scientific_configuration import ScientificConfiguration

    configuration = ScientificConfiguration.inspect(adopted)
    assert configuration.covariate_authority() == "researcher_stated"
    assert configuration.decision_is_resolved("ADJUSTMENT_SET_NOT_USER_CONFIRMED")
    assert adopted["covariate_rationales"] == {name: "Stated in the research question." for name in ("age", "sex")}
    assert {key for key in saved if key != "updated_at" and saved[key] != adopted[key]} == {
        "revision", "execution_concepts", "time_window", "covariates", "covariate_selection",
        "covariate_authority", "covariate_rationales", "covariate_operationalizations", "sensitivity_specs", "confirmations",
    }
    assert validate_landmark_survival_declaration(adopted) == hours
    preferences = agent_pipeline_runs._research_user_preferences(adopted)
    from easyicu.research_agent.schema import UserPreferences

    assert UserPreferences.model_validate(preferences).covariate_authority == "user"
    from easyicu.research_agent.planning.adjustment_authority import AdjustmentSetAuthority

    rebound = context.model_copy(update={"user_preferences": UserPreferences.model_validate(preferences)})
    roster = AdjustmentSetAuthority.from_context(rebound).prompt_projection()
    assert roster["authority"] == "user"
    assert "User-authored" in roster["authorship_boundary"]
    assert roster["scientific_covariates"] == ["age", "sex"]
    specs = normalize_prespecified_sensitivities(adopted["sensitivity_specs"])
    assert [spec.strategy for spec in specs] == ["landmark", *(["restricted_mean_survival"] if rmst else [])]
    columns = ["stay_id", exposure, f"{exposure}_onset_time", f"mort_{horizon}d", f"followup_days_{horizon}d", "age", "sex"]
    catalog = tmp_path / "zero_rows.parquet"
    pd.DataFrame({name: pd.Series(dtype="float64") for name in columns}).to_parquet(catalog)
    projection = compile_web_scientific_runtime_projection(
        study=adopted, sensitivity_specs=specs, primary_exposure=exposure, primary_exposure_source=exposure,
        target_outcome=f"mort_{horizon}d", declared_covariates=adopted["covariates"],
        covariate_operationalizations=adopted["covariate_operationalizations"], target_is_event_status=True,
        universe_path=catalog, scientific_configuration_sha256=study_contexts.scientific_configuration_sha256(adopted),
    )
    assert projection.authority["plan_method"] == "signed_landmark_survival_suite"
    assert projection.authority["adjustment_columns"] == ["age", "sex"]
    assert projection.authority["landmark_hours"] == hours
    if rmst:
        assert card.rmst_alternative.tau == horizon - hours / 24
    if (exposure, rmst) == ("rrt", True):
        from easyicu.research_agent.authority.current_case_scientific_runtime import load_current_case_scientific_runtime_authority
        from easyicu.research_agent.orchestration.scientific_runtime import ScientificRuntimeAuthorities
        from easyicu.research_agent.planning.primary_result_contract import validate_required_primary_result
        from easyicu.research_agent.schema import ConceptDescriptor
        from tests.support.survival_sealed import bound_survival_plan

        authority = load_current_case_scientific_runtime_authority(projection.authority)
        rebound = rebound.model_copy(update={
            "endpoint": authority.research_context_endpoint(),
            "variables": [*rebound.variables, ConceptDescriptor(name="rrt_onset_time", role="other", dtype="float64", unit="hours")],
        })
        plan = bound_survival_plan(rebound, ScientificRuntimeAuthorities(trajectory=None, current_case=authority))
        assert [step.method for step in plan.steps] == [
            "host_materialized_locked_cohort", "signed_landmark_survival_suite", "signed_landmark_survival_figure",
        ]
        validate_required_primary_result(plan=plan, context=rebound)


@pytest.mark.parametrize("change", [
    "unresolved_adjustment", "missing_adjustment", "no_statements", "duplicate_adjustment",
    "no_run", "invalid_context", "ambiguous_run", "missing_exposure", "missing_endpoint", "missing_followup",
    "continuous_exposure", "uncatalogued_exposure", "unsupported_endpoint", "wrong_database", "no_window", "wrong_anchor", "wrong_window_role",
    "horizon_window", "offset_window", "unsupported_unit", "coordinate_collision", "followup_collision",
    "study_timing", "run_timing", "run_rmst", "run_landmark",
])
def test_unclosable_conversation_coordinates_keep_the_existing_stop(tmp_path, change):
    from easyicu.research_agent.planning.sensitivity_authority import PrespecifiedSensitivitySpec

    study = _conversation_study()
    context = _run_context(study)
    constraints = json.loads(context.user_preferences.data_constraints)
    if change == "unresolved_adjustment":
        study["stated_adjustments"][1]["concept"] = None
    elif change.startswith("missing_"):
        name = {"missing_adjustment": "sex", "missing_exposure": "rrt", "missing_endpoint": "mort_90d", "missing_followup": "followup_days_90d"}[change]
        context = context.model_copy(update={"variables": [variable for variable in context.variables if variable.name != name]})
    elif change == "no_statements":
        study["stated_adjustments"] = []
    elif change == "duplicate_adjustment":
        study["stated_adjustments"].append({**study["stated_adjustments"][0], "variable": "age again"})
    elif change == "continuous_exposure":
        context = context.model_copy(update={"primary_exposure": "lact", "variables": [
            *context.variables, context.variable("age").model_copy(update={"name": "lact"}),
        ]})
    elif change == "uncatalogued_exposure":
        context = context.model_copy(update={"primary_exposure": "synthetic_event", "variables": [
            *context.variables, context.variable("rrt").model_copy(update={"name": "synthetic_event"}),
        ]})
    elif change == "unsupported_endpoint":
        context = context.model_copy(update={"target_outcome": "death", "variables": [
            *context.variables, context.variable("mort_90d").model_copy(update={"name": "death"}),
        ]})
    elif change == "wrong_database":
        context = context.model_copy(update={"cohort": context.cohort.model_copy(update={"database": "eicu_demo"})})
    elif change == "no_window":
        constraints = {}
    elif change == "wrong_anchor":
        constraints["materialization_window"]["anchor"] = "hospital_discharge"
    elif change == "wrong_window_role":
        constraints["materialization_window"]["role"] = "outcome_followup"
    elif change == "horizon_window":
        constraints["materialization_window"]["hours"] = 90 * 24
    elif change == "offset_window":
        constraints["materialization_window"]["start_hours"] = 6
    elif change == "unsupported_unit":
        study["analysis_design"]["analysis_unit"] = "patient"
    elif change in {"coordinate_collision", "followup_collision"}:
        study["stated_adjustments"][1]["concept"] = "rrt" if change == "coordinate_collision" else "followup_days_90d"
    elif change == "study_timing":
        study["sensitivity_specs"] = [{"spec_id": "saved_landmark", "axis": "timing", "strategy": "landmark", "landmark_hours": 12}]
    elif change in {"run_timing", "run_rmst"}:
        raw = {"spec_id": "run_landmark", "axis": "timing", "strategy": "landmark", "landmark_hours": 12}
        if change == "run_rmst":
            raw = {"spec_id": "run_rmst", "axis": "estimand", "strategy": "restricted_mean_survival",
                   "rmst_execution": {"time_column": "time", "event_column": "event", "group_column": "group", "group_levels": ["1", "0"], "tau": 89}}
        context = context.model_copy(update={"user_preferences": context.user_preferences.model_copy(update={
            "sensitivity_specs": [PrespecifiedSensitivitySpec.model_validate(raw)],
        })})
    elif change == "run_landmark":
        context = context.model_copy(update={"user_preferences": context.user_preferences.model_copy(update={"landmark_hours": 12})})
    context = context.model_copy(update={"user_preferences": context.user_preferences.model_copy(update={"data_constraints": json.dumps(constraints)})})
    if change != "no_run":
        path = _write_context(tmp_path, context)
        if change == "invalid_context":
            path.write_text("{}")
        if change == "ambiguous_run":
            _write_context(tmp_path, context, "run_other")
    before = copy.deepcopy(study)
    manifest = _record_stop(tmp_path, study)
    assert manifest["status"] == "failed"
    assert "plan_decision" not in manifest
    gate = json.loads((tmp_path / "quality_gate.json").read_text())["gate"]
    assert gate["reason"] == "research_pipeline_progressive_compile_failed"
    assert gate["detail"]["reason_code"] == "progressive_family_result_contract_unwritable"
    assert study == before


@pytest.mark.parametrize("change", ["pipeline_symlink", "run_symlink", "context_symlink", "oversized_context"])
def test_the_run_reader_rejects_escaping_or_unbounded_metadata(tmp_path, change):
    study = _conversation_study()
    wrapper = tmp_path / "wrapper"
    wrapper.mkdir()
    path = _write_context(wrapper, _run_context(study))
    if change == "oversized_context":
        with path.open("ab") as handle:
            for _ in range(128):
                handle.write(b" " * 65536)
    else:
        selected = {"pipeline_symlink": path.parent.parent, "run_symlink": path.parent, "context_symlink": path}[change]
        target = tmp_path / "metadata_outside_wrapper"
        selected.rename(target)
        selected.symlink_to(target, target_is_directory=target.is_dir())
    assert _record_stop(wrapper, study)["status"] == "failed"


@pytest.mark.parametrize("change", [None, "exposure", "endpoint", "window"])
def test_a_structured_study_must_agree_with_its_resolved_run(tmp_path, change):
    study = _conversation_study()
    study.update(
        covariates=["age", "sex"], covariate_selection="exact", covariate_authority="user",
        execution_concepts={"primary_exposure": "rrt", "outcome": "mort_90d", "covariates": ["age", "sex"]},
        time_window={"anchor": "icu_admission", "hours": 24},
    )
    context = _run_context(study)
    if change == "exposure":
        study["execution_concepts"]["primary_exposure"] = "abx"
    elif change == "endpoint":
        study["execution_concepts"]["outcome"] = "mort_365d"
    elif change == "window":
        study["time_window"]["hours"] = 12
    _write_context(tmp_path, context)
    manifest = _record_stop(tmp_path, study)
    assert manifest["status"] == ("plan_decision_pending" if change is None else "failed")


@pytest.mark.parametrize("change", ["window", "exposure", "roster", "run_id", "run_bytes", "run_removed", "study"])
def test_a_changed_run_or_study_withdraws_the_card_and_rejects_the_click(tmp_path, change):
    study = _conversation_study()
    context = _run_context(study)
    path = _write_context(tmp_path, context)
    manifest = _record_stop(tmp_path, study)
    card = manifest["plan_decision"]
    if change == "window":
        context = context.model_copy(update={"user_preferences": context.user_preferences.model_copy(update={
            "data_constraints": json.dumps({"materialization_window": {"anchor": "icu_admission", "hours": 12, "role": "outer_observation_window"}}),
        })})
    elif change == "exposure":
        context = context.model_copy(update={"primary_exposure": "abx"})
    elif change == "roster":
        context = context.model_copy(update={"variables": [variable for variable in context.variables if variable.name != "sex"]})
    elif change == "run_id":
        path.parent.rename(path.parent.with_name("run_replaced"))
    elif change == "run_bytes":
        path.write_text(path.read_text() + "\n")
    elif change == "run_removed":
        path.unlink()
    else:
        study["question"] += " Changed intent."
    if change in {"window", "exposure", "roster"}:
        path.write_text(context.model_dump_json())
    assert pending_survival_decision(study, {"source_run_manifest.json": manifest}, wrapper_dir=tmp_path) is None
    before = copy.deepcopy(study)
    with pytest.raises(plan_decisions.PlanDecisionError) as rejected:
        plan_decisions.compile_plan_decision(
            decision_code=DECISION_CODE, option_id=OPTION_ID, study=study, agent_plan={},
            decision_card=card, card_sha256=card["card_sha256"], wrapper_dir=tmp_path,
        )
    assert rejected.value.code == "plan_decision_source_superseded"
    assert study == before


@pytest.mark.parametrize("guard", ["session", "revision"])
def test_a_changed_study_supersedes_the_card_at_the_service_boundary(tmp_path, monkeypatch, guard):
    from easyicu.webserver import study_contexts
    from easyicu.webserver.pi_copilot import service as service_module
    from easyicu.webserver.pi_copilot.contracts import AuthorityBinding, PiSessionRecord, PiCopilotError

    monkeypatch.setenv("EASYICU_HOME", str(tmp_path))
    monkeypatch.setattr(study_contexts, "_CONFIG_PATH", tmp_path / "studies.json")
    raw = _conversation_study()
    raw.pop("revision")
    saved = study_contexts.upsert_context(
        raw, _server_question_wording_write=True, _server_study_family_reading_write=True,
    )
    _write_context(tmp_path, _run_context(saved))
    card = _record_stop(tmp_path, saved)["plan_decision"]
    changed = study_contexts.upsert_context({
        "id": saved["id"], "question": saved["question"] + " Restrict to a different target population.",
    }, expected_revision=saved["revision"])
    service = service_module.PiCopilotService.__new__(service_module.PiCopilotService)
    record = PiSessionRecord(session_id="synthetic-session", project_id="synthetic-project", binding=AuthorityBinding(
        study_context_id=saved["id"], study_revision=saved["revision"], run_id="synthetic-run",
    ))
    monkeypatch.setattr(service, "_scoped_record", lambda *a, **kw: record)
    monkeypatch.setattr(service, "_stale_details", lambda *a: {"stale": guard == "session"})
    with pytest.raises(PiCopilotError) as rejected:
        service.confirm_plan_decision(
            record.session_id, project_id=record.project_id, decision_code=DECISION_CODE,
            option_id=OPTION_ID, expected_revision=saved["revision"], run_id=record.binding.run_id,
            card_sha256=card["card_sha256"],
        )
    assert rejected.value.code == "plan_decision_source_superseded"
    assert study_contexts.get_context(saved["id"]) == changed
