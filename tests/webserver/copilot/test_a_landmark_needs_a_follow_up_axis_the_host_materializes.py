"""B2 (11-19 UX walk): the host never compiles or offers a landmark it cannot launch.

A 24-hour landmark on 28-day mortality named ``mort_28d_followup_time_hours``,
a column no owner materializes; the compiled study was then refused at launch
with no way on.  Only in-hospital death has a host follow-up axis, so a
landmark on another outcome routes to the timing decision without a
landmark option (descriptive, or re-extract for a time-varying model).
"""

from __future__ import annotations

import pytest

from easyicu.webserver.pi_copilot.plan_decisions import (
    PlanDecisionError,
    agent_plan_configuration_available,
    compile_agent_plan_configuration,
    compile_plan_decision,
    landmark_followup_axis_available,
    plan_decision_context,
)

_TIMING = "POST_BASELINE_EXPOSURE_TIMING_NOT_CLOSED"


def _plan(outcome: str) -> dict:
    return {
        "design_selection": {"candidates": [{
            "design_id": "landmark_adjusted_association", "disposition": "selected",
            "required_variables": ["lact_max", outcome, "age", "sex"],
        }]},
        "steps": [{"model_requirements": [{
            "analysis_role": "primary", "exposure_source": "lact_max", "outcome": outcome,
            "covariates": ["age", "sex"],
            "covariate_rationales": {"age": "Baseline age precedes the window.", "sex": "Baseline sex precedes the window."},
            "covariate_temporal_roles": {"age": "baseline_static", "sex": "baseline_static"},
        }]}],
    }


def _study() -> dict:
    return {
        "id": "b2-study", "revision": 1,
        "question": "Is the first-day maximum lactate associated with 28-day mortality?",
        "data_source": {"database": "miiv", "path": "/synthetic/export"},
        "cohort": {"preset": "all_icu"}, "confirmations": {},
        "time_window": {"hours": 24, "anchor": "ICU admission"},
    }


def test_only_in_hospital_death_has_a_host_follow_up_axis() -> None:
    assert landmark_followup_axis_available("death")
    assert not landmark_followup_axis_available("mort_28d")


def test_the_agent_landmark_on_28_day_mortality_is_not_compiled() -> None:
    with pytest.raises(PlanDecisionError) as raised:
        compile_agent_plan_configuration(
            study=_study(), agent_plan=_plan("mort_28d"), runtime_finding_codes=(_TIMING,),
            patient_cluster_available=True,
        )
    assert raised.value.code == "agent_plan_landmark_not_compilable"
    assert not agent_plan_configuration_available(
        study=_study(), agent_plan=_plan("mort_28d"), runtime_finding_codes=(_TIMING,))
    # In-hospital death keeps the host landmark.
    compiled = compile_agent_plan_configuration(
        study=_study(), agent_plan=_plan("death"), runtime_finding_codes=(_TIMING,),
        patient_cluster_available=True,
    )
    assert any(spec["strategy"] == "landmark" for spec in compiled.patch["sensitivity_specs"])


@pytest.mark.parametrize(("outcome", "profile"), [("death", "fixed_24h_landmark"), ("mort_28d", "unspecified_post_baseline")])
def test_the_timing_card_offers_a_landmark_only_where_it_can_run(outcome: str, profile: str) -> None:
    assert plan_decision_context(_plan(outcome), _TIMING, _study())["timing_profile"] == profile


def test_a_forced_landmark_choice_on_28_day_mortality_is_refused() -> None:
    with pytest.raises(PlanDecisionError) as raised:
        compile_plan_decision(decision_code=_TIMING, option_id="landmark_24h",
                              study=_study(), agent_plan=_plan("mort_28d"))
    assert raised.value.code == "plan_decision_option_not_applicable"
    assert raised.value.details["outcome"] == "mort_28d"


@pytest.mark.parametrize("option", ["descriptive_only", "time_varying_reextract"])
def test_the_offered_choices_on_28_day_mortality_compile(option: str) -> None:
    compiled = compile_plan_decision(decision_code=_TIMING, option_id=option,
                                     study=_study(), agent_plan=_plan("mort_28d"))
    assert not any(spec.get("strategy") == "landmark" for spec in compiled.patch.get("sensitivity_specs") or [])


def _outcome_catalogue() -> list[str]:
    from easyicu.concept.catalog import CONCEPT_GROUPS_INTERNAL

    return sorted({"death", *CONCEPT_GROUPS_INTERNAL.get("outcome", ())})


@pytest.mark.parametrize("outcome", _outcome_catalogue())
def test_no_outcome_compiles_a_landmark_data_readiness_would_refuse(outcome: str) -> None:
    """S1414(1): every landmark producer goes through one spec owner, for every outcome."""
    from easyicu.research_agent.planning.sensitivity_authority import PrespecifiedSensitivitySpec
    from easyicu.webserver.pi_copilot.plan_decisions import landmark_timing_specification

    if not landmark_followup_axis_available(outcome):
        with pytest.raises(PlanDecisionError):
            landmark_timing_specification(outcome=outcome, landmark_hours=24, spec_id="probe")
        return
    spec = PrespecifiedSensitivitySpec.model_validate(
        landmark_timing_specification(outcome=outcome, landmark_hours=24, spec_id="probe"))
    # Nothing left for the source catalog to resolve: the launch gate has no column to refuse.
    assert spec.source_materialization_variables == ()


def _service_and_projection(monkeypatch, outcome: str):
    """The card and its route, end to end, for a reviewed plan on ``outcome`` (B2 re-walk)."""
    from easyicu.webserver.pi_copilot import service as service_module, workflow as workflow_owner
    from easyicu.webserver.pi_copilot.contracts import AuthorityBinding, PiSessionRecord

    current = {**_study(), "id": "b2-route-study", "revision": 1}
    from easyicu.webserver import study_contexts

    run = {"run_id": "b2-run", "scientific_configuration_sha256": study_contexts.scientific_configuration_sha256(current)}
    findings = [{"code": _TIMING, "severity": "blocker", "requires_user_authorization": False,
                 "message": "Exposure is classified after ICU time zero."}]
    review = {"artifact_payloads": {"agent_plan.json": _plan(outcome),
                                    "scientific_plan_review.json": {"findings": findings}}}
    service = service_module.PiCopilotService.__new__(service_module.PiCopilotService)
    record = PiSessionRecord(session_id="b2", project_id="project", binding=AuthorityBinding(
        study_context_id=current["id"], study_revision=1, run_id=run["run_id"]))
    monkeypatch.setattr(service, "_scoped_record", lambda *a, **kw: record)
    monkeypatch.setattr(service, "_stale_details", lambda *a: {})
    monkeypatch.setattr(service, "_save_record", lambda *a: None)
    monkeypatch.setattr(service_module.study_contexts, "get_context", lambda *a: dict(current))
    monkeypatch.setattr(service_module, "list_bound_run_history", lambda **kw: [{**run, "project_dir": "/test/plan"}])
    monkeypatch.setattr(service_module, "research_pipeline_project_root", lambda *a: "/test")
    monkeypatch.setattr(service_module.agent_runs, "read_run_review", lambda *a: review)
    monkeypatch.setattr(service_module.plan_review_progress, "validate_choice_source", lambda *a: None)
    monkeypatch.setattr(service_module.plan_review_progress, "record_choice", lambda **kw: None)

    def update(patch, **kw):
        current.update(patch)
        current["revision"] += 1
        return dict(current)

    monkeypatch.setattr(service_module.study_contexts, "upsert_context", update)
    live_run = {**run, "project_dir": "/test/plan", "run_type": "full", "engine": "easyicu.research_agent.pipeline",
                "gate_status": "blocked", "run_status": "human_review_pending",
                "pending_review_reason_codes": ["plan_scientific_changes_required"],
                "artifact_names": ["agent_plan.json", "scientific_plan_review.json", "source_run_manifest.json"]}
    monkeypatch.setattr(workflow_owner, "list_bound_run_history", lambda **kw: [live_run])
    monkeypatch.setattr(workflow_owner, "workflow_authoritative_run", lambda rows: rows[0])
    monkeypatch.setattr(workflow_owner.agent_pipeline_runs, "pending_review", lambda _id: {
        **run, "resumable_here": True, "scientific_plan_review": {"findings": findings}})
    monkeypatch.setattr(workflow_owner.agent_runs, "read_run_review", lambda *a: review)
    monkeypatch.setattr(workflow_owner, "project_run_outcome", lambda _review, **_kwargs: {})
    monkeypatch.setattr(workflow_owner.sources, "load_registry", lambda **_kwargs: {})
    projection = workflow_owner.build_project_workflow_projection(study_context_id=current["id"], study_override=current)
    return service, projection, current, run


def test_28_day_mortality_reaches_the_timing_card_and_its_default_choice(monkeypatch) -> None:
    service, projection, current, run = _service_and_projection(monkeypatch, "mort_28d")
    workflow = projection.workflow
    assert workflow.next_action_code == "plan_scientific_changes_required"
    question = workflow.plan_review_summary["authorization_questions"][0]
    assert question["code"] == _TIMING
    assert question["decision_context"]["timing_profile"] == "unspecified_post_baseline"

    result = service.confirm_plan_decision(
        "b2", project_id="project", decision_code=_TIMING, option_id="descriptive_only",
        expected_revision=1, run_id=run["run_id"])
    assert result["next_action"] == "replan"
    assert current["confirmations"]["plan_timing_descriptive_only"] is True


def test_in_hospital_death_keeps_the_host_landmark_and_asks_nothing(monkeypatch) -> None:
    from easyicu.webserver.pi_copilot.contracts import PiCopilotError

    service, projection, _current, run = _service_and_projection(monkeypatch, "death")
    questions = (projection.workflow.plan_review_summary or {}).get("authorization_questions") or []
    assert _TIMING not in [q.get("code") for q in questions]
    with pytest.raises(PiCopilotError) as raised:
        service.confirm_plan_decision(
            "b2", project_id="project", decision_code=_TIMING, option_id="descriptive_only",
            expected_revision=1, run_id=run["run_id"])
    assert raised.value.code == "plan_decision_not_required_by_review"


@pytest.mark.parametrize("case", ["another_studys_review", "stale_revision", "already_resolved"])
def test_the_handed_timing_choice_is_refused_outside_this_studys_current_review(monkeypatch, case) -> None:
    """S1420: the route accepts the handed code only for this study's own, current, unresolved review."""
    from easyicu.webserver.pi_copilot import service as service_module
    from easyicu.webserver.pi_copilot.contracts import PiCopilotError

    service, _projection, current, run = _service_and_projection(monkeypatch, "mort_28d")
    revision = 1
    if case == "another_studys_review":
        # The bound review run is not in this study's run history.
        monkeypatch.setattr(service_module, "list_bound_run_history", lambda **kw: [
            {**run, "run_id": "other-study-run", "project_dir": "/other/plan"}])
    elif case == "stale_revision":
        current["revision"] = 2
    else:
        service.confirm_plan_decision(
            "b2", project_id="project", decision_code=_TIMING, option_id="descriptive_only",
            expected_revision=1, run_id=run["run_id"])
        revision = current["revision"]
        record = service._scoped_record("b2")
        moved = record.model_copy(update={"binding": record.binding.model_copy(update={"study_revision": revision})})
        monkeypatch.setattr(service, "_scoped_record", lambda *a, **kw: moved)
    with pytest.raises(PiCopilotError) as raised:
        service.confirm_plan_decision(
            "b2", project_id="project", decision_code=_TIMING, option_id="descriptive_only",
            expected_revision=revision, run_id=run["run_id"])
    assert raised.value.code in {
        "another_studys_review": {"plan_decision_run_not_found"},
        "stale_revision": {"plan_decision_revision_conflict", "plan_decision_source_superseded"},
        "already_resolved": {"plan_decision_not_required_by_review"},
    }[case]
