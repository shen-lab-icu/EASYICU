"""Repair contradictory legacy setup without weakening a valid study design."""

from copy import deepcopy

import pytest

from easyicu.webserver import study_contexts
from easyicu.webserver.pi_copilot import study_context_update as owner
from easyicu.webserver.pi_copilot.contracts import AuthorityBinding, PiSessionRecord, ToolExecutionContext


def _saved():
    return {
        "id": "study-legacy-prediction", "revision": 5, "active_job_id": None,
        "question": "Build a prediction model for hospital mortality using first-day features.",
        "analysis_goal": "Prediction with calibration and external validation",
        "analysis_design": {"analysis_family": "prediction_model", "analysis_unit": "icu_stay", "variance_estimator": "none_counts_only"},
        "time_window": {"hours": 24}, "cohort": {},
        "data_source": {"path": "/unchanged/export", "database": "miiv"},
        "confirmations": {"feature_time_window": True},
    }


def _update(monkeypatch, current, params, message):
    writes = []
    monkeypatch.setattr(owner.study_contexts, "upsert_context", lambda raw, **kw: writes.append(deepcopy(raw)) or {**current, **raw, "revision": 6})
    context = ToolExecutionContext(
        session=PiSessionRecord(session_id="pi-legacy-recovery", binding=AuthorityBinding(study_context_id=current["id"], study_revision=5)),
        user_message=message, allowed_actions={"configure"},
    )
    result = owner.update_study_context(
        context, params, load_context=lambda binding: deepcopy(current),
        project_workflow=lambda *a, **kw: {"next_action_code": "provider_ready_to_generate_plan"},
    )
    return result, writes


def _proposal():
    return {"analysis_design": {"analysis_family": "prediction_model", "analysis_unit": "icu_stay", "variance_estimator": "model_based"}}


def test_prediction_intent_recovers_only_contradictory_ceiling_for_new_plan(monkeypatch):
    current = _saved()
    before = deepcopy(current)
    result, writes = _update(monkeypatch, current, _proposal(), "保留预测研究目标，修复旧配置并重新生成完整计划，保留校准和外部验证。")
    assert result["code"] == "study_context_updated"
    assert writes[0]["analysis_design"] == _proposal()["analysis_design"]
    revised = {**current, **writes[0]}
    for key in ("question", "analysis_goal", "cohort", "time_window", "confirmations", "data_source"):
        assert revised[key] == before[key]
    assert len(study_contexts.scientific_configuration_sha256(revised)) == 64
    # The contradictory historical envelope cannot confer current approval.
    with pytest.raises(study_contexts.StudyContextError):
        study_contexts.scientific_configuration_sha256(current)
    assert result["details"]["rebind_required"] is True
    assert result["details"]["analysis_design_recovery"]["requires_full_plan_review"] is True
    assert result["details"]["analysis_design_recovery"]["execution_authorized"] is False
    assert current == before
    from easyicu.webserver.pi_copilot.workflow import build_research_workflow_snapshot

    workflow = build_research_workflow_snapshot(
        study=revised, active_export_present=False, active_job=None,
        latest_run={
            "run_id": "old-plan", "run_status": "human_review_pending",
            "artifact_names": ["agent_plan.json"],
            "pending_review_reason_codes": ["operator_plan_approval_required"],
            "scientific_configuration_sha256": "0" * 64,
        },
    )
    assert workflow.next_action_code == "plan_configuration_superseded"
    assert workflow.plan_execution_ready is False


@pytest.mark.parametrize("case", ["no_intent", "negated", "valid_counts", "cluster_choice", "repeated_stays"])
def test_recovery_cannot_replace_authority_or_choose_new_science(monkeypatch, case):
    current, proposal = _saved(), _proposal()
    message = "保留预测研究目标，修复配置并重新生成计划。"
    if case == "no_intent":
        message = "继续，修复配置。"
    if case == "negated":
        message = "Not a prediction study; describe the cohort."
    if case == "valid_counts":
        current["analysis_design"]["analysis_family"] = "descriptive_epidemiology"
    if case == "cluster_choice":
        proposal["analysis_design"].update(variance_estimator="cluster_robust", cluster_unit="patient")
    if case == "repeated_stays":
        current["cohort"] = {"exclude_readmissions": False}
    result, writes = _update(monkeypatch, current, proposal, message)
    assert result["status"] == "blocked"
    assert writes == []


def test_empty_message_does_not_invoke_conversational_recovery():
    # Existing host-only typed updates have a separate Configure authority;
    # this exception must not pretend that their absent user text is intent.
    assert not owner._repairs_contradictory_prediction_ceiling(
        _saved()["analysis_design"], _proposal()["analysis_design"], "",
    )
