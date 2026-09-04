"""Sibling plan choices remain visible without reviving old execution authority."""

import json

import pytest

from easyicu.webserver import study_contexts
from easyicu.webserver.pi_copilot import plan_review_progress as owner
from easyicu.webserver.pi_copilot.plan_decisions import PlanDecisionError
from easyicu.webserver.pi_copilot.workflow import build_research_workflow_snapshot


@pytest.fixture
def states(tmp_path, monkeypatch):
    monkeypatch.setenv("EASYICU_HOME", str(tmp_path))
    before = {
        "id": "review-study",
        "revision": 1,
        "question": "Exposure and outcome association",
        "data_source": {"database": "test", "path": "/test/source"},
    }
    after = {
        **before,
        "revision": 2,
        "analysis_design": {
            "analysis_unit": "icu_stay",
            "variance_estimator": "cluster_robust",
            "cluster_unit": "patient",
        },
    }
    run = {
        "run_id": "review-run",
        "scientific_configuration_sha256": study_contexts.scientific_configuration_sha256(
            before
        ),
    }
    return before, after, run


def _record(before, after, run, code="REPEATED_STAY_IDENTITY_UNAVAILABLE"):
    owner.record_choice(
        before=before, after=after, run=run, decision_code=code, option_id="selected"
    )


def test_host_choices_survive_reloading_and_accumulate_in_one_review(states):
    before, after, run = states
    owner.validate_choice_source(before, run)
    _record(before, after, run)
    progress = owner.matching_progress(json.loads(json.dumps(after)), run)
    assert progress is not None and progress.revision == 2
    owner.validate_choice_source(after, run)
    final = {**after, "revision": 3, "covariates": ["baseline_factor"]}
    _record(after, final, run, "ADJUSTMENT_SET_NOT_USER_CONFIRMED")
    assert len(owner.matching_progress(final, run).choices) == 2
    assert owner.matching_progress(after, run) is None  # old click cannot replay


@pytest.mark.parametrize(
    "change",
    ["question", "revision", "run", "source_digest", "study", "missing", "corrupt"],
)
def test_other_edits_and_missing_or_corrupt_receipts_cannot_revive_review(
    states, change
):
    before, after, run = states
    _record(before, after, run)
    if change == "question":
        after = {**after, "question": "Different study"}
    elif change == "revision":
        after = {**after, "revision": 3}
    elif change == "study":
        after = {**after, "id": "another-study"}
    elif change == "run":
        run = {**run, "run_id": "another-run"}
    elif change == "source_digest":
        run = {**run, "scientific_configuration_sha256": "b" * 64}
    elif change == "missing":
        owner._path(after["id"]).unlink()
    else:
        owner._path(after["id"]).write_text("{}")
    assert owner.matching_progress(after, run) is None
    with pytest.raises(PlanDecisionError, match="outside this plan review"):
        owner.validate_choice_source(after, run)


def test_revision_choice_projection_does_not_grant_old_plan_execution(states):
    before, after, run = states
    _record(before, after, run)
    run = {
        **run,
        "run_type": "full",
        "engine": "easyicu.research_agent.pipeline",
        "gate_status": "blocked",
        "run_status": "human_review_pending",
        "pending_review_reason_codes": ["plan_scientific_changes_required"],
        "artifact_names": [
            "agent_plan.json",
            "scientific_plan_review.json",
            "source_run_manifest.json",
        ],
    }
    snapshot = build_research_workflow_snapshot(
        study=after,
        active_export_present=True,
        active_job=None,
        latest_run=run,
        plan_review_authority={**run, "resumable_here": True, "approval_allowed": True},
        continuing_review_choices=owner.matching_progress(after, run) is not None,
    )
    assert snapshot.current_stage == "plan"
    assert snapshot.next_action_code == "plan_scientific_changes_required"
    assert snapshot.plan_execution_ready is False
    assert "cohort_eligibility" in snapshot.missing_setup_fields
    assert next(s for s in snapshot.stages if s.id == "analysis").status == "blocked"


def test_unrelated_edit_remains_superseded_even_when_original_plan_was_approved(states):
    before, after, run = states
    _record(before, after, run)
    after = {**after, "question": "Unrelated new question"}
    assert owner.matching_progress(after, run) is None


def test_host_choice_handler_persists_progress_and_rejects_unrelated_drift(
    states, monkeypatch
):
    from easyicu.webserver.pi_copilot import service as service_module
    from easyicu.webserver.pi_copilot.contracts import (
        AuthorityBinding,
        PiSessionRecord,
        PiCopilotError,
    )

    before, _after, run = states
    service = service_module.PiCopilotService.__new__(service_module.PiCopilotService)
    record = PiSessionRecord(
        session_id="test",
        project_id="project",
        binding=AuthorityBinding(
            study_context_id=before["id"],
            study_revision=1,
            run_id=run["run_id"],
        ),
    )
    current = dict(before)
    monkeypatch.setattr(service, "_scoped_record", lambda *a, **kw: record)
    monkeypatch.setattr(service, "_stale_details", lambda *a: {})
    monkeypatch.setattr(service, "_save_record", lambda *a: None)
    monkeypatch.setattr(
        service_module.study_contexts, "get_context", lambda *a: dict(current)
    )
    monkeypatch.setattr(
        service_module,
        "list_bound_run_history",
        lambda **kw: [{**run, "project_dir": "/test/plan"}],
    )
    monkeypatch.setattr(
        service_module, "research_pipeline_project_root", lambda *a: "/test"
    )
    monkeypatch.setattr(
        service_module.agent_runs,
        "read_run_review",
        lambda *a: {
            "artifact_payloads": {
                "agent_plan.json": {
                    "steps": [
                        {
                            "model_requirements": [
                                {"analysis_role": "primary", "covariates": ["age"]}
                            ]
                        }
                    ]
                },
                "scientific_plan_review.json": {
                    "findings": [
                        {
                            "code": "REPEATED_STAY_IDENTITY_UNAVAILABLE",
                            "requires_user_authorization": True,
                        },
                        {
                            "code": "ADJUSTMENT_SET_NOT_USER_CONFIRMED",
                            "requires_user_authorization": True,
                        },
                    ]
                },
            }
        },
    )

    def update(patch, **kw):
        assert kw["expected_revision"] == current["revision"]
        current.update(patch)
        current["revision"] += 1
        return dict(current)

    monkeypatch.setattr(service_module.study_contexts, "upsert_context", update)
    result = service.confirm_plan_decision(
        "test",
        project_id="project",
        decision_code="REPEATED_STAY_IDENTITY_UNAVAILABLE",
        option_id="all_icu_stays_clustered",
        expected_revision=1,
        run_id=run["run_id"],
    )
    assert result["next_action"] == "continue_review"
    assert result["remaining_decision_codes"] == ["ADJUSTMENT_SET_NOT_USER_CONFIRMED"]
    assert owner.matching_progress(current, run) is not None

    from easyicu.webserver.pi_copilot import workflow as workflow_owner

    findings = [
        {
            "code": "REPEATED_STAY_IDENTITY_UNAVAILABLE",
            "requires_user_authorization": True,
            "authorization_question": "Select the analysis unit",
        },
        {
            "code": "ADJUSTMENT_SET_NOT_USER_CONFIRMED",
            "requires_user_authorization": True,
            "authorization_question": "Review the adjustment set",
        },
    ]
    review = {
        "artifact_payloads": {
            "agent_plan.json": {
                "steps": [
                    {
                        "model_requirements": [
                            {"analysis_role": "primary", "covariates": ["age"]}
                        ]
                    }
                ]
            },
            "scientific_plan_review.json": {"findings": findings},
        }
    }
    live_run = {
        **run,
        "project_dir": "/test/plan",
        "run_type": "full",
        "engine": "easyicu.research_agent.pipeline",
        "gate_status": "blocked",
        "run_status": "human_review_pending",
        "pending_review_reason_codes": ["plan_scientific_changes_required"],
        "artifact_names": [
            "agent_plan.json",
            "scientific_plan_review.json",
            "source_run_manifest.json",
        ],
    }
    monkeypatch.setattr(
        workflow_owner, "list_bound_run_history", lambda **kw: [live_run]
    )
    monkeypatch.setattr(
        workflow_owner, "workflow_authoritative_run", lambda rows: rows[0]
    )
    monkeypatch.setattr(
        workflow_owner.agent_pipeline_runs,
        "pending_review",
        lambda _id: {
            **run,
            "resumable_here": True,
            "scientific_plan_review": {"findings": findings},
        },
    )
    monkeypatch.setattr(workflow_owner.agent_runs, "read_run_review", lambda *a: review)
    monkeypatch.setattr(workflow_owner, "project_run_outcome", lambda _: {})
    monkeypatch.setattr(workflow_owner.sources, "load_registry", lambda: {})
    projection = workflow_owner.build_project_workflow_projection(
        study_context_id=current["id"], study_override=current
    )
    assert projection.workflow.next_action_code == "plan_scientific_changes_required"
    assert [
        q["code"]
        for q in projection.workflow.plan_review_summary["authorization_questions"]
    ] == ["ADJUSTMENT_SET_NOT_USER_CONFIRMED"]
    assert projection.workflow.plan_execution_ready is False

    # A duplicate choice cannot mint another revision or restart planning.
    with pytest.raises(PiCopilotError) as duplicate:
        service.confirm_plan_decision(
            "test",
            project_id="project",
            decision_code="REPEATED_STAY_IDENTITY_UNAVAILABLE",
            option_id="all_icu_stays_clustered",
            expected_revision=2,
            run_id=run["run_id"],
        )
    assert duplicate.value.code == "plan_decision_not_required_by_review"

    current["question"] = "Changed outside the review"
    with pytest.raises(PiCopilotError) as raised:
        service.confirm_plan_decision(
            "test",
            project_id="project",
            decision_code="ADJUSTMENT_SET_NOT_USER_CONFIRMED",
            option_id="accept_proposed_adjustment",
            expected_revision=2,
            run_id=run["run_id"],
        )
    assert raised.value.code == "plan_decision_source_superseded"
    assert current["revision"] == 2
