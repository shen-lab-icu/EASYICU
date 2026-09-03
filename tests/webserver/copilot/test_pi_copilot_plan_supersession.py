"""Focused stale-plan authority projection contracts."""

from easyicu.webserver.pi_copilot import workflow as workflow_module
from easyicu.webserver.pi_copilot.workflow import build_research_workflow_snapshot

from .test_pi_copilot_research_workflow import _complete_study


def test_stale_plan_review_questions_do_not_replace_fresh_plan_action() -> None:
    study = _complete_study()
    review = {
        "status": "changes_required",
        "findings": [
            {
                "code": "REQUIRED_SENSITIVITY_IS_PROTOCOL_ONLY",
                "requires_user_authorization": True,
                "authorization_question": "Keep the requested sensitivity?",
                "message": "A legacy sensitivity is absent from this old plan.",
            }
        ],
    }
    snapshot = build_research_workflow_snapshot(
        study=study,
        active_export_present=True,
        active_job=None,
        latest_run={
            "run_id": "run-old-plan",
            "run_type": "full",
            "engine": "easyicu.research_agent.pipeline",
            "gate_status": "blocked",
            "run_status": "human_review_pending",
            "pending_review_reason_codes": ["plan_scientific_changes_required"],
            "scientific_configuration_sha256": "a" * 64,
            "artifact_names": ["agent_plan.json", "scientific_plan_review.json"],
        },
        plan_review_authority={
            "run_id": "run-old-plan",
            "resumable_here": True,
            "scientific_configuration_sha256": "a" * 64,
            "scientific_plan_review": review,
        },
    )

    enriched = workflow_module._enrich_plan_review(
        snapshot,
        study=study,
        review={"artifact_payloads": {"agent_plan.json": {}}},
    )

    assert enriched.next_action_code == "plan_configuration_superseded"
