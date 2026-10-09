"""A host decision is the state a button answers, named once.

The workflow projection offers the decision each job-starting button can
answer, the browser echoes it with the submission, and the submission owner
compares it field by field and derives its key (``host_action_contracts``).
The key names the decision within its study, so it holds no conversation.
"""

from __future__ import annotations

from typing import Any

import pytest
from fastapi import HTTPException

from easyicu.webserver import study_contexts
from easyicu.webserver.host_action_contracts import (
    ExecutionRetryDecision,
    HostDecisionOffers,
    PlanReviewDecision,
    PlanTransitionDecision,
    decision_key,
    host_action_id,
    mismatched_fields,
    retry_options_sha256,
    review_authority_sha256,
)
from easyicu.webserver.pi_copilot.workflow import (
    build_research_workflow_snapshot,
    host_decision_offers,
)
from easyicu.webserver.routes import agent as agent_routes

DIGEST = "a" * 64
PREPARE = PlanTransitionDecision(
    next_action_code="plan_execution_upgrade_required",
    scientific_configuration_sha256=DIGEST,
    source_run_id="run_candidate",
)
REVIEW = PlanReviewDecision(
    run_id="run_candidate",
    scientific_configuration_sha256=DIGEST,
    review_authority_sha256="b" * 64,
)
RETRY = ExecutionRetryDecision(
    source_run_id="run_failed",
    gate_reason="research_pipeline_execution_failed",
    scientific_configuration_sha256=DIGEST,
)


def test_each_family_has_its_key_and_no_key_names_a_conversation() -> None:
    assert decision_key(PREPARE) == (
        "plan:aaaaaaaaaaaaaaaa:run_candidate:plan_execution_upgrade_required"
    )
    assert decision_key(REVIEW, review_decision="approved") == (
        "review:run_candidate:approved:aaaaaaaaaaaaaaaa:bbbbbbbbbbbbbbbb"
    )
    assert decision_key(REVIEW, review_decision="approved") != decision_key(
        REVIEW, review_decision="rejected"
    )
    with pytest.raises(ValueError):
        decision_key(REVIEW)
    openai = retry_options_sha256(
        report_only=False, provider="openai", credential_source="pi_verified"
    )
    codex = retry_options_sha256(
        report_only=False, provider="codex", credential_source="codex_user_auth"
    )
    assert decision_key(RETRY, retry_options=openai) != decision_key(
        RETRY, retry_options=codex
    )
    with pytest.raises(ValueError):
        decision_key(RETRY)
    key = decision_key(PREPARE)
    # The conversation row is per conversation; the decision is not.
    assert host_action_id("pi_a", "prepare_analysis_data", key) != host_action_id(
        "pi_b", "prepare_analysis_data", key
    )


def test_a_paused_review_digest_ignores_order_and_refuses_an_unbound_request() -> None:
    first = {"review_id": "review-0123456789abcdef", "authority_sha256": "1" * 64}
    second = {"review_id": "review-fedcba9876543210", "authority_sha256": "2" * 64}

    assert review_authority_sha256([first, second]) == review_authority_sha256(
        [second, first]
    )
    assert review_authority_sha256([first]) != review_authority_sha256([first, second])
    assert review_authority_sha256([first, {"review_id": "review-x"}]) is None
    assert review_authority_sha256([]) is None


def test_mismatched_fields_name_what_changed() -> None:
    assert mismatched_fields(PREPARE, PREPARE) == ()
    assert mismatched_fields(PREPARE, None) == ("family",)
    assert mismatched_fields(PREPARE, RETRY) == ("family",)
    moved = PREPARE.model_copy(
        update={
            "source_run_id": "run_newer",
            "next_action_code": "operator_plan_approval_required",
        }
    )
    assert mismatched_fields(PREPARE, moved) == ("next_action_code", "source_run_id")


def _approval_state(study: dict[str, Any]) -> tuple[dict, dict]:
    digest = study_contexts.scientific_configuration_sha256(study)
    candidate = {
        "run_id": "run_candidate",
        "study_id": study["id"],
        "run_type": "full",
        "engine": "easyicu.research_agent.pipeline",
        "gate_status": "blocked",
        "gate_reason": "human_plan_review_required",
        "run_status": "human_review_pending",
        "pending_review_reason_codes": ["operator_plan_approval_required"],
        "scientific_configuration_sha256": digest,
        "artifact_names": ["agent_plan.json", "source_run_manifest.json"],
    }
    review = {
        "run_id": "run_candidate",
        "resumable_here": True,
        "scientific_configuration_sha256": digest,
        "budget_mode": "full_reviewed",
        "research_input_state": "prepared",
        "plan_approval_allowed": True,
        "requests": [
            {
                "review_id": "review-0123456789abcdef",
                "authority_sha256": "c" * 64,
                "reason_code": "operator_plan_approval_required",
            }
        ],
    }
    return candidate, review


def _snapshot(study: dict, candidate: dict, review: dict, active_job: Any = None):
    return build_research_workflow_snapshot(
        study=study,
        active_export_present=True,
        active_job=active_job,
        latest_run=candidate,
        plan_review_authority=review,
    )


def test_the_projection_offers_each_family_its_decision(
    workflow_complete_study: dict,
) -> None:
    study = workflow_complete_study
    candidate, review = _approval_state(study)
    snapshot = _snapshot(study, candidate, review)
    assert snapshot.next_action_code == "operator_plan_approval_required"
    digest = study_contexts.scientific_configuration_sha256(study)

    offers = host_decision_offers(
        snapshot, study=study, latest_run=candidate, plan_review_authority=review
    )

    assert offers.plan_review == PlanReviewDecision(
        run_id="run_candidate",
        scientific_configuration_sha256=digest,
        review_authority_sha256=review_authority_sha256(review["requests"]),
    )
    # The same state can also be planned afresh, or its run retried.
    assert offers.plan_transition == PlanTransitionDecision(
        next_action_code="operator_plan_approval_required",
        scientific_configuration_sha256=digest,
        source_run_id="run_candidate",
    )
    assert offers.execution_retry.source_run_id == "run_candidate"
    assert offers.execution_retry.gate_reason == "human_plan_review_required"
    unbound = host_decision_offers(
        snapshot,
        study=study,
        latest_run=candidate,
        plan_review_authority={**review, "requests": []},
    )
    assert unbound.plan_review is None
    # Coordinates outside the contract offer nothing to echo and do not raise.
    odd = host_decision_offers(
        snapshot,
        study=study,
        latest_run={**candidate, "run_id": "/private/run"},
        plan_review_authority=review,
    )
    assert odd == HostDecisionOffers()
    assert (
        host_decision_offers(
            snapshot, study={}, latest_run=candidate, plan_review_authority=review
        )
        == HostDecisionOffers()
    )


def test_a_running_job_outranks_the_review_it_paused_for(
    workflow_complete_study: dict,
) -> None:
    study = workflow_complete_study
    candidate, review = _approval_state(study)

    running = _snapshot(
        study,
        candidate,
        review,
        active_job={"kind": "agent-run", "status": "running", "events": []},
    )

    assert running.next_action_code == "research_planning_running"
    assert running.plan_execution_ready is False
    assert (
        host_decision_offers(
            running, study=study, latest_run=candidate, plan_review_authority=review
        ).plan_review
        is None
    )


_RETRY_BODY = {
    "engine": "research_agent_pipeline",
    "study_context_id": "study-1",
    "llm_provider": "openai",
    "credential_source": "pi_verified",
    "planner_start_mode": "auto",
    "execution_resume_source_run_id": "run_failed",
    "report_only": True,
    "host_action": {
        "session_id": "pi_1",
        "project_id": "project-1",
        "action_code": "retry_analysis",
        "decision": RETRY.model_dump(),
    },
}


@pytest.fixture
def owner_calls(monkeypatch: pytest.MonkeyPatch) -> list:
    calls: list = []

    def submit_with_host_action(host_action, *, study_context_id, submit, **key_inputs):
        calls.append((host_action.action_code, study_context_id, key_inputs))
        return {"job_id": "job-host", "reused": False}

    monkeypatch.setattr(
        agent_routes.host_action_jobs,
        "submit_with_host_action",
        submit_with_host_action,
    )
    monkeypatch.setattr(
        agent_routes, "submit_agent_run", lambda body, **_: {"job_id": "job-legacy"}
    )
    monkeypatch.setattr(
        agent_routes,
        "submit_agent_run_review",
        lambda body, **_: {"job_id": "job-legacy-review"},
    )
    return calls


def test_the_agent_run_route_hands_a_host_decision_to_its_owner(
    owner_calls: list,
) -> None:
    assert (
        agent_routes.jobs_agent_run(dict(_RETRY_BODY), request=None)["job_id"]
        == "job-host"
    )
    assert owner_calls == [
        (
            "retry_analysis",
            "study-1",
            {
                "retry_options": retry_options_sha256(
                    report_only=True, provider="openai", credential_source="pi_verified"
                )
            },
        )
    ]
    legacy = {key: value for key, value in _RETRY_BODY.items() if key != "host_action"}
    assert agent_routes.jobs_agent_run(legacy, request=None) == {"job_id": "job-legacy"}
    assert len(owner_calls) == 1


@pytest.mark.parametrize(
    "change",
    [
        {"engine": "native_summary"},
        {"execution_resume_source_run_id": "run_other"},
        {
            "host_action": {
                **_RETRY_BODY["host_action"],
                "action_code": "prepare_analysis_data",
                "decision": PREPARE.model_dump(),
            }
        },
    ],
)
def test_an_agent_run_shaped_for_another_decision_is_refused(
    owner_calls: list, change: dict
) -> None:
    with pytest.raises(HTTPException) as caught:
        agent_routes.jobs_agent_run({**_RETRY_BODY, **change}, request=None)
    assert caught.value.status_code == 400
    assert caught.value.detail["error"] == "host_action_request_mismatch"
    assert owner_calls == []


def test_the_review_route_hands_its_decision_and_answer_to_its_owner(
    owner_calls: list,
) -> None:
    body = {
        "run_id": "run_candidate",
        "study_context_id": "study-1",
        "decision": "Approved",
        "external_llm_opt_in": True,
        "host_action": {
            "session_id": "pi_1",
            "project_id": "project-1",
            "action_code": "execute_plan",
            "decision": REVIEW.model_dump(),
        },
    }
    assert (
        agent_routes.jobs_agent_run_review(dict(body), request=None)["job_id"]
        == "job-host"
    )
    assert owner_calls == [("execute_plan", "study-1", {"review_decision": "approved"})]

    with pytest.raises(HTTPException) as caught:
        agent_routes.jobs_agent_run_review(
            {**body, "run_id": "run_other"}, request=None
        )
    assert caught.value.detail["error"] == "host_action_request_mismatch"
    with pytest.raises(HTTPException) as caught:
        agent_routes.jobs_agent_run_review(
            {
                **body,
                "host_action": {**body["host_action"], "decision_key": "review:forged"},
            },
            request=None,
        )
    assert (caught.value.status_code, caught.value.detail["error"]) == (
        400,
        "host_action_invalid",
    )
    legacy = {key: value for key, value in body.items() if key != "host_action"}
    assert agent_routes.jobs_agent_run_review(legacy, request=None) == {
        "job_id": "job-legacy-review"
    }
