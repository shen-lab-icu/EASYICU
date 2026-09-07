"""Shared approval authority and actionable owner refusals."""
from __future__ import annotations

import pytest
from fastapi import HTTPException

from easyicu.webserver.pi_copilot.contracts import plan_approval_allowed


def test_plan_approval_authority_is_fail_closed_for_every_consumer() -> None:
    """Only an explicit ``True`` is approval authority.

    The model-facing projection once read a missing flag as permissive while
    the submit route read it as blocking, so a manifest that predates the field
    advertised an approvable plan that the route then rejected with 409.
    """

    assert plan_approval_allowed({"plan_approval_allowed": True}) is True
    assert plan_approval_allowed({"plan_approval_allowed": False}) is False
    assert plan_approval_allowed({"plan_approval_allowed": None}) is False
    assert plan_approval_allowed({}) is False
    assert plan_approval_allowed(None) is False
    # A truthy non-boolean is not a compiled decision either.
    assert plan_approval_allowed({"plan_approval_allowed": "true"}) is False


def test_plan_approval_refusal_names_the_blockers_it_is_refusing_on(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A bare 409 is a dead end for the researcher.

    Approval is refused by the same fail-closed reading every consumer shares,
    so the refusal has to carry the blocker codes that produced it.
    """

    from easyicu.webserver.routes import agent as agent_routes

    monkeypatch.setattr(
        agent_routes.settings_store,
        "load_settings",
        lambda: {"ai_enabled": True},
    )
    monkeypatch.setattr(
        agent_routes.agent_pipeline_runs,
        "pending_review",
        lambda _run_id: {
            "run_id": "run-blocked-plan",
            "study_id": "study-blocked-plan",
            "resumable_here": True,
            "budget_mode": "full_reviewed",
            "plan_approval_allowed": False,
            "scientific_plan_review": {
                "status": "changes_required",
                "findings": [
                    {
                        "code": "REPEATED_STAY_IDENTITY_UNAVAILABLE",
                        "severity": "blocker",
                    },
                    {
                        "code": "UNADJUSTED_ASSOCIATION_NOT_ARTICLE_GRADE",
                        "severity": "major",
                    },
                ],
            },
        },
    )

    with pytest.raises(HTTPException) as exc:
        agent_routes.submit_agent_run_review(
            {
                "run_id": "run-blocked-plan",
                "study_context_id": "study-blocked-plan",
                "decision": "approved",
                "external_llm_opt_in": True,
            }
        )

    assert exc.value.status_code == 409
    assert exc.value.detail["error"] == "scientific_plan_review_changes_required"
    assert exc.value.detail["blocking_codes"] == [
        "REPEATED_STAY_IDENTITY_UNAVAILABLE"
    ]
    assert exc.value.detail["scientific_plan_review_status"] == "changes_required"
