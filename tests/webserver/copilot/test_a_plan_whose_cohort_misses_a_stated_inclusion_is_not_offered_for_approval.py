"""A plan whose cohort misses an inclusion the study states is not offered for approval.

Step 2b of the population spec design: the host compiles the plan's cohort
from the population the Planner states, and an inclusion the cohort does not
apply no longer stops planning.  The plan is reviewed, but its approval is
refused with a typed reason: ``population_inclusion_requires_extraction``
when an extraction of the study's own population can apply it, and
``population_inclusion_not_applied`` when nothing can as stated.

The workflow read neither reason as a plan review, so it showed such a plan
as ready and named preparing its data as the next action.  It now shows the
plan under review with the reason as its next action, offers no approval,
and lets a fresh candidate plan be generated; once the study binds the
export extracted for its own population, the plan is superseded and planned
again on that export.  Synthetic studies only.
"""

from __future__ import annotations

from typing import Any, Mapping

from easyicu.research_agent.planning.population_compile import (
    POPULATION_APPROVAL_STOPS,
)
from easyicu.webserver import agent_pipeline_runs
from easyicu.webserver import study_contexts as study_context_owner
from easyicu.webserver.pi_copilot.workflow import (
    build_research_workflow_snapshot,
    host_decision_offers,
)
from easyicu.webserver.routes import agent as agent_routes
from tests.webserver.copilot.research_workflow_fixtures import complete_study

_EXTRACT = "population_inclusion_requires_extraction"
_UNAPPLIED = "population_inclusion_not_applied"
_RUN = "run-population-stop"


def _run_and_review(
    study: Mapping[str, Any], reasons: list[str], *, digest: str
) -> tuple[dict[str, Any], dict[str, Any]]:
    run = {
        "run_id": _RUN,
        "run_type": "full",
        "engine": "easyicu.research_agent.pipeline",
        "gate_status": "blocked",
        "run_status": "human_review_pending",
        "pending_review_reason_codes": list(reasons),
        "scientific_configuration_sha256": digest,
        "artifact_names": ["agent_plan.json", "scientific_plan_review.json"],
    }
    review = {
        "run_id": _RUN,
        "resumable_here": True,
        "scientific_configuration_sha256": digest,
        "budget_mode": "full_reviewed",
        "requests": [
            {
                "review_id": f"review-{index}",
                "kind": "scientific_stop",
                "summary": "This plan cannot be approved.",
                "authority_sha256": "a" * 64,
                "reason_code": reason,
                "approval_allowed": False,
            }
            for index, reason in enumerate(reasons)
        ],
        "plan_approval_allowed": False,
        "scientific_plan_review": {
            "status": "ready_for_approval",
            "approval_allowed": True,
            "score": 90,
            "findings": [],
        },
    }
    return run, review


def _workflow(study: Mapping[str, Any], reasons: list[str], *, digest: str = ""):
    digest = digest or study_context_owner.scientific_configuration_sha256(dict(study))
    run, review = _run_and_review(study, reasons, digest=digest)
    snapshot = build_research_workflow_snapshot(
        study=study,
        active_export_present=True,
        active_job=None,
        latest_run=run,
        plan_review_authority=review,
    )
    offers = host_decision_offers(
        snapshot, study=study, latest_run=run, plan_review_authority=review
    )
    return snapshot, offers


def _plan_stage(snapshot) -> tuple[str, str]:
    stage = next(row for row in snapshot.stages if row.id == "plan")
    return stage.status, stage.reason_code


def test_an_inclusion_an_extraction_can_apply_names_that_extraction() -> None:
    snapshot, offers = _workflow(complete_study(), [_EXTRACT])

    assert _plan_stage(snapshot) == ("review_required", _EXTRACT)
    assert snapshot.next_action_code == _EXTRACT
    assert snapshot.plan_execution_ready is False
    assert offers.plan_review is None


def test_an_inclusion_nothing_can_apply_leaves_the_plan_unapprovable() -> None:
    snapshot, offers = _workflow(complete_study(), [_UNAPPLIED])

    assert _plan_stage(snapshot) == ("review_required", _UNAPPLIED)
    assert snapshot.next_action_code == _UNAPPLIED
    assert snapshot.plan_execution_ready is False
    assert offers.plan_review is None
    # The analysis waits for a plan that can be approved.
    analysis = next(row for row in snapshot.stages if row.id == "analysis")
    assert analysis.status == "blocked"


def test_the_extraction_is_named_first_when_both_stop_the_plan() -> None:
    snapshot, _offers = _workflow(complete_study(), [_UNAPPLIED, _EXTRACT])

    assert snapshot.next_action_code == _EXTRACT


def test_the_export_extracted_for_the_study_supersedes_the_plan() -> None:
    study = complete_study()
    planned_on = study_context_owner.scientific_configuration_sha256(dict(study))
    # The study binds the export extracted for its own population (BX).
    rebound = {
        **study,
        "data_source": {
            "path": "/private/prepared/own-population",
            "database": "mimiciv",
        },
    }

    snapshot, _offers = _workflow(rebound, [_EXTRACT], digest=planned_on)

    assert _plan_stage(snapshot) == ("ready", "plan_configuration_superseded")
    assert snapshot.next_action_code in agent_routes._CANDIDATE_PLAN_WORKFLOW_CODES


def test_a_fresh_candidate_plan_from_either_stop_reads_no_patient_row() -> None:
    assert set(POPULATION_APPROVAL_STOPS.values()) == {_EXTRACT, _UNAPPLIED}
    assert {_EXTRACT, _UNAPPLIED} <= agent_routes._CANDIDATE_PLAN_WORKFLOW_CODES


class _Request:
    def __init__(self, reason: str) -> None:
        self.payload = {"reason": reason, "approval_allowed": False}


def test_the_pending_review_keeps_the_typed_reason() -> None:
    current = {
        "schema_version": agent_pipeline_runs.CURRENT_SCIENTIFIC_REVIEW_SCHEMA_VERSION,
        "approval_allowed": True,
    }

    for reason in (_EXTRACT, _UNAPPLIED):
        assert (
            agent_pipeline_runs._pending_review_reason_code(
                request=_Request(reason),
                plan_recommendation_complete=True,
                scientific_plan_review=current,
            )
            == reason
        )
