"""Web job projection for recoverable scientific-action gaps."""

from __future__ import annotations

from easyicu.research_agent.planning.scientific_action_catalog import (
    ScientificActionGapError,
    resolve_scientific_action_request,
)
from easyicu.webserver.jobs import Job, JobManager


def _unavailable_action_runner(_job: Job) -> None:
    resolution = resolve_scientific_action_request(
        analysis_type="prediction_model",
        action_id="prediction.external_validation",
    )
    raise ScientificActionGapError(resolution)


def test_scientific_action_gap_reaches_web_as_a_user_choice() -> None:
    job = Job("gap-projection", "agent-run")

    JobManager()._run(job, _unavailable_action_runner)

    snapshot = job.snapshot()
    assert snapshot["status"] == "failed"
    assert snapshot["error"].startswith("scientific_action_requires_user_choice:")
    action_events = [
        event
        for event in snapshot["events"]
        if event.get("step") == "scientific_action_gap"
    ]
    assert len(action_events) == 1
    projection = action_events[0]["action"]
    assert action_events[0]["type"] == "progress"
    assert action_events[0]["status"] == "action_required"
    assert "prediction.internal_validation" in action_events[0]["label"]
    assert projection["schema_version"] == "easyicu.scientific_action_resolution/1"
    assert projection["requires_user_confirmation"] is True
    assert projection["alternative_action_ids"] == [
        "prediction.internal_validation"
    ]
    assert snapshot["result"] == {"action_required": projection}


def test_user_cancellation_still_wins_over_late_action_gap() -> None:
    job = Job("gap-cancelled", "agent-run")
    assert job.request_cancel("user_requested") is True

    JobManager()._run(job, _unavailable_action_runner)

    snapshot = job.snapshot()
    assert snapshot["status"] == "cancelled"
    assert snapshot["result"] is None


def test_arbitrary_exception_dict_is_not_projected_to_the_browser() -> None:
    class UnsafeError(RuntimeError):
        user_action_required = {"secret": "must-not-reach-browser"}

    def runner(_job: Job) -> None:
        raise UnsafeError("ordinary failure")

    job = Job("unsafe-projection", "agent-run")
    JobManager()._run(job, runner)

    snapshot = job.snapshot()
    assert snapshot["status"] == "failed"
    assert snapshot["result"] is None
    assert all(event.get("step") != "scientific_action_gap" for event in snapshot["events"])
