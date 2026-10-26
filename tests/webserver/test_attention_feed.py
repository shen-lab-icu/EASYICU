"""Attention is derived from real job/replay receipts, never scientific data."""

from concurrent.futures import ThreadPoolExecutor
import json

import pytest

from easyicu.webserver.pi_copilot.replay_store import PiConversationReplayStore
from tests.webserver.attention_fixtures import archive, result_ready


@pytest.fixture
def replay(tmp_path):
    return PiConversationReplayStore(tmp_path / "replay")


def page(replay, client="browser-1", after=0, limit=100):
    return replay.attention.page(client_id=client, after=after, limit=limit)


@pytest.mark.parametrize("status,result,error,kind,reason", [
    ("done", {"run_id": "run-1", "human_review_pending": True,
              "gate": {"status": "blocked", "reason": "human_plan_review_required"}},
     None, "needs_review", "human_plan_review_required"),
    ("done", {"run_id": "run-1", "human_review_pending": True,
              "gate": {"status": "blocked", "reason": "capability_substitution_requires_approval"}},
     None, "needs_review", "capability_substitution_requires_approval"),
    ("done", result_ready(), None, "completed", "result_ready"),
    ("failed", {"run_id": "run-1"}, "research_pipeline_execution_failed",
     "failed", "research_pipeline_execution_failed"),
    ("cancelled", {"run_id": "run-1"}, None, "cancelled", "user_cancelled"),
])
def test_each_transition_is_published_once(replay, status, result, error, kind, reason):
    projected = archive(replay, status=status, result=result, error=error)
    for _ in range(3):
        replay.archive_child_job(session_id="conversation-1", project_id="project-1", job=projected)
    items = page(replay)["items"]
    assert len(items) == 1
    assert items[0] == {
        "project_id": "project-1", "conversation_id": "conversation-1", "run_id": "run-1",
        "kind": kind, "reason_code": reason, "sequence": 1, "timestamp": items[0]["timestamp"],
    }
    assert items[0]["timestamp"].endswith("Z")


def test_acknowledgements_survive_restart_and_are_per_client(replay):
    projected = archive(replay, result=result_ready())
    replay.attention.acknowledge(client_id="browser-1", sequences=[1])
    restarted = PiConversationReplayStore(replay.root)
    restarted.archive_child_job(session_id="conversation-1", project_id="project-1", job=projected)
    assert page(restarted)["items"] == []
    assert page(restarted)["unacknowledged_count"] == 0
    assert len(page(restarted, client="desktop-1")["items"]) == 1


def test_review_then_result_on_same_run_are_distinct(replay):
    archive(replay, job_id="plan", result={"run_id": "run-1", "human_review_pending": True})
    archive(replay, job_id="execute", result=result_ready())
    assert [item["kind"] for item in page(replay)["items"]] == ["needs_review", "completed"]


def test_concurrent_runs_and_store_instances_have_one_strict_order(replay):
    def finish(index):
        other = PiConversationReplayStore(replay.root.parent / f"replay-{index}")
        # Independent writers to the same host-owned attention journal.
        other.attention = type(replay.attention)(replay.attention.path)
        archive(other, job_id=f"job-{index}", conversation=f"conversation-{index}",
                result=result_ready(f"run-{index}"))

    with ThreadPoolExecutor(max_workers=4) as pool:
        list(pool.map(finish, range(16)))
    items = page(replay)["items"]
    assert [item["sequence"] for item in items] == list(range(1, 17))
    assert len({item["run_id"] for item in items}) == 16


def test_bounded_paging_does_not_skip_unseen_items(replay):
    for index in range(5):
        archive(replay, job_id=f"job-{index}", result=result_ready(f"run-{index}"))
    replay.attention.acknowledge(client_id="browser-1", sequences=[2])
    first = page(replay, limit=2)
    assert [item["sequence"] for item in first["items"]] == [1, 3]
    assert first["next_after"] == 3 and first["has_more"]
    assert first["unacknowledged_count"] == 4
    second = page(replay, after=first["next_after"], limit=2)
    assert [item["sequence"] for item in second["items"]] == [4, 5]
    assert not second["has_more"]
    # Reading a page never acknowledges it.
    assert page(replay)["unacknowledged_count"] == 4


def test_ack_is_exact_idempotent_and_cannot_ack_a_future_item(replay):
    archive(replay, result=result_ready())
    with pytest.raises(ValueError, match="attention_sequence_unknown"):
        replay.attention.acknowledge(client_id="browser-1", sequences=[1, 2])
    assert len(page(replay)["items"]) == 1  # Entire request rolls back.
    for _ in range(2):
        replay.attention.acknowledge(client_id="browser-1", sequences=[1])
    archive(replay, job_id="job-2", result=result_ready("run-2"))
    assert [item["sequence"] for item in page(replay)["items"]] == [2]


@pytest.mark.parametrize("compile_result,expected,reason", [
    ({"status": "compiled", "approvable": True, "compile_sha256": "a" * 64},
     "needs_review", "target_trial_review"),
    ({"status": "stopped", "reason_code": "target_trial_source_missing"},
     "failed", "target_trial_source_missing"),
    ({"status": "failed", "reason_code": "target_trial_compile_failed"},
     "failed", "target_trial_compile_failed"),
    ({"status": "compiled", "approvable": False}, "needs_review", "target_trial_review"),
])
def test_target_trial_card_is_attention_not_study_completion(replay, compile_result, expected, reason):
    archive(replay, kind="target-trial-compile", result={"target_trial_compile": compile_result})
    item, = page(replay)["items"]
    assert (item["kind"], item["reason_code"], item["run_id"]) == (expected, reason, "job-1")


@pytest.mark.parametrize("result", [None, {}, {"run_id": "run-1"}])
def test_a_finished_reply_or_unproven_result_is_not_a_completion(replay, result):
    archive(replay, result=result)
    assert page(replay)["items"] == []


def test_extraction_is_a_completed_task_with_a_job_coordinate(replay):
    archive(replay, kind="extract", result={"status": "ok"})
    item, = page(replay)["items"]
    assert (item["kind"], item["reason_code"], item["run_id"]) == ("completed", "extraction_ready", "job-1")


def test_blocked_run_keeps_reason_and_never_announces_success(replay):
    archive(replay, result={"run_id": "run-1", "gate": {
        "status": "blocked", "reason": "scientific_validation_failed",
    }})
    item, = page(replay)["items"]
    assert (item["kind"], item["reason_code"]) == ("failed", "scientific_validation_failed")


def test_receipt_recovery_does_not_replace_the_original_failure(replay):
    projected = archive(replay, status="failed", error="research_pipeline_execution_failed")
    replay.archive_child_job(session_id="conversation-1", project_id="project-1", job={
        **projected, "recovered": True, "result_omitted": True,
        "error_code": "job_failed_before_server_restart",
    })
    item, = page(replay)["items"]
    assert item["reason_code"] == "research_pipeline_execution_failed"


def test_feed_never_copies_free_text_or_paths(replay):
    archive(replay, status="failed", error="Synthetic error at /private/example.csv",
            result={"question": "Synthetic question", "path": "/private/example.csv"})
    item, = page(replay)["items"]
    assert item["reason_code"] == "job_failed"
    assert "private" not in json.dumps(item) and "question" not in json.dumps(item)


def test_duplicate_receipt_writers_cannot_assign_two_sequences(replay):
    projected = archive(replay, result=result_ready())
    payload = {"project_id": "project-1", "session_id": "conversation-1", "child_jobs": [projected]}
    def observe(_index):
        type(replay.attention)(replay.attention.path).observe_replay(payload)
    with ThreadPoolExecutor(max_workers=4) as pool:
        list(pool.map(observe, range(12)))
    assert [item["sequence"] for item in page(replay)["items"]] == [1]


@pytest.mark.parametrize("changes", [
    {"status": "running", "cancel_requested": True},
    {"recovered": True, "result_omitted": True},
    {"job_id": "../private/example"},
    {"finished_at_epoch": float("nan")},
    {"finished_at_epoch": True},
])
def test_incomplete_or_invalid_receipts_cannot_announce_completion(replay, changes):
    from easyicu.webserver.jobs import Job
    from easyicu.webserver.pi_copilot.projections import project_job

    job = Job("synthetic", "agent-run")
    job.finish("done", result=result_ready())
    replay.attention.observe_replay({
        "project_id": "project-1", "session_id": "conversation-1",
        "child_jobs": [{**project_job(job.snapshot()), **changes}],
    })
    assert page(replay)["items"] == []


def test_new_attempt_after_a_failure_on_the_same_run_is_a_new_transition(replay):
    archive(replay, job_id="attempt-1", status="failed", result={"run_id": "run-1"}, error="execution_failed")
    archive(replay, job_id="attempt-2", status="failed", result={"run_id": "run-1"}, error="execution_failed")
    assert [item["sequence"] for item in page(replay)["items"]] == [1, 2]


def test_rereading_one_terminal_state_keeps_its_first_reason_and_sequence(replay):
    projected = archive(replay, status="failed", error="specific_execution_failed")
    replay.attention.acknowledge(client_id="browser-1", sequences=[1])
    replay.archive_child_job(session_id="conversation-1", project_id="project-1", job={
        **projected, "error_code": "job_failed_before_server_restart",
    })
    assert page(replay)["items"] == []
    item, = page(replay, client="another-browser")["items"]
    assert (item["sequence"], item["reason_code"]) == (1, "specific_execution_failed")
