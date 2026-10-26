"""Full local durable service across process death, with synthetic science only."""

from __future__ import annotations

import json
import os
import sqlite3
import subprocess
import sys

import pytest

pytest.importorskip("dbos")

_PROCESS = r'''
import json, os, sys, time
from pathlib import Path
from easyicu.research_agent.authority.provider_hard_stop import ProviderHardStopLedger, ProviderHardStopLimits
from easyicu.webserver import durable_research as runtime
from easyicu.webserver import durable_research_store as store
from easyicu.webserver.durable_research_store import DurableResearchError
from easyicu.webserver.jobs import JobManager

root, mode = Path(sys.argv[1]), sys.argv[2]
runtime.runtime_digest = lambda: "changed-code" if mode == "code_drift" else "code"
store.runtime_digest = lambda: "code"
runtime.study_contexts.clear_active_job_if = lambda *args, **kwargs: {}
limits = ProviderHardStopLimits(
    max_provider_attempts_per_run=2, max_provider_attempts_per_batch=2,
    max_total_tokens_per_run=1000000, max_total_tokens_per_batch=1000000,
    max_estimated_cost_usd_per_batch=10, max_wall_clock_seconds_per_task=600,
    input_cost_usd_per_million_tokens=1, output_cost_usd_per_million_tokens=2,
)
ledger_path = root / "original-budget.json"
def ledger(resume):
    return ProviderHardStopLedger(
        path=ledger_path, task_ids=("original-task",), limits=limits,
        batch_id="synthetic-process-test", declaration_sha256="a" * 64,
        resume_existing=resume, require_existing=resume,
    )
def charge(book):
    attempt = book.reserve_transport_attempt(
        task_id="original-task", role="planner", model="synthetic-no-provider",
        messages=[{"role": "user", "content": "synthetic request, no transport"}],
        max_tokens=8, prior_attempt_id=None,
    )
    book.finish_transport_attempt(
        task_id="original-task", attempt_id=attempt,
        usage={"prompt_tokens": 8, "completion_tokens": 3, "total_tokens": 11},
        error_type=None,
    )
def recovery(payload, job):
    with (root / "recovery-calls.jsonl").open("a") as output:
        output.write(json.dumps({"job_id": job.id, "attempt": payload["_resume_attempt"]}) + "\n")
    if not (root / "safe-checkpoint.json").exists():
        raise DurableResearchError("research_auto_resume_safe_checkpoint_missing")
    binding = payload["provider_binding"]
    assert binding["hard_stop_ledger_path"] == str(ledger_path)
    assert binding["hard_stop_task_id"] == "original-task"
    assert binding["hard_stop_declaration_sha256"] == "a" * 64
    book = ledger(True)
    book.start_task("original-task")
    assert len(book.snapshot()["tasks"][0]["calls"]) == 1
    def continuation(bound_job):
        charge(book)
        book.finish_task("original-task")
        bound_job.emit({"type": "progress", "private": "private-scientific-progress"})
        return {"gate": {"status": "passed"}, "private_result": "private-scientific-result"}
    return continuation

manager = JobManager()
manager.enable_persistence(root / "jobs.sqlite3")
service = runtime.DurableResearchService(root, manager=manager, recover_runner=recovery)
if mode == "missing_checkpoint":
    (root / "safe-checkpoint.json").unlink()
service.launch()
if mode in {"crash", "cancel_crash"}:
    job = service.reserve({"study_context_id": "synthetic-study"})
    def initial(bound_job):
        book = ledger(False)
        book.start_task("original-task")
        service.store.bind_provider(bound_job.id, {
            "wrapper_dir": str(root), "hard_stop_ledger_path": str(ledger_path),
            "hard_stop_task_id": "original-task", "hard_stop_declaration_sha256": "a" * 64,
        })
        charge(book)
        (root / "safe-checkpoint.json").write_text(json.dumps({"completed_step": 1}))
        bound_job.emit({"type": "progress", "private": "private-scientific-progress"})
        if mode == "cancel_crash":
            bound_job.request_cancel("user_requested")
        saved = {"id": bound_job.id, "next_seq": bound_job.snapshot()["next_event_seq"],
                 "first_call": book.snapshot()["tasks"][0]["calls"][0]}
        (root / "initial.json").write_text(json.dumps(saved))
        os._exit(74)
    service.activate(job.id, initial)
    deadline = time.monotonic() + 5
    while job.status == "running" and time.monotonic() < deadline:
        time.sleep(.02)
    failure = {"status": job.status, "error": job.error,
               "dbos": service.dispatcher.status(job.id)}
    service.close()
    raise RuntimeError("synthetic_initial_runner_did_not_crash: " + json.dumps(failure))

identity = json.loads((root / "initial.json").read_text())["id"]
deadline = time.monotonic() + 8
while time.monotonic() < deadline:
    job = manager.get(identity)
    state = service.dispatcher.status(identity)
    command = service.store.get(identity)
    if job.status != "running" and command["status"] == "terminal" and state and state["status"] not in {"PENDING", "ENQUEUED", "DELAYED"}:
        break
    time.sleep(.02)
else:
    raise RuntimeError("synthetic_recovered_service_did_not_finish")
snapshot = job.snapshot()
print(json.dumps({"id": identity, "status": snapshot["status"], "error": snapshot["error"],
                  "next_seq": snapshot["next_event_seq"], "attempts": command["attempts"],
                  "binding": command["payload"]["provider_binding"], "dbos": state}))
assert service.close()
'''


def _run(tmp_path, mode, *, crash=False):
    env = os.environ.copy()
    env["EASYICU_HOME"] = str(tmp_path / "home")
    result = subprocess.run(
        [sys.executable, "-c", _PROCESS, str(tmp_path), mode],
        env=env, text=True, capture_output=True, timeout=25,
    )
    assert result.returncode == (74 if crash else 0), result.stdout + result.stderr
    return None if crash else json.loads(result.stdout.strip().splitlines()[-1])


def test_service_restarts_same_job_with_original_ledger_without_resubmission(tmp_path):
    _run(tmp_path, "crash", crash=True)
    original = json.loads((tmp_path / "initial.json").read_text())
    result = _run(tmp_path, "recover")
    assert result["id"] == original["id"]
    assert result["status"] == "done"
    assert result["attempts"] == 2
    assert result["next_seq"] > original["next_seq"]
    assert result["binding"]["hard_stop_task_id"] == "original-task"
    accounting = json.loads((tmp_path / "original-budget.json").read_text())
    calls = accounting["tasks"][0]["calls"]
    assert len(calls) == 2
    assert calls[0] == original["first_call"]
    assert accounting["task_order"] == ["original-task"]
    assert accounting["tasks"][0]["status"] == "completed"
    assert accounting["totals"]["provider_attempts"] == 2
    assert accounting["totals"]["accounted_tokens"] == 22
    recovered = [json.loads(line) for line in (tmp_path / "recovery-calls.jsonl").read_text().splitlines()]
    assert recovered == [{"job_id": original["id"], "attempt": 2}]
    database = tmp_path / "research-workflows.sqlite3"
    with sqlite3.connect(database) as conn:
        inputs = json.loads(conn.execute("SELECT inputs FROM workflow_input").fetchone()[0])
        assert inputs == {"positionalArgs": [original["id"]], "namedArgs": {}}
        assert conn.execute("SELECT DISTINCT serialization FROM operation_outputs").fetchall() == [("portable_json",)]
    raw = database.read_bytes()
    for private in (b"private-scientific-progress", b"private-scientific-result", b"original-budget.json"):
        assert private not in raw


@pytest.mark.parametrize("mode,reason", [
    ("code_drift", "research_auto_resume_runtime_changed"),
    ("missing_checkpoint", "research_auto_resume_safe_checkpoint_missing"),
])
def test_service_recovery_refuses_changed_code_or_missing_checkpoint(tmp_path, mode, reason):
    _run(tmp_path, "crash", crash=True)
    result = _run(tmp_path, mode)
    assert result["status"] == "failed"
    assert result["error"] == reason
    assert result["attempts"] == 2
    calls = json.loads((tmp_path / "original-budget.json").read_text())["tasks"][0]["calls"]
    assert len(calls) == 1
    if mode == "code_drift":
        assert not (tmp_path / "recovery-calls.jsonl").exists()


def test_service_does_not_resume_cancelled_command_after_process_death(tmp_path):
    _run(tmp_path, "cancel_crash", crash=True)
    result = _run(tmp_path, "recover")
    assert result["status"] == "cancelled"
    assert result["attempts"] == 1
    assert not (tmp_path / "recovery-calls.jsonl").exists()
    calls = json.loads((tmp_path / "original-budget.json").read_text())["tasks"][0]["calls"]
    assert len(calls) == 1
