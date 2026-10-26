"""Real DBOS subprocess recovery; only synthetic descriptor owners execute."""

from __future__ import annotations

import json
import os
import sqlite3
import subprocess
import sys

import pytest

pytest.importorskip("dbos")

_CHILD = r'''
import json, os, sys, threading, time
from pathlib import Path
from easyicu.webserver.durable_workflows import DurableWorkflowDispatcher

root = Path(sys.argv[1])
mode = sys.argv[2]
started = threading.Event()
release = threading.Event()
attempts = root / "attempts.jsonl"
checkpoint = root / "owner-checkpoint.json"
def execute(identifier):
    with attempts.open("a") as f:
        f.write(json.dumps({"id": identifier, "mode": mode}) + "\n")
        f.flush()
        os.fsync(f.fileno())
    # This represents the scientific owner resolving current authority and
    # original remaining budget, before any synthetic operation.
    authority = json.loads((root / "authority.json").read_text())
    if not authority["approved"]:
        return {"status": "blocked", "reason_code": "authority_revoked"}
    if mode == "exception":
        raise RuntimeError("provider-secret-must-never-be-persisted")
    if mode in {"cancel", "drain"}:
        started.set()
        release.wait(10)
        return {"status": "cancelled"}
    state = json.loads(checkpoint.read_text()) if checkpoint.exists() else {"spent": 0}
    if state["spent"] == 0:
        state["spent"] = 1
        checkpoint.write_text(json.dumps(state))
    if mode in {"crash", "always_crash"}:
        os._exit(73)
    if state["spent"] >= authority["budget"]:
        return {"status": "blocked", "reason_code": "budget_exhausted"}
    state["spent"] += 1
    checkpoint.write_text(json.dumps(state))
    return {"status": "done", "job_id": identifier, "secret": "drop-this-field"}

dispatcher = DurableWorkflowDispatcher(root / "workflows.sqlite3", execute).launch()
identifier = "descriptor-one"
if mode in {"crash", "always_crash", "start", "exception", "cancel", "drain"}:
    dispatcher.submit(identifier)
if mode in {"cancel", "drain"}:
    assert started.wait(5)
    if mode == "cancel":
        dispatcher.cancel(identifier)
    else:
        before = time.monotonic()
        assert dispatcher.close(timeout=.05) is False
        assert time.monotonic() - before < .5
    release.set()
    if mode == "drain":
        assert dispatcher.close()
        print(json.dumps({"drained": True}))
        sys.exit(0)
deadline = time.monotonic() + 8
while time.monotonic() < deadline:
    status = dispatcher.status(identifier)
    if status and status["status"] not in {"PENDING", "ENQUEUED", "DELAYED"}:
        break
    time.sleep(.02)
else:
    raise RuntimeError("synthetic_workflow_did_not_finish")
if mode == "recover":
    # Duplicate delivery must return the same durable completion, not execute
    # a second copy of the callback or spend budget again.
    assert dispatcher.submit(identifier) == status["workflow_id"]
    time.sleep(.1)
print(json.dumps(status))
assert dispatcher.close()
'''


def _child(tmp_path, mode, *, crash=False):
    result = subprocess.run(
        [sys.executable, "-c", _CHILD, str(tmp_path), mode],
        env=os.environ.copy(), text=True, capture_output=True, timeout=20,
    )
    assert result.returncode == (73 if crash else 0), result.stdout + result.stderr
    return None if crash else json.loads(result.stdout.strip().splitlines()[-1])


def _authority(tmp_path, *, approved=True, budget=2):
    (tmp_path / "authority.json").write_text(json.dumps({"approved": approved, "budget": budget}))


def test_process_death_automatically_continues_original_descriptor_once(tmp_path):
    _authority(tmp_path)
    _child(tmp_path, "crash", crash=True)
    result = _child(tmp_path, "recover")  # no submission before startup recovery
    assert result["receipt"] == {"status": "done", "job_id": "descriptor-one"}
    assert len((tmp_path / "attempts.jsonl").read_text().splitlines()) == 2
    assert json.loads((tmp_path / "owner-checkpoint.json").read_text()) == {"spent": 2}
    path = tmp_path / "workflows.sqlite3"
    with sqlite3.connect(path) as conn:
        assert conn.execute("SELECT DISTINCT serialization FROM workflow_status").fetchall() == [("portable_json",)]
        assert conn.execute("SELECT DISTINCT serialization FROM operation_outputs").fetchall() == [("portable_json",)]
        inputs = conn.execute("SELECT inputs FROM workflow_input").fetchone()[0]
        assert json.loads(inputs) == {"positionalArgs": ["descriptor-one"], "namedArgs": {}}
    assert b"drop-this-field" not in path.read_bytes()
    if os.name != "nt":
        assert path.stat().st_mode & 0o777 == 0o600


@pytest.mark.parametrize("approved,budget,reason", [(False, 2, "authority_revoked"), (True, 1, "budget_exhausted")])
def test_recovery_rechecks_owner_authority_and_original_budget(tmp_path, approved, budget, reason):
    _authority(tmp_path)
    _child(tmp_path, "crash", crash=True)
    _authority(tmp_path, approved=approved, budget=budget)
    result = _child(tmp_path, "recover")
    assert result["receipt"] == {"status": "blocked", "reason_code": reason}
    assert json.loads((tmp_path / "owner-checkpoint.json").read_text()) == {"spent": 1}


def test_ordinary_exception_is_sanitized_and_not_retried(tmp_path):
    _authority(tmp_path)
    first = _child(tmp_path, "exception")
    second = _child(tmp_path, "recover")
    assert first == second
    assert first["receipt"] == {"status": "failed", "reason_code": "durable_callback_failed"}
    assert len((tmp_path / "attempts.jsonl").read_text().splitlines()) == 1
    assert b"provider-secret" not in (tmp_path / "workflows.sqlite3").read_bytes()


def test_repeated_process_death_has_a_finite_recovery_limit(tmp_path):
    _authority(tmp_path)
    # One original attempt and at most three crash-recovery attempts. Calling
    # submit again cannot reset DBOS's persisted recovery counter.
    for _ in range(4):
        _child(tmp_path, "always_crash", crash=True)
    result = _child(tmp_path, "recover")
    assert result["status"] == "MAX_RECOVERY_ATTEMPTS_EXCEEDED"
    assert len((tmp_path / "attempts.jsonl").read_text().splitlines()) == 4


def test_explicit_cancellation_is_not_recovered(tmp_path):
    _authority(tmp_path)
    assert _child(tmp_path, "cancel")["status"] == "CANCELLED"
    assert _child(tmp_path, "recover")["status"] == "CANCELLED"
    assert len((tmp_path / "attempts.jsonl").read_text().splitlines()) == 1


def test_close_is_bounded_and_reports_callback_still_draining(tmp_path):
    _authority(tmp_path)
    assert _child(tmp_path, "drain") == {"drained": True}


def test_non_json_persistent_rows_are_rejected_before_deserialization(tmp_path):
    from easyicu.webserver.durable_workflows import DurableWorkflowDispatcher, DurableWorkflowError

    _authority(tmp_path)
    _child(tmp_path, "start")
    path = tmp_path / "workflows.sqlite3"
    with sqlite3.connect(path) as conn:
        conn.execute("UPDATE workflow_status SET serialization='py_pickle'")
    with pytest.raises(DurableWorkflowError, match="durable_database_serialization_invalid"):
        DurableWorkflowDispatcher(path, lambda _: pytest.fail("unsafe callback")).launch()


def test_closing_unused_dispatcher_does_not_destroy_active_owner(tmp_path):
    from easyicu.webserver.durable_workflows import DurableWorkflowDispatcher

    owner = DurableWorkflowDispatcher(tmp_path / "owner.sqlite3", lambda _: {"status": "done"}).launch()
    try:
        unused = DurableWorkflowDispatcher(tmp_path / "unused.sqlite3", lambda _: {})
        assert unused.close()
        assert owner.submit("still-owned") == "easyicu-recovery:still-owned"
    finally:
        assert owner.close()
