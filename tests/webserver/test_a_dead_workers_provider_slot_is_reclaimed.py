"""A model-request slot whose worker is verifiably gone is reclaimed (0.1.3.1).

pilot2 (dd5e3f29b): quitting or restarting the app during a model request left
its slot, and every later request was refused with no way out. A slot is now
reclaimed only on proof: its age exceeds the request timeout, the process that
reserved it no longer exists, its task cannot still be resumed, and its worker
never started or shows a quiesced lifecycle receipt bound to the slot's nonce.
"""

from __future__ import annotations

import json
import subprocess
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import pytest

from easyicu.webserver import metadata_provider_concurrency as slots
from easyicu.webserver.durable_research import DurableResearchService
from easyicu.webserver.metadata_provider_shared_scope import SHARED_ROOT_ENV

SHA = "a" * 64
ACCOUNT = "account-1"
OLD = 4000.0  # seconds; beyond the 1800 s default plus the margin


@pytest.fixture(autouse=True)
def _state(tmp_path, monkeypatch):
    monkeypatch.setenv("EASYICU_HOME", str(tmp_path))
    monkeypatch.delenv(SHARED_ROOT_ENV, raising=False)
    slots.set_protected_tasks_provider(None)
    yield
    slots.set_protected_tasks_provider(None)


def _dead_process() -> tuple[int, float]:
    """A process identity that has already exited."""
    child = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(30)"])
    import psutil

    created = psutil.Process(child.pid).create_time()
    child.kill()
    child.wait()
    return child.pid, created


def _prepare_guard(path: Path) -> str:
    """A worker lease prepared by a process that then died, as after a crash."""
    code = (
        "import sys; from pathlib import Path\n"
        "from easyicu.runtime.host_execution_lease import prepare_host_execution\n"
        f"lease = prepare_host_execution(Path({str(path)!r}), script_sha256={SHA!r})\n"
        "print(lease.nonce, flush=True)\n"
    )
    done = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, check=True)
    return done.stdout.strip()


def _put(tmp_path, name, *, age=OLD, nonce=None, owner=None, timeout=None, task="task-a", guard=None):
    connection = slots._open()
    try:
        pid, created = owner or _dead_process()
        connection.execute(
            "INSERT INTO slots (attempt_id, account, task_id, ledger_path, guard_path, guard_nonce,"
            " worker_sha256, created, request_sha256, request_timeout, owner_pid, owner_created)"
            " VALUES (?,?,?,?,?,?,?,?,?,?,?,?)",
            (name, ACCOUNT, task, str(tmp_path / "ledger"), str(guard or tmp_path / "guards" / name),
             nonce, SHA, time.time() - age, None, timeout, pid, created),
        )
    finally:
        connection.close()
    return name


def _held() -> set[str]:
    connection = slots._open()
    try:
        return {row[0] for row in connection.execute("SELECT attempt_id FROM slots")}
    finally:
        connection.close()


def _log(tmp_path) -> list[dict]:
    path = tmp_path / ".easyicu" / "metadata-provider-reclaims.jsonl"
    return [json.loads(line) for line in path.read_text().splitlines()] if path.exists() else []


def test_a_slot_whose_worker_is_gone_is_reclaimed_with_its_age_and_timeout(tmp_path):
    guard = tmp_path / "guards" / "gone"
    nonce = _prepare_guard(guard)
    _put(tmp_path, "gone", nonce=nonce, guard=guard, timeout=600.0, age=900.0)

    assert slots.reclaim_orphaned() == 1

    assert _held() == set()
    record = _log(tmp_path)[0]
    assert record["reason"] == "slot_reclaimed_worker_gone" and record["attempt_id"] == "gone"
    assert record["timeout_seconds"] == 600.0 and record["age_seconds"] >= 900.0


def test_a_slot_that_never_prepared_a_worker_is_reclaimed_as_never_sent(tmp_path):
    _put(tmp_path, "unsent", nonce=None)

    assert slots.reclaim_orphaned() == 1
    assert _log(tmp_path)[0]["reason"] == "slot_reclaimed_never_sent"


def test_a_slot_saved_without_timeout_or_owner_uses_the_default_timeout(tmp_path):
    _put(tmp_path, "legacy", owner=(None, None), age=1000.0)
    assert slots.reclaim_orphaned() == 0 and _held() == {"legacy"}

    _put(tmp_path, "legacy-old", owner=(None, None), age=OLD)
    assert slots.reclaim_orphaned() == 1 and _held() == {"legacy"}


def test_a_slot_younger_than_the_request_timeout_is_kept(tmp_path):
    _put(tmp_path, "young", age=500.0, timeout=3000.0)

    assert slots.reclaim_orphaned() == 0 and _held() == {"young"}


def test_a_slot_reserved_by_a_running_instance_is_never_reclaimed(tmp_path):
    """Two instances may share one state folder; a live owner keeps its slot."""
    import psutil

    me = psutil.Process()
    _put(tmp_path, "mine", owner=(me.pid, me.create_time()))

    assert slots.reclaim_orphaned() == 0 and _held() == {"mine"}


def test_a_reused_process_id_does_not_count_as_the_owner(tmp_path):
    import psutil

    me = psutil.Process()
    _put(tmp_path, "pid-reuse", owner=(me.pid, me.create_time() - 100.0))

    assert slots.reclaim_orphaned() == 1


@pytest.mark.parametrize("case", ("nonce_mismatch", "receipt_missing", "worker_still_running"))
def test_a_slot_without_proof_that_its_worker_is_gone_is_kept(tmp_path, case):
    guard = tmp_path / "guards" / case
    if case == "receipt_missing":
        nonce = "b" * 32
    else:
        nonce = _prepare_guard(guard)
        if case == "nonce_mismatch":
            nonce = "c" * 32
        else:
            # A launch whose parent is this live process is still unresolved.
            import shutil

            shutil.rmtree(guard)
            from easyicu.runtime.host_execution_lease import prepare_host_execution

            lease = prepare_host_execution(guard, script_sha256=SHA)
            nonce = lease.nonce
    _put(tmp_path, case, nonce=nonce, guard=guard)

    assert slots.reclaim_orphaned() == 0 and _held() == {case}


def test_a_slot_of_a_task_that_can_still_resume_keeps_its_attempt(tmp_path):
    _put(tmp_path, "resumable", task="web-resume")
    _put(tmp_path, "finished", task="web-done")
    slots.set_protected_tasks_provider(lambda: frozenset({"web-resume"}))

    assert slots.reclaim_orphaned() == 1

    assert _held() == {"resumable"}
    assert slots.get("resumable")["task_id"] == "web-resume"


def test_the_service_protects_the_tasks_of_active_commands(tmp_path):
    store = SimpleNamespace(
        active_ids=lambda: ["job-1", "job-2", "job-3"],
        get=lambda job_id: {
            "job-1": {"payload": {"provider_binding": {"hard_stop_task_id": "web-1"}}},
            "job-2": {"payload": {}},
            "job-3": None,
        }[job_id],
    )

    protected = DurableResearchService._resumable_provider_tasks(SimpleNamespace(store=store))

    assert protected == frozenset({"web-1"})


def test_an_imported_shared_scope_slot_is_left_for_its_own_terminal_proof(tmp_path, monkeypatch):
    _put(tmp_path, "imported")
    monkeypatch.setenv(SHARED_ROOT_ENV, str(tmp_path / "shared"))

    assert slots.reclaim_orphaned() == 0


def test_a_new_request_reserves_after_an_orphan_is_reclaimed(tmp_path):
    _put(tmp_path, "orphan")

    slots.reserve(attempt_id="fresh", account=ACCOUNT, task_id="task-b", ledger_path=tmp_path / "l",
                  guard_path=tmp_path / "g", worker_sha256=SHA, timeout_seconds=900.0)

    assert _held() == {"fresh"}
    assert slots.get("fresh")["request_timeout"] == 900.0


def test_a_new_request_is_still_refused_while_the_held_slot_may_be_in_flight(tmp_path):
    _put(tmp_path, "young", age=100.0, timeout=900.0)

    with pytest.raises(slots.MetadataProviderCapacityExhausted):
        slots.reserve(attempt_id="fresh", account=ACCOUNT, task_id="task-b", ledger_path=tmp_path / "l",
                      guard_path=tmp_path / "g", worker_sha256=SHA)

    assert _held() == {"young"}
