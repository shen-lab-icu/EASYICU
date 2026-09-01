from __future__ import annotations

import multiprocessing
import time
from queue import Empty
from pathlib import Path


def _workspace_create_worker(root: str, ready, start, queue, content: str) -> None:
    from easyicu.webserver.pi_copilot.contracts import PiCopilotError
    from easyicu.webserver.pi_copilot.workspace import ProjectWorkspace

    original = ProjectWorkspace._atomic_write

    def delayed(candidate: Path, encoded: bytes) -> None:
        time.sleep(0.2)
        original(candidate, encoded)

    ProjectWorkspace._atomic_write = staticmethod(delayed)
    # Import and patching are done: only now is this process able to race.
    ready.put(content)
    start.wait()
    try:
        ProjectWorkspace(Path(root)).write_file("project", "result.md", content)
        queue.put(("ok", content))
    except PiCopilotError as exc:
        queue.put(("error", exc.code))


def _authority_bind_worker(path: str, ready, start, queue, study_id: str) -> None:
    from easyicu.webserver.pi_copilot.contracts import PiCopilotError
    from easyicu.webserver.pi_copilot.project_authority import ProjectAuthorityStore

    store = ProjectAuthorityStore(Path(path))
    original = store._write

    def delayed(rows) -> None:
        time.sleep(0.2)
        original(rows)

    store._write = delayed
    # Import and patching are done: only now is this process able to race.
    ready.put(study_id)
    start.wait()
    try:
        queue.put(("ok", store.bind("project", study_id)))
    except PiCopilotError as exc:
        queue.put(("error", exc.code))


#: Total budget for BOTH children to finish booting and patching.
#:
#: A "spawn" child starts a fresh interpreter and imports easyicu before it can
#: race for anything, and CI runs the suite under ``-n auto``. This budget is
#: transport plumbing and scales with machine load; it is deliberately separate
#: from the two below so widening it cannot mask a slow lock or a slow exit.
_WORKER_READY_TIMEOUT_SECONDS = 120

#: Total budget for the contended operation once both children are
#: parked on the start gate. Short on purpose: the work is one 0.2s delayed
#: write plus lock contention, so overrunning this is a lock problem.
_RESULT_TIMEOUT_SECONDS = 30

#: How long a child that has already reported may take to exit. Short on
#: purpose: it has nothing left to do but unwind, so a slow exit is a hang.
_EXIT_TIMEOUT_SECONDS = 10


def _drain(
    queue,
    count: int,
    budget: float,
    what: str,
    workers,
    *,
    stalled_hint: str,
) -> list:
    """Collect ``count`` reports within ONE shared budget for the whole phase.

    Passing the budget to each ``queue.get`` separately meant a phase declared
    at 120s could legitimately take 240s with two workers, so the constants
    named a per-worker wait while reading like a phase budget. The deadline is
    absolute and computed once: whatever the phase is declared to cost is what
    it can cost in total, regardless of how many workers it collects from.
    """

    deadline = time.monotonic() + budget
    collected: list = []
    for index in range(count):
        remaining = deadline - time.monotonic()
        try:
            if remaining <= 0:
                raise Empty
            collected.append(queue.get(timeout=remaining))
        except Empty:
            states = [
                f"pid={w.pid} alive={w.is_alive()} exitcode={w.exitcode}"
                for w in workers
            ]
            for worker in workers:
                worker.kill()
                worker.join(timeout=_EXIT_TIMEOUT_SECONDS)
            raise AssertionError(
                f"only {index} of {count} workers reported {what} within the "
                f"{budget}s phase budget; worker states at timeout: "
                f"{'; '.join(states)}. {stalled_hint} A non-zero exitcode "
                "means the child crashed before it could report."
            ) from None
    return collected


def _run_pair(target, args: tuple[str, ...]) -> list[tuple[str, str]]:
    """Race two processes for one file lock, and actually make them race.

    The previous version called ``start.set()`` immediately after ``start()``,
    so the gate was already open before either child had finished importing
    easyicu and installing its delay patch. Whichever child booted first could
    complete the whole operation before the second one existed, and the
    assertions still passed -- the loser hit "already created" rather than the
    concurrent path the test is named for. Under load that also blew the 10s
    result wait, so the same defect produced both a false green and a false
    red.

    Now each child reports ``ready`` once its imports and patching are done,
    the parent collects both under a generous boot budget, and only then opens
    the gate -- at which point both children are provably parked on it.
    """

    context = multiprocessing.get_context("spawn")
    ready = context.Queue()
    start = context.Event()
    queue = context.Queue()
    workers = [
        context.Process(target=target, args=(*args[:-2], ready, start, queue, value))
        for value in args[-2:]
    ]
    for worker in workers:
        worker.start()

    # 1-2. Both children are booted and patched before the gate opens.
    _drain(
        ready,
        len(workers),
        _WORKER_READY_TIMEOUT_SECONDS,
        "ready",
        workers,
        stalled_hint=(
            "alive=True here means the child was still starting its "
            "interpreter or importing easyicu, not that the lock misbehaved."
        ),
    )

    # 3. Release them together: this is the only moment either can proceed.
    start.set()

    # 4. Result wait, exit wait and the business assertions stay separate.
    results = _drain(
        queue,
        len(workers),
        _RESULT_TIMEOUT_SECONDS,
        "a result",
        workers,
        stalled_hint=(
            "Both children had already reported ready, so alive=True here is "
            "NOT slow startup: the child is contending, blocked on the lock, "
            "or blocked reporting its result."
        ),
    )
    for worker in workers:
        worker.join(timeout=_EXIT_TIMEOUT_SECONDS)
        assert worker.exitcode == 0, (
            f"worker pid={worker.pid} exited {worker.exitcode!r} after reporting"
        )
    return results


def test_workspace_create_only_contract_is_atomic_across_processes(tmp_path) -> None:
    results = _run_pair(
        _workspace_create_worker,
        (str(tmp_path / "workspace"), "first", "second"),
    )

    assert [status for status, _ in results].count("ok") == 1
    assert ("error", "pi_workspace_write_create_only") in results


def test_project_authority_binding_is_atomic_across_processes(tmp_path) -> None:
    results = _run_pair(
        _authority_bind_worker,
        (str(tmp_path / "authority.json"), "study-a", "study-b"),
    )

    assert [status for status, _ in results].count("ok") == 1
    assert ("error", "pi_project_study_context_mismatch") in results
