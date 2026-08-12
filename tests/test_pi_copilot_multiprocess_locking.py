from __future__ import annotations

import multiprocessing
import time
from pathlib import Path


def _workspace_create_worker(root: str, start, queue, content: str) -> None:
    from easyicu.webserver.pi_copilot.contracts import PiCopilotError
    from easyicu.webserver.pi_copilot.workspace import ProjectWorkspace

    original = ProjectWorkspace._atomic_write

    def delayed(candidate: Path, encoded: bytes) -> None:
        time.sleep(0.2)
        original(candidate, encoded)

    ProjectWorkspace._atomic_write = staticmethod(delayed)
    start.wait()
    try:
        ProjectWorkspace(Path(root)).write_file("project", "result.md", content)
        queue.put(("ok", content))
    except PiCopilotError as exc:
        queue.put(("error", exc.code))


def _authority_bind_worker(path: str, start, queue, study_id: str) -> None:
    from easyicu.webserver.pi_copilot.contracts import PiCopilotError
    from easyicu.webserver.pi_copilot.project_authority import ProjectAuthorityStore

    store = ProjectAuthorityStore(Path(path))
    original = store._write

    def delayed(rows) -> None:
        time.sleep(0.2)
        original(rows)

    store._write = delayed
    start.wait()
    try:
        queue.put(("ok", store.bind("project", study_id)))
    except PiCopilotError as exc:
        queue.put(("error", exc.code))


def _run_pair(target, args: tuple[str, ...]) -> list[tuple[str, str]]:
    context = multiprocessing.get_context("spawn")
    start = context.Event()
    queue = context.Queue()
    workers = [
        context.Process(target=target, args=(*args[:-2], start, queue, value))
        for value in args[-2:]
    ]
    for worker in workers:
        worker.start()
    start.set()
    results = [queue.get(timeout=10) for _ in workers]
    for worker in workers:
        worker.join(timeout=10)
        assert worker.exitcode == 0
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
