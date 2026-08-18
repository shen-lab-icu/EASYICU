from __future__ import annotations

import multiprocessing
import queue
import time
from pathlib import Path
from typing import Any

import pytest

from easyicu.research_agent.pipeline import ResearchAgentPipeline
from easyicu.research_agent.authority.run_lock import (
    RunExecutionLockError,
    acquire_run_execution_lock,
)


def _try_lock_in_child(workdir: str, run_id: str, result_queue: Any) -> None:
    started = time.monotonic()
    try:
        with acquire_run_execution_lock(workdir=Path(workdir), run_id=run_id):
            result_queue.put(("acquired", time.monotonic() - started, ""))
    except RunExecutionLockError as exc:
        result_queue.put(("blocked", time.monotonic() - started, str(exc)))


#: How long the PARENT waits for a spawned child to boot and report back.
#:
#: This is transport plumbing, not the contract under test. A "spawn" child
#: starts a fresh interpreter and imports easyicu before it can even attempt
#: the lock, and CI now runs the suite under ``-n auto``: on a loaded machine
#: that boot alone exceeded the previous 10s and the queue read raised
#: ``_queue.Empty``, which reads exactly like a lock regression while being
#: nothing of the sort (observed twice in the 2026-08-18 exact-head xdist runs,
#: 588s and 826s wall clock; the same tests pass 3/3 under ``-n auto`` when the
#: machine is idle).
#:
#: The atomicity assertion is deliberately NOT this number. Each test asserts
#: ``elapsed < 1.0``, measured inside the child around
#: ``acquire_run_execution_lock`` itself, so a lock that actually became slow
#: still fails no matter how long the child took to start. Widening the wait
#: below cannot hide that.
_CHILD_REPORT_TIMEOUT_SECONDS = 120

#: How long we wait for a child that has ALREADY reported to exit.
#:
#: Deliberately short and deliberately separate from the boot wait above. Once
#: the result is on the queue the child has nothing left to do but unwind, so a
#: slow exit here is a real hang, not a busy machine -- reusing the 120s boot
#: budget would make a genuinely stuck teardown cost two minutes per test to
#: discover.
_CHILD_EXIT_TIMEOUT_SECONDS = 10


def _child_lock_result(tmp_path: Path, run_id: str) -> tuple[str, float, str]:
    context = multiprocessing.get_context("spawn")
    result_queue = context.Queue()
    process = context.Process(
        target=_try_lock_in_child,
        args=(str(tmp_path), run_id, result_queue),
    )
    started = time.monotonic()
    process.start()
    try:
        result = result_queue.get(timeout=_CHILD_REPORT_TIMEOUT_SECONDS)
    except queue.Empty:  # pragma: no cover - only on a genuine child failure
        waited = time.monotonic() - started
        alive = process.is_alive()
        exitcode = process.exitcode
        process.kill()
        process.join(timeout=10)
        pytest.fail(
            "the run-lock child never reported: "
            f"waited={waited:.1f}s limit={_CHILD_REPORT_TIMEOUT_SECONDS}s "
            f"alive_at_timeout={alive} exitcode_at_timeout={exitcode} "
            f"run_id={run_id!r} workdir={tmp_path}. "
            "alive_at_timeout=True means the child was still booting or hung "
            "rather than the lock misbehaving; a non-zero exitcode means it "
            "crashed before it could report."
        )
    process.join(timeout=_CHILD_EXIT_TIMEOUT_SECONDS)
    assert process.exitcode == 0, (
        f"run-lock child exited {process.exitcode!r} after reporting {result!r}"
    )
    result_queue.close()
    return result


def test_same_run_second_writer_is_rejected_immediately(tmp_path: Path) -> None:
    with acquire_run_execution_lock(workdir=tmp_path, run_id="run_same"):
        status, elapsed, message = _child_lock_result(tmp_path, "run_same")

    assert status == "blocked"
    assert elapsed < 1.0
    assert "already being written" in message
    assert "run_same" in message
    assert "pid=" in message


def test_run_lock_release_allows_later_writer_without_deleting_lock(
    tmp_path: Path,
) -> None:
    with acquire_run_execution_lock(workdir=tmp_path, run_id="run_reenter") as first:
        lock_path = first.path

    assert lock_path.exists()
    with acquire_run_execution_lock(workdir=tmp_path, run_id="run_reenter") as second:
        assert second.path == lock_path


def test_different_runs_do_not_block_each_other(tmp_path: Path) -> None:
    with acquire_run_execution_lock(workdir=tmp_path, run_id="run_a"):
        status, elapsed, message = _child_lock_result(tmp_path, "run_b")

    assert status == "acquired"
    assert elapsed < 1.0
    assert message == ""


@pytest.mark.parametrize(
    "run_id",
    ["..", "../outside", "run/child", r"run\child", "/absolute", "\x00bad"],
)
def test_run_lock_rejects_non_component_run_ids(
    tmp_path: Path,
    run_id: str,
) -> None:
    with pytest.raises(ValueError, match="one non-empty path component"):
        acquire_run_execution_lock(workdir=tmp_path, run_id=run_id)


def test_pipeline_run_is_wrapped_by_whole_call_execution_lock() -> None:
    assert ResearchAgentPipeline.run.__wrapped__.__name__ == "run"
    assert ResearchAgentPipeline.run.__easyicu_run_execution_locked__ is True


def test_pipeline_rejects_conflicting_resume_before_input_processing(
    tmp_path: Path,
) -> None:
    pipeline = object.__new__(ResearchAgentPipeline)
    pipeline.workdir = tmp_path

    with acquire_run_execution_lock(workdir=tmp_path, run_id="run_resume"):
        with pytest.raises(RunExecutionLockError):
            pipeline.run(
                cohort=tmp_path / "not_read.parquet",
                resume_run_id="run_resume",
            )


def test_pipeline_exception_releases_resume_writer_lock(tmp_path: Path) -> None:
    pipeline = object.__new__(ResearchAgentPipeline)
    pipeline.workdir = tmp_path

    with pytest.raises(ValueError, match="question"):
        pipeline.run(
            cohort=tmp_path / "not_read.parquet",
            resume_run_id="run_exception",
        )

    with acquire_run_execution_lock(workdir=tmp_path, run_id="run_exception"):
        pass
