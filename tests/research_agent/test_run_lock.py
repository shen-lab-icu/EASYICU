from __future__ import annotations

import multiprocessing
import time
from pathlib import Path
from typing import Any

import pytest

from easyicu.research_agent.pipeline import ResearchAgentPipeline
from easyicu.research_agent.run_lock import (
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


def _child_lock_result(tmp_path: Path, run_id: str) -> tuple[str, float, str]:
    context = multiprocessing.get_context("spawn")
    result_queue = context.Queue()
    process = context.Process(
        target=_try_lock_in_child,
        args=(str(tmp_path), run_id, result_queue),
    )
    process.start()
    result = result_queue.get(timeout=10)
    process.join(timeout=10)
    assert process.exitcode == 0
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
