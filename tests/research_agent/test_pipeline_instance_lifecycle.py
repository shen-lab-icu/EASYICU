"""Concurrency regressions for one ResearchAgentPipeline object.

Run-id file locks deliberately permit different fresh run ids.  These tests
therefore block before the first run body and prove that the instance boundary,
not a shared output directory, rejects the second caller immediately.
"""

from __future__ import annotations

import asyncio
from contextlib import contextmanager, nullcontext
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FutureTimeout
from pathlib import Path
from threading import Event
from time import monotonic
from types import SimpleNamespace

import pandas as pd
import pytest

from easyicu.research_agent.orchestration.instance_lifecycle import (
    PipelineInstanceLifecycleBusy,
    PipelineInstanceLifecycleError,
)
from easyicu.research_agent.pipeline import ResearchAgentPipeline
from easyicu.research_agent.providers.mocks import MockLLMClient


class _StopFirstRun(RuntimeError):
    pass


def test_pause_return_reserves_instance_until_resume_finishes() -> None:
    """The adapter must retain the lease after the run call itself returns."""

    from easyicu.research_agent.pipeline import _pipeline_instance_lifecycle

    class _Harness:
        def __init__(self) -> None:
            self._pending_human_review = None

        @_pipeline_instance_lifecycle("run")
        def run(self):
            self._pending_human_review = {
                "pending": SimpleNamespace(run_id="run-awaiting-review")
            }
            return "paused"

        @_pipeline_instance_lifecycle("resume")
        def resume(self):
            self._pending_human_review = None
            return "completed"

    pipeline = _Harness()
    assert pipeline.run() == "paused"
    paused = pipeline._instance_lifecycle_lease.snapshot()
    assert paused.state == "paused"
    assert paused.paused_run_id == "run-awaiting-review"

    with pytest.raises(PipelineInstanceLifecycleBusy, match="paused for human review"):
        pipeline.run()

    assert pipeline.resume() == "completed"
    assert pipeline._instance_lifecycle_lease.snapshot().state == "idle"


def _run_kwargs() -> dict[str, object]:
    return {
        "question": "Does this instance reject concurrent orchestration?",
        "cohort": pd.DataFrame({"patient_id": [1, 2], "death": [0, 1]}),
    }


def _install_blocking_run_lock(monkeypatch):
    """Stop the first call before run-body validation without holding the lease."""

    import easyicu.research_agent.authority.run_lock as run_lock

    entered = Event()
    release = Event()

    @contextmanager
    def _blocking_lock(**_kwargs):
        entered.set()
        if not release.wait(timeout=5):
            raise AssertionError("test did not release the blocked first run")
        raise _StopFirstRun("stop after lifecycle concurrency assertion")
        yield  # pragma: no cover - makes this a context-manager generator

    monkeypatch.setattr(run_lock, "acquire_run_execution_lock", _blocking_lock)
    return entered, release


def test_two_threaded_runs_on_one_instance_fail_fast(monkeypatch, tmp_path) -> None:
    pipeline = ResearchAgentPipeline(workdir=tmp_path, llm=MockLLMClient())
    entered, release = _install_blocking_run_lock(monkeypatch)

    with ThreadPoolExecutor(max_workers=2) as pool:
        first = pool.submit(pipeline.run, **_run_kwargs())
        assert entered.wait(timeout=2)
        started = monotonic()
        second = pool.submit(pipeline.run, **_run_kwargs())
        try:
            with pytest.raises(PipelineInstanceLifecycleBusy) as caught:
                second.result(timeout=0.5)
        except FutureTimeout:
            pytest.fail("the second pipeline call waited instead of failing fast")
        finally:
            release.set()

        assert caught.value.reason_code == "pipeline_instance_lifecycle_busy"
        assert monotonic() - started < 1.0
        with pytest.raises(_StopFirstRun):
            first.result(timeout=2)

    assert pipeline._instance_lifecycle_lease.snapshot().state == "idle"


def test_two_run_async_calls_on_one_instance_fail_fast(monkeypatch, tmp_path) -> None:
    pipeline = ResearchAgentPipeline(workdir=tmp_path, llm=MockLLMClient())
    entered, release = _install_blocking_run_lock(monkeypatch)

    async def _exercise() -> None:
        first = asyncio.create_task(pipeline.run_async(**_run_kwargs()))
        assert await asyncio.to_thread(entered.wait, 2)
        second = asyncio.create_task(pipeline.run_async(**_run_kwargs()))
        try:
            with pytest.raises(PipelineInstanceLifecycleBusy):
                await asyncio.wait_for(second, timeout=0.5)
        finally:
            release.set()
        with pytest.raises(_StopFirstRun):
            await asyncio.wait_for(first, timeout=2)

    asyncio.run(_exercise())
    assert pipeline._instance_lifecycle_lease.snapshot().state == "idle"


class _Pending:
    resumable_here = True

    def __init__(self, *, run_id: str, run_dir: Path) -> None:
        self.run_id = run_id
        self.run_dir = str(run_dir)


def _install_resume_runtime_stubs(monkeypatch, pipeline, run_dir: Path) -> None:
    import easyicu.research_agent.pipeline as pipeline_module

    run_dir.mkdir(parents=True, exist_ok=True)
    monkeypatch.setattr(
        pipeline_module,
        "run_heartbeat_scope",
        lambda **_kwargs: nullcontext(),
    )
    monkeypatch.setattr(
        pipeline_module,
        "bind_active_run_heartbeat",
        lambda *_args, **_kwargs: None,
    )
    monkeypatch.setattr(
        pipeline_module,
        "acquire_run_execution_lock",
        lambda **_kwargs: nullcontext(),
    )
    monkeypatch.setattr(
        pipeline,
        "_heartbeat_wall_clock_remaining",
        lambda: None,
    )


def test_active_resume_excludes_run_and_another_resume(monkeypatch, tmp_path) -> None:
    entered = Event()
    release = Event()

    class _Workflow:
        state = "paused"

        def resume(self, _payload):
            entered.set()
            if not release.wait(timeout=5):
                raise AssertionError("test did not release resume")
            self.state = "completed"
            return "finished"

    pipeline = ResearchAgentPipeline(
        workdir=tmp_path / "wd", llm=MockLLMClient()
    )
    run_dir = tmp_path / "paused-run"
    _install_resume_runtime_stubs(monkeypatch, pipeline, run_dir)
    pipeline._pending_human_review = {
        "workflow": _Workflow(),
        "pending": _Pending(run_id="paused-run", run_dir=run_dir),
        "runtime_capabilities": (),
        "runtime_bundle": None,
    }

    def _finish(outcome, **_kwargs):
        pipeline._pending_human_review = None
        return outcome

    monkeypatch.setattr(pipeline, "_pipeline_result_or_pending", _finish)

    with ThreadPoolExecutor(max_workers=2) as pool:
        resumed = pool.submit(pipeline.resume_human_review, [])
        assert entered.wait(timeout=2)
        with pytest.raises(PipelineInstanceLifecycleBusy):
            pipeline.run(**_run_kwargs())
        with pytest.raises(PipelineInstanceLifecycleBusy):
            pipeline.resume_human_review([])
        release.set()
        assert resumed.result(timeout=2) == "finished"

    assert pipeline._instance_lifecycle_lease.snapshot().state == "idle"
    with pytest.raises(PipelineInstanceLifecycleError, match="same_process"):
        pipeline.resume_human_review([])


def test_correctable_resume_error_retains_pause_until_retry(
    monkeypatch, tmp_path
) -> None:
    class _Workflow:
        state = "paused"
        attempts = 0

        def resume(self, _payload):
            self.attempts += 1
            if self.attempts == 1:
                raise ValueError("human review decision authority digest mismatch")
            self.state = "completed"
            return "finished"

    pipeline = ResearchAgentPipeline(
        workdir=tmp_path / "wd", llm=MockLLMClient()
    )
    run_dir = tmp_path / "paused-run"
    _install_resume_runtime_stubs(monkeypatch, pipeline, run_dir)
    workflow = _Workflow()
    pipeline._pending_human_review = {
        "workflow": workflow,
        "pending": _Pending(run_id="paused-run", run_dir=run_dir),
        "runtime_capabilities": (),
        "runtime_bundle": None,
    }

    def _finish(outcome, **_kwargs):
        pipeline._pending_human_review = None
        return outcome

    monkeypatch.setattr(pipeline, "_pipeline_result_or_pending", _finish)

    with pytest.raises(ValueError, match="authority digest mismatch"):
        pipeline.resume_human_review([])

    paused = pipeline._instance_lifecycle_lease.snapshot()
    assert paused.state == "paused"
    assert paused.paused_run_id == "paused-run"
    with pytest.raises(PipelineInstanceLifecycleBusy, match="paused for human review"):
        pipeline.run(**_run_kwargs())

    assert pipeline.resume_human_review([]) == "finished"
    assert pipeline._instance_lifecycle_lease.snapshot().state == "idle"
