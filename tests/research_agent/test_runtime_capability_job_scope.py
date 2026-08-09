"""One job's runtime-capability publication must not outlive that job.

``DockerRunner`` publishes its image's package allow-list into a ContextVar so
the coder prompt — rendered after the runner is constructed — offers only
packages the sandbox can actually import. Publication is deliberately
long-lived within a job. What was missing was the other end: nothing put the
ContextVar back, so a value published by one job stayed visible to whatever ran
next in the same context.

The runner constructors each call ``set_runtime_capability_snapshot_provider(None)``
defensively, which covers the ordinary path — but only paths that build a
runner. Anything that does not (a job that raises before construction, a
resumed review, one test after another in a single process) inherited the
previous job's allow-list, and the coder was then told to ``import shap`` in an
interpreter without it.
"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from contextvars import copy_context
from pathlib import Path

import pytest

from easyicu.research_agent.execution import method_capabilities as mc
from easyicu.research_agent.execution.method_capabilities import (
    runtime_capability_job_scope,
    runtime_capability_snapshot,
    set_runtime_capability_snapshot_provider,
)


DOCKER_PACKAGES = frozenset({"shap", "xgboost", "lifelines"})
HOST_PACKAGES = frozenset({"lifelines"})


@pytest.fixture(autouse=True)
def _clean_context():
    """Leave the ContextVar as this module found it, whatever a test does."""

    with runtime_capability_job_scope():
        yield


def _publish(packages: frozenset[str]) -> None:
    set_runtime_capability_snapshot_provider(lambda: packages)


# ---------------------------------------------------------------------------
# The leak itself
# ---------------------------------------------------------------------------


def test_a_job_that_raises_does_not_leave_its_allow_list_behind() -> None:
    """The failure mode that reached production, in four lines.

    A Docker-backed job that raises between publishing its snapshot and
    finishing used to leave that snapshot in place. On a long-lived process —
    a web request, an MCP session, a benchmark loop — the next job then
    rendered a coder prompt promising packages its own interpreter lacked.
    """

    with pytest.raises(RuntimeError):
        with runtime_capability_job_scope():
            _publish(DOCKER_PACKAGES)
            raise RuntimeError("step failed")

    assert runtime_capability_snapshot() is None


def test_a_second_job_does_not_inherit_the_first_ones_packages() -> None:
    with runtime_capability_job_scope():
        _publish(DOCKER_PACKAGES)
        assert runtime_capability_snapshot() == DOCKER_PACKAGES

    with runtime_capability_job_scope():
        # A host runner that never publishes must not see Docker's list.
        assert runtime_capability_snapshot() is None
        _publish(HOST_PACKAGES)
        assert runtime_capability_snapshot() == HOST_PACKAGES

    assert runtime_capability_snapshot() is None


def test_a_nested_job_restores_the_outer_publication_not_none() -> None:
    """``run_from_spec`` calls ``run``; the outer value has to come back.

    This is why the scope resets with the ``Token`` rather than assigning
    ``None`` on the way out, which is what the runner constructors do and what
    would otherwise silently clear an enclosing job.
    """

    with runtime_capability_job_scope():
        _publish(DOCKER_PACKAGES)

        with runtime_capability_job_scope():
            _publish(HOST_PACKAGES)
            assert runtime_capability_snapshot() == HOST_PACKAGES

        assert runtime_capability_snapshot() == DOCKER_PACKAGES


def test_publication_survives_inside_the_job_because_the_coder_reads_it_later() -> None:
    """The scope must not be mistaken for a bracket around the setter.

    A ``try/finally`` reset around ``set_runtime_capability_snapshot_provider``
    would be tidier-looking and wrong: the runner publishes in its constructor
    and the coder prompt is rendered afterwards, so the value has to outlive
    the call that set it.
    """

    with runtime_capability_job_scope():
        _publish(DOCKER_PACKAGES)
        block = mc.coder_method_capability_block()

    assert "shap" in block
    assert "lifelines" in block


def test_a_runner_that_fails_to_construct_does_not_disturb_the_outer_job() -> None:
    with runtime_capability_job_scope():
        _publish(DOCKER_PACKAGES)

        with pytest.raises(ValueError):
            with runtime_capability_job_scope():
                # DockerRunner.__init__ clears first, then may raise on a
                # missing docker executable before it ever publishes.
                set_runtime_capability_snapshot_provider(None)
                raise ValueError("docker executable not found")

        assert runtime_capability_snapshot() == DOCKER_PACKAGES


# ---------------------------------------------------------------------------
# Worker threads
# ---------------------------------------------------------------------------


def test_a_worker_thread_inherits_the_publishing_context() -> None:
    """Concurrent steps must see the allow-list, and a fresh thread does not.

    A thread starts with an empty context, so ``executor.submit(fn)`` would
    read ``None`` in the worker. ``execution/phase.py`` submits through
    ``copy_context().run`` for this reason; the assertion below is what makes
    that a contract instead of an incidental detail.
    """

    def read_snapshot() -> object:
        return runtime_capability_snapshot()

    with runtime_capability_job_scope():
        _publish(DOCKER_PACKAGES)
        with ThreadPoolExecutor(max_workers=2) as executor:
            inherited = executor.submit(copy_context().run, read_snapshot).result()
            bare = executor.submit(read_snapshot).result()

    assert inherited == DOCKER_PACKAGES
    assert bare is None, "a bare submit sees an empty context — hence copy_context"


def test_the_submit_helper_the_pipeline_uses_propagates_the_context() -> None:
    from easyicu.research_agent.execution.phase import _submit_in_current_context

    def read_snapshot() -> object:
        return runtime_capability_snapshot()

    with runtime_capability_job_scope():
        _publish(DOCKER_PACKAGES)
        with ThreadPoolExecutor(max_workers=2) as executor:
            futures = [
                _submit_in_current_context(executor, read_snapshot) for _ in range(4)
            ]
            results = [future.result() for future in futures]

    # Each submission copies the context separately, so several workers can run
    # at once — one shared Context object could not be entered concurrently.
    assert results == [DOCKER_PACKAGES] * 4


# ---------------------------------------------------------------------------
# The pipeline entry points
# ---------------------------------------------------------------------------


def test_run_clears_on_entry_and_restores_the_caller_publication(
    tmp_path, monkeypatch
) -> None:
    """``run`` is the job boundary, observed rather than asserted structurally.

    Checking ``hasattr(run, "__wrapped__")`` would pass on
    ``@exclusive_run_execution`` alone and prove nothing about this scope. So
    publish from outside, look at what the inside sees, and look at what
    survives: entry must show ``None`` (the outer publication is not inherited)
    and exit must restore the outer value rather than clearing it.
    """

    import easyicu.research_agent.pipeline as pipeline_module
    from easyicu.research_agent.pipeline import ResearchAgentPipeline

    seen_inside: list[object] = []

    def _probe(_name: str):
        seen_inside.append(runtime_capability_snapshot())
        _publish(HOST_PACKAGES)
        raise RuntimeError("stop here — the scope is what is under test")

    monkeypatch.setattr(pipeline_module, "get_skill", _probe)
    pipeline = ResearchAgentPipeline(workdir=tmp_path)

    with runtime_capability_job_scope():
        _publish(DOCKER_PACKAGES)

        with pytest.raises(RuntimeError):
            pipeline.run(skill="anything", cohort=tmp_path / "cohort.parquet")

        assert seen_inside == [None], "run inherited the caller's publication"
        assert (
            runtime_capability_snapshot() == DOCKER_PACKAGES
        ), "run reset to None instead of restoring what the caller published"


def test_scoping_run_keeps_its_signature_introspectable() -> None:
    """Callers and tests read ``run``'s keyword-only parameters."""

    import inspect

    from easyicu.research_agent.pipeline import ResearchAgentPipeline

    parameters = inspect.signature(ResearchAgentPipeline.run).parameters
    for expected in ("question", "cohort", "database", "progress_callback"):
        assert expected in parameters


def test_resume_uses_the_paused_runs_snapshot_not_the_instances_latest(
    tmp_path,
) -> None:
    """A later run must not change the environment an approved review resumes into.

    ``run`` returns when it pauses, so its writer lease is released and a second
    run can start on the same pipeline. That second run's preflight overwrites
    ``_validated_runtime_capabilities``. Reading the instance field at resume
    would finish run A's analysis under run B's image allow-list — an
    environment the reviewer never approved. The snapshot is therefore captured
    into the pending state at the pause.

    Driving this with a stub workflow rather than a real LLM run: what is under
    test is which snapshot resume publishes, and the workflow only records
    what it saw.
    """

    from easyicu.research_agent.pipeline import ResearchAgentPipeline

    observed: list[object] = []

    class _RecordingWorkflow:
        def resume(self, *_args, **_kwargs):
            observed.append(runtime_capability_snapshot())
            return "done"

    class _Pending:
        run_id = "20260725T120000_abcdef"
        run_dir = str(tmp_path / "run")
        resumable_here = True

    pipeline = ResearchAgentPipeline(workdir=tmp_path)
    Path(_Pending.run_dir).mkdir(parents=True, exist_ok=True)

    paused_snapshot = ("shap", "xgboost")
    pipeline._pending_human_review = {
        "workflow": _RecordingWorkflow(),
        "pending": _Pending(),
        "runtime_capabilities": paused_snapshot,
        "runtime_bundle": None,
    }
    # A second run happened while the reviewer was deciding.
    pipeline._validated_runtime_capabilities = ("lifelines",)
    pipeline._pipeline_result_or_pending = lambda outcome, **_kwargs: outcome

    pipeline.resume_human_review([])

    assert observed == [
        frozenset(paused_snapshot)
    ], "resume published the instance's current snapshot, not the paused run's"
    assert runtime_capability_snapshot() is None, "resume leaked its own scope"


def test_resume_takes_the_writer_lease_for_the_paused_run() -> None:
    """``run`` released its lease when it paused, so resume needs its own.

    Without it, an approval could be replayed into a run directory another
    call is writing to.
    """

    import inspect

    from easyicu.research_agent.pipeline import ResearchAgentPipeline

    source = inspect.getsource(ResearchAgentPipeline.resume_human_review)
    assert "acquire_run_execution_lock" in source
    assert "run_id=pending.run_id" in source
