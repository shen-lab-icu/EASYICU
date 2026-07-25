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


def test_the_public_entry_points_are_scoped() -> None:
    """``run`` and ``resume_human_review`` are the job boundary.

    ``_build_runner`` is not: the published value has to survive until the
    coder prompt is rendered and the step executes, both of which happen after
    the runner is built.
    """

    from easyicu.research_agent.pipeline import ResearchAgentPipeline

    for name in ("run", "resume_human_review"):
        method = getattr(ResearchAgentPipeline, name)
        assert hasattr(method, "__wrapped__"), f"{name} is not scoped"


def test_scoping_run_keeps_its_signature_introspectable() -> None:
    """Callers and tests read ``run``'s keyword-only parameters."""

    import inspect

    from easyicu.research_agent.pipeline import ResearchAgentPipeline

    parameters = inspect.signature(ResearchAgentPipeline.run).parameters
    for expected in ("question", "cohort", "database", "progress_callback"):
        assert expected in parameters


def test_resume_republishes_the_validated_snapshot_after_the_pause() -> None:
    """The pause ends the scope, so resume cannot rely on ambient state.

    It republishes from ``_validated_runtime_capabilities`` — an immutable
    tuple of import names kept on the instance — rather than from a provider
    callable captured by the paused job.
    """

    import inspect

    from easyicu.research_agent.pipeline import ResearchAgentPipeline

    source = inspect.getsource(ResearchAgentPipeline.resume_human_review)
    assert "_validated_runtime_capabilities" in source
    assert "set_runtime_capability_snapshot_provider" in source
