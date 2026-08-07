"""Asking once, right after a timeout, reads a container already going away.

``_teardown_container`` ran ``docker rm --force`` with a 10 s timeout and, if
that did not return 0, asked ``docker container inspect`` exactly once before
declaring "container presence could not be excluded" and refusing to collect
the step's outputs.

Refusing is right in principle -- outputs must not be collected while a
container may still be writing into the bind mount.  Asking only once is not:
``rm --force`` can outlast its own timeout while the removal it started
proceeds, and a container whose bind mounts sit on a slow volume takes longer
to unmount than to stop.

Measured 2026-08-02 over every recorded run: 2 steps hit this, both the
robustness replay -- 17 output files, the largest set any step writes -- on the
external-drive run root, and each cost 3 steps (itself plus the figure and the
panel that depend on it).  One was the full-cohort E1 run whose science had
already completed: ``primary_or.json`` held 1.6076904571975383
(1.5392928783882884-1.679127242419454) over 94,425 stays, and
``robustness_matrix.csv`` held two converged rows -- all sitting in the staging
directory, refused collection.

The fix is patience, not permission: poll for absence within a bounded budget.
A container that is genuinely stuck still refuses.
"""

from __future__ import annotations

import subprocess
from typing import List, Sequence

import pytest

from easyicu.research_agent.execution import runner as runner_module
from easyicu.research_agent.execution.runner import DockerRunner


class _FakeDocker:
    """Replays a recorded ``docker`` control sequence.

    ``rm`` raises TimeoutExpired the way the real client did; ``inspect``
    reports the container present for ``present_inspects`` calls and absent
    afterwards, modelling a removal that completes while we wait.
    """

    def __init__(self, *, present_inspects: int, ever_absent: bool = True) -> None:
        self.present_inspects = present_inspects
        self.ever_absent = ever_absent
        self.calls: List[str] = []

    def __call__(self, argv: Sequence[str], **kwargs):  # noqa: ANN001, ANN201
        verb = " ".join(str(part) for part in argv[1:3])
        self.calls.append(verb)
        if verb.startswith("rm"):
            raise subprocess.TimeoutExpired(cmd=list(argv), timeout=10.0)
        if verb.startswith("container inspect"):
            if self.present_inspects > 0:
                self.present_inspects -= 1
                return subprocess.CompletedProcess(list(argv), 0, "[{}]", "")
            if not self.ever_absent:
                return subprocess.CompletedProcess(list(argv), 0, "[{}]", "")
            return subprocess.CompletedProcess(
                list(argv), 1, "", "Error: No such object: abc123"
            )
        return subprocess.CompletedProcess(list(argv), 0, "", "")


@pytest.fixture
def _no_real_sleep(monkeypatch):
    """Keep the poll loop's shape, drop its wall clock."""

    monkeypatch.setattr(runner_module.time, "sleep", lambda _seconds: None)


def _teardown(monkeypatch, fake: _FakeDocker):
    monkeypatch.setattr(runner_module.subprocess, "run", fake)
    runner = DockerRunner.__new__(DockerRunner)
    runner.docker_executable = "/usr/bin/docker"
    return runner._teardown_container("abc123")


# ---------------------------------------------------------------------------
# The recorded failure
# ---------------------------------------------------------------------------


def test_a_removal_that_completes_while_we_wait_is_confirmed(
    monkeypatch, _no_real_sleep
):
    """Two "still present" answers, then gone -- the recorded shape."""

    fake = _FakeDocker(present_inspects=2)
    confirmed, note = _teardown(monkeypatch, fake)

    assert confirmed is True
    assert "teardown confirmed before output collection" in note
    assert fake.calls.count("container inspect") == 3


def test_one_inspect_is_no_longer_the_whole_answer(monkeypatch, _no_real_sleep):
    """The defect in one assertion: a single ask used to decide it."""

    fake = _FakeDocker(present_inspects=1)
    confirmed, _note = _teardown(monkeypatch, fake)

    assert confirmed is True
    assert fake.calls.count("container inspect") > 1


# ---------------------------------------------------------------------------
# The invariant that must not move
# ---------------------------------------------------------------------------


def test_a_container_that_never_goes_away_still_refuses(monkeypatch, _no_real_sleep):
    """Patience, not permission."""

    fake = _FakeDocker(present_inspects=0, ever_absent=False)
    confirmed, note = _teardown(monkeypatch, fake)

    assert confirmed is False
    assert "container presence could not be excluded" in note
    assert "timed-out container cleanup" in note


def test_the_budget_is_bounded(monkeypatch):
    """A stuck container must not hold the run forever.

    Anchored on the real clock rather than the constant, so raising the budget
    without noticing this bound fails here.
    """

    slept: List[float] = []
    monkeypatch.setattr(runner_module.time, "sleep", slept.append)
    fake = _FakeDocker(present_inspects=0, ever_absent=False)

    confirmed, _note = _teardown(monkeypatch, fake)

    assert confirmed is False
    assert sum(slept) <= runner_module._TEARDOWN_ABSENCE_BUDGET_SECONDS
    assert slept, "a refusal that never waited did not poll at all"


def test_a_clean_removal_never_polls(monkeypatch):
    """The fast path must not pay for the slow one."""

    class _CleanDocker(_FakeDocker):
        def __call__(self, argv, **kwargs):  # noqa: ANN001, ANN201
            verb = " ".join(str(part) for part in argv[1:3])
            self.calls.append(verb)
            return subprocess.CompletedProcess(list(argv), 0, "", "")

    fake = _CleanDocker(present_inspects=0)
    confirmed, note = _teardown(monkeypatch, fake)

    assert confirmed is True
    assert "container inspect" not in fake.calls
    assert "teardown confirmed" in note


def test_absence_is_proved_by_the_daemons_own_words(monkeypatch, _no_real_sleep):
    """A non-zero exit alone is not proof; the reason has to say "no such"."""

    class _AmbiguousDocker(_FakeDocker):
        def __call__(self, argv, **kwargs):  # noqa: ANN001, ANN201
            verb = " ".join(str(part) for part in argv[1:3])
            self.calls.append(verb)
            if verb.startswith("rm"):
                raise subprocess.TimeoutExpired(cmd=list(argv), timeout=10.0)
            if verb.startswith("container inspect"):
                return subprocess.CompletedProcess(
                    list(argv), 1, "", "Error: daemon connection refused"
                )
            return subprocess.CompletedProcess(list(argv), 0, "", "")

    fake = _AmbiguousDocker(present_inspects=0)
    confirmed, note = _teardown(monkeypatch, fake)

    assert confirmed is False
    assert "container presence could not be excluded" in note


def test_the_deadline_exit_refuses_too(monkeypatch):
    """The other way out of the loop, which the poll-count tests never take.

    With `time.sleep` stubbed the clock never advances, so every test above
    leaves the loop by the COUNT bound. A mutation that flips the DEADLINE
    branch to "confirmed" therefore survived them all. Advance the clock
    instead, so this test leaves by the deadline and still refuses.
    """

    ticks = iter([0.0] + [1000.0] * 50)
    monkeypatch.setattr(runner_module.time, "monotonic", lambda: next(ticks))
    monkeypatch.setattr(runner_module.time, "sleep", lambda _seconds: None)
    fake = _FakeDocker(present_inspects=0, ever_absent=False)

    confirmed, note = _teardown(monkeypatch, fake)

    assert confirmed is False
    assert "container presence could not be excluded" in note
    assert (
        fake.calls.count("container inspect") == 1
    ), "an expired deadline must stop after the ask that observed it"
