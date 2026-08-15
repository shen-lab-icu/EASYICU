"""Regression: a subprocess timeout returning BYTES output must not crash (G-3).

Root cause of the intermittent M3 ``TypeError: can't concat str to bytes``:
``CodeRunner.run`` caught ``subprocess.TimeoutExpired`` and did
``(exc.stderr or "") + "<timeout msg>"``. On timeout the partial capture can be
``bytes`` even under ``text=True`` (a CPython wrinkle), so ``bytes + str`` blew
up. It was timing-dependent (hence non-reproducible on resume) and surfaced on a
slow clustering step. The DockerRunner sibling already decoded defensively; both
now share ``_as_text``.
"""

from __future__ import annotations

import subprocess
import io
import os
import signal
import sys
import time
from pathlib import Path

import pandas as pd
import pytest

import easyicu.research_agent.execution.runner as runner_mod
from easyicu.research_agent.execution.runner import CodeRunner, _as_text


def test_as_text_handles_bytes_str_and_none():
    assert _as_text(b"abc") == "abc"
    assert _as_text("abc") == "abc"
    assert _as_text(None) == ""
    assert _as_text(b"\xff\xfe") == "��"  # undecodable -> replacement


def _cohort(tmp_path: Path) -> Path:
    p = tmp_path / "cohort.parquet"
    pd.DataFrame({"age": [50, 60], "death": [0, 1]}).to_parquet(p)
    return p


def test_code_runner_timeout_with_bytes_output_does_not_crash(tmp_path, monkeypatch):
    runner = CodeRunner(
        workdir=tmp_path / "run",
        cohort_parquet=_cohort(tmp_path),
        timeout_seconds=1.0,
    )

    def fake_run(*args, **kwargs):
        # Mimic CPython returning the partial capture as *bytes* on timeout.
        raise subprocess.TimeoutExpired(
            cmd=["python", "script.py"],
            timeout=1.0,
            output=b"partial stdout bytes",
            stderr=b"partial stderr bytes",
        )

    monkeypatch.setattr(runner_mod, "_run_capturing_with_descendant_reaping", fake_run)

    # Must not raise TypeError: can't concat str to bytes.
    result = runner.run(step_id="01_slow_clustering", code="print('hi')")

    assert result.timed_out is True
    assert result.returncode == -1
    assert isinstance(result.stdout, str) and "partial stdout bytes" in result.stdout
    assert isinstance(result.stderr, str)
    assert "partial stderr bytes" in result.stderr
    assert "timed out after" in result.stderr


def test_code_runner_timeout_with_none_output_is_clean(tmp_path, monkeypatch):
    runner = CodeRunner(
        workdir=tmp_path / "run",
        cohort_parquet=_cohort(tmp_path),
        timeout_seconds=1.0,
    )

    def fake_run(*args, **kwargs):
        raise subprocess.TimeoutExpired(cmd=["python"], timeout=1.0)

    monkeypatch.setattr(runner_mod, "_run_capturing_with_descendant_reaping", fake_run)
    result = runner.run(step_id="01_x", code="print('hi')")
    assert result.timed_out is True
    assert result.stdout == ""
    assert "timed out after" in result.stderr


def test_run_capturing_reaps_whole_process_group_on_timeout(monkeypatch):
    # On timeout the executor must signal the child's whole process group, not
    # only the direct child, so a background descendant of generated code dies
    # instead of surviving to mutate step outputs after evidence collection.
    events: dict = {}

    class _FakeProc:
        pid = 4242
        returncode = None

        def __init__(self):
            self._calls = 0
            self.stdout = io.BytesIO(b"partial-out")
            self.stderr = io.BytesIO(b"partial-err")

        def wait(self, timeout=None):
            self._calls += 1
            if self._calls == 1:
                raise subprocess.TimeoutExpired(
                    cmd=["python"],
                    timeout=timeout,
                )
            self.returncode = -9
            return self.returncode

        def kill(self):
            events["direct_kill"] = True

    def _fake_popen(cmd, *, start_new_session=False, **kwargs):
        events["start_new_session"] = start_new_session
        return _FakeProc()

    monkeypatch.setattr(runner_mod.subprocess, "Popen", _fake_popen)
    monkeypatch.setattr(
        runner_mod.os,
        "killpg",
        lambda pgid, sig: events.__setitem__("killpg", (pgid, sig)),
    )

    with pytest.raises(subprocess.TimeoutExpired) as excinfo:
        runner_mod._run_capturing_with_descendant_reaping(
            ["python", "x.py"], cwd="/tmp", env={}, timeout=0.01
        )

    # Group leader established, then the whole group SIGKILLed (not just kill()).
    assert events["start_new_session"] is True
    assert events["killpg"] == (4242, signal.SIGKILL)
    assert "direct_kill" not in events
    # Partial capture is preserved for the caller's bytes-safe timeout handler.
    assert excinfo.value.output == "partial-out"
    assert excinfo.value.stderr == "partial-err"


def test_run_capturing_returns_completed_process_for_real_command(tmp_path):
    # Exercise the real Popen path: a normal command yields a CompletedProcess
    # with captured text stdout/stderr and returncode.
    import os
    import sys

    result = runner_mod._run_capturing_with_descendant_reaping(
        [sys.executable, "-c", "import sys; print('hi'); sys.stderr.write('err')"],
        cwd=str(tmp_path),
        env=dict(os.environ),
        timeout=30,
    )
    assert result.returncode == 0
    assert "hi" in result.stdout
    assert "err" in result.stderr


def test_normal_completion_still_terminates_the_dedicated_process_group(monkeypatch):
    import signal

    events = {}

    class _FakeProc:
        pid = 4343
        returncode = 0
        stdout = io.BytesIO(b"ok")
        stderr = io.BytesIO()

        def wait(self, timeout=None):
            return self.returncode

        def poll(self):
            return self.returncode

    monkeypatch.setattr(runner_mod.subprocess, "Popen", lambda *a, **k: _FakeProc())
    monkeypatch.setattr(
        runner_mod.os,
        "killpg",
        lambda pgid, sig: events.__setitem__("killpg", (pgid, sig)),
    )

    result = runner_mod._run_capturing_with_descendant_reaping(
        ["python", "x.py"], cwd="/tmp", env={}, timeout=1
    )

    assert result.returncode == 0
    assert events["killpg"] == (4343, signal.SIGKILL)


def test_run_capturing_bounds_generated_output_without_temp_backing(
    tmp_path, monkeypatch
):
    def forbid_temporary_file(*args, **kwargs):
        raise AssertionError("runner capture must not use unbounded temporary files")

    monkeypatch.setattr(runner_mod.tempfile, "TemporaryFile", forbid_temporary_file)
    size = runner_mod.MAX_RUNNER_CAPTURE_BYTES * 2
    result = runner_mod._run_capturing_with_descendant_reaping(
        [sys.executable, "-c", f"print('x' * {size})"],
        cwd=str(tmp_path),
        env=dict(os.environ),
        timeout=30,
    )

    assert "earlier runner output truncated" in result.stdout
    assert len(result.stdout.encode("utf-8")) <= (
        runner_mod.MAX_RUNNER_CAPTURE_BYTES + 64
    )


@pytest.mark.parametrize("error", [RuntimeError("wait failed"), KeyboardInterrupt()])
def test_wait_exceptions_still_reap_the_process_group(monkeypatch, error):
    events = {}

    class _FakeProc:
        pid = 4545
        returncode = None
        stdout = io.BytesIO(b"partial")
        stderr = io.BytesIO()

        def __init__(self):
            self.calls = 0

        def wait(self, timeout=None):
            self.calls += 1
            if self.calls == 1:
                raise error
            self.returncode = -9
            return self.returncode

        def kill(self):
            events["direct_kill"] = True

        def poll(self):
            return self.returncode

    monkeypatch.setattr(runner_mod.subprocess, "Popen", lambda *a, **k: _FakeProc())
    monkeypatch.setattr(
        runner_mod.os,
        "killpg",
        lambda pgid, sig: events.__setitem__("killpg", (pgid, sig)),
    )

    with pytest.raises(type(error), match=str(error) if str(error) else None):
        runner_mod._run_capturing_with_descendant_reaping(
            ["python", "x.py"], cwd="/tmp", env={}, timeout=1
        )

    assert events["killpg"] == (4545, signal.SIGKILL)
    assert "direct_kill" not in events


@pytest.mark.skipif(os.name != "posix", reason="requires POSIX process groups")
def test_real_keyboard_interrupt_reaps_the_spawned_process_group(tmp_path):
    pid_path = tmp_path / "child.pid"
    previous = signal.getsignal(signal.SIGALRM)

    def interrupt(_signum, _frame):
        raise KeyboardInterrupt

    signal.signal(signal.SIGALRM, interrupt)
    signal.setitimer(signal.ITIMER_REAL, 0.2)
    try:
        with pytest.raises(KeyboardInterrupt):
            runner_mod._run_capturing_with_descendant_reaping(
                [
                    sys.executable,
                    "-c",
                    (
                        "import os,time; "
                        f"open({str(pid_path)!r}, 'w').write(str(os.getpid())); "
                        "time.sleep(30)"
                    ),
                ],
                cwd=str(tmp_path),
                env=dict(os.environ),
                timeout=30,
            )
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
        signal.signal(signal.SIGALRM, previous)

    pid = int(pid_path.read_text(encoding="utf-8"))
    for _ in range(50):
        try:
            os.kill(pid, 0)
        except ProcessLookupError:
            break
        time.sleep(0.01)
    else:
        pytest.fail(f"interrupted runner process {pid} survived cleanup")


def test_runner_redacts_explicit_secrets_from_result_and_log(tmp_path, monkeypatch):
    secret = "runner-secret-value"
    runner = CodeRunner(
        workdir=tmp_path / "run",
        cohort_parquet=_cohort(tmp_path),
        extra_env={"SERVICE_API_KEY": secret},
        allow_unsafe_host_fallback=True,
    )
    monkeypatch.setattr(
        runner_mod,
        "_run_capturing_with_descendant_reaping",
        lambda *a, **k: subprocess.CompletedProcess(
            a[0], 1, f"token={secret}", f"SERVICE_API_KEY={secret}"
        ),
    )

    result = runner.run(step_id="01_secret", code="raise RuntimeError()")
    log = result.runner_log_path.read_text(encoding="utf-8")

    assert secret not in result.stdout
    assert secret not in result.stderr
    assert secret not in log
    assert "[REDACTED]" in result.stdout + result.stderr + log
