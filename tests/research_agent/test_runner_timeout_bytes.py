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
from pathlib import Path

import pandas as pd

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

    monkeypatch.setattr(subprocess, "run", fake_run)

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

    monkeypatch.setattr(subprocess, "run", fake_run)
    result = runner.run(step_id="01_x", code="print('hi')")
    assert result.timed_out is True
    assert result.stdout == ""
    assert "timed out after" in result.stderr
