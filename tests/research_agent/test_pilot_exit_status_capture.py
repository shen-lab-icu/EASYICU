"""Pilot terminal-status capture tests.

These tests exercise the pilot runner's always-on exit-status artifact without
calling a real LLM or running the full research pipeline.
"""

from __future__ import annotations

import importlib.util
import json
import signal
import subprocess
import sys
import time
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
PILOT_SCRIPT = REPO_ROOT / "scripts" / "pilot_real_llm.py"


def _read_exit_status(run_dir: Path) -> dict:
    return json.loads((run_dir / "pilot_exit_status.json").read_text())


def test_normal_exit_writes_exit_status(tmp_path: Path) -> None:
    run_dir = tmp_path / "normal_exit"

    result = subprocess.run(
        [
            sys.executable,
            str(PILOT_SCRIPT),
            "--_test-exit-status-run-dir",
            str(run_dir),
            "--_test-exit-status-mode",
            "normal",
        ],
        cwd=REPO_ROOT,
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    payload = _read_exit_status(run_dir)
    assert payload["schema_version"] == "easyicu.pilot_exit/1"
    assert payload["run_id"] == "normal_exit"
    assert payload["exit_kind"] == "normal"
    assert payload["exit_code"] == 0
    assert payload["captured_at"]


def test_sigterm_exit_writes_status(tmp_path: Path) -> None:
    run_dir = tmp_path / "sigterm_exit"
    proc = subprocess.Popen(
        [
            sys.executable,
            str(PILOT_SCRIPT),
            "--_test-exit-status-run-dir",
            str(run_dir),
            "--_test-exit-status-mode",
            "sleep",
        ],
        cwd=REPO_ROOT,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    try:
        time.sleep(1.0)
        proc.send_signal(signal.SIGTERM)
        stdout, stderr = proc.communicate(timeout=10)
    finally:
        if proc.poll() is None:
            proc.kill()

    assert proc.returncode != 0, (stdout, stderr)
    payload = _read_exit_status(run_dir)
    assert payload["exit_kind"] == "signal"
    assert payload["exit_signal"] == signal.SIGTERM
    assert payload["exit_code"] == 128 + signal.SIGTERM


def test_exception_in_execute_writes_status(tmp_path: Path) -> None:
    run_dir = tmp_path / "exception_exit"

    result = subprocess.run(
        [
            sys.executable,
            str(PILOT_SCRIPT),
            "--_test-exit-status-run-dir",
            str(run_dir),
            "--_test-exit-status-mode",
            "exception",
        ],
        cwd=REPO_ROOT,
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 1
    payload = _read_exit_status(run_dir)
    assert payload["exit_kind"] == "exception"
    assert payload["exit_code"] == 1
    assert payload["exception_class"] == "RuntimeError"
    assert "synthetic pilot failure" in payload["exception_message"]
    assert "RuntimeError" in payload["traceback_tail"]


def test_dump_failure_does_not_mask_original_error(
    tmp_path: Path,
    capsys,
) -> None:
    spec = importlib.util.spec_from_file_location("pilot_real_llm", PILOT_SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    recorder = module.PilotExitRecorder()
    recorder.configure(run_dir=tmp_path / "dump_failure", run_id="dump_failure")
    original = RuntimeError("original failure")
    recorder.mark_exception(original)

    def _boom():
        raise RuntimeError("dump failure")

    recorder._payload = _boom
    recorder.dump()

    captured = capsys.readouterr()
    assert "failed to write pilot_exit_status.json" in captured.err
    assert recorder.exit_kind == "exception"
    assert recorder.exception_class == "RuntimeError"
    assert recorder.exception_message == "original failure"

