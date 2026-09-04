"""Process-tree memory evidence must remain machine-readable on success/failure."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import sys


SCRIPT = (
    Path(__file__).resolve().parents[2]
    / "tools"
    / "run_with_memory_evidence.py"
)
SPEC = importlib.util.spec_from_file_location("memory_evidence_runner", SCRIPT)
assert SPEC and SPEC.loader
runner = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(runner)


def test_memory_evidence_runner_records_success(tmp_path: Path) -> None:
    output = tmp_path / "memory.json"

    exit_code = runner.run(
        [
            sys.executable,
            "-c",
            "import time; x = [0] * 100000; time.sleep(0.15)",
        ],
        output=output,
        interval=0.05,
    )

    evidence = json.loads(output.read_text(encoding="utf-8"))
    assert exit_code == 0
    assert evidence["status"] == "complete"
    assert evidence["process_exit_code"] == 0
    assert evidence["peak_process_count"] >= 1
    assert evidence["peak_process_tree_rss_mb"] > 0
    assert evidence["samples"] >= 1


def test_memory_evidence_runner_records_child_failure(tmp_path: Path) -> None:
    output = tmp_path / "memory.json"

    exit_code = runner.run(
        [sys.executable, "-c", "raise SystemExit(7)"],
        output=output,
        interval=0.05,
    )

    evidence = json.loads(output.read_text(encoding="utf-8"))
    assert exit_code == 7
    assert evidence["status"] == "failed"
    assert evidence["process_exit_code"] == 7
    assert evidence["ended_at_utc"] is not None


def test_memory_evidence_runner_stops_process_tree_at_rss_limit(
    tmp_path: Path,
) -> None:
    output = tmp_path / "memory.json"

    exit_code = runner.run(
        [
            sys.executable,
            "-c",
            "import time; x = bytearray(64 * 1024 * 1024); time.sleep(30)",
        ],
        output=output,
        interval=0.05,
        rss_limit_mb=32,
    )

    evidence = json.loads(output.read_text(encoding="utf-8"))
    assert exit_code == 137
    assert evidence["status"] == "rss_limit_exceeded"
    assert evidence["process_exit_code"] == 137
    assert evidence["rss_limit_mb"] == 32
    assert evidence["stopped_for_rss"] is True
    assert evidence["peak_process_tree_rss_mb"] >= 32
