"""Ownership and executable contracts for raw Cross-DB job reconnects."""

from __future__ import annotations

import json
import shutil
import subprocess
from pathlib import Path

import pytest


ROOT = Path(__file__).parents[1]
STATIC = ROOT / "src" / "easyicu" / "webserver" / "static"


def _read(relative: str) -> str:
    return (STATIC / relative).read_text(encoding="utf-8")


def _node_binary() -> str | None:
    direct = shutil.which("node")
    if direct:
        return direct
    candidates = sorted((Path.home() / ".nvm" / "versions" / "node").glob("*/bin/node"))
    return str(candidates[-1]) if candidates else None


def test_crossdb_raw_job_continuity_has_one_explicit_owner() -> None:
    viz = _read("js/screens-viz.js")
    setup = _read("js/screens-viz-crossdb-setup.js")
    owner = _read("js/screens-viz-crossdb-job-continuity.js")
    index = _read("index.html")

    owner_src = "js/screens-viz-crossdb-job-continuity.js?v=20260710-cancel-fence"
    assert owner_src in index
    assert index.index("js/screens-viz-crossdb-setup.js?") < index.index("js/screens-viz.js?")
    assert index.index("js/screens-viz.js?") < index.index(owner_src)
    assert index.index(owner_src) < index.index("js/screens-viz-crossdb-source.js?")

    for marker in (
        "window.EU_CROSSDB_JOB_CONTINUITY",
        "easyicu_crossdb_raw_job_v1",
        "loadJobSnapshot",
        "restoreIfNeeded",
        "onSourceChanged",
        "new window.EventSource",
        "snapshot.status === 'done'",
        "snapshot.status === 'failed'",
        "snapshot.status === 'cancelled'",
        "cancelFenceJobId",
        "RECONNECT_DELAYS_MS",
        "maxEventSeq",
        "HTTP\\s+404",
    ):
        assert marker in owner

    for marker in (
        "window.EU_CROSSDB_JOB_HOST",
        "jobContinuity.start({",
        "source_identity: crossSetup.sourceIdentity(rawDatabases)",
        "crossSetup.disconnectJob({ forget: true })",
        "This saved raw Cross-DB job is no longer available",
    ):
        assert marker in viz

    for marker in (
        "continuity.restoreIfNeeded()",
        "continuity.onSourceChanged(pathValue(nextRoot), sourceIdentity(), nextMode || state.sampleMode)",
        "disconnectJob({ forget: true })",
    ):
        assert marker in setup

    assert "easyicu_crossdb_raw_job_v1" not in viz
    assert "new EventSource('/api/jobs/' + r.job_id + '/events')" not in viz
    assert "loadJobSnapshot" not in viz


def test_crossdb_raw_job_owner_stays_route_pure_and_bounded() -> None:
    owner = _read("js/screens-viz-crossdb-job-continuity.js")

    assert "rawRoot.length > 4096" in owner
    assert "raw.length > 8192" in owner
    assert "sourceIdentity.length > 256" in owner
    assert "JOB_ID_RE" in owner
    assert "SAMPLE_MODES" in owner
    assert "JSON.stringify(meta)" in owner
    assert "JSON.stringify(result" not in owner

    for foreign_marker in (
        "EU_CROSSDB_SOURCE_CHOICE",
        "data-crossdb-run-registered",
        "loadCrossdbReviewSummary",
        "data-study-handoff",
        "EU_VIZ_CONTEXT",
        "data-patient-",
        "data-cohort-",
        "data-ag-",
        "startExtractionJob",
        "startAgentRun",
    ):
        assert foreign_marker not in owner


def test_crossdb_raw_job_continuity_executes_lifecycle_contract() -> None:
    node = _node_binary()
    if not node:
        pytest.skip("node is required for the executable Cross-DB job contract")

    owner = STATIC / "js" / "screens-viz-crossdb-job-continuity.js"
    subprocess.run([node, "--check", str(owner)], check=True, capture_output=True, text=True)
    test_file = ROOT / "tests" / "js" / "crossdb_job_continuity.test.js"
    result = subprocess.run(
        [node, str(test_file), str(owner)],
        check=True,
        capture_output=True,
        text=True,
    )
    assert json.loads(result.stdout) == {
        "restored": True,
        "terminal_statuses": 3,
        "missing_cleared": True,
        "root_guard": True,
        "late_progress_blocked": True,
        "stale_stream_blocked": True,
        "replay_watermark": True,
        "reconnect_backoff": True,
        "same_job_stale_stream": True,
    }
