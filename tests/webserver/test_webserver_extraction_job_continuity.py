"""Ownership and executable contracts for Data Extraction job restoration."""

import json
import shutil
import subprocess
from pathlib import Path

import pytest


ROOT = Path(__file__).parents[2]
STATIC = ROOT / "src" / "easyicu" / "webserver" / "static"


def _read(relative: str) -> str:
    return (STATIC / relative).read_text(encoding="utf-8")


def _node_binary() -> str | None:
    direct = shutil.which("node")
    if direct:
        return direct
    candidates = sorted((Path.home() / ".nvm" / "versions" / "node").glob("*/bin/node"))
    return str(candidates[-1]) if candidates else None


def test_extraction_job_continuity_has_one_explicit_owner() -> None:
    owner = _read("js/screens-extraction-job-continuity.js")
    extraction = _read("js/screens-extraction.js")
    i18n = _read("js/i18n.js")
    index = _read("index.html")

    owner_src = "js/screens-extraction-job-continuity.js?v=20260802-job-404"
    assert owner_src in index
    assert index.index("js/screens-extraction.js?") < index.index(owner_src)
    assert index.index(owner_src) < index.index("js/screens-extraction-study-context.js?")

    for marker in (
        "easyicu.extractionJob.v1",
        "api.loadJobSnapshot(record.job_id)",
        "new EventSource('/api/jobs/'",
        "function reconcile(",
        "function cleanRecord(",
        "isRunning: () => !!(active && running)",
        "source_changed_before_tracking",
    ):
        assert marker in owner
        assert marker not in extraction

    assert "window.EU_EXTRACTION_JOB_HOST" in extraction
    assert "window.EU_EXTRACTION_JOB_CONTINUITY" in extraction
    assert "localStorage" not in extraction[extraction.index("window.EU_EXTRACTION_JOB_HOST") :]
    assert "continuity.isRunning && continuity.isRunning()" in i18n
    assert "EU_EXTRACTION_JOB_CONTINUITY.active()" not in i18n

    for foreign_marker in (
        "crossdb",
        "StudyContext",
        "agent-run",
        "guided",
        "patient-review",
    ):
        assert foreign_marker not in owner


def test_explicit_terminal_actions_discard_saved_job_pointer() -> None:
    extraction = _read("js/screens-extraction.js")

    assert "window.__euExtractReset = function () {\n    abandonExtractionContinuity();" in extraction
    assert "[data-ex-rescan]').forEach(b => b.addEventListener('click', () => { abandonExtractionContinuity();" in extraction
    assert "[data-ex-convdone]'); if (convDoneBtn) convDoneBtn.addEventListener('click', () => { abandonExtractionContinuity();" in extraction
    assert "[data-ex-reset]').forEach(b => b.addEventListener('click', () => { abandonExtractionContinuity();" in extraction


def test_extraction_job_continuity_executes_refresh_and_race_contracts() -> None:
    node = _node_binary()
    if not node:
        pytest.skip("Node.js is unavailable")

    owner = STATIC / "js" / "screens-extraction-job-continuity.js"
    subprocess.run([node, "--check", str(owner)], check=True, capture_output=True, text=True)
    subprocess.run(
        [node, "--check", str(STATIC / "js" / "screens-extraction.js")],
        check=True,
        capture_output=True,
        text=True,
    )
    result = subprocess.run(
        [node, str(ROOT / "tests" / "js" / "extraction_job_continuity.test.js"), str(owner)],
        check=False,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr or result.stdout
    assert json.loads(result.stdout) == {
        "restored": True,
        "bounded": True,
        "missingCleared": True,
    }
