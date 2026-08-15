"""Ownership and executable contracts for Cross-DB progress UI."""

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


def test_crossdb_progress_has_one_explicit_owner() -> None:
    index = _read("index.html")
    owner = _read("js/screens-viz-crossdb-progress.js")
    setup = _read("js/screens-viz-crossdb-setup.js")
    viz = _read("js/screens-viz.js")
    owner_src = "js/screens-viz-crossdb-progress.js?v=20260812-crossdb-jobs"

    assert owner_src in index
    assert index.index("js/screens-viz-crossdb-raw.js?") < index.index(owner_src)
    assert index.index(owner_src) < index.index("js/screens-viz-crossdb-setup.js?")
    assert index.index("js/screens-viz-crossdb-setup.js?") < index.index("js/screens-viz.js?")
    assert "window.EU_CROSSDB_PROGRESS" in owner
    assert "const crossRawProgress = window.EU_CROSSDB_PROGRESS" in viz
    assert "let crossRawJobId" not in viz
    assert "let crossRawProg" not in viz
    assert "let crossRawCancelRequested" not in viz
    assert "let crossRawJobStarting" not in viz
    assert "scanRequestSeq: 0" in setup
    assert "requestSeq !== state.scanRequestSeq" in setup
    assert "missingSelectedKeys.length === 0" in setup
    assert 'data-db="${index}"' in setup
    assert 'aria-pressed="${row.selected ? \'true\' : \'false\'}"' in setup
    assert "const rawLoaded = window.EU_CROSSDB_WORKSPACE" in setup
    assert "${rawLoaded ? '' : `<button" in setup
    assert "const preserveScan = window.EU_CROSSDB_WORKSPACE" in setup
    assert "if (!preserveScan) invalidateScan();" in setup
    for marker in (
        "rawRootDraft:",
        "scanRequestSeq:",
        "sampleMode:",
        "registeredLoading:",
        "data-crossdb-root",
        "data-crossdb-sample-mode",
    ):
        assert marker not in viz
    for marker in (
        "data-crossdb-cancel",
        "crossdb-progress-databases",
        "crossdb-progress-error",
        "Cancellation requested. EasyICU will stop after the current bounded read returns.",
        "flushCancel",
        "state.cancelRequested) return false",
    ):
        assert marker in owner


def test_crossdb_progress_owner_and_css_stay_route_pure() -> None:
    owner = _read("js/screens-viz-crossdb-progress.js")
    crossdb_css = _read("css/crossdb.css")
    for marker in (
        "data-patient-",
        "data-cohort-",
        "data-ag-",
        "startExtractionJob",
        "startAgentRun",
        "EU_CROSSDB_SOURCE_CHOICE",
        "loadCrossdbReviewSummary",
    ):
        assert marker not in owner
    for selector in (
        ".crossdb-progress-card",
        ".crossdb-progress-bar",
        ".crossdb-progress-databases",
        ".crossdb-progress-db",
        ".crossdb-progress-error",
    ):
        assert selector in crossdb_css
        for non_owner in (
            "agent.css",
            "cohort.css",
            "extraction.css",
            "guided.css",
            "ideas.css",
            "patient.css",
            "settings.css",
        ):
            assert selector not in _read(f"css/{non_owner}")
    assert crossdb_css.count("{") == crossdb_css.count("}")
    assert owner.count("{") == owner.count("}")


def test_crossdb_progress_owner_executes_behavior_contract() -> None:
    node = _node_binary()
    if not node:
        pytest.skip("node is required for the Cross-DB progress contract")
    owner = STATIC / "js" / "screens-viz-crossdb-progress.js"
    subprocess.run([node, "--check", str(owner)], check=True, capture_output=True, text=True)
    result = subprocess.run(
        [node, str(ROOT / "tests" / "js" / "crossdb_progress_owner.test.js"), str(owner)],
        check=True,
        capture_output=True,
        text=True,
    )
    assert json.loads(result.stdout) == {
        "deferred_cancel": True,
        "cancel_api_guard": True,
        "cancel_error_visible": True,
        "cancel_history_restored": True,
        "late_progress_blocked": True,
        "structured_progress": True,
    }
