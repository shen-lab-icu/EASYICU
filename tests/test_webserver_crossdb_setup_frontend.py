"""Ownership and executable contracts for the Cross-DB setup/scan owner."""

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


def test_crossdb_setup_owner_is_unique_and_wired_in_dependency_order() -> None:
    index = _read("index.html")
    setup = _read("js/screens-viz-crossdb-setup.js")
    shell = _read("js/screens-viz.js")
    setup_src = "js/screens-viz-crossdb-setup.js?v=20260710-setup-owner"

    assert index.count(setup_src) == 1
    ordered = (
        "js/screens-viz-crossdb-raw.js?",
        "js/screens-viz-crossdb-progress.js?",
        setup_src,
        "js/screens-viz.js?",
        "js/screens-viz-crossdb-job-continuity.js?",
        "js/screens-viz-crossdb-source.js?",
    )
    positions = [index.index(marker) for marker in ordered]
    assert positions == sorted(positions)

    assert "window.EU_CROSSDB_SETUP = {" in setup
    assert "const crossSetup = window.EU_CROSSDB_SETUP" in shell
    for state_marker in (
        "const DATABASES = [",
        "rawRootDraft: ''",
        "rawRootScan: null",
        "rawRootScanPath: ''",
        "rawRootScanning: false",
        "scanRequestSeq: 0",
        "operationSeq: 0",
        "sampleMode: 'quick'",
        "registeredLoading: false",
    ):
        assert state_marker in setup
        assert state_marker not in shell

    for removed_shell_marker in (
        "const CROSS_DBS = [",
        "let crossView = 'idle'",
        "let crossRawRootDraft",
        "let crossRawRootScan",
        "let crossRawRootScanPath",
        "let crossRawRootScanning",
        "let crossRawScanRequestSeq",
        "let crossRawSampleMode",
        "let crossRegisteredLoading",
        "function scanCrossdbRawRoot(",
        "function rawCrossdbSetup(",
        "function crossLoadingState(",
    ):
        assert removed_shell_marker not in shell


def test_crossdb_setup_owner_stays_route_pure_and_bounded() -> None:
    setup = _read("js/screens-viz-crossdb-setup.js")
    shell = _read("js/screens-viz.js")
    source = _read("js/screens-viz-crossdb-source.js")
    crossdb_css = _read("css/crossdb.css")

    for owner_marker in (
        "function scan(",
        "requestSeq !== state.scanRequestSeq",
        "missingSelectedKeys.length === 0",
        "function acceptResume(",
        "function beginOperation(",
        "function invalidateOperations(",
        "function operationCurrent(",
        "function renderReal(",
        "function renderDemo(",
        "function bind(",
        "data-crossdb-run-raw",
        "data-crossdb-root-scan",
        "data-crossdb-select-detected",
        'role="status" aria-live="polite" aria-atomic="true"',
        'aria-disabled="true" aria-busy="true"',
        "repaint(config, `[data-db=",
    ):
        assert owner_marker in setup

    for foreign_marker in (
        "data-patient-",
        "data-cohort-",
        "data-ag-",
        "startExtractionJob",
        "startAgentRun",
        "loadPatientReview",
        "loadCohortReview",
        "EU_AGENT",
        "EU_VIZ_CONTEXT",
    ):
        assert foreign_marker not in setup

    assert len(setup.splitlines()) < 1500
    assert setup.count("{") == setup.count("}")
    assert setup.count("/*") == setup.count("*/")
    assert 'aria-modal="true" aria-labelledby="eu-source-picker-title"' in shell
    assert "sourcePickerReturnFocus" in shell
    assert "e.key !== 'Tab' || !sourcePickerEl" in shell
    assert "Close folder picker" in shell
    assert "<details class=\"card crossdb-source-option" in source
    for selector in (
        ".crossdb-source-option",
        ".crossdb-source-summary",
        ".crossdb-source-detail",
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
    assert crossdb_css.count("/*") == crossdb_css.count("*/")


def test_crossdb_setup_owner_executes_state_and_reset_contract() -> None:
    node = _node_binary()
    if not node:
        pytest.skip("node is required for the Cross-DB setup contract")

    owner = STATIC / "js" / "screens-viz-crossdb-setup.js"
    subprocess.run([node, "--check", str(owner)], check=True, capture_output=True, text=True)
    result = subprocess.run(
        [node, str(ROOT / "tests" / "js" / "crossdb_setup_owner.test.js"), str(owner)],
        check=True,
        capture_output=True,
        text=True,
    )
    assert json.loads(result.stdout) == {
        "bounded_profile": True,
        "identity_resume_fail_closed": True,
        "missing_api_visible": True,
        "operation_reset_fence": True,
        "raw_completion_loaded": True,
        "raw_registry_guard": True,
        "raw_reset_cancel": True,
        "scan_reused": True,
        "selection_revalidated": True,
        "server_text_escaped": True,
        "stale_scan_blocked": True,
    }
