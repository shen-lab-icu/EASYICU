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
    results = _read("js/screens-viz-crossdb-results.js")
    shell = _read("js/screens-viz.js")
    setup_src = "js/screens-viz-crossdb-setup.js?"
    charts_src = "js/screens-viz-crossdb-charts.js?"
    results_src = "js/screens-viz-crossdb-results.js?"

    assert index.count(setup_src) == 1
    assert index.count(results_src) == 1
    ordered = (
        "js/screens-viz-crossdb-raw.js?",
        "js/screens-viz-crossdb-progress.js?",
        setup_src,
        charts_src,
        results_src,
        "js/screens-viz.js?",
        "js/screens-viz-crossdb-job-continuity.js?",
        "js/screens-viz-crossdb-source.js?",
    )
    positions = [index.index(marker) for marker in ordered]
    assert positions == sorted(positions)

    assert "window.EU_CROSSDB_SETUP = {" in setup
    assert "window.EU_CROSSDB_RESULTS = {" in results
    assert index.count(charts_src) == 1
    assert "const crossSetup = window.EU_CROSSDB_SETUP" in shell
    assert "const crossResults = window.EU_CROSSDB_RESULTS" in shell
    for state_marker in (
        "const DATABASES = [",
        "sourceMethod: 'registered'",
        "rawRootDraft: ''",
        "rawRootScan: null",
        "rawRootScanPath: ''",
        "rawRootScanning: false",
        "scanRequestSeq: 0",
        "operationSeq: 0",
        "sampleMode: 'quick'",
        "featureScope: 'all'",
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
        "function crossRealLoaded(",
        "function crossFeatureDensityPanel(",
        "let crossDensityModule",
        "let crossDensityFeature",
    ):
        assert removed_shell_marker not in shell


def test_crossdb_setup_owner_stays_route_pure_and_bounded() -> None:
    setup = _read("js/screens-viz-crossdb-setup.js")
    shell = _read("js/screens-viz.js")
    source = _read("js/screens-viz-crossdb-source.js")
    results = _read("js/screens-viz-crossdb-results.js")
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
        "data-crossdb-source-method",
        "data-crossdb-feature-scope",
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
    assert "data-crossdb-run-registered" in source
    assert "compactOfficialPair" in source
    assert "data-crossdb-result-tab" in results
    assert "data-crossdb-feature-query" in results
    assert "data-crossdb-feature-workspace" not in shell
    assert len(results.splitlines()) < 600
    for foreign_marker in ("data-patient-", "data-cohort-", "data-ag-", "EU_API"):
        assert foreign_marker not in results
    for selector in (
        ".crossdb-method-grid",
        ".crossdb-source-summary",
        ".crossdb-source-detail",
        ".xdb-result-tabs",
        ".xdb-feature-workspace",
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
        "explicit_feature_scope": True,
        "identity_resume_fail_closed": True,
        "missing_api_visible": True,
        "one_click_full_default": True,
        "typed_root_enables_primary": True,
        "operation_reset_fence": True,
        "progressive_source_choice": True,
        "raw_completion_loaded": True,
        "raw_registry_guard": True,
        "raw_reset_cancel": True,
        "scan_reused": True,
        "selection_revalidated": True,
        "server_text_escaped": True,
        "stale_scan_blocked": True,
    }


def test_crossdb_results_owner_executes_navigation_and_single_chart_contract() -> None:
    node = _node_binary()
    if not node:
        pytest.skip("node is required for the Cross-DB result contract")

    owner = STATIC / "js" / "screens-viz-crossdb-results.js"
    subprocess.run([node, "--check", str(owner)], check=True, capture_output=True, text=True)
    result = subprocess.run(
        [node, str(ROOT / "tests" / "js" / "crossdb_results_owner.test.js"), str(owner)],
        check=True,
        capture_output=True,
        text=True,
    )
    assert json.loads(result.stdout) == {
        "accessible_tabs": True,
        "complete_catalog_filter": True,
        "full_scope_handoff": True,
        "no_duplicate_actions": True,
        "one_main_chart": True,
        "partial_scope_disclosed": True,
        "result_tabs": True,
    }


def test_review_routes_share_the_echarts_renderer_contract() -> None:
    node = _node_binary()
    if not node:
        pytest.skip("node is required for the shared ECharts contract")

    sources = [
        STATIC / "js" / "screens-viz-echarts.js",
        STATIC / "js" / "screens-viz-crossdb-charts.js",
        STATIC / "js" / "screens-viz-cohort-charts.js",
    ]
    for source in sources:
        subprocess.run(
            [node, "--check", str(source)],
            check=True,
            capture_output=True,
            text=True,
        )
    result = subprocess.run(
        [
            node,
            str(ROOT / "tests" / "js" / "review_echarts_owners.test.js"),
            *(str(source) for source in sources),
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    assert json.loads(result.stdout) == {
        "cohort_heatmap": True,
        "cohort_survival": True,
        "crossdb_density": True,
        "fail_closed_fallback": True,
        "resize_dispose": True,
        "shared_svg_renderer": True,
    }
