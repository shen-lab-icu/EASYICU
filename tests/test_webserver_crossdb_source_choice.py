"""Ownership and executable contracts for the Cross-DB source-choice UI."""

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


def test_crossdb_source_choice_owner_is_explicitly_wired() -> None:
    index = _read("index.html")
    viz = _read("js/screens-viz.js")
    owner = _read("js/screens-viz-crossdb-source.js")

    owner_src = "js/screens-viz-crossdb-source.js?v=20260710-source-choice"
    assert owner_src in index
    assert index.index("js/screens-viz.js?") < index.index(owner_src)
    assert index.index(owner_src) < index.index("js/screens-viz-study-context.js?")

    assert "window.EU_CROSSDB_SOURCE_HOST" in viz
    assert "window.EU_CROSSDB_SOURCE_CHOICE" in viz
    assert "sourceChoice.render({ registryHtml: sourceRegistryBlock('multi') })" in viz
    assert "sourceChoice.renderLoading()" in viz
    assert "window.EU_CROSSDB_SOURCE_CHOICE.wire(root)" in viz
    assert "data-crossdb-run-registered" not in viz
    assert "explicitRegistryCrossdbPaths" in viz
    assert 'type="button" data-src-cross=' in viz
    assert 'aria-pressed="${on ? \'true\' : \'false\'}"' in viz
    assert "const cur = explicitRegistryCrossdbPaths();" in viz

    for marker in (
        "window.EU_CROSSDB_SOURCE_CHOICE",
        "data-crossdb-source-choice",
        "data-crossdb-registered-option",
        "data-crossdb-run-registered",
        "data-crossdb-registered-loading",
        "Registered EasyICU exports",
        "Run registered exports",
        "Add and select at least two EasyICU exports below.",
    ):
        assert marker in owner


def test_crossdb_source_choice_owner_stays_route_pure() -> None:
    owner = _read("js/screens-viz-crossdb-source.js")
    for foreign_marker in (
        "data-crossdb-root",
        "scanCrossdbRawRoot",
        "startCrossdbRawDistributionJob",
        "loadCrossdbReviewSummary",
        "data-study-handoff",
        "EU_VIZ_CONTEXT",
        "data-patient-",
        "data-cohort-",
        "data-ag-",
    ):
        assert foreign_marker not in owner


def test_registered_export_host_bypasses_raw_root_scan() -> None:
    viz = _read("js/screens-viz.js")
    host = viz.split("window.EU_CROSSDB_SOURCE_HOST =", 1)[1].split(
        "function loadDemoCrossdb", 1
    )[0]

    assert "registeredPaths()" in host
    assert "explicitRegistryCrossdbPaths()" in host
    assert "runRegistered()" in host
    assert "crossView = 'loading'" in host
    assert "loadRealCrossdb(ok =>" in host
    assert "{ registeredPaths: paths }" in host
    assert "scanCrossdbRawRoot" not in host
    assert "rawRoot" not in host

    assert "const registeredPathOverride" in viz
    assert "window.EU_API.loadCrossdbReviewSummary({ paths: paths })" in viz
    assert "loadedCrossdb.source_type !== 'raw_database_root'" in viz
    assert "window.EU_CROSSDB_SOURCE_HOST.runRegistered();" in viz
    assert "loadRealCrossdb(() => { crossView = 'idle'; repaintScreen('crossdb'); }, { rawRoot });" in viz


def test_crossdb_registered_action_executes_owner_contract() -> None:
    node = _node_binary()
    if not node:
        pytest.skip("node is required for the executable Cross-DB UI contract")

    owner = STATIC / "js" / "screens-viz-crossdb-source.js"
    subprocess.run([node, "--check", str(owner)], check=True, capture_output=True, text=True)
    test_file = ROOT / "tests" / "js" / "crossdb_source_choice.test.js"
    result = subprocess.run(
        [node, str(test_file), str(owner)],
        check=True,
        capture_output=True,
        text=True,
    )
    assert json.loads(result.stdout) == {"ready_sources": 2, "run_count": 1}
