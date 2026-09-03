"""Ownership and executable contracts for the Cross-DB source-choice UI."""

from __future__ import annotations

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


def test_crossdb_source_choice_owner_is_explicitly_wired() -> None:
    index = _read("index.html")
    viz = _read("js/screens-viz.js")
    setup = _read("js/screens-viz-crossdb-setup.js")
    owner = _read("js/screens-viz-crossdb-source.js")

    owner_src = "js/screens-viz-crossdb-source.js?v=20260812-crossdb-jobs"
    assert owner_src in index
    assert index.index("js/screens-viz.js?") < index.index(owner_src)
    assert index.index(owner_src) < index.index("js/screens-viz-study-context.js?")

    assert "window.EU_CROSSDB_SOURCE_HOST" in viz
    assert "window.EU_CROSSDB_SOURCE_CHOICE" in setup
    assert "sourceChoice.render({ registryHtml })" in setup
    assert "sourceChoice.renderLoading()" in setup
    assert "sourceChoice.wire(root)" in setup
    assert "data-crossdb-run-registered" not in viz
    assert "registryHtml() { return sourceRegistryBlock('multi'); }" in viz
    assert "explicitRegistryCrossdbPaths" in viz
    assert 'type="button" data-src-cross=' in viz
    assert 'aria-pressed="${on ? \'true\' : \'false\'}"' in viz
    assert "const cur = explicitRegistryCrossdbPaths();" in viz
    assert "function registeredPaths()" in owner
    assert "registeredPaths().length < 2" in owner

    for marker in (
        "window.EU_CROSSDB_SOURCE_CHOICE",
        "data-crossdb-source-choice",
        "data-crossdb-run-registered",
        "data-crossdb-registered-loading",
        "selected exports",
        "Start consistency check",
        "Add and select at least two EasyICU exports.",
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
    setup = _read("js/screens-viz-crossdb-setup.js")
    host = viz.split("window.EU_CROSSDB_SOURCE_HOST =", 1)[1].split(
        "function loadDemoCrossdb", 1
    )[0]

    assert "registeredPaths()" in host
    assert "explicitRegistryCrossdbPaths()" in host
    assert "runRegistered()" in host
    assert "crossSetup.setRegisteredLoading(true)" in host
    assert "crossSetup.setView('loading')" in host
    assert "loadRealCrossdb(ok =>" in host
    assert "{ operationId, registeredPaths: paths }" in host
    assert "scanCrossdbRawRoot" not in host
    assert "rawRoot" not in host

    assert "const registeredPathOverride" in viz
    assert "window.EU_API.startCrossdbReviewSummaryJob" in viz
    assert "deadline_seconds: 120" in viz
    registered_branch = viz.split("if (!requestedRawRoot && paths.length >= 2", 1)[1].split(
        "if (!requestedRawRoot && registeredPathOverride)", 1
    )[0]
    assert "loadCrossdbReviewSummary" not in registered_branch
    assert "runRaw: loadRealCrossdb" in viz
    assert "config.runRaw(ok =>" in setup
    assert "{ operationId, rawRoot: rootValue, setup: runSnapshot }" in setup


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
    assert json.loads(result.stdout) == {
        "official_pair_ready": True,
        "official_run_count": 1,
        "ready_sources": 2,
        "run_count": 1,
    }
