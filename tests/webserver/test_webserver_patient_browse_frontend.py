"""Executable and static ownership contracts for Patient Review browsing."""

import json
import shutil
import subprocess
from pathlib import Path

import pytest


ROOT = Path(__file__).parents[2]
STATIC = ROOT / "src" / "easyicu" / "webserver" / "static"


def _node_binary() -> str | None:
    direct = shutil.which("node")
    if direct:
        return direct
    candidates = sorted((Path.home() / ".nvm" / "versions" / "node").glob("*/bin/node"))
    return str(candidates[-1]) if candidates else None


def _read(relative: str) -> str:
    return (STATIC / relative).read_text(encoding="utf-8")


def test_patient_browse_owners_reject_stale_responses_and_preserve_old_tables() -> None:
    node = _node_binary()
    if not node:
        pytest.skip("Node.js is unavailable")
    result = subprocess.run(
        [
            node,
            str(ROOT / "tests" / "js" / "patient_browse_owners.test.js"),
            str(STATIC / "js" / "screens-viz-patient-navigation.js"),
            str(STATIC / "js" / "screens-viz-patient-tables.js"),
        ],
        check=False,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr or result.stdout
    assert json.loads(result.stdout) == {"ok": True, "table_calls": 4}


def test_patient_feature_loader_is_entity_scoped_and_rejects_stale_responses() -> None:
    node = _node_binary()
    if not node:
        pytest.skip("Node.js is unavailable")
    result = subprocess.run(
        [
            node,
            str(ROOT / "tests" / "js" / "patient_feature_loader_owner.test.js"),
            str(STATIC / "js" / "screens-viz-patient-feature-loader.js"),
        ],
        check=False,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr or result.stdout
    assert json.loads(result.stdout) == {
        "bounded_module_batch_supported": True,
        "cache_scoped_to_entity": True,
        "projected_request": True,
        "stale_response_rejected": True,
    }


def test_patient_browse_css_has_route_pure_owners_and_balanced_syntax() -> None:
    navigation = _read("css/patient-navigation.css")
    tables = _read("css/patient-tables.css")
    non_owners = {
        name: _read(f"css/{name}")
        for name in ("pages.css", "patient.css", "cohort.css", "crossdb.css")
    }

    assert ".pt-entity-nav" in navigation
    assert ".pt-entity-pager" in navigation
    assert ".patient-table-scroll" in tables
    assert ".patient-table-pager" in tables
    assert ".pt-entity" not in tables
    assert ".patient-table" not in navigation
    for css in (navigation, tables):
        assert css.count("{") == css.count("}")
        assert css.count("/*") == css.count("*/")
    for css in non_owners.values():
        assert ".pt-entity-nav" not in css
        assert ".patient-table-scroll" not in css
        assert ".patient-table-pager" not in css
        assert ".pt-feature-inventory-item.loadable" not in css


def test_patient_browse_owners_are_wired_before_the_host_screen() -> None:
    index = _read("index.html")
    api = _read("js/api.js")
    host = _read("js/screens-viz-patient.js")
    navigation = _read("js/screens-viz-patient-navigation.js")
    tables = _read("js/screens-viz-patient-tables.js")
    feature_loader = _read("js/screens-viz-patient-feature-loader.js")
    patient_series = _read("js/screens-viz-patient-series.js")
    patient_series_css = _read("css/patient-series.css")

    assert index.index("js/screens-viz-patient-navigation.js?") < index.index(
        "js/screens-viz-patient.js?"
    )
    assert index.index("js/screens-viz-patient-tables.js?") < index.index(
        "js/screens-viz-patient.js?"
    )
    assert index.index("js/screens-viz-patient-feature-loader.js?") < index.index(
        "js/screens-viz-patient.js?"
    )
    for method, endpoint in (
        ("loadPatientReviewEntities", "/api/patient-review/entities"),
        ("loadPatientReviewEntity", "/api/patient-review/entity"),
        ("loadPatientReviewTablePreview", "/api/patient-review/table-preview"),
        ("loadPatientReviewFeature", "/api/patient-review/feature"),
    ):
        assert method in api
        assert endpoint in api
    assert "window.EU_PATIENT_REVIEW.navigation" in host
    assert "window.EU_PATIENT_REVIEW.tables" in host
    assert "window.EU_PATIENT_REVIEW.features" in host
    assert "requestSeq" in navigation
    assert "requestSeq" in tables
    assert "MAX_CACHE_ENTRIES = 12" in tables
    assert "MAX_CACHE_ENTRIES = 320" in feature_loader
    assert "MAX_PARALLEL_LOADS = 4" in feature_loader
    assert "previewCache.clear()" in tables
    assert 'role="status" aria-live="polite"' in navigation
    assert 'role="status" aria-live="polite"' in tables
    assert 'role="alert"' in tables
    assert "aria-pressed" in navigation
    assert "aria-pressed" in host
    assert "restoreFocus" in navigation
    assert "restoreFocus" in tables
    assert "Table page controls" in host
    assert "bounded table preview" in host
    assert "loadPatientReviewDrilldown" not in navigation
    assert "loadPatientReviewDrilldown" not in tables
    assert "loadPatientReviewDrilldown" not in feature_loader
    assert 'data-patient-feature-load="' in patient_series
    assert 'data-patient-module-load="' in patient_series
    assert 'data-patient-inventory-toggle="open"' in patient_series
    assert ".pt-feature-inventory-item.loadable" in patient_series_css
    assert patient_series_css.count("{") == patient_series_css.count("}")
    assert patient_series_css.count("/*") == patient_series_css.count("*/")
