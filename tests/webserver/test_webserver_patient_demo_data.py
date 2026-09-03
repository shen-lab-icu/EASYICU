from __future__ import annotations

import json
from pathlib import Path
import shutil
import subprocess

import pytest


STATIC_DIR = (
    Path(__file__).resolve().parents[2] / "src" / "easyicu" / "webserver" / "static"
)


def _static_js(name: str) -> str:
    return (STATIC_DIR / "js" / name).read_text(encoding="utf-8")


def _static_html(name: str) -> str:
    return (STATIC_DIR / name).read_text(encoding="utf-8")


def test_patient_seeded_demo_uses_clinical_table_shape() -> None:
    demo_js = _static_js("screens-viz-demo.js")
    demo_drilldown_js = _static_js("screens-viz-demo-drilldown.js")
    viz_js = _static_js("screens-viz.js")
    patient_js = _static_js("screens-viz-patient.js")
    index_html = _static_html("index.html")

    assert (
        "const DEMO_CHART_HOURS = [-1, 0.5, 2, 4, 7, 11, 16, 22, 29, 36, 43, 48];"
        in demo_js
    )
    assert "function demoCharttimeAt(rowIndex)" in demo_js
    assert "function demoTableValue(feature, entityIndex, timeIndex)" in demo_js
    assert "function demoScenarioState(entityIndex, hour)" in demo_js
    assert "function demoHasClinicalModel(feature)" in demo_js
    assert "const DEMO_SCORED_COMPONENTS = new Set" in demo_js
    assert "'sofa2_resp'" in demo_js
    assert "return demoClamp(Math.round(raw), 0, 4);" in demo_js
    assert "Math.sin(" not in demo_js
    assert "Math.random(" not in demo_js

    assert "if (timeIndexed) out.charttime = context.charttime;" in demo_drilldown_js
    assert (
        "out[feature] = demoTableValue(feature, context.entityIndex, context.timeIndex);"
        in demo_drilldown_js
    )
    assert "function buildPatientDrilldown(selectedRef)" in demo_drilldown_js
    assert "function buildPatientDrilldown(selectedRef)" not in viz_js
    assert "featureOwner.augmentLanes(rawLanes, drill)" in patient_js
    assert "const lanes = patientCatalogLanes(" in patient_js
    assert "drill && drill.feature_coverage" in patient_js
    assert "2026-01-01" not in patient_js

    assert "js/screens-viz-demo.js?" in index_html
    assert "js/screens-viz-demo-drilldown.js?" in index_html
    assert (
        index_html.index("js/screens-viz-demo.js?")
        < index_html.index("js/screens-viz-demo-drilldown.js?")
        < index_html.index("js/screens-viz.js?")
    )
    assert "js/screens-viz.js?" in index_html
    # A stale pin must stay gone. This asserts the OLD string is absent — my
    # bulk re-pin briefly rewrote it to the current one, which made the two
    # lines contradict each other.
    assert "js/screens-viz.js?v=20260627-survival-outcome-summary" not in index_html


def test_patient_demo_fidelity_contract_executes() -> None:
    node = shutil.which("node")
    if not node:
        pytest.skip("Node.js is unavailable")
    root = Path(__file__).resolve().parents[2]
    result = subprocess.run(
        [
            node,
            str(root / "tests" / "js" / "patient_demo_fidelity.test.js"),
            str(STATIC_DIR / "js" / "data-catalog.js"),
            str(STATIC_DIR / "js" / "screens-viz-demo.js"),
            str(STATIC_DIR / "js" / "screens-viz-demo-drilldown.js"),
        ],
        check=False,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr or result.stdout
    assert json.loads(result.stdout) == {
        "all_catalog_features_grouped": True,
        "clinically_correlated": True,
        "derived_scores_consistent": True,
        "deterministic": True,
        "irregular_cadence": True,
        "unmodeled_not_invented": True,
    }


def test_patient_echarts_owner_contract_executes() -> None:
    node = shutil.which("node")
    if not node:
        pytest.skip("Node.js is unavailable")
    root = Path(__file__).resolve().parents[2]
    result = subprocess.run(
        [
            node,
            str(root / "tests" / "js" / "patient_echarts_owner.test.js"),
            str(STATIC_DIR / "js" / "screens-viz-patient-charts.js"),
        ],
        check=False,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr or result.stdout
    assert json.loads(result.stdout) == {
        "aria": True,
        "elapsed_time_alignment": True,
        "html_tooltip_disabled": True,
        "irregular_spacing": True,
        "local_svg_renderer": True,
        "renderer_failure_fallback": True,
        "repaint_disposes": True,
        "step_interventions": True,
        "thresholds": True,
    }


def test_patient_series_owner_contract_executes() -> None:
    node = shutil.which("node")
    if not node:
        pytest.skip("Node.js is unavailable")
    root = Path(__file__).resolve().parents[2]
    result = subprocess.run(
        [
            node,
            str(root / "tests" / "js" / "patient_series_owner.test.js"),
            str(STATIC_DIR / "js" / "data-catalog.js"),
            str(STATIC_DIR / "js" / "screens-viz-demo.js"),
            str(STATIC_DIR / "js" / "screens-viz-patient-features.js"),
            str(STATIC_DIR / "js" / "screens-viz-patient-series.js"),
        ],
        check=False,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr or result.stdout
    assert json.loads(result.stdout) == {
        "all_features_discoverable": True,
        "categorical_observations_distinct": True,
        "numeric_trajectory_count_truthful": True,
        "paired_time_filtering": True,
    }


def test_patient_official_demo_source_owner_contract_executes() -> None:
    node = shutil.which("node")
    if not node:
        pytest.skip("Node.js is unavailable")
    root = Path(__file__).resolve().parents[2]
    result = subprocess.run(
        [
            node,
            str(root / "tests" / "js" / "patient_demo_sources_owner.test.js"),
            str(STATIC_DIR / "js" / "screens-viz-patient-demo-sources.js"),
        ],
        check=False,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr or result.stdout
    assert json.loads(result.stdout) == {
        "done_status_supported": True,
        "official_sources_rendered": True,
        "progress_telemetry_visible": True,
        "prepare_single_flight": True,
        "refresh_reconnect_supported": True,
        "prepared_source_openable": True,
        "provenance_visible": True,
        "official_pair_resolved": True,
        "shared_owner_contract": True,
        "user_mode_remains_demo": True,
        "synthetic_fallback_explicit": True,
    }
