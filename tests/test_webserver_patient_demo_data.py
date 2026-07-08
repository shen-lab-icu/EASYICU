from __future__ import annotations

from pathlib import Path


STATIC_DIR = (
    Path(__file__).resolve().parents[1] / "src" / "easyicu" / "webserver" / "static"
)


def _static_js(name: str) -> str:
    return (STATIC_DIR / "js" / name).read_text(encoding="utf-8")


def _static_html(name: str) -> str:
    return (STATIC_DIR / name).read_text(encoding="utf-8")


def test_patient_seeded_demo_uses_clinical_table_shape() -> None:
    demo_js = _static_js("screens-viz-demo.js")
    viz_js = _static_js("screens-viz.js")
    index_html = _static_html("index.html")

    assert "const DEMO_CHART_HOURS = [0, 1, 2, 3, 4, 6, 8, 12, 18, 24, 36, 48];" in demo_js
    assert "function demoCharttimeAt(rowIndex)" in demo_js
    assert "function demoTableValue(feature, entityIndex)" in demo_js
    assert "const DEMO_SCORED_COMPONENTS = new Set" in demo_js
    assert "'sofa2_resp'" in demo_js
    assert "return demoClamp(Math.round(raw), 0, 4);" in demo_js

    assert "out.charttime = demoCharttimeAt(idx);" in viz_js
    assert "out[feature] = demoTableValue(feature, idx + featureIdx + moduleIdx);" in viz_js
    assert "2026-01-01" not in viz_js

    assert "js/screens-viz-demo.js?v=20260627-demo-clinical-shape" in index_html
    assert "js/screens-viz.js?v=20260707-residuals2" in index_html
    assert "js/screens-viz.js?v=20260627-survival-outcome-summary" not in index_html
