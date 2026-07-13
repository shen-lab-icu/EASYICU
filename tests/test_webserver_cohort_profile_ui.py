from __future__ import annotations

from pathlib import Path


STATIC_DIR = (
    Path(__file__).resolve().parents[1] / "src" / "easyicu" / "webserver" / "static"
)


def _static_js(name: str) -> str:
    return (STATIC_DIR / "js" / name).read_text(encoding="utf-8")


def _static_css(name: str) -> str:
    return (STATIC_DIR / "css" / name).read_text(encoding="utf-8")


def _static_html(name: str) -> str:
    return (STATIC_DIR / name).read_text(encoding="utf-8")


def test_demo_cohort_profile_has_clinical_domains() -> None:
    viz_js = _static_js("screens-viz.js")
    cohort_css = _static_css("cohort.css")
    index_html = _static_html("index.html")

    assert "function demoCohortClinicalProfile()" in viz_js
    assert "demo_cohort_aggregate_no_patient_rows" in viz_js
    assert "Treatments and organ support" in viz_js
    assert "Diagnoses and comorbidities" in viz_js
    assert "Vitals and laboratory profile" in viz_js
    assert "Mechanical ventilation" in viz_js
    assert "Vasopressor exposure" in viz_js
    assert "AKI / renal dysfunction" in viz_js
    assert "Lactate" in viz_js
    assert "Data completeness" in viz_js
    assert "cohortClinicalProfile(demoProfile)" in viz_js

    assert ".cprof-spark-grid" in cohort_css
    assert "css/cohort.css?v=20260707-design" in index_html
    assert "js/screens-viz.js?v=20260712-ux-fixes" in index_html
