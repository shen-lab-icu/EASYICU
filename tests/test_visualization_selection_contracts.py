from __future__ import annotations

from pathlib import Path

from easyicu.webserver.catalog import build_concept_lineage


ROOT = Path(__file__).resolve().parents[1]
STATIC = ROOT / "src" / "easyicu" / "webserver" / "static"


def _text(relative: str) -> str:
    return (STATIC / relative).read_text(encoding="utf-8")


def test_patient_visualization_selections_have_route_owned_markers() -> None:
    assert "data-pt-table-dashboard" in _text("js/screens-viz-patient-tables.js")
    assert "data-pt-module-pareto" in _text("js/screens-viz-patient-tables.js")
    assert "slice(0, 5)" in _text("js/screens-viz-patient-series.js")
    assert "data-patient-trajectory-summary" in _text("js/screens-viz-patient-series.js")
    assert "data-patient-distribution-latest" in _text("js/screens-viz-patient-overview.js")
    assert "data-patient-missing-record-scatter" in _text("js/screens-viz-patient-overview.js")


def test_cohort_visualization_selections_have_route_owned_markers() -> None:
    source = _text("js/screens-viz.js")
    assert "data-cohort-table-one" in source
    assert "data-cohort-coverage-forest" in source
    assert "cohortSofaMatrixGranularity = 'exact'" in source
    assert "pairedExact" in _text("js/screens-viz-cohort-charts.js")


def test_crossdb_visualization_selections_have_route_owned_markers() -> None:
    source = _text("js/screens-viz-crossdb-results.js")
    assert "data-crossdb-three-scale" in source
    assert "data-crossdb-paired-coverage" in source
    assert "data-crossdb-table-one" in source
    assert "Feature distribution comparison" in source


def test_dictionary_lineage_uses_declared_mapping_metadata() -> None:
    lineage = build_concept_lineage("glu")
    assert lineage is not None
    assert lineage["scope"] == "declared_dictionary_lineage_not_observed_data"
    assert lineage["canonical"]["unit"] == "mg/dL"
    assert lineage["canonical"]["minimum"] == 0
    assert lineage["canonical"]["maximum"] == 1000
    lanes = {lane["database"]: lane for lane in lineage["lanes"]}
    assert set(lanes) == {"miiv", "mimic", "eicu", "aumc", "hirid", "sic"}
    assert lanes["miiv"]["mappings"][0]["table"] == "labevents"
    assert lanes["aumc"]["mappings"][0]["callback"]


def test_dictionary_lineage_ui_contains_audit_track_and_database_lanes() -> None:
    source = _text("js/screens-dict.js")
    assert "data-dict-audit-track" in source
    assert "data-dict-lineage-lanes" in source
    assert "loadConceptLineage" in _text("js/api.js")


def test_visualization_typography_does_not_use_tiny_new_labels() -> None:
    owner_css = "\n".join(
        _text(path)
        for path in (
            "css/patient-tables.css",
            "css/patient-series.css",
            "css/cohort-charts.css",
            "css/crossdb.css",
            "css/deepdive.css",
        )
    )
    assert "pt-table-dashboard" in owner_css
    assert "dict-audit-track" in owner_css
    assert "xdb-three-scale" in owner_css

