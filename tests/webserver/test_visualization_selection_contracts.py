from __future__ import annotations

from pathlib import Path

from easyicu.concept import catalog as cc
from easyicu.webserver.catalog import build_concept_lineage


ROOT = Path(__file__).resolve().parents[2]
STATIC = ROOT / "src" / "easyicu" / "webserver" / "static"


def _text(relative: str) -> str:
    return (STATIC / relative).read_text(encoding="utf-8")


def test_patient_visualization_selections_have_route_owned_markers() -> None:
    assert "data-pt-table-dashboard" in _text("js/screens-viz-patient-tables.js")
    assert "data-pt-module-pareto" in _text("js/screens-viz-patient-tables.js")
    assert "gallerySignals(lanes, 5)" in _text("js/screens-viz-patient-series.js")
    assert "buckets.forEach(bucket" in _text("js/screens-viz-patient-series.js")
    assert "data-patient-trajectory-summary" in _text("js/screens-viz-patient-series.js")
    assert "data-patient-distribution-latest" in _text("js/screens-viz-patient-overview.js")
    assert "data-patient-missing-record-scatter" in _text("js/screens-viz-patient-overview.js")


def test_cohort_visualization_selections_have_route_owned_markers() -> None:
    view = _text("js/screens-viz-cohort-view.js")
    route = _text("js/screens-viz-cohort.js")
    assert "data-cohort-table-one" in view
    assert "data-cohort-coverage-forest" in view
    assert "sofaMatrixGranularity: 'exact'" in route
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
    # The overlays extraction applies are part of the lineage: the community
    # sources' own mappings sit beside the six public databases.
    assert {"miiv", "mimic", "eicu", "aumc", "hirid", "sic"} <= set(lanes)
    assert set(lanes) <= set(cc.SUPPORTED_DB_KEYS)
    assert lanes["miiv"]["mappings"][0]["table"] == "labevents"
    assert lanes["aumc"]["mappings"][0]["callback"]
    assert lineage["inputs"] == [] and lineage["producer"] is None


def test_derived_and_module_variables_name_what_they_come_from() -> None:
    """A derived variable names its inputs; a module output names its module.

    The dictionary showed "no raw-source mapping declared" for SOFA-2 and every
    other derived variable, and failed for the SOFA-2 overlay ones.
    """
    sofa2 = build_concept_lineage("sofa2")
    assert sofa2 is not None and sofa2["lanes"] == []
    assert sofa2["inputs"] == ["sofa2_resp", "sofa2_coag", "sofa2_liver", "sofa2_cardio", "sofa2_cns", "sofa2_renal"]
    assert build_concept_lineage("sofa2_coag")["inputs"] == ["plt"]
    assert build_concept_lineage("pafi")["inputs"] == ["po2", "fio2"]
    # SOFA-2 overlay variables with raw sources keep their lanes.
    assert build_concept_lineage("adv_resp")["lanes"]
    aki = build_concept_lineage("aki_reference")
    assert aki is not None and aki["producer"] == cc.COMPOSITE_CONCEPT_OUTPUT_SOURCES["aki_reference"]
    assert build_concept_lineage("not_a_concept") is None


def test_a_variable_mapped_in_some_databases_names_how_the_others_compute_it() -> None:
    """Anion gap reads a table in AUMC but is computed from Na, Cl and HCO3 in MIMIC-IV."""
    lineage = build_concept_lineage("anion_gap")
    mapped = {lane["database"] for lane in lineage["lanes"]}
    derived = {entry["database"] for entry in lineage["derived_in"]}
    assert "aumc" in mapped and "miiv" in derived and not mapped & derived
    assert lineage["inputs"] == ["na", "cl", "bicar"]
    # A variable without declared inputs claims no derivation anywhere.
    lact = build_concept_lineage("lact")
    assert lact["inputs"] == [] and lact["derived_in"] == []


def test_source_map_groups_declared_rows_by_table_and_field() -> None:
    from easyicu.webserver.catalog_source_map import build_source_map

    source_map = build_source_map("miiv")
    assert source_map["scope"] == "declared_dictionary_lineage_not_observed_data"
    tables = {table["table"]: table for table in source_map["tables"]}
    labs = tables["labevents"]
    assert "lact" in labs["concepts"] and "glu" in labs["concepts"]
    roles = {row["field"]: row["roles"] for row in labs["used_fields"]}
    assert roles["itemid"] == ["selector_field"] and roles["valuenum"] == ["value_field"]
    assert set(roles) <= set(labs["columns"])
    lact = next(row for row in labs["rows"] if row["concept"] == "lact")
    assert lact["selector_field"] == "itemid" and lact["selector_values"]
    assert "" not in tables
    assert {"concept": "los_icu", "callback": "los_callback"} in source_map["functions"]
    derived = {row["concept"]: row["inputs"] for row in source_map["derived"]}
    assert derived["sofa2_coag"] == ["plt"]
    assert {row["database"] for row in source_map["databases"]} >= {"miiv", "eicu", "sic"}
    assert build_source_map("not_a_database") is None


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
    patient_series_css = _text("css/patient-series.css")
    patient_charts_js = _text("js/screens-viz-patient-charts.js")
    for tiny in ("font-size:9px", "font-size:10.5px", "font-size:11px"):
        assert tiny not in patient_series_css
    assert "fontSize: 9" not in patient_charts_js


def test_visualization_css_markers_stay_with_their_route_owners() -> None:
    patient = _text("css/patient.css") + _text("css/patient-series.css") + _text("css/patient-tables.css")
    cohort = _text("css/cohort.css") + _text("css/cohort-charts.css")
    crossdb = _text("css/crossdb.css")
    dictionary = _text("css/deepdive.css")

    assert "pt-distribution-latest" in patient
    assert "cohort-coverage-forest" not in patient
    assert "xdb-three-scale" not in patient
    assert "dict-audit-track" not in patient

    assert "cohort-coverage-forest" in cohort
    assert "pt-distribution-latest" not in cohort
    assert "xdb-three-scale" not in cohort

    assert "xdb-three-scale" in crossdb
    assert "pt-distribution-latest" not in crossdb
    assert "cohort-coverage-forest" not in crossdb

    assert "dict-audit-track" in dictionary
    assert "xdb-three-scale" not in dictionary
