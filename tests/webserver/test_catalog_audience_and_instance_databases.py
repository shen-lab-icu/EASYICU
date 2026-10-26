"""The dictionary exposes display audience and registered-instance support."""

from __future__ import annotations

import json
import subprocess
import sys

import pytest

from easyicu.concept import catalog as cc
from easyicu.concept import loader
from easyicu.webserver import catalog
from easyicu.webserver import catalog_metadata
from easyicu.webserver import sources


@pytest.fixture(autouse=True)
def _isolated_definitions():
    catalog_metadata.catalog_definitions.cache_clear()
    yield
    catalog_metadata.catalog_definitions.cache_clear()


@pytest.mark.parametrize(
    ("concept", "definition", "audience"),
    [
        ("screen_ascertainment", {"category": "outcome"}, "internal"),
        ("aki_reference_creatinine", {"category": "renal"}, "internal"),
        ("old_pulse", {"deprecated": True}, "internal"),
        ("legacy_pulse", {"alias_of": "hr"}, "internal"),
        ("legacy_signal", {"clinical_status": "deprecated"}, "internal"),
        ("working_series", {"derived_helper": True}, "internal"),
        ("pipeline_state", {"category": "bookkeeping"}, "internal"),
        ("screen_state", {"role": "receipt"}, "internal"),
        ("working_callback", {"callback_type": "baseline_helper"}, "internal"),
        ("candidate_state", {"callback": "screen_ascertainment"}, "internal"),
        ("legacy_therapy", {"callback": "delirium_tx_alias"}, "internal"),
        (
            "receipt_state",
            {"levels": ["unavailable", "confirmed", "not_detected", "proxy_only"]},
            "internal",
        ),
        ("working_input", {"callback": "kdigo_creatinine_input"}, "internal"),
        ("urine_input", {"callback": "kdigo_urine_input"}, "internal"),
        ("therapy_input", {"callback": "acute_rrt_input"}, "internal"),
        ("sample_observed", {"unit": "boolean"}, "internal"),
        ("sample_available", {}, "internal"),
        ("screen_receipt", {}, "internal"),
        ("sofa2_cns_proxy_sensitivity", {}, "internal"),
        ("creat_baseline_n_48h", {}, "internal"),
        ("creat_low_past_48hr", {}, "internal"),
        ("aki_source_native_profile", {}, "internal"),
        ("creatinine_evidence_status", {}, "internal"),
        ("urine_evidence_reason", {}, "internal"),
        ("kidney_observation_window_coverage", {}, "internal"),
        ("measurement_offset", {}, "internal"),
        ("raw_measurement_time", {}, "internal"),
        ("raw_measurement_timestamp", {}, "internal"),
        ("explicit_internal", {"audience": "internal"}, "internal"),
        ("explicit_receipt", {"audience": "clinical"}, "clinical"),
        ("hr", {"category": "vitals", "unit": "bpm"}, "clinical"),
        ("lact", {"category": "blood gas", "unit": "mmol/L"}, "clinical"),
        ("sofa2", {"category": "outcome", "callback": "sofa2_score"}, "clinical"),
        ("death", {"category": "outcome", "class_name": "lgl_cncpt"}, "clinical"),
        ("tidal_vol_observed", {"unit": ["mL", "L"]}, "clinical"),
        ("vent_start", {"unit": "datetime"}, "clinical"),
        ("aki_reference", {}, "clinical"),
        ("aki_stage_reference", {}, "clinical"),
        ("modern_signal", {"deprecated": False, "derived_helper": False}, "clinical"),
        ("pulse", {"aliases": ["heart rate"]}, "clinical"),
        ("severity_class", {"levels": ["none", "mild", "moderate", "severe"]}, "clinical"),
        (
            "sep3_sofa2",
            {"canonical_definition": False, "requires_explicit_opt_in": True},
            "clinical",
        ),
    ],
)
def test_catalog_marks_audience_without_removing_concepts(
    monkeypatch: pytest.MonkeyPatch, concept: str, definition: dict, audience: str
) -> None:
    monkeypatch.setattr(cc, "CONCEPT_DICTIONARY", {concept: ("Synthetic", "测试", "")})
    monkeypatch.setattr(cc, "CONCEPT_GROUPS_INTERNAL", {"synthetic": [concept]})
    monkeypatch.setattr(loader, "load_concept_dict_cached", lambda: {concept: definition})
    monkeypatch.setattr(sources, "load_registry", lambda: {"sources": [], "active_path": None})

    body = catalog.build_catalog()
    assert body["conceptCoverage"][concept]["audience"] == audience
    assert body["dict"][concept] == ["Synthetic", "测试", ""]
    assert body["totalConcepts"] == 1
    assert body["groupConcepts"] == {"synthetic": [concept]}
    assert definition == loader.load_concept_dict_cached()[concept]


def test_catalog_uses_registered_instance_databases(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(cc, "CONCEPT_DICTIONARY", {"local_signal": ("Signal", "信号", "")})
    monkeypatch.setattr(
        loader,
        "load_concept_dict_cached",
        lambda: {"local_signal": {"sources": {"eicu": [{"table": "synthetic"}]}}},
    )
    monkeypatch.setattr(
        sources,
        "load_registry",
        lambda: {
            "sources": [
                {"id": "synthetic_miiv", "database": "miiv", "ok": True},
                {"id": "synthetic_eicu", "database": "eicu", "ok": True},
            ],
            "active_path": None,
        },
    )

    body = catalog.build_catalog()
    assert body["instance_databases"] == ["eicu", "miiv"]
    assert body["conceptCoverage"]["local_signal"]["instance_databases_covered"] == ["eicu"]
    assert body["supportedDbs"] == list(cc.SUPPORTED_DB_KEYS)
    assert body["cov"] == cc.CONCEPT_DB_COVERAGE


def test_instance_support_is_distinct_from_the_legacy_audit_count(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(cc, "CONCEPT_DICTIONARY", {"hr": ("Pulse", "心率", "bpm")})
    monkeypatch.setattr(
        loader, "load_concept_dict_cached",
        lambda: {"hr": {"sources": {"eicu": [{"table": "synthetic"}]}}},
    )
    registry = {
        "sources": [
            {"database": "miiv", "ok": True},
            {"database": "eicu", "ok": True},
            {"database": "eicu", "ok": True},
            {"database": "hirid", "ok": False},
            {"database": None, "ok": True},
        ],
        "active_path": None,
    }
    calls = []

    def registered():
        calls.append(True)
        return registry

    monkeypatch.setattr(sources, "load_registry", registered)
    body = catalog.build_catalog()
    assert calls == [True]
    assert body["instance_databases"] == ["eicu", "miiv"]
    coverage = body["conceptCoverage"]["hr"]
    assert coverage["instance_databases_covered"] == ["eicu"]
    # The dictionary names the mapped database; the earlier audit's count stays beside it.
    assert (coverage["kind"], coverage["declared_databases"], coverage["basis"]) == (
        "mapped", ["eicu"], "concept_dictionary",
    )
    assert coverage["data_audit_databases"] == 6
    assert body["coverageSummary"]["supportedDatabases"] == 6
    assert body["activeExportCoverage"]["status"] == "no_active_source"

    # Registration changes are visible without restarting or reloading assets.
    registry["sources"] = []
    again = catalog.build_catalog()
    assert calls == [True, True]
    assert again["instance_databases"] == []
    assert again["conceptCoverage"]["hr"]["instance_databases_covered"] == []


def test_catalog_reads_sources_registered_through_the_registry_owner(
    tmp_path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(sources, "_CONFIG_PATH", tmp_path / "sources.json")
    monkeypatch.setattr(sources, "_autodiscovered_paths", lambda: [])
    monkeypatch.setattr(sources, "_configured_export_base", lambda: None)
    monkeypatch.setattr(cc, "CONCEPT_DICTIONARY", {"local_signal": ("Signal", "信号", "")})
    monkeypatch.setattr(
        loader, "load_concept_dict_cached",
        lambda: {"local_signal": {"sources": {"eicu": [{"table": "synthetic"}]}}},
    )
    for database in ("miiv", "eicu"):
        export = tmp_path / database
        export.mkdir()
        (export / "demographics.csv").write_text("stay_id,age\n1,50\n2,60\n")
        (export / "_manifest.json").write_text(json.dumps({
            "database": database,
            "format": "csv",
            "files": [{"file": "demographics.csv", "module": "demographics", "rows": 2}],
        }))
        sources.register_source(str(export))

    body = catalog.build_catalog()
    assert body["instance_databases"] == ["eicu", "miiv"]
    assert body["conceptCoverage"]["local_signal"]["instance_databases_covered"] == ["eicu"]


def test_first_request_reuses_assets_loaded_at_startup() -> None:
    result = subprocess.run(
        [sys.executable, "-c", """
from easyicu.concept import loader
from easyicu.webserver import catalog, catalog_metadata, sources

def no_asset_reads(*args, **kwargs):
    raise AssertionError('the first request must not open dictionary assets')

loader.load_concept_dict_cached = no_asset_reads
catalog_metadata.package_path = no_asset_reads
sources.load_registry = lambda: {'sources': [], 'active_path': None}
assert catalog.build_catalog()['conceptCoverage']['hr']['audience'] == 'clinical'
"""],
        capture_output=True, text=True, timeout=15,
    )
    assert result.returncode == 0, result.stdout + result.stderr


@pytest.mark.parametrize(
    ("concept", "definitions", "expected"),
    [
        ("signal", {"signal": {"sources": {"eicu": [{}], "miiv": []}}}, {"eicu"}),
        (
            "score",
            {"score": {"concepts": ["signal", "missing"]}, "signal": {"sources": {"eicu": [{}]}}},
            {"eicu"},
        ),
        (
            "score",
            {"score": {"depends_on": "signal"}, "signal": {"sources": {"miiv": [{}]}}},
            {"miiv"},
        ),
        (
            "old_signal",
            {"old_signal": {"alias_of": "signal"}, "signal": {"sources": {"eicu": [{}]}}},
            {"eicu"},
        ),
        ("missing", {}, set()),
        ("cycle", {"cycle": {"concepts": ["cycle"]}}, set()),
        ("aki_reference_status", {"kdigo_aki": {"sources": {"eicu": [{}]}}}, {"eicu"}),
        ("mort_28d", {}, {"miiv", "miiv_demo", "sic", "sic_demo"}),
        ("charlson", {}, {"eicu", "miiv", "mimic", "eicu_demo", "miiv_demo", "mimic_demo"}),
        ("culture_positive", {}, {"eicu", "miiv", "mimic", "eicu_demo", "miiv_demo", "mimic_demo"}),
        ("circ_failure", {"lact": {"sources": {"eicu": [{}]}}}, set()),
        (
            "circ_failure",
            {"lact": {"sources": {"eicu": [{}]}}, "map": {"sources": {"eicu": [{}]}}},
            {"eicu"},
        ),
    ],
)
def test_support_follows_declared_inputs_and_loader_constraints(
    concept: str, definitions: dict, expected: set[str]
) -> None:
    assert catalog_metadata.concept_databases(concept, definitions) == expected


def test_packaged_metadata_is_cached_and_response_metadata_is_fresh(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(sources, "load_registry", lambda: {"sources": [], "active_path": None})
    first = catalog.build_catalog()

    def no_asset_reads(*args, **kwargs):
        raise AssertionError("page loads must reuse packaged metadata")

    monkeypatch.setattr(loader, "load_concept_dict_cached", no_asset_reads)
    monkeypatch.setattr(catalog_metadata, "package_path", no_asset_reads)
    first["conceptCoverage"]["hr"]["audience"] = "internal"
    second = catalog.build_catalog()
    assert second["conceptCoverage"]["hr"]["audience"] == "clinical"
    assert second["dict"] == first["dict"]


def test_overlay_definitions_merge_sources_without_changing_the_base(
    tmp_path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    from contextlib import nullcontext

    base = {"signal": {"deprecated": True, "sources": {"eicu": [{"table": "base"}]}}}
    overlay = tmp_path / "overlay.json"
    overlay.write_text('{"signal": {"unit": "bpm", "sources": {"miiv": [{}]}}}')
    monkeypatch.setattr(loader, "load_concept_dict_cached", lambda: base)
    monkeypatch.setattr(catalog_metadata, "package_path", lambda _: nullcontext(overlay))

    definition = catalog_metadata.catalog_definitions()["signal"]
    assert definition["deprecated"] is True
    assert definition["unit"] == "bpm"
    assert set(definition["sources"]) == {"eicu", "miiv"}
    assert base == {"signal": {"deprecated": True, "sources": {"eicu": [{"table": "base"}]}}}
