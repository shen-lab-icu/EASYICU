"""Every publication path counts non-numeric source values and shares one gate (B-IMPLICIT-GATE, delta23v).

- A republication or a staging recovery passes each module's full counts
  through one path; a module without counts is ``audit_unknown``, never zero.
- The run-level count is a union keyed by stay (a private ledger, never
  published): disjoint readers add up, and a clean reader of more stays
  cannot hide an anomalous one from the classless refusal, which checks every
  module first. Published counts 1-9 read ``"<10"``.
- A callback's first parse of raw values counts (and, for a classless
  concept, refuses) text before it becomes NaN.
- The gate judges a concept by the run's effective (per-source) dictionary.
- The web export and the flat (non-native) grouped export run the same gate
  before they commit. Synthetic rows only.
"""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import pandas as pd
import pytest

from easyicu.api import extraction
from easyicu.concept.value_kind import ConceptValueKindUndeclared
from easyicu.io.data_quality import source_value_coercion_warnings
from easyicu.source_bounds import collect_bounds_drops, record_coercion_counts

pytestmark = pytest.mark.filterwarnings("ignore")

ROOT = Path(__file__).resolve().parents[2]


def _load_script(name: str, filename: str):
    spec = importlib.util.spec_from_file_location(name, ROOT / "scripts" / "releases" / filename)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _read(stays, bad_stays, *, concept="temp", table="t", bad_per_stay=1):
    """One module's counts for ``concept``: one value per stay, ``bad_per_stay`` text in ``bad_stays``."""
    frame = pd.DataFrame({
        "stay": list(stays),
        "values": [1 + (bad_per_stay if stay in bad_stays else 0) for stay in stays],
        "bad": [bad_per_stay if stay in bad_stays else 0 for stay in stays],
    })
    with collect_bounds_drops() as drops:
        record_coercion_counts(
            database="unit", concept=concept, table=table, stays=list(stays), frame=frame,
            id_column="stay", values_column="values", non_numeric_column="bad",
        )
    return drops


def _definition(name, *, class_name=None, minimum=None, maximum=None):
    return SimpleNamespace(name=name, class_name=class_name, minimum=minimum, maximum=maximum)


_CLASSLESS = {"temp": _definition("temp")}
_DECLARED = {"temp": _definition("temp", class_name="num_cncpt")}


# -- D23V-ROOT-UNION ---------------------------------------------------------


def test_disjoint_readers_add_up_in_the_run_level_count():
    a = _read(range(1, 21), set(range(1, 21)))
    b = _read(range(21, 41), set(range(21, 41)))
    [row] = source_value_coercion_warnings({"a": a, "b": b})
    assert (row["stays_with_non_numeric"], row["stays_with_values"]) == (40, 40)
    assert row["non_numeric_values"] == 40 and row["union"] == "exact"
    fields = extraction.source_value_publication(
        {"a": {"source_bounds": a}, "b": {"source_bounds": b}}, ["a", "b"],
        database="unit", dictionary=_DECLARED,
    )
    assert fields["source_values_not_numeric"]["temp"]["t"]["non_numeric_values"] == 40


def test_the_same_stays_read_twice_count_once():
    a = _read(range(1, 21), set(range(1, 21)))
    [row] = source_value_coercion_warnings({"a": a, "b": a, "c": a})
    assert (row["non_numeric_values"], row["stays_with_non_numeric"]) == (20, 20)


def test_a_clean_reader_of_more_stays_cannot_hide_an_anomalous_one():
    clean = _read(range(1, 31), set())
    anomalous = _read(range(31, 51), set(range(31, 51)))
    drops = {"clean": clean, "anomalous": anomalous}
    [row] = source_value_coercion_warnings(drops)
    assert (row["stays_with_non_numeric"], row["stays_with_values"]) == (20, 50)
    with pytest.raises(ConceptValueKindUndeclared, match="'temp'.*20 source value.*anomalous"):
        extraction._refuse_classless_non_numeric(drops, database="unit", dictionary=_CLASSLESS)
    # Whatever the module names' order.
    with pytest.raises(ConceptValueKindUndeclared):
        extraction._refuse_classless_non_numeric(
            {"z_clean": clean, "a_anomalous": anomalous}, database="unit", dictionary=_CLASSLESS
        )


def test_published_counts_carry_no_ledger_and_small_counts_read_below_ten():
    few = _read(range(1, 30), {3, 4, 5})
    fields = extraction.source_value_publication(
        {"m": {"source_bounds": few}}, ["m"], database="unit", dictionary=_DECLARED,
    )
    published = json.dumps(fields)
    assert "_stays" not in published and '"seen"' not in published
    counts = fields["source_bounds_drops"]["m"]["temp"]["t"]
    assert counts["non_numeric_values"] == "<10" and counts["stays_with_non_numeric"] == "<10"
    assert counts["stays_with_values"] == 29
    union = fields["source_values_not_numeric"]["temp"]["t"]
    assert union["non_numeric_values"] == "<10"


# -- D23V-REFRESH-RECEIPT -----------------------------------------------------


def _module_manifests(root: Path, modules, counts):
    root.mkdir(parents=True, exist_ok=True)
    for module in modules:
        manifest = {"module": module, "errors": [], "elapsed_sec": 1.0}
        if module in counts:
            manifest["source_bounds"] = counts[module]
        (root / f"{module}.manifest.json").write_text(json.dumps(manifest), encoding="utf-8")


_PUBLISHED = {"temp": {"t": {
    "dropped_values": 0, "stays_all_out_of_bounds": 0, "stays_read": 0,
    "non_numeric_values": 25, "stays_with_non_numeric": 12, "stays_with_values": 40,
}}}


def test_a_republication_passes_every_module_count_and_marks_old_receipts_unknown(tmp_path, monkeypatch):
    republisher = _load_script("full6_republisher_counts", "EX-A02_republish_full6_candidate.py")
    modules = list(republisher.MODULES)
    root = tmp_path / "eicu"
    _module_manifests(root, modules, {modules[0]: _PUBLISHED})
    receipts = {
        field: {name: 1.0 for name in modules}
        for field in ("module_timings_seconds", "module_peak_rss_mb", "module_peak_working_set_mb")
    }
    (root / "_manifest.json").write_text(json.dumps({**receipts, "column_metadata": {}}), encoding="utf-8")
    captured = {}

    def fake_publish(**kwargs):
        captured.update(kwargs)
        (root / "_manifest.json").write_text(json.dumps(receipts), encoding="utf-8")

    monkeypatch.setattr(republisher.extraction_api, "_publish_native_export_v2", fake_publish)
    republisher._republish_database(root, database="eicu", source_receipt={}, publication_commit="2" * 40)
    results = captured["result"]["modules"]
    assert results[modules[0]]["source_bounds"] == _PUBLISHED
    assert "source_bounds" not in results[modules[1]]
    fields = extraction.source_value_publication(
        results, modules, database="eicu", dictionary=_DECLARED,
    )
    assert fields["source_bounds_audit"][modules[1]] == "audit_unknown"
    assert modules[0] not in fields["source_bounds_audit"]
    assert fields["source_values_not_numeric"]["temp"]["t"]["non_numeric_values"] == 25
    assert fields["qc_warnings"]["source_value_counts_audit_unknown"]


def test_a_staging_recovery_passes_every_module_count(tmp_path, monkeypatch):
    refresher = _load_script("selected_module_refresh_counts", "EX-A03_refresh_selected_modules.py")
    root = tmp_path / "staging"
    _module_manifests(root, ["outcome", "vitals"], {"vitals": _PUBLISHED})
    captured = {}
    monkeypatch.setattr(refresher, "_publish_native_export_v2", lambda **kwargs: captured.update(kwargs))
    refresher._recover_native_staging_publication(
        database="eicu", data_path="unused", staging_root=root,
        modules=["outcome", "vitals"], resource_budget_mb=8192.0,
    )
    results = captured["result"]["modules"]
    assert results["vitals"]["source_bounds"] == _PUBLISHED
    assert "source_bounds" not in results["outcome"]
    with pytest.raises(ConceptValueKindUndeclared):
        extraction.source_value_publication(
            results, ["outcome", "vitals"], database="eicu", dictionary=_CLASSLESS,
        )


# -- D23V-GATE-CONTEXT --------------------------------------------------------


def test_the_gate_judges_by_the_runs_effective_dictionary():
    drops = {"m": _read(range(1, 21), set(range(1, 21)), concept="cvp_raw")}
    base = extraction.load_dictionary(include_sofa2=True)
    extraction._refuse_classless_non_numeric(drops, database="sic", dictionary=base)  # bounded there
    effective = extraction.publication_dictionary("sic")  # SIC overlay: no bounds, no class
    with pytest.raises(ConceptValueKindUndeclared, match="'cvp_raw'"):
        extraction._refuse_classless_non_numeric(drops, database="sic", dictionary=effective)


def test_the_native_publisher_hands_its_own_dictionary_to_the_gate():
    import inspect

    source = inspect.getsource(extraction._publish_native_export_v2)
    assert "dictionary = publication_dictionary(normalized_database)" in source
    assert "database=database, dictionary=dictionary" in source


# -- D23V-RAW-EARLY -----------------------------------------------------------


def _callback(expr, values, *, definition):
    from easyicu.concept.callback_apply import _apply_callback
    from easyicu.concept.schema import ConceptSource

    frame = pd.DataFrame({
        "patientunitstayid": list(range(1, len(values) + 1)),
        "observationoffset": 60.0,
        "temperature": pd.Series(values, dtype=object),
    })
    source = ConceptSource(table="vitalperiodic", value_var="temperature", callback=expr)
    resolver = SimpleNamespace(dictionary={"temp_unit": definition})
    data_source = SimpleNamespace(config=SimpleNamespace(name="eicu"))
    with collect_bounds_drops() as drops:
        out = _apply_callback(frame, source, "temp_unit", resolver=resolver, data_source=data_source)
    return out, drops


_TEXT20 = ["see note"] * 20 + ["37.0", "98.6", "36.5"]


def test_the_eicu_temperature_callback_counts_text_before_it_becomes_nan():
    out, drops = _callback(
        "transform_fun(fahr_between(85, 110))", _TEXT20,
        definition=_definition("temp_unit", minimum=32.0, maximum=42.0),
    )
    assert out["temperature"].isna().sum() == 20
    [counts] = drops["temp_unit"].values()
    assert (counts["non_numeric_values"], counts["stays_with_non_numeric"]) == (20, 20)


def test_a_classless_concept_refuses_text_at_the_callbacks_first_parse():
    with pytest.raises(ConceptValueKindUndeclared, match="'temp_unit'.*vitalperiodic"):
        _callback("transform_fun(fahr_between(85, 110))", _TEXT20, definition=_definition("temp_unit"))


def test_a_callback_without_the_helper_is_counted_at_its_boundary(monkeypatch):
    from easyicu.concept import callback_apply

    def opaque(frame, source, concept_name, *args, **kwargs):
        frame = frame.copy()
        frame["temperature"] = pd.to_numeric(frame["temperature"], errors="coerce")
        return frame

    monkeypatch.setattr(callback_apply, "_apply_callback_body", opaque)
    _out, drops = _callback(
        "transform_fun(opaque)", _TEXT20, definition=_definition("temp_unit", class_name="num_cncpt"),
    )
    [counts] = drops["temp_unit"].values()
    assert counts["non_numeric_values"] == 20
    with pytest.raises(ConceptValueKindUndeclared):
        _callback("transform_fun(opaque)", _TEXT20, definition=_definition("temp_unit"))


def test_a_declared_text_concept_is_a_deliberate_parse():
    _out, drops = _callback(
        "transform_fun(fahr_between(85, 110))", _TEXT20,
        definition=_definition("temp_unit", class_name="chr_cncpt"),
    )
    assert drops == {}


# -- D23V-WEB-GATE ------------------------------------------------------------


def _inject_unlisted(real):
    """Totals plus a non-numeric count for a concept no dictionary declares."""

    def totals(records):
        out = dict(real(records))
        out["x_undeclared"] = {"t": {
            "dropped_values": 0, "stays_all_out_of_bounds": 0, "stays_read": 0,
            "non_numeric_values": 3, "stays_with_non_numeric": 1, "stays_with_values": 1,
        }}
        return out

    return totals


def test_a_flat_grouped_export_runs_the_gate_before_it_commits(tmp_path, monkeypatch):
    from easyicu.api.concepts import clear_global_loader
    from easyicu.synthetic_sources import generate_source
    from tests.support.inprocess_extraction import inprocess_mp_context

    root = tmp_path / "sic"
    generate_source(root, "sic", stays=6)
    monkeypatch.setattr(extraction, "bounds_totals", _inject_unlisted(extraction.bounds_totals))
    monkeypatch.setattr(extraction, "_get_extraction_mp_context", inprocess_mp_context)
    out = tmp_path / "flat"
    try:
        with pytest.raises(ConceptValueKindUndeclared, match="'x_undeclared'.*<10 source value"):
            extraction.extract_database(
                "sic", data_path=str(root), output_dir=str(out), modules=["vitals"],
                native_export_v2=False, verbose=False,
            )
    finally:
        clear_global_loader()
    assert not list(out.glob("*.manifest.json"))
    assert not (out / "_manifest.json").exists()


def test_the_web_export_runs_the_gate_before_its_root_manifest(tmp_path, monkeypatch):
    import easyicu.source_bounds as source_bounds
    from easyicu.webserver import dataio
    from easyicu.webserver.demo_source_contracts import SOURCES
    from tests.support.windows_laptop_demo import isolated_demo_paths, synthetic_demo_download

    class Job:
        def emit(self, _payload):
            pass

    source = next(s for s in SOURCES if s.database.removesuffix("_demo") == "eicu")
    paths = isolated_demo_paths(tmp_path, source)
    synthetic_demo_download(source, paths, Job())
    assert dataio.make_convert_runner(str(paths.raw), "eicu")(Job())["failed"] == 0
    monkeypatch.setattr(source_bounds, "bounds_totals", _inject_unlisted(source_bounds.bounds_totals))
    runner = dataio.make_export_runner(
        str(paths.raw), "eicu", modules=["vitals"], out_dir=str(paths.export),
        export_format="parquet", include_feature_definitions=False,
    )
    with pytest.raises(ConceptValueKindUndeclared, match="'x_undeclared'"):
        runner(Job())
    assert not (paths.export / "_manifest.json").exists()
