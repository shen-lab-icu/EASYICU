"""Native fluid records are events, including duplicate minute coordinates."""

import json

import pandas as pd
import pytest

from easyicu.concept.fluid_administrations import fluid_sources, administration_events
from easyicu.resources import load_dictionary


@pytest.mark.parametrize("database", ["aumc", "miiv", "sic"])
def test_native_records_classes_modes_volumes_and_gates(database):
    sources = fluid_sources(load_dictionary(), database)
    spec = next(s for s in sources if s.fluid_class == "crystalloid")
    item = spec.source.ids[0]
    if database == "aumc":
        import re
        category = int(re.search(r"\(\?:([0-9]+)", spec.source.params["require_regex"]["ordercategoryid"])[1])
        raw = pd.DataFrame({"admissionid": 1, "itemid": item, "start": 7., "stop": 31.,
                            "fluidin": [100., 200., 300., 400., 500.], "iscontinuous": [1, 0, 1, 1, 1],
                            "isadditive": [0, 0, 1, 0, 0], "ordercategoryid": category,
                            "action": ["Nieuwe zak", "Nieuwe zak", "Nieuwe zak", "Flush", "Nieuwe zak"],
                            "ordercategory": ["Infuus - Crystalloid"] * 4 + ["Niet IV"]})
        origins = pd.DataFrame({"admissionid": [1], "admittedat": [0.]})
        expected = ["continuous", "unknown"]
    elif database == "miiv":
        origin = pd.Timestamp("2150-01-01")
        raw = pd.DataFrame({"stay_id": 1, "itemid": item, "starttime": origin + pd.Timedelta(minutes=7),
                            "endtime": origin + pd.Timedelta(minutes=31), "amount": [100., 200., 300., 400.],
                            "amountuom": "mL", "statusdescription": ["FinishedRunning"] * 2 + ["Rewritten", "FinishedRunning"],
                            "ordercategorydescription": ["Continuous IV", "Bolus", "Continuous IV", "Continuous IV"],
                            "ordercategoryname": ["02-Fluids (Crystalloids)"] * 3 + ["OR"]})
        origins = pd.DataFrame({"stay_id": [1], "intime": [origin]})
        expected = ["continuous", "bolus"]
    else:
        raw = pd.DataFrame({"CaseID": 1, "DrugID": item, "Offset": 1320.,
                            "OffsetDrugEnd": [2760., 1380., 2760., 1320.],
                            "Amount": [100., 200., -1., 400.], "IsSingleDose": [0, 1, 0, 0]})
        origins = pd.DataFrame({"CaseID": [1], "ICUOffset": [900.]})
        expected = ["continuous", "bolus"]
    out = administration_events(raw, spec, database=database, origins=origins)
    assert out.volume_ml.tolist() == [100., 200.]
    assert out.fluid_class.tolist() == ["crystalloid"] * 2
    assert out.start.tolist() == pytest.approx([7 / 60] * 2)
    assert out["mode"].tolist() == expected
    assert out.mode_evidence.notna().all()
    assert out.end.iloc[0] == pytest.approx(31 / 60)
    assert out.end.iloc[1] == pytest.approx(7 / 60 if database == "sic" else 31 / 60)


def test_accessor_uses_exact_fluid_io_sources_without_enteral_or_counter_events():
    dictionary = load_dictionary()
    for db in ("aumc", "miiv", "sic"):
        sources = fluid_sources(dictionary, db)
        assert {s.concept for s in sources} <= set(dictionary.keys())
        for spec in sources:
            assert spec.source in dictionary[spec.concept].sources[db]
        assert {s.fluid_class for s in sources} <= {
            "crystalloid", "colloid", "blood_rbc", "blood_ffp", "blood_platelet", "parenteral"}
    assert fluid_sources(dictionary, "hirid") == ()


@pytest.mark.parametrize("database", ["aumc", "sic", "hirid", "miiv"])
def test_explicit_native_v2_module_receipt_and_opt_in_isolation(tmp_path, database):
    from easyicu.api import extraction as api
    from easyicu.api.concepts import clear_global_loader
    from easyicu.synthetic_sources import generate_source

    assert "fluid_events" not in api.EXTRACT_MODULE_ORDER
    assert api.EXTRACT_MODULES["fluid_events"] == []  # Event properties are not scalar concepts.
    root = tmp_path / "source"
    if database == "miiv":
        from tests.support.source_mortality import source
        source(root, "miiv")
        icu = pd.read_parquet(root / "icustays.parquet").assign(los=1.)
        icu.to_parquet(root / "icustays.parquet", index=False)
        patients = pd.read_parquet(root / "patients.parquet").assign(dod=pd.NaT)
        patients.to_parquet(root / "patients.parquet", index=False)
        origin = pd.Timestamp("2020-01-01")
        # Identical administrations are separate records, even after publication.
        pd.DataFrame({"stay_id": [1, 1], "hadm_id": 11, "subject_id": 101,
                      "itemid": 220949, "starttime": origin + pd.Timedelta(minutes=7),
                      "endtime": origin + pd.Timedelta(minutes=31), "amount": 100., "amountuom": "mL",
                      "rate": 250., "rateuom": "mL/hour", "linkorderid": [1, 2],
                      "ordercategoryname": "02-Fluids (Crystalloids)",
                      "ordercategorydescription": "Continuous IV", "statusdescription": "FinishedRunning"}).to_parquet(root / "inputevents.parquet", index=False)
    else:
        generate_source(root, database, stays=3)
    out = tmp_path / "out"
    try:
        result = api.extract_database(database, data_path=root, output_dir=out,
                                      modules=["outcome", "fluid_events"], native_export_v2=True,
                                      resource_budget_mb=2048, verbose=False)
    finally:
        clear_global_loader()
    assert all(not m["errors"] for m in result["modules"].values())
    manifest = json.loads((out / "_manifest.json").read_text())
    entry = next(e for e in manifest["files"] if e["module"] == "fluid_events")
    assert entry["row_grain"] == "one_row_per_native_administration"
    assert entry["primary_key"] == ["stay_id", "event_index"]
    assert len(entry["event_receipt"]["source_declarations_sha256"]) == 64
    frame = pd.read_parquet(out / "fluid_events.parquet")
    sidecar = json.loads((out / manifest["column_metadata"]["file"]).read_text())
    binding = next(f for f in sidecar["files"] if f["module"] == "fluid_events")
    assert {t["column"] for t in binding["time_coordinates"]} == {"start", "end"}
    assert binding["columns"]["volume_ml"]["metadata"]["canonical_unit"] == "mL"
    assert binding["columns"]["mode"]["metadata"]["canonical_unit"] == "category"
    if database == "hirid":
        assert frame.empty
        assert entry["event_receipt"]["status"] == "NOT_AVAILABLE"
        assert "counter" in entry["event_receipt"]["reason"]
    else:
        assert entry["event_receipt"]["status"] == "AVAILABLE"
        assert not frame.empty
        assert not frame.duplicated(["stay_id", "event_index"]).any()
        if database == "miiv":
            assert len(frame) == 2
            assert frame.start.nunique() == 1
            assert frame.volume_ml.tolist() == [100., 100.]
            assert frame.event_index.tolist() == [0, 1]


@pytest.mark.parametrize("database", ["aumc", "miiv", "sic"])
def test_each_class_uses_the_fluid_io_amount_field(database):
    import re
    for spec in fluid_sources(load_dictionary(), database):
        source = spec.source
        if database == "aumc":
            category = int(re.search(r"\(\?:([0-9]+)", source.params["require_regex"]["ordercategoryid"])[1])
            raw = pd.DataFrame({"admissionid": [1], "itemid": source.ids[0], "start": [7.], "stop": [31.],
                                "fluidin": [123.], "dose": [9999.], "doseunit": "mg", "isadditive": [0],
                                "ordercategoryid": [category], "iscontinuous": [1]})
            origins = pd.DataFrame({"admissionid": [1], "admittedat": [0.]})
        elif database == "miiv":
            origin = pd.Timestamp("2150-01-01")
            raw = pd.DataFrame({"stay_id": [1], "itemid": source.ids[0], "starttime": [origin],
                                "endtime": [origin], "amount": [123.], "amountuom": "mL", "rate": [9999.],
                                "ordercategoryname": "01-Drips", "statusdescription": "FinishedRunning",
                                "ordercategorydescription": "Drug Push"})
            origins = pd.DataFrame({"stay_id": [1], "intime": [origin]})
        else:
            raw = pd.DataFrame({"CaseID": [1], "DrugID": source.ids[0], "Offset": [900.],
                                "OffsetDrugEnd": [960.], "Amount": [123.], "IsSingleDose": [1]})
            origins = pd.DataFrame({"CaseID": [1], "ICUOffset": [900.]})
        out = administration_events(raw, spec, database=database, origins=origins)
        assert out.volume_ml.tolist() == [123.]
        assert out.fluid_class.tolist() == [spec.fluid_class]


@pytest.mark.parametrize("label", [None, "unexpected coded category"])
def test_unknown_mode_is_not_inferred_from_duration_or_copied_to_evidence(label):
    spec = fluid_sources(load_dictionary(), "miiv")[0]
    origin = pd.Timestamp("2150-01-01")
    raw = pd.DataFrame({"stay_id": [1], "itemid": spec.source.ids[0], "starttime": [origin],
                        "endtime": [origin], "amount": [123.], "amountuom": "mL",
                        "ordercategoryname": "01-Drips", "statusdescription": "FinishedRunning",
                        "ordercategorydescription": [label]})
    out = administration_events(raw, spec, database="miiv",
                                origins=pd.DataFrame({"stay_id": [1], "intime": [origin]}))
    assert out["mode"].tolist() == ["unknown"]
    assert out.mode_evidence.tolist() == ["unrecognized_or_missing_mode"]


def test_sic_single_dose_does_not_require_a_usable_synthetic_end():
    spec = fluid_sources(load_dictionary(), "sic")[0]
    raw = pd.DataFrame({"CaseID": [1, 1], "DrugID": spec.source.ids[0], "Offset": [900., 900.],
                        "OffsetDrugEnd": [0., None], "Amount": [123., 456.], "IsSingleDose": 1})
    out = administration_events(raw, spec, database="sic",
                                origins=pd.DataFrame({"CaseID": [1], "ICUOffset": [900.]}))
    assert out.volume_ml.tolist() == [123., 456.]
    assert out.end.tolist() == [0., 0.]


def test_hirid_unavailable_events_alone_do_not_require_longitudinal_bounds(tmp_path):
    from easyicu.api import extraction as api
    from easyicu.api.concepts import clear_global_loader
    from easyicu.synthetic_sources import generate_source

    root = tmp_path / "source"
    generate_source(root, "hirid", stays=3)
    try:
        result = api.extract_database("hirid", data_path=root, output_dir=tmp_path / "out",
                                      modules=["fluid_events"], native_export_v2=True,
                                      resource_budget_mb=2048, verbose=False)
    finally:
        clear_global_loader()
    assert not result["modules"]["fluid_events"]["errors"]
    assert result["modules"]["fluid_events"]["event_receipt"]["status"] == "NOT_AVAILABLE"


def test_event_contract_rejects_duplicate_record_keys_and_unknown_codes():
    from easyicu.concept.native_events import NATIVE_EVENT_MODULES
    spec = NATIVE_EVENT_MODULES["fluid_events"]
    row = pd.DataFrame({"stay_id": [1], "event_index": [0], "start": [0.], "end": [0.],
                        "volume_ml": [100.], "fluid_class": ["crystalloid"], "mode": ["bolus"],
                        "mode_evidence": ["Bolus"]})
    with pytest.raises(ValueError, match="duplicate record identities"):
        spec.canonical(pd.concat([row, row], ignore_index=True))
    with pytest.raises(ValueError, match="unrecognized event property code"):
        spec.canonical(row.assign(fluid_class="enteral"))


def test_event_metadata_cannot_bind_missing_physical_properties():
    from easyicu.concept.export_metadata import build_export_file_metadata_binding
    from easyicu.concept.native_events import NATIVE_EVENT_MODULES

    with pytest.raises(ValueError, match="event schema mismatch"):
        build_export_file_metadata_binding(
            relative_path="fluid_events.parquet", module="fluid_events",
            frame=NATIVE_EVENT_MODULES["fluid_events"].empty().drop(columns="mode_evidence"),
            concept_ids=[], database="miiv", database_class_prefixes=(), dictionary=load_dictionary(),
        )


def test_event_identity_and_publication_bytes_do_not_depend_on_batch_size(tmp_path):
    from easyicu.api.event_extraction import run_event_extraction, publish_event_table
    from easyicu.api.extraction import _canonicalise_native_export_row_order
    from tests.support.mimic_synthetic_source import mimic_synthetic_source

    root = tmp_path / "source"
    root.mkdir()
    origin = pd.Timestamp("2150-01-01")
    raw = pd.DataFrame({"stay_id": [1, 1, 2, 2], "subject_id": [11, 11, 12, 12],
                        "hadm_id": [21, 21, 22, 22], "itemid": 220949,
                        "starttime": origin, "endtime": origin, "amount": 100., "amountuom": "mL",
                        "rate": 100., "rateuom": "mL/hour", "linkorderid": range(4),
                        "ordercategoryname": "03-IV Fluid Bolus", "ordercategorydescription": "Bolus",
                        "statusdescription": "FinishedRunning"})
    mimic_synthetic_source(root, "miiv", inputevents=raw)
    payloads = []
    for size in (1, 3):
        out = tmp_path / str(size)
        run_event_extraction(module="fluid_events", database="miiv", data_path=str(root),
                             patient_ids={"stay_id": [2, 1, 1]}, batch_size=size, output_dir=out)
        published = out / "published.parquet"
        audit = publish_event_table(source_parquet=out / "fluid_events.parquet", temporary_parquet=published,
                                    module="fluid_events", stay_time_upper_bounds={1: 48., 2: 48.})
        assert audit["rows"] == 4
        _canonicalise_native_export_row_order(published, module="fluid_events")
        frame = pd.read_parquet(published)
        assert frame.event_index.tolist() == [0, 1, 0, 1]
        payloads.append(published.read_bytes())
    assert payloads[0] == payloads[1]
