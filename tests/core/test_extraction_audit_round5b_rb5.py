"""Round-5b boolean evidence regressions; every record is synthetic."""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from easyicu import resources
from easyicu.concept.callback_apply import _apply_callback
from easyicu.concept.schema import ConceptSource

# Resolve beside the imported package so the copied test also checks the base.
DATA = Path(resources.__file__).parent / "data"
DICTIONARIES = ("concept-dict.json", "sofa2-dict.json")
DEFAULTS = {
    "inputevents": "rate", "inputevents_mv": "rate", "inputevents_cv": "rate",
    "chartevents": "valuenum", "outputevents": "value", "drugitems": "dose",
    "medication": "Amount", "infusiondrug": "drugrate",
    "intakeoutput": "cellvaluenumeric", "respiratorycharting": "respchartvalue",
    "listitems": "value",
}


def _dictionary(filename):
    return json.loads((DATA / filename).read_text())


def _sources(filename, concept, db, tables):
    definition = _dictionary(filename)[concept]
    assert definition["aggregate"] == "max"
    sources = definition["sources"].get(db, definition["sources"].get(db.removesuffix("_demo"), []))
    return [s for s in sources if s["table"] in tables]


def _apply(spec, concept, frame, projected, **kwargs):
    frame = frame.copy()
    value = spec.get("val_var", spec.get("value_var", DEFAULTS[spec["table"]]))
    if projected:
        frame = frame.rename(columns={value: concept})
    out = _apply_callback(frame, ConceptSource.from_mapping(spec), concept, **kwargs)
    return out if concept in out else out.rename(columns={value: concept})


def _true_rows(out, concept):
    return set(out.loc[out[concept].eq(True).fillna(False), "row"])  # noqa: E712


@pytest.mark.parametrize("projected", [False, True])
def test_rb_01_other_vaso_uses_positive_nonrewritten_intervals(projected):
    spec = _sources("sofa2-dict.json", "other_vaso", "miiv", {"inputevents"})[0]
    frame = pd.DataFrame({
        "stay_id": 1, "itemid": 221749, "starttime": [0., 4., 8., 12., 16., 20.],
        "endtime": [2., 6., 10., 14., 18., 22.], "rate": [1., 0., None, -1., 1., None],
        "amount": [1., 0., None, 0., 1., 2.],
        "statusdescription": ["FinishedRunning"] * 4 + ["Rewritten", "FinishedRunning"],
    })
    out = _apply(spec, "other_vaso", frame, projected)
    active = out.loc[out.other_vaso.eq(True), "starttime"]  # noqa: E712
    assert set(active) == {0., 1., 20.}
    assert spec["callback"] == "mimic_other_vaso"
    assert {"amount", "statusdescription"} <= set(spec["extra_vars"])
    assert spec["val_var"] == "rate"


@pytest.mark.parametrize("filename", DICTIONARIES)
@pytest.mark.parametrize("db", ["miiv", "mimic", "mimic_demo"])
@pytest.mark.parametrize("projected", [False, True])
def test_rb_02_rrt_needs_positive_flow_or_finite_nonzero_pressure(filename, db, projected):
    specs = _sources(filename, "rrt", db, {"chartevents", "outputevents"})
    # 582 is a separately regex-selected procedure record, not a numeric channel.
    specs = [s for s in specs if 582 not in s["ids"]]
    assert specs
    pressures = {224149, 224150, 224151, 224152, 173, 192}
    texts = {225976: "Replacement fluid", 225977: "Dialysate fluid", 227290: "CVVH",
             150: "Dialysis machine", 152: "CVVHD"}
    for spec in specs:
        records = []
        expected = set()
        for item in spec["ids"]:
            for value in [None, "", "0", "0.0", "-2", "4", "inf", "-inf", "nan"]:
                row = len(records)
                records.append({"row": row, "itemid": item, "value": value,
                                "valuenum": pd.to_numeric(value, errors="coerce")})
                if value == "4" or (item in pressures and value == "-2"):
                    expected.add(row)
            if item in texts:
                row = len(records)
                records.append({"row": row, "itemid": item, "value": texts[item], "valuenum": np.nan})
                expected.add(row)
        out = _apply(spec, "rrt", pd.DataFrame(records), projected)
        assert _true_rows(out, "rrt") == expected, (filename, db, spec["ids"])
        assert spec["val_var"] == "value"
        assert _apply(spec, "rrt", pd.DataFrame(), projected).empty


@pytest.mark.parametrize("db", ["miiv", "mimic", "mimic_demo", "eicu", "eicu_demo"])
@pytest.mark.parametrize("projected", [False, True])
def test_rb_03_advanced_respiratory_settings_require_positive_values(db, projected):
    specs = _sources("sofa2-dict.json", "adv_resp", db, {"chartevents", "respiratorycharting"})
    specs = [s for s in specs if s.get("callback") == "transform_fun(set_val(TRUE))"
             or s.get("callback") in {"mimic_support_evidence", "eicu_positive_value"}]
    numeric_items = {223848, 223849, 229314, 220339, 224700, 227577, 227578,
                     227579, 227580, 227583, 225949, 505, 506, 686}
    modes = {223848: "Servo", 223849: "Assist Control", 229314: "PC-AC",
             227577: "S/T", 227578: "Full face", 227583: "CPAP", 225949: "NIV mask"}
    found = False
    for spec in specs:
        items = set(spec.get("ids", [])) & numeric_items
        if db.startswith("eicu") and spec.get("regex", "").startswith("CPAP"):
            items = {"PEEP"}
        if not items:
            continue
        found = True
        for item in items:
            values = [None, "", "0", "0.0", "-2", "5", "inf"]
            frame = pd.DataFrame({"row": range(7), "value": values,
                                  "valuenum": pd.to_numeric(values, errors="coerce"),
                                  "respchartvalue": values, "respchartvaluelabel": "PEEP",
                                  "itemid": item})
            out = _apply(spec, "adv_resp", frame, projected)
            assert _true_rows(out, "adv_resp") == {5}, (db, item)
        for item in set(spec.get("ids", [])) & set(modes):
            frame = pd.DataFrame({"row": range(4), "itemid": item,
                                  "value": [modes[item], "Standby", "off", None],
                                  "valuenum": np.nan})
            assert _true_rows(_apply(spec, "adv_resp", frame, projected), "adv_resp") == {0}
    assert found
    if db in {"mimic", "mimic_demo"}:
        definition = _dictionary("concept-dict.json")["vent_start"]
        spec = definition["sources"][db][0]
        records = [{"row": i * 4 + j, "itemid": item, "value": value}
                   for i, item in enumerate(spec["ids"])
                   for j, value in enumerate(["0", None, "-2", "5"])]
        row = len(records)
        records.append({"row": row, "itemid": 223849, "value": "Assist Control"})
        out = _apply(spec, "vent_start", pd.DataFrame(records), projected)
        assert _true_rows(out, "vent_start") == {r["row"] for r in records if r["value"] in {"5", "Assist Control"}}
        assert definition["aggregate"] == "max"
        # A failed numeric reading is not a documented ventilator mode.
        for item in (444, 445, 448, 450, 501, 505, 506, 535, 543, 681, 682, 683, 686,
                     220339, 224684, 224685, 224687, 224695, 224696, 224697, 224700, 224701):
            frame = pd.DataFrame({"row": [0], "itemid": [item], "value": ["unmeasured"]})
            assert not _true_rows(_apply(spec, "vent_start", frame, projected), "vent_start")


@pytest.mark.parametrize("filename", DICTIONARIES)
@pytest.mark.parametrize("db", ["miiv", "mimic", "mimic_demo"])
@pytest.mark.parametrize("projected", [False, True])
def test_rb_04_mechanical_circulatory_numeric_channels_reject_zero(filename, db, projected):
    for spec in _sources(filename, "mech_circ_support", db, {"chartevents"}):
        for item in spec["ids"]:
            values = [None, "0", "0.0", "-1", "3", "Off", "Not applicable", "inf"]
            frame = pd.DataFrame({"row": range(8), "itemid": item, "value": values,
                                  "valuenum": pd.to_numeric(values, errors="coerce")})
            out = _apply(spec, "mech_circ_support", frame, projected)
            assert _true_rows(out, "mech_circ_support") == {4}, (filename, db, item)
        if 227980 in spec["ids"] or 225 in spec["ids"]:
            item = 227980 if 227980 in spec["ids"] else 225
            frame = pd.DataFrame({"row": [0], "itemid": [item], "value": ["1:1"], "valuenum": [np.nan]})
            assert _true_rows(_apply(spec, "mech_circ_support", frame, projected), "mech_circ_support") == {0}


def _exposure_specs():
    tables = {"miiv": {"inputevents"}, "mimic": {"inputevents_mv", "inputevents_cv"},
              "mimic_demo": {"inputevents_mv", "inputevents_cv"},
              "aumc": {"drugitems"}, "sic": {"medication"}}
    for filename in DICTIONARIES:
        for concept, definition in _dictionary(filename).items():
            if definition.get("class_name") != "lgl_cncpt" or concept == "other_vaso":
                continue
            for db, sources in definition.get("sources", {}).items():
                for i, spec in enumerate(sources):
                    if spec.get("callback", "").startswith("transform_fun(comp_na("):
                        continue  # Existing explicit positive comparisons are unchanged.
                    if spec.get("table") in tables.get(db, set()):
                        assert definition["aggregate"] == "max"
                        yield pytest.param(concept, spec, id=f"{filename}:{concept}:{db}:{i}")


@pytest.mark.parametrize("concept,spec", list(_exposure_specs()))
@pytest.mark.parametrize("projected", [False, True])
def test_rb_05_every_drug_and_blood_product_needs_a_positive_dose(concept, spec, projected):
    if spec["table"] in {"inputevents", "inputevents_mv", "inputevents_cv"}:
        item = spec["ids"][0] if isinstance(spec["ids"], list) else spec["ids"]
        frame = pd.DataFrame({"row": range(7), "amount": [0., None, -1., 2., None, 0., 2.],
                              "rate": [0., None, -1., None, 2., 0., 2.],
                              "itemid": item,
                              "statusdescription": ["FinishedRunning"] * 6 + ["Rewritten"]})
        expected = {3, 4, 6} if spec["table"] == "inputevents_cv" else {3, 4}
        # Exercise the existing antibiotic label whitelist, as well as the dose rule.
        ds = SimpleNamespace(load_table=lambda *a, **k: pd.DataFrame({
            "itemid": [item], "label": ["Vancomycin"]}))
        out = _apply(spec, concept, frame, projected, data_source=ds)
        assert {"amount", "rate"} <= {spec.get("val_var"), *spec.get("extra_vars", [])}
        if spec["table"] != "inputevents_cv":
            assert "statusdescription" in spec["extra_vars"]
            assert spec["status_var"] == "statusdescription"
            assert "Rewritten" in spec["excluded_statuses"]
    else:
        column = DEFAULTS[spec["table"]]
        frame = pd.DataFrame({"row": range(5), column: [0., None, -1., 2., 0.], "IsSingleDose": 1})
        for extra in spec.get("extra_vars", []):
            if extra not in frame:
                frame[extra] = "Injecties spuitpomp"
        expected = {3}
        out = _apply(spec, concept, frame, projected)
    assert _true_rows(out, concept) == expected


@pytest.mark.parametrize("concept", ["abx", "ketamine", "mannitol", "morphine", "rocuronium", "vecuronium", "acetazolamide"])
@pytest.mark.parametrize("db", ["eicu", "eicu_demo"])
@pytest.mark.parametrize("projected", [False, True])
def test_rb_06_name_matched_infusions_read_positive_drugrate(concept, db, projected):
    specs = _sources("concept-dict.json", concept, db, {"infusiondrug"})
    assert specs
    for spec in specs:
        frame = pd.DataFrame({"row": range(6), "drugname": concept,
                              "drugrate": ["0", None, "paused", "-1", "2", "inf"]})
        assert _true_rows(_apply(spec, concept, frame, projected), concept) == {4}
        assert spec["val_var"] == "drugrate"
        assert spec["callback"] == "eicu_positive_infusion"


@pytest.mark.parametrize("filename", DICTIONARIES)
@pytest.mark.parametrize("db", ["eicu", "eicu_demo"])
@pytest.mark.parametrize("projected", [False, True])
def test_rb_07_intake_output_exposure_requires_positive_volume(filename, db, projected):
    for concept in ["ffp", "packed_rbc", "platelets", "rrt"]:
        if concept not in _dictionary(filename):
            continue
        for spec in _sources(filename, concept, db, {"intakeoutput"}):
            frame = pd.DataFrame({"row": range(6), "cellvaluenumeric": [0., None, -2., 5., np.inf, -np.inf]})
            assert _true_rows(_apply(spec, concept, frame, projected), concept) == {3}
            assert spec["val_var"] == "cellvaluenumeric"


@pytest.mark.parametrize("filename", DICTIONARIES)
@pytest.mark.parametrize("concept,item,positive", [("ecmo", 20662, "VA ECMO"), ("mech_circ_support", 6658, "Ja")])
@pytest.mark.parametrize("projected", [False, True])
def test_rb_08_categorical_support_excludes_explicit_negative_and_empty_values(filename, concept, item, positive, projected):
    spec = _sources(filename, concept, "aumc", {"listitems"})[0]
    assert spec["ids"] == [item]
    values = [positive, "Nee", " no ", "FALSE", "0", None, "", "  "]
    frame = pd.DataFrame({"row": range(8), "itemid": item, "value": values})
    assert _true_rows(_apply(spec, concept, frame, projected), concept) == {0}
    assert spec["val_var"] == "value"


PATH_CASES = [
    ("RB-01", "miiv", "other_vaso", "inputevents", 221749),
    ("RB-02", "miiv", "rrt", "chartevents", 224144),
    ("RB-02", "mimic", "rrt", "chartevents", 173),
    ("RB-02", "mimic", "rrt", "outputevents", 40286),
    ("RB-03", "miiv", "adv_resp", "chartevents", 220339),
    ("RB-03", "mimic", "adv_resp", "chartevents", 505),
    ("RB-03", "eicu", "adv_resp", "respiratorycharting", "PEEP"),
    ("RB-04", "miiv", "mech_circ_support", "chartevents", 224309),
    ("RB-04", "mimic", "mech_circ_support", "chartevents", 225),
    ("RB-05", "miiv", "albumin_iv", "inputevents", 220862),
    ("RB-05", "mimic", "albumin_iv", "inputevents_mv", 220862),
    ("RB-05", "mimic", "albumin_iv", "inputevents_cv", 30008),
    ("RB-05", "aumc", "albumin_iv", "drugitems", 8933),
    ("RB-05", "sic", "albumin_iv", "medication", 2040),
    ("RB-06", "eicu", "ketamine", "infusiondrug", "ketamine"),
    ("RB-07", "eicu", "ffp", "intakeoutput", "FFP"),
    ("RB-08", "aumc", "ecmo", "listitems", 20662),
    ("RB-08", "aumc", "mech_circ_support", "listitems", 6658),
]


def _disk_source(root, database, concept, table, item, bucketed):
    """Six invented stays; only columns present in the real table schemas."""
    from easyicu.config import load_src_cfg
    from easyicu.datasource import ICUDataSource

    config = next(c for c in json.loads((DATA / "data-sources.json").read_text()) if c["name"] == database)
    ids = list(range(1, 7))
    origin = pd.Timestamp("2140-01-01 00:17")
    identifier = {"miiv": "stay_id", "mimic": "icustay_id", "eicu": "patientunitstayid",
                  "aumc": "admissionid", "sic": "CaseID"}[database]
    if database in {"miiv", "mimic"}:
        stays_table = "icustays"
        stays = pd.DataFrame({identifier: ids, "subject_id": ids, "hadm_id": ids,
                              "intime": origin, "outtime": origin + pd.Timedelta(days=2), "los": 2.})
    elif database == "eicu":
        stays_table = "patient"
        stays = pd.DataFrame({identifier: ids, "patienthealthsystemstayid": ids,
                              "unitdischargeoffset": 2880,
                              "hospitaladmitoffset": 0, "hospitaldischargeoffset": 2880})
    elif database == "aumc":
        stays_table = "admissions"
        stays = pd.DataFrame({identifier: ids, "patientid": ids,
                              "admittedat": 0, "dischargedat": 172800000})
    else:
        stays_table = "cases"
        stays = pd.DataFrame({identifier: ids, "PatientID": ids, "ICUOffset": 21600,
                              "TimeOfStay": 194400, "HospitalUnit": 3})
    stays.to_parquet(root / f"{stays_table}.parquet", index=False)
    filename = "sofa2-dict.json" if concept in {"other_vaso", "adv_resp"} else "concept-dict.json"
    spec = next(s for s in _dictionary(filename)[concept]["sources"][database]
                if s["table"] == table and ("ids" not in s or item in s["ids"]))
    defaults = config["tables"][table]["defaults"]
    value = defaults["val_var"]
    clock = spec.get("index_var", defaults["index_var"])
    values = [0., None, -2., 4., 4., 4.]
    expected = {4, 5, 6}
    rows = pd.DataFrame({identifier: ids, spec["sub_var"]: item, value: values})
    if database in {"miiv", "mimic"}:
        rows["subject_id"] = ids
        rows["hadm_id"] = ids
        rows[clock] = origin + pd.Timedelta(hours=1.25)
        rows["valueuom" if table in {"chartevents", "outputevents"} else "rateuom"] = ""
        if table == "chartevents":
            rows["value"] = pd.Series(values).astype("string")
            rows["valuenum"] = values
            if item == 173:
                expected.add(3)  # Negative finite circuit pressure is support.
        if table.startswith("inputevents"):
            rows["amountuom"] = "mg"
            rows["linkorderid"] = ids
            rows["rate"] = [0., None, -2., 4., None, 4.]
            rows["amount"] = [0., None, -2., None, 4., 4.]
            if table != "inputevents_cv":
                rows["ordercategoryname"] = "Continuous Med"
                rows["endtime"] = origin + pd.Timedelta(hours=2.25)
                rows["statusdescription"] = ["FinishedRunning"] * 5 + ["Rewritten"]
                expected.remove(6)
    else:
        rows[clock] = {"eicu": 75., "aumc": 4500000., "sic": 26100.}[database]
        if table == "listitems":
            rows[value] = ["Nee", None, "", "VA ECMO" if concept == "ecmo" else "Ja", "no", "0"]
            expected = {4}
        elif table == "drugitems":
            rows["doseunit"] = "mg"
        elif table in {"infusiondrug", "respiratorycharting"}:
            rows[value] = pd.Series(values).astype("string")
    for table_name, frame in ((stays_table, stays), (table, rows)):
        schema = {k.lower() for k in config["tables"][table_name]["cols"]}
        assert {c.lower() for c in frame} <= schema
    path = root / f"{table}.parquet"
    if bucketed:
        path = root / f"{table}_bucket" / "bucket_id=0" / "part.parquet"
        path.parent.mkdir(parents=True)
    rows.to_parquet(path, index=False)
    # Empty ancillary sources retain their real schemas, as an empty patient
    # projection would. Never invent columns to make a callback succeed.
    from easyicu.concept.loader import load_dictionary
    dictionary = load_dictionary(include_sofa2=True)
    names = [concept, "rrt" if concept != "rrt" else "mech_circ_support"]
    related = {s.table for name in names for s in dictionary[name].sources.get(database, [])}
    for name in related - {table, stays_table}:
        columns = config["tables"][name]["cols"]
        empty = pd.DataFrame({spec["name"]: pd.Series(dtype={
            "col_integer": "Int64", "col_double": "float64", "col_datetime": "datetime64[ns]",
            "col_logical": "boolean",
        }.get(spec["spec"], "string")) for spec in columns.values()})
        empty.to_parquet(root / f"{name}.parquet", index=False)
    return ICUDataSource(load_src_cfg(database), base_path=root, enable_cache=False), identifier, expected


@pytest.mark.parametrize("item_id,database,concept,table,item", PATH_CASES)
@pytest.mark.parametrize("path", ["flat", "bucket", "batch", "native"])
def test_rb_01_to_08_real_schema_loader_and_native_paths(tmp_path, monkeypatch, item_id, database, concept, table, item, path):
    from easyicu.api import extraction as api
    from easyicu.api.concepts import clear_global_loader
    from easyicu.concept import ConceptResolver
    from easyicu.concept.loader import load_dictionary
    from easyicu.datasource import _close_duckdb_connections

    monkeypatch.setenv("EASYICU_HOME", str(tmp_path / "state"))
    monkeypatch.setenv("EASYICU_DUCKDB_THREADS", "1")
    monkeypatch.setenv("EASYICU_DUCKDB_MEMORY_LIMIT_GB", "1")
    raw = tmp_path / "source"
    raw.mkdir()
    source, identifier, expected = _disk_source(raw, database, concept, table, item, path != "flat")
    patients = {identifier: list(range(1, 7))}
    try:
        if path == "native":
            output = tmp_path / "native"
            output.mkdir()
            module = {"other_vaso": "vasopressors", "rrt": "renal", "adv_resp": "respiratory",
                      "ecmo": "respiratory", "mech_circ_support": "circulatory"}.get(concept, "medications")
            api._run_module_extraction(
                module_name=module, concepts=[concept], database=database, data_path=str(raw),
                patient_ids_filter=patients, batch_size=6, output_dir=str(output),
                use_sofa2=True, stream_output_batches=False,
            )
            manifest = json.loads((output / "_manifest.json").read_text())
            assert manifest["errors"] == []
            (output / "_manifest.json").rename(output / f"{module}.manifest.json")
            api._publish_native_export_v2(
                database=database, data_path=str(raw), output_dir=str(output), modules=[module],
                max_patients=None, result={"modules": {module: manifest}},
                concept_projection={module: [concept]},
            )
            frame = pd.read_parquet(output / f"{module}.parquet")
            identifier = "stay_id"
        else:
            dictionary = load_dictionary(include_sofa2=True)
            # A second source-backed concept makes the resolver consider batch
            # aggregation. Non-inlineable callbacks must still see raw rows.
            names = [concept, "rrt" if concept != "rrt" else "mech_circ_support"] if path == "batch" else [concept]
            frame = ConceptResolver(dictionary).load_concepts(
                names, source, patient_ids=patients, interval=pd.Timedelta(hours=1),
                merge=False, r_compatible=False, concept_workers=1, verbose=False,
            )[concept].data
        assert set(frame.loc[frame[concept].eq(True).fillna(False), identifier]) == expected, item_id  # noqa: E712
    finally:
        clear_global_loader()
        _close_duckdb_connections()
