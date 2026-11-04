"""Synthetic round-5 bounds receipts; flagged tails are not clinical gold standards."""

import json
from pathlib import Path
from types import SimpleNamespace

import pandas as pd
import numpy as np
import pytest
import easyicu

from easyicu.concept import ConceptDefinition, ConceptDictionary, ConceptResolver
from easyicu.concept.callbacks import ConceptCallbackContext, execute_concept_callback
from easyicu.table import ICUTable


DICTIONARY_PATH = Path(easyicu.__file__).parent / "data/concept-dict.json"


def _retained_tail(name, values, database):
    dictionary = json.loads(DICTIONARY_PATH.read_text())
    raw_name = f"{name}_raw"
    raw = dictionary[raw_name]
    # The audit channel must not discard the very observations it documents.
    assert "min" not in raw and "max" not in raw
    assert raw["sources"][database]
    definition = ConceptDefinition.from_name_and_payload(f"{name}_qc", dictionary[f"{name}_qc"])
    assert definition.aggregate == "max"
    assert raw_name in definition.sub_concepts
    frame = pd.DataFrame({"stay_id": 1, "charttime": range(len(values)), raw_name: values})
    table = ICUTable(frame, id_columns=["stay_id"], index_column="charttime", value_column=raw_name)
    context = ConceptCallbackContext(
        concept_name=f"{name}_qc", target="ts_tbl", interval=pd.Timedelta(hours=1),
        resolver=SimpleNamespace(dictionary={
            name: ConceptDefinition.from_name_and_payload(name, dictionary[name]),
        }),
        data_source=SimpleNamespace(config=SimpleNamespace(name=database)), patient_ids=None,
    )
    result = execute_concept_callback(definition.callback, {raw_name: table}, context).data
    assert table.data[raw_name].tolist() == values
    assert result[f"{name}_qc"].tolist() == [1] * len(values)
    return result


def test_r5c_aumc_010_low_core_temperature_remains_auditable():
    _retained_tail("temp", [30.5, 31.5], "aumc")


def test_r5c_hirid_014_high_etco2_remains_auditable():
    _retained_tail("etco2", [65.0, 90.0], "hirid")


def test_r5c_hirid_015_repeatable_hemoglobin_tails_remain_auditable():
    _retained_tail("hgb", [3.0, 20.0], "hirid")


def test_r5_eicu_010_paired_blood_count_tails_remain_auditable():
    _retained_tail("hgb", [3.0, 20.0], "eicu")
    _retained_tail("hct", [10.0, 65.0], "eicu")


def test_r5_eicu_024_severe_acidosis_remains_auditable():
    _retained_tail("ph", [6.7], "eicu")


def test_r5_eicu_027_severe_renal_panel_remains_auditable():
    _retained_tail("bun", [220.0], "eicu")
    _retained_tail("crea", [28.0], "eicu")


def test_r5_eicu_029_repeatable_sodium_tails_remain_auditable():
    _retained_tail("na", [105.0, 170.0], "eicu")


def test_r5_sic_014_ck_supported_myoglobin_remains_auditable():
    _retained_tail("myoglobin", [35000.0], "sic")


def test_r5c_iv_007_paired_blood_count_tails_remain_auditable():
    _retained_tail("hgb", [3.0, 20.0], "miiv")
    _retained_tail("hct", [10.0, 65.0], "miiv")


def test_r5c_iii_007_paired_blood_count_tails_remain_auditable():
    _retained_tail("hgb", [3.0, 20.0], "mimic")
    _retained_tail("hct", [10.0, 65.0], "mimic")


NAMES = ("temp", "etco2", "hgb", "hct", "ph", "bun", "crea", "na", "myoglobin")


@pytest.mark.parametrize("name", NAMES)
def test_audit_sources_keep_existing_units_specimens_and_channel_precedence(name):
    dictionary = json.loads(DICTIONARY_PATH.read_text())
    public, raw = dictionary[name], dictionary[f"{name}_raw"]
    for field in ("sources", "unit", "source_channel_priority", "source_channel_window_hours"):
        assert raw.get(field) == public.get(field)
    # Round 5 supplies candidate receipts, not new clinically reviewed limits.
    assert (public.get("min"), public.get("max")) == {
        "temp": (32, 42), "etco2": (10, 60), "hgb": (4, 18), "hct": (15, 60),
        "ph": (6.8, 8), "bun": (0, 200), "crea": (0, 25),
        "na": (110, 165), "myoglobin": (0, 20000),
    }[name]


def _quality(name, values):
    dictionary = ConceptDictionary.from_json(DICTIONARY_PATH)
    raw_name = f"{name}_raw"
    frame = pd.DataFrame({"stay_id": 1, "charttime": range(len(values)), raw_name: values})
    table = ICUTable(frame, id_columns=["stay_id"], index_column="charttime", value_column=raw_name)
    context = ConceptCallbackContext(
        concept_name=f"{name}_qc", target="ts_tbl", interval=pd.Timedelta(hours=1),
        resolver=SimpleNamespace(dictionary=dictionary), data_source=None, patient_ids=None,
    )
    return execute_concept_callback(dictionary[f"{name}_qc"].callback, {raw_name: table}, context).data


def test_quality_distinguishes_inclusive_range_tails_missing_and_invalid_values():
    values = [4.0, 18.0, 3.0, 20.0, np.nan, np.inf, -np.inf, "unreadable"]
    result = _quality("hgb", values)
    assert result.hgb_qc.iloc[:4].tolist() == [0, 0, 1, 1]
    assert pd.isna(result.hgb_qc.iloc[4])
    assert result.hgb_qc.iloc[5:].tolist() == [2, 2, 2]
    assert result.charttime.tolist() == list(range(len(values)))


def test_empty_audit_preserves_typed_schema():
    result = _quality("hgb", [])
    assert result.empty
    assert result.columns.tolist() == ["stay_id", "charttime", "hgb_qc"]
    assert pd.api.types.is_numeric_dtype(result.hgb_qc)


def test_audit_channels_require_explicit_selection_in_research_catalog():
    from easyicu.research_agent.agents.planner import _format_concept_id_allowlist
    from easyicu.research_agent.concept_catalog import load_concept_catalog
    from easyicu.research_agent.discovery.idea_mining import _build_concept_lookup
    from easyicu.research_agent.planning.cohort_contract import known_concept_ids

    names = {f"{name}_{suffix}" for name in NAMES for suffix in ("raw", "qc")}
    assert names <= known_concept_ids()
    assert not names.intersection(known_concept_ids(include_audit=False))
    catalog = load_concept_catalog()
    assert not names.intersection(catalog.available_concepts)
    # A manifest's columns are not user consent to use audit values in research.
    selected = load_concept_catalog(restrict_to=["hgb", *sorted(names)])
    assert selected.available_concepts == ("hgb",)
    lookup = _build_concept_lookup(catalog.available_concepts, concept_aliases=catalog.concept_aliases)
    # R5C-HIRID-014: etco2_raw's stripped variant must not steal etco2's alias.
    assert lookup["etco"] == "etco2"
    menu = _format_concept_id_allowlist([])
    assert all(f"- `{name}`" not in menu for name in names)
    assert all(f"- `{name}`" not in _format_concept_id_allowlist(["hgb", *names]) for name in names)


def test_eicu_public_filter_and_raw_quality_receipt_are_independent(tmp_path):
    from easyicu.config import load_src_cfg
    from easyicu.datasource import ICUDataSource

    pd.DataFrame({
        "patientunitstayid": [1], "patienthealthsystemstayid": [11], "uniquepid": ["synthetic"],
        "hospitaladmitoffset": [0], "hospitaldischargeoffset": [720], "unitdischargeoffset": [720],
    }).to_parquet(tmp_path / "patient.parquet", index=False)
    pd.DataFrame({
        "patientunitstayid": 1, "labid": [1, 2, 3, 4], "labname": "Hgb",
        "labresultoffset": [0, 5, 60, 120], "labresultrevisedoffset": [0, 5, 60, 120],
        "labresult": [3.0, 12.0, 12.0, 20.0], "labmeasurenameinterface": "g/dL",
    }).to_parquet(tmp_path / "lab.parquet", index=False)
    source = ICUDataSource(load_src_cfg("eicu"), base_path=tmp_path, enable_cache=False)
    resolver = ConceptResolver(ConceptDictionary.from_json(DICTIONARY_PATH))
    result = resolver.load_concepts(
        ["hgb", "hgb_raw", "hgb_qc"], source, interval=pd.Timedelta(hours=1),
        merge=True, r_compatible=False, concept_workers=1, verbose=False,
    ).sort_values("charttime")
    assert result.hgb_qc.tolist() == [1, 0, 1]
    assert result.hgb.dropna().tolist() == [12.0, 12.0]
    assert result.hgb_raw.tolist() == [7.5, 12.0, 20.0]
    assert result.hgb_qc.tolist() == [1, 0, 1]
    native = resolver.load_concepts(
        ["hgb_raw"], source, interval=pd.Timedelta(hours=1), aggregate=False,
        merge=False, r_compatible=False, concept_workers=1, verbose=False,
    )["hgb_raw"].data
    assert native.hgb_raw.tolist() == [3.0, 12.0, 12.0, 20.0]


def _source_fixture(root, database, name, values, layout):
    """Only synthetic observations, including two readings in the first hour."""
    from easyicu.config import load_src_cfg
    from easyicu.datasource import ICUDataSource

    dictionary = json.loads(DICTIONARY_PATH.read_text())
    source = dictionary[name]["sources"][database][0]
    item = source["ids"]
    item = item[0] if isinstance(item, list) else item
    minutes = [0, 5, 60, 120]
    start = pd.Timestamp("2020-01-01")
    times = start + pd.to_timedelta(minutes, unit="m")
    unit = dictionary[name].get("unit", "")
    unit = unit[0] if isinstance(unit, list) else unit
    if database == "eicu":
        pd.DataFrame({"patientunitstayid": [1], "patienthealthsystemstayid": [11],
                      "uniquepid": ["synthetic"], "hospitaladmitoffset": [0],
                      "hospitaldischargeoffset": [720], "unitdischargeoffset": [720]}).to_parquet(root / "patient.parquet")
        rows = pd.DataFrame({"patientunitstayid": 1, "labid": range(4), "labname": item,
                             "labresultoffset": minutes, "labresultrevisedoffset": minutes,
                             "labresult": values, "labmeasurenameinterface": unit})
    elif database in {"miiv", "mimic"}:
        stay = "stay_id" if database == "miiv" else "icustay_id"
        pd.DataFrame({stay: [1], "subject_id": [1], "hadm_id": [11], "intime": [start],
                      "outtime": [start + pd.Timedelta(hours=12)], "los": [0.5]}).to_parquet(root / "icustays.parquet")
        rows = pd.DataFrame({"subject_id": 1, "hadm_id": 11, "itemid": item,
                             "charttime": times, "valuenum": values, "valueuom": unit})
    elif database == "aumc":
        pd.DataFrame({"admissionid": [1], "patientid": [1], "admittedat": [0],
                      "dischargedat": [43_200_000]}).to_parquet(root / "admissions.parquet")
        rows = pd.DataFrame({"admissionid": 1, "itemid": item,
                             "measuredat": np.array(minutes) * 60_000, "value": values, "unit": unit,
                             "tag": "+", "registeredat": np.array(minutes) * 60_000,
                             "updatedat": np.array(minutes) * 60_000})
    elif database == "hirid":
        pd.DataFrame({"patientid": [1], "admissiontime": [str(start)], "age": [60],
                      "sex": ["M"]}).to_parquet(root / "general.parquet")
        rows = pd.DataFrame({"patientid": 1, "variableid": item, "datetime": times,
                             "entertime": times, "value": np.array(values) * (10 if name == "hgb" else 1),
                             "status": 8, "type": "N", "stringvalue": None})
    else:
        pd.DataFrame({"CaseID": [1], "PatientID": [1], "ICUOffset": [0],
                      "TimeOfStay": [43_200]}).to_parquet(root / "cases.parquet")
        rows = pd.DataFrame({"CaseID": 1, "LaboratoryID": item, "Offset": np.array(minutes) * 60,
                             "LaboratoryValue": values, "LaboratoryType": 2296})
    table = source["table"]
    if layout == "file":
        path = root / f"{table}.parquet"
    elif layout == "bucket":
        path = root / f"{table}_bucket" / "bucket_id=0" / "part.parquet"
        path.parent.mkdir(parents=True)
        (path.parent.parent / "_COMPLETE").write_text("ready")
    else:
        # Flat parquet directories exercise the same DuckDB aggregate owners as buckets.
        path = root / table / "part.parquet"
        path.parent.mkdir()
    rows.to_parquet(path, index=False)
    return ICUDataSource(load_src_cfg(database), base_path=root, enable_cache=False)


CASES = [
    ("aumc", "temp", 31.0, 37.0), ("hirid", "etco2", 90.0, 30.0),
    ("hirid", "hgb", 3.0, 12.0), ("eicu", "hgb", 3.0, 12.0),
    ("eicu", "hct", 10.0, 40.0), ("eicu", "ph", 6.7, 7.4),
    ("eicu", "bun", 220.0, 10.0), ("eicu", "crea", 28.0, 1.0),
    ("eicu", "na", 105.0, 140.0), ("sic", "myoglobin", 35000.0, 100.0),
    ("miiv", "hgb", 3.0, 12.0), ("miiv", "hct", 10.0, 40.0),
    ("mimic", "hgb", 3.0, 12.0), ("mimic", "hct", 10.0, 40.0),
]


@pytest.mark.parametrize("database,name,tail,normal", CASES)
@pytest.mark.parametrize("layout", ["file", "duckdb", "bucket"])
def test_source_tails_and_flags_survive_individual_batch_and_native_export(
    tmp_path, database, name, tail, normal, layout,
):
    from easyicu.api import extraction

    values = [tail, normal, normal, tail]
    source = _source_fixture(tmp_path, database, name, values, layout)
    dictionary = ConceptDictionary.from_json(DICTIONARY_PATH)
    raw, qc = f"{name}_raw", f"{name}_qc"

    def load(names, **kwargs):
        return ConceptResolver(dictionary).load_concepts(
            names, source, interval=pd.Timedelta(hours=1), merge=False,
            r_compatible=False, concept_workers=1, verbose=False, **kwargs,
        )

    # Unaggregated audit retrieval must preserve the original four observations.
    exact = load([raw], aggregate=False)[raw].data
    assert exact[raw].tolist() == pytest.approx(values)
    exact_batch = load([name, raw], aggregate=False)
    assert exact_batch[raw].data[raw].tolist() == pytest.approx(values)
    # Each source observation is flagged before any hourly SQL/Pandas aggregate.
    assert load([qc])[qc].data[qc].tolist() == [1, 0, 1]
    batch = load([name, raw, qc])
    assert batch[raw].data[raw].tolist() == pytest.approx([(tail + normal) / 2, normal, tail])
    assert batch[qc].data[qc].tolist() == [1, 0, 1]
    assert batch[name].data[name].dropna().tolist() == pytest.approx([normal, normal])

    # Explicit native-v2 audit module, using the real resolver output and publisher.
    frame = batch[raw].data.copy()
    time = batch[raw].index_column
    frame[qc] = batch[qc].data[qc].to_numpy()
    frame = frame.rename(columns={batch[raw].id_columns[0]: "stay_id", time: "charttime"})
    out = tmp_path / "out"
    out.mkdir()
    frame.to_parquet(out / "bounds_audit.parquet", index=False)
    extraction._publish_native_export_v2(
        database=database, data_path=str(tmp_path), output_dir=str(out),
        modules=["bounds_audit"], max_patients=None,
        result={"modules": {"bounds_audit": {"errors": []}}},
        concept_projection={"bounds_audit": [raw, qc]},
    )
    published = pd.read_parquet(out / "bounds_audit.parquet")
    assert published[raw].tolist() == pytest.approx([(tail + normal) / 2, normal, tail])
    assert published[qc].tolist() == [1, 0, 1]
    assert (out / "_manifest.json").exists()


@pytest.mark.parametrize("name", NAMES)
def test_qc_documents_and_tests_its_own_inclusive_public_range(name):
    dictionary = json.loads(DICTIONARY_PATH.read_text())
    low, high = dictionary[name]["min"], dictionary[name]["max"]
    comment = dictionary[f"{name}_qc"]["_comment"]
    assert f"[{low}, {high}]" in comment
    values = [low - 1, low, (low + high) / 2, high, high + 1, np.nan]
    quality = _quality(name, values)[f"{name}_qc"]
    assert quality.iloc[:5].tolist() == [1, 0, 0, 0, 1]
    assert pd.isna(quality.iloc[5])


def test_audit_module_is_explicit_and_absent_from_default_modules():
    from easyicu.api.extraction import EXTRACT_MODULE_ORDER, EXTRACT_MODULES
    from easyicu.concept.catalog import CONCEPT_GROUPS_INTERNAL

    assert "bounds_audit" not in EXTRACT_MODULE_ORDER
    assert "bounds_audit" not in CONCEPT_GROUPS_INTERNAL
    assert set(EXTRACT_MODULES["bounds_audit"]) == {f"{n}_{s}" for n in NAMES for s in ("raw", "qc")}


@pytest.mark.parametrize("database", ["hirid", "aumc", "sic"])
def test_explicit_audit_module_runs_through_native_worker(tmp_path, monkeypatch, database):
    import os
    import tempfile
    from easyicu.api.extraction import extract_database
    from easyicu.synthetic_sources import generate_source

    monkeypatch.setattr(tempfile, "tempdir", tempfile.tempdir)
    for key in ("TMPDIR", "EASYICU_DUCKDB_TEMP_DIR"):
        if key in os.environ:
            monkeypatch.setenv(key, os.environ[key])
        else:
            monkeypatch.delenv(key, raising=False)
    source, output = tmp_path / "source", tmp_path / "out"
    generate_source(source, database, stays=3)
    result = extract_database(
        database, data_path=source, output_dir=output, modules=["outcome", "bounds_audit"],
        native_export_v2=True, resource_budget_mb=512, verbose=False,
    )
    assert not result["modules"]["bounds_audit"].get("errors")
    assert (output / "_manifest.json").exists()
    frame = pd.read_parquet(output / "bounds_audit.parquet")
    assert {"hgb_raw", "hgb_qc", "etco2_raw", "etco2_qc"} <= set(frame)
    assert frame.hgb_raw.notna().any()
    assert frame.hgb_qc.dropna().isin([0, 1, 2]).all()


def test_audit_columns_in_a_manifest_do_not_enter_planner_or_idea_rosters():
    from easyicu.research_agent.agents.planner import _build_planner_user_prompt
    from easyicu.research_agent.discovery.idea_mining import _build_concept_lookup
    from easyicu.research_agent.schema import CohortDescriptor, ConceptDescriptor, ResearchContext

    clinical = [ConceptDescriptor(name="hgb", dtype="float64"),
                ConceptDescriptor(name="death", dtype="int64", role="outcome")]
    dictionary = json.loads(DICTIONARY_PATH.read_text())
    audit = [ConceptDescriptor(name=f"{n}_{s}", source_concept=f"{n}_{s}", dtype="float64",
                               description=dictionary[f"{n}_{s}"]["description"])
             for n in NAMES for s in ("raw", "qc")]
    # Materialized aggregates may rename a column, keeping its source identity.
    audit.append(ConceptDescriptor(name="audit_hgb_mean", source_concept="hgb_raw", dtype="float64"))
    context = ResearchContext(
        research_question="Estimate hemoglobin association with death.",
        cohort=CohortDescriptor(cohort_name="synthetic", database="eicu", n_stays=3),
        variables=clinical + audit, target_outcome="death",
    )
    prompt = _build_planner_user_prompt(context)
    assert all(variable.name not in prompt for variable in audit)
    assert _build_concept_lookup(context.variables) == _build_concept_lookup(clinical)
    assert len(context.variables) == len(clinical) + len(audit)  # host authority is unchanged
