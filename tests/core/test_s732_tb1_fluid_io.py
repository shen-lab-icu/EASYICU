"""Invented fluid sources exercise gates, amount allocation and opt-in isolation."""

import json
from pathlib import Path
from types import SimpleNamespace

import pandas as pd
import pytest

from easyicu.concept import ConceptResolver, _apply_callback
from easyicu.config import load_src_cfg
from easyicu.datasource import ICUDataSource
from easyicu.resources import load_dictionary
from easyicu.api.extraction import EXTRACT_MODULES, EXTRACT_MODULE_ORDER
from easyicu.concept.catalog import FLUID_IO_CONCEPTS, HIDDEN_DICTIONARY_CONCEPTS

NAMES = ("crystalloid_input_ml", "colloid_input_ml", "ffp_input_ml", "packed_rbc_input_ml",
         "parenteral_input_ml", "platelet_input_ml")
DICTIONARY = json.loads((Path(__file__).parents[2] / "src/easyicu/data/concept-dict.json").read_text())


def _load(root, database, names):
    source = ICUDataSource(load_src_cfg(database), base_path=root, enable_cache=False)
    return ConceptResolver(load_dictionary()).load_concepts(names, source, merge=False,
        r_compatible=False, interval=pd.Timedelta(hours=1), verbose=False, concept_workers=1)


def test_module_is_opt_in_and_does_not_add_a_balance_sum():
    assert tuple(EXTRACT_MODULES["fluid_io"]) == FLUID_IO_CONCEPTS
    assert set(FLUID_IO_CONCEPTS) == {*NAMES, "other_output_ml"}
    assert set(FLUID_IO_CONCEPTS) <= HIDDEN_DICTIONARY_CONCEPTS
    assert "fluid_io" not in EXTRACT_MODULE_ORDER
    assert all(not set(NAMES) & set(EXTRACT_MODULES[m]) for m in EXTRACT_MODULE_ORDER)
    balance = load_dictionary()["fluid_balance"]
    assert not set(NAMES) & set(balance.sub_concepts)


@pytest.mark.parametrize("name,entry", [(name, entry) for name in NAMES
    for entry in DICTIONARY[name]["sources"]["miiv"]])
def test_miiv_amounts_through_loader(tmp_path, name, entry):
    from tests.support.mimic_synthetic_source import mimic_synthetic_load, mimic_synthetic_source

    origin = pd.Timestamp("2150-01-01")
    start = origin + pd.Timedelta(minutes=30)
    frame = pd.DataFrame({"subject_id": 11, "hadm_id": 21, "stay_id": 1,
        "itemid": entry["ids"][0], "orderid": range(7), "linkorderid": range(7),
        "starttime": start, "endtime": origin + pd.Timedelta(hours=2),
        "amount": [150.] * 7, "amountuom": ["mL"] * 6 + ["mg"],
        "rate": 999., "rateuom": "mL/hour", "storetime": start,
        "statusdescription": ["FinishedRunning", "Rewritten", "Cancelled", "Canceled",
                              "FinishedRunning", "FinishedRunning", "FinishedRunning"],
        "ordercategoryname": ["01-Drips"] * 4 + ["OR", "PACU", "01-Drips"]})
    source = mimic_synthetic_source(tmp_path, "miiv", inputevents=frame)
    result = mimic_synthetic_load(source, [name])[name].data
    assert result[name].tolist() == pytest.approx([50., 100.])
    assert result.charttime.tolist() == [0., 1.]


@pytest.mark.parametrize("name", NAMES)
def test_aumc_boolean_category_and_fluidin_through_loader(tmp_path, name):
    entry = DICTIONARY[name]["sources"]["aumc"][0]
    import re
    category = int(re.search(r"\(\?:([0-9]+)", entry["require_regex"]["ordercategoryid"]).group(1))
    pd.DataFrame({"admissionid": [1], "patientid": [1], "admittedat": [0],
        "dischargedat": [24 * 3600000]}).to_parquet(tmp_path / "admissions.parquet", index=False)
    frame = pd.DataFrame({"admissionid": 1, "itemid": entry["ids"][0],
        "start": 1800000, "stop": 7200000, "fluidin": 150., "dose": 999., "doseunit": "mg",
        "isadditive": pd.Series([False, True, False, None], dtype="boolean"),
        "ordercategoryid": [category, category, -1, category]})
    frame.to_parquet(tmp_path / "drugitems.parquet", index=False)
    result = _load(tmp_path, "aumc", [name])[name].data
    assert result[name].tolist() == pytest.approx([50., 100.])
    assert result.charttime.tolist() == [0., 1.]


@pytest.mark.parametrize("name", NAMES)
def test_aumc_integer_isadditive_as_in_the_converted_source(tmp_path, name):
    """The converted AUMCdb drugitems store isadditive as BIGINT 0/1 (TB1 real-data check)."""
    entry = DICTIONARY[name]["sources"]["aumc"][0]
    import re
    category = int(re.search(r"\(\?:([0-9]+)", entry["require_regex"]["ordercategoryid"]).group(1))
    pd.DataFrame({"admissionid": [1], "patientid": [1], "admittedat": [0],
        "dischargedat": [24 * 3600000]}).to_parquet(tmp_path / "admissions.parquet", index=False)
    frame = pd.DataFrame({"admissionid": 1, "itemid": entry["ids"][0],
        "start": 1800000, "stop": 7200000, "fluidin": 150., "dose": 999., "doseunit": "mg",
        "isadditive": pd.Series([0, 1, 0, None], dtype="Int64"),
        "ordercategoryid": [category, category, -1, category]})
    frame.to_parquet(tmp_path / "drugitems.parquet", index=False)
    result = _load(tmp_path, "aumc", [name])[name].data
    assert result[name].tolist() == pytest.approx([50., 100.])
    assert result.charttime.tolist() == [0., 1.]


@pytest.mark.parametrize("name", [n for n in NAMES if "sic" in DICTIONARY[n]["sources"]])
def test_sic_loader_retains_bolus_and_conserves_continuous_volume(tmp_path, name):
    entry = DICTIONARY[name]["sources"]["sic"][0]
    pd.DataFrame({"CaseID": [1], "PatientID": [1], "ICUOffset": [900.],
                  "TimeOfStay": [86400.], "OffsetOfDeath": [None]}).to_parquet(tmp_path / "cases.parquet", index=False)
    frame = pd.DataFrame({"CaseID": 1, "DrugID": entry["ids"][0],
        "Offset": [2700., 4500., 2700., 2700., 2700., 2700.],
        "OffsetDrugEnd": [8100., 11700., 8100., 8100., 8100., 2700.],
        "Amount": [150., 25., 900., -10., 0., 900.],
        "IsSingleDose": [0, 1, 2, 0, 0, 0]})
    frame.to_parquet(tmp_path / "medication.parquet", index=False)
    result = _load(tmp_path, "sic", [name])[name].data
    assert result[name].tolist() == pytest.approx([50., 125.])
    assert result.Offset.tolist() == [0., 1.]
    assert result[name].sum() == pytest.approx(175.)


def test_sic_missing_flag_fails_explicitly():
    source = load_dictionary()["crystalloid_input_ml"].sources["sic"][0]
    frame = pd.DataFrame({"CaseID": [1], "Offset": [0.], "OffsetDrugEnd": [3600.],
                          "crystalloid_input_ml": [100.]})
    data = SimpleNamespace(load_table=lambda *a, **k: pd.DataFrame({"CaseID": [1], "ICUOffset": [0.]}))
    with pytest.raises(ValueError, match="IsSingleDose"):
        _apply_callback(frame, source, "crystalloid_input_ml", data_source=data)


@pytest.fixture(scope="module")
def synthetic_sources(tmp_path_factory):
    from easyicu.synthetic_sources import generate_source
    root = tmp_path_factory.mktemp("fluid-sources")
    for db in ("aumc", "hirid", "sic"):
        generate_source(root / db, db, stays=12)
    return root


@pytest.mark.parametrize("name,db", [(n, db) for n in NAMES for db in DICTIONARY[n]["sources"]
                                    if db in {"aumc", "hirid", "sic"}])
def test_synthetic_extraction(synthetic_sources, name, db):
    result = _load(synthetic_sources / db, db, [name])[name].data
    assert not result.empty and result[name].notna().any()


@pytest.mark.parametrize("name,entry", [(name, entry) for name in NAMES
    for entry in DICTIONARY[name]["sources"].get("eicu", [])])
def test_eicu_exact_cellpath_has_no_alias_double_count(tmp_path, name, entry):
    from tests.support.eicu_source import _eicu_source, _hourly

    path = entry["ids"][0]
    frame = pd.DataFrame({"intakeoutputid": [1, 2], "intakeoutputoffset": [15, 75],
        "intakeoutputentryoffset": [16, 76], "cellpath": [path, path + " other"],
        "celllabel": [path.split("|")[-1], path.split("|")[-1] + " other"],
        "cellvaluenumeric": [100., 500.]})
    source = _eicu_source(tmp_path, "eicu", intakeoutput=frame)
    assert _hourly(source, name, "sum") == {0.: 100.}


@pytest.mark.parametrize("missing", [None, "rateuom", "storetime"])
def test_miiv_amounts_do_not_need_an_observed_rate(tmp_path, missing):
    from tests.support.mimic_synthetic_source import mimic_synthetic_load, mimic_synthetic_source

    start = pd.Timestamp("2150-01-01 00:30")
    # Converted inputevents always has its table-default value field `rate`.
    # Omitting that column made the earlier synthetic table unreadable even
    # for an amount source. Null rates are valid for delivered fluid amounts.
    frame = pd.DataFrame({"subject_id": [11], "hadm_id": [21], "stay_id": [1],
        "itemid": [220949], "orderid": [1], "linkorderid": [1], "starttime": [start],
        "endtime": [start + pd.Timedelta(hours=1)], "amount": [100.], "amountuom": ["mL"],
        "rate": [None], "rateuom": [None], "storetime": [start],
        "ordercategoryname": ["IV Fluid"], "statusdescription": ["FinishedRunning"]})
    if missing:
        frame = frame.drop(columns=missing)
    source = mimic_synthetic_source(tmp_path, "miiv", inputevents=frame)
    values = mimic_synthetic_load(source, ["crystalloid_input_ml"])["crystalloid_input_ml"].data
    assert values.crystalloid_input_ml.sum() == pytest.approx(100.)


def test_hirid_fluid_io_keeps_the_b1_batch_profile():
    from easyicu.api.extraction import plan_extraction_resources

    plan = plan_extraction_resources("hirid", ["fluid_io"], 33_905, available_memory_mb=12 * 1024)
    assert plan.batch_size == 10_000
    assert plan.measured_peak_rss_mb == pytest.approx(5_032.5)


def test_selected_fluid_channels_do_not_overlap_or_use_broad_category_sources():
    for db in ("aumc", "miiv", "sic"):
        seen = set()
        for name in NAMES:
            for entry in DICTIONARY[name]["sources"].get(db, []):
                assert entry.get("ids") and not entry.get("regex")
                channels = {(entry["table"], entry["sub_var"], item) for item in entry["ids"]}
                assert not channels & seen
                seen |= channels
