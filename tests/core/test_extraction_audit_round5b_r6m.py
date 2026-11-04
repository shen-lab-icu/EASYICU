"""Round-6 mappings and legacy GCS; all input records are invented."""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

import easyicu
from easyicu.callbacks import gcs
from easyicu.concept import ConceptDictionary, ConceptResolver
from easyicu.concept.metadata_projection import (
    ColumnProjectionSpec,
    ConceptColumnRole,
    project_concept_column_metadata,
)
from tests.core.test_extraction_audit_round5_aumc5 import _db
from tests.core.test_extraction_audit_round5_hirid5 import _source_root
from tests.support.blood_gas_sources import ADMISSION, _source, _write
from tests.support.eicu_source import _eicu_source


DATA = Path(easyicu.__file__).parent / "data"
AUMC_FIO2_ITEMS = (6699, 12279, 12282, 12329, 12369, 16246, 20134)


def _dictionary():
    return ConceptDictionary.from_json(DATA / "concept-dict.json")


def _load(source, names, id_column, *, interval=pd.Timedelta(hours=1)):
    dictionary = _dictionary()
    # Do not rely on identifier/value columns invented only by the fixtures.
    for name in names:
        for mapping in dictionary.get(name).for_data_source(source.config):
            table = source.config.tables[mapping.table]
            columns = {key.lower() for key in table.columns}
            columns.update(spec["name"].lower() for spec in table.columns.values())
            required = [mapping.sub_var, mapping.value_var,
                        *mapping.params.get("deduplicate_by", ()),
                        *mapping.params.get("extra_vars", ())]
            assert {key.lower() for key in required if key} <= columns
    return ConceptResolver(dictionary).load_concepts(
        names, source, merge=False, r_compatible=False, interval=interval,
        patient_ids={id_column: [1]}, concept_workers=1, verbose=False,
    )


def _bucket_tables(root, names):
    for name in names:
        original = root / f"{name}.parquet"
        if not original.exists():
            original = root / name / "part.parquet"
        target = root / f"{name}_bucket" / "bucket_id=0" / "part.parquet"
        target.parent.mkdir(parents=True)
        original.replace(target)
        if original.parent != root:
            original.parent.rmdir()


def _assert_single_loads(source, loaded, names, id_column):
    for name in names:
        single = _load(source, [name], id_column)[name]
        assert _values(single, name) == _values(loaded[name], name)


def _values(table, name):
    rows = table.data.dropna(subset=[name])
    return dict(zip(rows[table.index_column].astype(float), rows[name].astype(float)))


def _metadata(name, database):
    return project_concept_column_metadata(
        _dictionary().get(name), source_database=database,
        spec=ColumnProjectionSpec(column_name=name, source_concept=name,
                                  role=ConceptColumnRole.VALUE),
    )


@pytest.mark.parametrize("layout", ["flat", "bucket"])
def test_R6_F018_hirid_chart_matches_parent_percent_and_missing(tmp_path, layout):
    root = _source_root(tmp_path, "observations", [
        {"patientid": 1, "datetime": t, "variableid": item, "value": value}
        for t, item, value in [(0.1, 2010, 40.), (0.5, 2010, 60.),
                               (1.1, 2010, .5), (2.1, 2010, 21.),
                               (3.1, 2010, 100.), (4.1, 2010, None),
                               (5.1, 2010, 0.), (6.1, 2010, 101.),
                               (7.1, 99999, 70.)]
    ], layout=layout)
    loaded = _load(_source(root, "hirid"), ["fio2", "fio2_chart", "fio2_lab"], "patientid")
    assert _values(loaded["fio2_chart"], "fio2_chart") == {0.: 50., 1.: 50., 2.: 21., 3.: 100.}
    assert _values(loaded["fio2"], "fio2") == _values(loaded["fio2_chart"], "fio2_chart")
    assert loaded["fio2_lab"].data.empty
    assert _metadata("fio2_chart", "hirid").canonical_unit == "%"
    _assert_single_loads(_source(root, "hirid"), loaded, ["fio2", "fio2_chart"], "patientid")


@pytest.mark.parametrize("layout", ["file", "bucket"])
def test_R6_F019_aumc_item_columns_trace_the_unchanged_pool(tmp_path, layout, monkeypatch):
    rows = [(5 + i, item, 30. + 5 * i, "Geen") for i, item in enumerate(AUMC_FIO2_ITEMS)]
    rows += [(65 + i, item, 70. - 5 * i, "Geen") for i, item in enumerate(AUMC_FIO2_ITEMS)]
    rows += [(125, 6699, None, "Geen"), (185, 6699, 101., "Geen"),
             (245, 20656, 90., "Geen"), (305, 13076, 1., "Geen"),
             (5, 9560, 1.1, "mmol/L"), (5, 9561, 1.2, "mmol/L")]
    source = _db(tmp_path, numeric=rows, layout=layout)
    names = [f"fio2_chart_aumc_{item}" for item in AUMC_FIO2_ITEMS]
    from easyicu.api.extraction import _native_export_physical_value_columns
    from easyicu.concept.export_metadata import build_export_file_metadata_binding

    assert set(names) <= set(_native_export_physical_value_columns(["fio2_chart"]))
    loaded = _load(source, ["fio2", "fio2_chart"], "admissionid")
    table = loaded["fio2_chart"]
    frame = table.data.rename(columns={table.index_column: "charttime", "admissionid": "stay_id"})
    binding = build_export_file_metadata_binding(
        relative_path="respiratory.parquet", module="respiratory", frame=frame,
        concept_ids=["fio2_chart"], database="aumc", database_class_prefixes=(),
        dictionary=_dictionary(),
    )
    assert _values(loaded["fio2_chart"], "fio2_chart") == {0.: 45., 1.: 55.}
    assert _values(loaded["fio2"], "fio2") == _values(table, "fio2_chart")
    for i, (item, name) in enumerate(zip(AUMC_FIO2_ITEMS, names)):
        assert _values(table, name) == {0.: 30. + 5 * i, 1.: 70. - 5 * i}
        metadata = binding.columns[name].metadata.to_dict()
        assert metadata["canonical_unit"] == "%"
        assert metadata["source_lineage"][0]["item_ids"] == [item]
        params = metadata["source_lineage"][0]["semantic_parameters"]
        assert params["chart_channel"] == "unknown"
        assert params["chart_channel_reason"] == "device_channel_definition_not_verified"
        assert metadata["source_concept"] == "fio2_chart"
        assert metadata["extraction_bounds"] == _metadata("fio2_chart", "aumc").to_dict()["extraction_bounds"]
    # Neither source ordering nor the magnitudes assign a set/analyzed role.
    assert _dictionary().get("fio2_chart").sources["aumc"][0].params["chart_channel"] == "unknown"
    for merge, r_compatible in [(False, True), (True, False), (True, True)]:
        projected = ConceptResolver(_dictionary()).load_concepts(
            ["fio2_chart"], source, merge=merge, r_compatible=r_compatible,
            interval=pd.Timedelta(hours=1), patient_ids={"admissionid": [1]},
            concept_workers=1, verbose=False,
        )
        for name in ["fio2_chart", *names]:
            np.testing.assert_allclose(projected[name].dropna(), table.data[name].dropna())
    merged = ConceptResolver(_dictionary()).load_concepts(
        ["cai", "cai_ph74", "fio2_chart"], source, merge=True, r_compatible=True,
        interval=pd.Timedelta(hours=1), patient_ids={"admissionid": [1]},
        concept_workers=1, verbose=False,
    )
    for name in ["fio2_chart", *names]:
        np.testing.assert_allclose(merged[name].dropna(), table.data[name].dropna())
    from easyicu.api import extraction

    monkeypatch.setitem(extraction.EXTRACT_MODULES, "respiratory", ["fio2", "fio2_chart"])
    monkeypatch.setitem(extraction.EXTRACT_MODULES, "outcome", ["los_icu"])
    result = extraction.extract_database(
        "aumc", data_path=str(source.base_path), output_dir=str(tmp_path / "out"),
        modules=["respiratory", "outcome"], patient_ids={"admissionid": [1]},
        native_export_v2=True, verbose=False,
    )
    assert not result["modules"]["respiratory"].get("errors")
    published = pd.read_parquet(tmp_path / "out/respiratory.parquet")
    for name in ["fio2_chart", *names]:
        np.testing.assert_allclose(published[name].dropna(), table.data[name].dropna())


@pytest.mark.parametrize("database", ["eicu", "eicu_demo"])
@pytest.mark.parametrize("layout", ["flat", "bucket"])
def test_R6_F021_peak_pressure_measurements_keep_alarm_and_lab_exclusions(tmp_path, database, layout):
    respiratory = pd.DataFrame([
        (5, "Peak Pressure", "25"), (15, "Peak Insp. Pressure", "35"),
        (25, "Peak Pressure Alarm", "90"), (65, "Peak Pressure", "20"),
        (65, "Plateau Pressure", "23"), (125, "Peak Pressure", "0"),
        (185, "Peak Pressure", "101"), (245, "Peak Pressure", "error"),
        (305, "Peak Pressure High", "50"), (365, "Peak Pressure", None),
    ], columns=["respchartoffset", "respchartvaluelabel", "respchartvalue"])
    lab = pd.DataFrame({"labresultoffset": [425, 485, 545, 605],
                        "labname": "Peak Airway/Pressure", "labresult": [5., 6., 7., 30.],
                        "labmeasurenameinterface": ["cm h20", "cm/h2o", "cwp", "cmH2O"]})
    source = _eicu_source(tmp_path, database, respiratorycharting=respiratory, lab=lab)
    if layout == "bucket":
        _bucket_tables(tmp_path, ["respiratorycharting", "lab"])
        source = _source(tmp_path, database)
    loaded = _load(source, ["pip", "plateau_pres", "plateau_pres_raw"], "patientunitstayid")
    assert _values(loaded["pip"], "pip") == {0.: 30., 1.: 20., 2.: 0., 10.: 30.}
    assert _values(loaded["plateau_pres"], "plateau_pres") == {}
    assert _values(loaded["plateau_pres_raw"], "plateau_pres_raw") == {1.: 23.}
    _assert_single_loads(source, loaded, ["pip"], "patientunitstayid")


@pytest.mark.parametrize("database", ["mimic", "mimic_demo"])
@pytest.mark.parametrize("layout", ["flat", "bucket"])
def test_R6_F022_nephrostomy_volume_nets_irrigation_without_drain_or_rate(tmp_path, database, layout):
    rows = [(5, 41184, 80.), (5, 41184, 80.), (5, 40055, 80.),
            (5, 227488, 40.), (5, 227489, 50.), (65, 41184, 90.),
            (125, 40086, 900.), (185, 41868, 800.), (245, 99999, 600.),
            (1505, 41184, None)]
    rows += [(5 + 60 * hour, 41184, 0.) for hour in range(2, 25)]
    frame = pd.DataFrame(rows, columns=["minute", "itemid", "value"])
    frame["charttime"] = ADMISSION + pd.to_timedelta(frame.pop("minute"), unit="m")
    frame["icustay_id"], frame["subject_id"], frame["hadm_id"] = 1, 11, 21
    frame["valueuom"] = "mL"
    _write(frame, tmp_path / "outputevents/part.parquet")
    _write(pd.DataFrame({"icustay_id": [1], "subject_id": [11], "hadm_id": [21],
                         "intime": [ADMISSION], "outtime": [ADMISSION + pd.Timedelta(days=2)]}),
           tmp_path / "icustays.parquet")
    if layout == "bucket":
        _bucket_tables(tmp_path, ["outputevents"])
    source = _source(tmp_path, database)
    loaded = _load(source, ["urine"], "icustay_id")
    expected = {0.: 170., 1.: 90., **{float(hour): 0. for hour in range(2, 25)}}
    assert _values(loaded["urine"], "urine") == expected
    # Native-time selection must use the same signed, deduplicated net amount.
    native = _load(source, ["urine"], "icustay_id", interval=None)["urine"]
    assert list(_values(native, "urine").values()) == list(expected.values())
    from easyicu.scores.kdigo_aki import kdigo_uo

    assessed = kdigo_uo(native.data, pd.DataFrame({"icustay_id": [1], "weight": [70.]}),
                        id_col="icustay_id", time_col=native.index_column, time_unit="hours")
    last = assessed.iloc[-1]
    assert last.uo_rt_24hr_covered_h == pytest.approx(24.)
    assert last.uo_rt_24hr == pytest.approx(90. / 24. / 70.)
    assert last.aki_stage_uo == 3


@pytest.mark.parametrize("layout", ["flat", "bucket"])
def test_R6_F023_sic_enzymatic_creatinine_is_mgdl_and_excludes_hpm(tmp_path, layout):
    lab = pd.DataFrame({"id": range(8), "CaseID": 1,
                        "Offset": [7200, 7200, 9000, 10800, 14400, 18000, 21600, 25200],
                        "LaboratoryID": [367, 368, 368, 368, 369, 368, 368, 368],
                        "LaboratoryValue": [1., 1., 3., 4., 10., 26., None, -1.]})
    _write(lab, tmp_path / "laboratory/part.parquet")
    _write(pd.DataFrame({"CaseID": [1], "PatientID": [11], "ICUOffset": [0],
                         "TimeOfStay": [48 * 3600]}), tmp_path / "cases.parquet")
    if layout == "bucket":
        _bucket_tables(tmp_path, ["laboratory"])
    source = _source(tmp_path, "sic")
    loaded = _load(source, ["crea", "crea_raw", "kdigo_creat"], "CaseID")
    assert _values(loaded["crea"], "crea") == {2.: 2., 3.: 4.}
    single = _load(source, ["crea"], "CaseID")["crea"]
    assert _values(single, "crea") == _values(loaded["crea"], "crea")
    assert _values(loaded["crea_raw"], "crea_raw") == {2.: 2., 3.: 4., 5.: 26., 7.: -1.}
    assert _metadata("crea", "sic").canonical_unit == "mg/dL"
    staged = loaded["kdigo_creat"].data
    last = staged.dropna(subset=["crea"]).iloc[-1]
    assert last.creat_low_past_7day == 2.
    assert last.aki_stage_creat == 3


@pytest.mark.parametrize("database", ["eicu", "eicu_demo"])
@pytest.mark.parametrize("layout", ["flat", "bucket"])
def test_R6_F035_nursing_arterial_sbp_dbp_beat_cuff_with_existing_bounds(tmp_path, database, layout):
    rows = []
    for name, label, cuff, measured, high in [
        ("Systolic", "SBP", 145., 110., 301.), ("Diastolic", "DBP", 85., 60., 201.),
    ]:
        rows += [(5, f"Invasive BP {name}", f"Arterial Line {label} (mmHg)", str(measured)),
                 (15, "", f"Arterial Line {label} (mmHg)", str(measured + 10)),
                 (25, f"Non-Invasive BP {name}", "", str(cuff)),
                 (65, f"Non-Invasive BP {name}", "", str(cuff)),
                 (125, f"Invasive BP {name}", "", str(high)),
                 (125, f"Non-Invasive BP {name}", "", str(cuff)),
                 (185, f"Invasive BP {name} Alarm", "", "100"),
                 (245, f"Invasive BP {name}", "", "zeroing"),
                 (305, "", f"Arterial Line {label} (mmHg)", "0"),
                 (365, f"Invasive BP {name}", "", str(measured + 2))]
    rows += [(5, "Invasive BP Mean", "", "80"),
             (25, "Non-Invasive BP Mean", "", "95"),
             (65, "Non-Invasive BP Mean", "", "90")]
    nurse = pd.DataFrame(rows, columns=["nursingchartoffset", "nursingchartcelltypevalname",
                                       "nursingchartcelltypevallabel", "nursingchartvalue"])
    source = _eicu_source(tmp_path, database, nursecharting=nurse)
    if layout == "bucket":
        _bucket_tables(tmp_path, ["nursecharting"])
        source = _source(tmp_path, database)
    loaded = _load(source, ["sbp", "dbp", "map"], "patientunitstayid")
    assert _values(loaded["sbp"], "sbp") == {0.: 115., 1.: 145., 2.: 145., 6.: 112.}
    assert _values(loaded["dbp"], "dbp") == {0.: 65., 1.: 85., 2.: 85., 6.: 62.}
    assert _values(loaded["map"], "map") == {0.: 80., 1.: 90.}
    _assert_single_loads(source, loaded, ["sbp", "dbp"], "patientunitstayid")


@pytest.mark.parametrize("clock", ["hours", "timedelta", "datetime"])
@pytest.mark.parametrize("set_na_max", [False, True])
def test_R6_F046_legacy_gcs_ages_each_original_component_independently(clock, set_na_max):
    def component(name, hours, values, stay=1):
        time = pd.Series(hours, dtype=float)
        if clock != "hours":
            time = pd.to_timedelta(time, unit="h")
        if clock == "datetime":
            time = ADMISSION + time
        return pd.DataFrame({"stay_id": stay, "time": time, name: values})

    eye = pd.concat([component("egcs", [0.], [1.]), component("egcs", [8.], [4.], 2)])
    verbal = component("vgcs", [0., 6., 6.001, 7., 8.], [2., 3., np.nan, 4., np.nan])
    motor = component("mgcs", [0.], [3.])
    result = gcs(eye, verbal, motor, set_na_max=set_na_max).reset_index(drop=True)
    one = result.loc[result.stay_id.eq(1), "gcs"].to_numpy()
    expected = [6., 7., 13., 14., 14.] if set_na_max else [6., 7., np.nan, np.nan, np.nan]
    np.testing.assert_allclose(one, expected, equal_nan=True)
    # A recorded total expires separately from the asynchronously refreshed parts.
    total = component("tgcs", [0.], [5.])
    timed = gcs(eye, verbal, motor, tgcs=total, set_na_max=set_na_max)
    got = timed.loc[timed.stay_id.eq(1), "gcs"].to_numpy()
    np.testing.assert_allclose(got, [5., 5., *expected[2:]], equal_nan=True)
    short = gcs(eye, verbal, motor, valid_win=pd.Timedelta(hours=1), set_na_max=False)
    assert short.loc[short.stay_id.eq(1), "gcs"].iloc[1:].isna().all()


@pytest.mark.parametrize("layout", ["flat", "bucket"])
def test_R6_F047_hirid_insulin_is_conserved_hour_bin_amount_with_honest_metadata(tmp_path, layout):
    root = _source_root(tmp_path, "pharma", [
        {"patientid": 1, "givenat": t, "pharmaid": item, "givendose": dose,
         "doseunit": "U", "route": "iv", "infusionid": 7, "recordstatus": 0}
        for t, item, dose in [(.1, 15, 2.), (.4, 15, 3.), (.7, 1000724, 4.), (1.1, 15, 5.)]
    ], layout=layout)
    loaded = _load(_source(root, "hirid"), ["ins"], "patientid")
    assert _values(loaded["ins"], "ins") == {0.: 9., 1.: 5.}
    metadata = _metadata("ins", "hirid").to_dict()
    assert metadata["canonical_unit"] == "units/hr"
    assert "hour bin" in metadata["description"].lower()
    assert "not an instantaneous" in metadata["description"].lower()
    params = metadata["source_lineage"][0]["semantic_parameters"]
    assert params["time_basis"] == "one_hour_bin_total"
    assert params["quantity_kind"] == "administered_amount"
    assert params["continuous_iv_rate"] is False
