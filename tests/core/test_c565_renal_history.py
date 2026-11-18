"""Synthetic diagnosis-history windows, nullable absence and opt-in registration."""

import hashlib
import json

import pandas as pd
import pytest

from easyicu.api import EXTRACT_MODULE_ORDER, EXTRACT_MODULES
from easyicu.api.concepts import clear_global_loader, load_concepts
from easyicu.concept.c565_renal_history import FIELDS
from tests.support.bucket_storage import write_contract
from tests.support.source_mortality import source as mortality_source


@pytest.fixture(autouse=True)
def clear_source_cache():
    yield
    clear_global_loader()


def _write(root, table, records, columns):
    pd.DataFrame(records, columns=columns).to_parquet(root / f"{table}.parquet", index=False)


def _load(root, database, stays, **kwargs):
    return load_concepts(
        list(FIELDS), database=database, data_path=root, patient_ids=stays,
        r_compatible=False, concept_workers=1, parallel_workers=1, **kwargs,
    )


def test_miiv_version_split_and_index_admission_window(tmp_path):
    start = pd.Timestamp("2023-05-01")
    past = start - pd.Timedelta(hours=1)
    future = start + pd.Timedelta(seconds=1)
    _write(tmp_path, "icustays", [
        (1, 10, 101), (2, 20, 102), (3, 30, 103), (4, 40, 104), (5, 50, 105), (6, 60, 106),
    ], ["stay_id", "subject_id", "hadm_id"])
    _write(tmp_path, "admissions", [
        (101, 10, start), (111, 10, past), (112, 10, start),
        (102, 20, start), (122, 20, future),
        (103, 30, start), (104, 40, start), (105, 50, start), (106, 60, start),
    ], ["hadm_id", "subject_id", "admittime"])
    _write(tmp_path, "diagnoses_icd", [
        (10, 111, 1, "z99.2", 10), (10, 112, 1, "t86.199", 10),
        (20, 122, 1, "5856", 9), (20, 122, 2, "Z940", 10),
        (30, 103, 1, "N186", 9), (30, 103, 2, "T8610", 9),
        (30, 103, 3, "585.6", 10), (30, 103, 4, "V42.0", 10),
        (40, 104, 1, "I13.11", 10), (40, 104, 2, "V420", 9),
        (50, 105, 1, "N186", 10),
    ], ["subject_id", "hadm_id", "seq_num", "icd_code", "icd_version"])
    result = _load(tmp_path, "miiv", {"stay_id": [1, 2, 3, 4, 5, 6]}).set_index("stay_id")
    assert "charttime" not in result
    assert result[list(FIELDS)].values.tolist() == [
        [True, True, False, False, True],
        [False, False, False, False, False],
        [False, False, False, False, False],
        [True, True, False, False, True],
        [True, False, False, False, True],
        [False, False, False, False, False],
    ]
    assert _load(tmp_path, "miiv", [1]).set_index("stay_id").loc[1, list(FIELDS)].tolist() == result.loc[1, list(FIELDS)].tolist()


@pytest.mark.parametrize("code", ["Z91.15", "I120", "I13.11", "I132"])
def test_miiv_all_four_v2_additions(tmp_path, code):
    _write(tmp_path, "icustays", [(1, 10, 101)], ["stay_id", "subject_id", "hadm_id"])
    _write(tmp_path, "admissions", [(101, 10, pd.Timestamp("2023-05-01"))],
           ["hadm_id", "subject_id", "admittime"])
    _write(tmp_path, "diagnoses_icd", [(101, code, 10)], ["hadm_id", "icd_code", "icd_version"])
    result = _load(tmp_path, "miiv", [1])
    assert result[list(FIELDS)].values.tolist() == [[True, False, False, False, True]]


def test_eicu_all_sibling_units_contribute_codes_and_past_history(tmp_path):
    _write(tmp_path, "patient", [(1, 20), (2, 21), (3, 20), (4, 22), (5, 23), (6, 24)],
           ["patientunitstayid", "patienthealthsystemstayid"])
    _write(tmp_path, "diagnosis", [
        (3, " A000, z99.2 , T86-1.1 "),
        (2, "Z94.1"), (2, None), (4, "i13.2"),
        (5, "V45.12, 99681"),
    ], ["patientunitstayid", "icd9code"])
    _write(tmp_path, "pasthistory", [
        (3, "renal failure - Hemodialysis"), (3, "history / S/P renal transplant"),
        (4, "RENAL FAILURE - PERITONEAL DIALYSIS"), (2, None),
    ], ["patientunitstayid", "pasthistorypath"])
    result = _load(tmp_path, "eicu", [1, 2, 4, 5, 6]).set_index("patientunitstayid")
    assert result[list(FIELDS)].values.tolist() == [
        [True, True, True, True, True],
        [False, False, False, False, False],
        [True, False, True, False, True],
        [True, True, False, False, True],
        [False, False, False, False, False],
    ]


@pytest.mark.parametrize("database,reason", [
    ("aumc", "no_icd_diagnosis_source"),
    ("hirid", "no_icd_diagnosis_source"),
    ("sic", "no prior-history diagnosis source"),
])
def test_absent_diagnosis_history_has_typed_reason(tmp_path, database, reason):
    table, stay_column = {
        "aumc": ("admissions", "admissionid"),
        "hirid": ("general_table", "patientid"),
        "sic": ("cases", "CaseID"),
    }[database]
    _write(tmp_path, table, [(1,)], [stay_column])
    _assert_absent(tmp_path, database, reason)


@pytest.fixture(params=["raw", "bucket"], ids=["raw", "bucket-root"])
def carevue_view(tmp_path, request):
    if request.param == "raw":
        root = tmp_path / "mimicview" / "raw"
    else:
        root = tmp_path / "b6" / "mimic_cv_bkt_s123456789"
    root.mkdir(parents=True)
    return root


def _carevue_diagnoses(root, diagnoses):
    _write(root, "diagnoses_icd", [
        (row_id, 200 if hadm_id == 20 else 100, hadm_id, row_id, code)
        for row_id, (hadm_id, code) in enumerate(diagnoses, 1)
    ], ["ROW_ID", "SUBJECT_ID", "HADM_ID", "SEQ_NUM", "ICD9_CODE"])


def _carevue_history(root, stays, diagnoses, *, dbsource=True):
    start = pd.Timestamp("2007-05-01")
    _write(root, "icustays", stays, ["ICUSTAY_ID", "SUBJECT_ID", "HADM_ID"] + (["DBSOURCE"] if dbsource else []))
    admissions = [
        (10, 100, start), (11, 100, start - pd.Timedelta(hours=1)),
        (12, 100, start), (13, 100, start + pd.Timedelta(seconds=1)),
        (14, 100, start - pd.Timedelta(hours=2)), (15, 100, start - pd.Timedelta(hours=3)),
        (20, 200, start),
    ]
    _write(root, "admissions", [
        (row_id, subject_id, hadm_id, admit, admit + pd.Timedelta(days=2))
        for row_id, (hadm_id, subject_id, admit) in enumerate(admissions, 1)
    ], ["ROW_ID", "SUBJECT_ID", "HADM_ID", "ADMITTIME", "DISCHTIME"])
    _carevue_diagnoses(root, diagnoses)


@pytest.mark.parametrize("r_compatible", [False, True])
def test_carevue_era_and_admittime_window(carevue_view, r_compatible):
    _carevue_history(carevue_view, [
        (1, 100, 10, "carevue"), (11, 100, 11, "carevue"),
        (12, 100, 12, "carevue"), (13, 100, 13, "carevue"),
        (14, 100, 14, "carevue"), (140, 100, 14, "metavision"),
        (2, 200, 20, "CAREVue"),
    ], [
        (11, "585.6"), (12, "v42.0"), (13, "99681"), (14, "99681"),
        (15, "99681"), (20, "N186"), (20, "T8610"),
    ])
    availability = {}
    result = load_concepts(
        list(FIELDS), database="mimic", data_path=carevue_view, patient_ids=[1, 2, 11, 14, 140],
        r_compatible=r_compatible, concept_workers=1, parallel_workers=1, availability_sink=availability,
    ).set_index("icustay_id")
    assert set(result.index) == {1, 2, 11}
    assert result.loc[[1, 2, 11], list(FIELDS)].values.tolist() == [
        [True, True, False, False, True],    # Equal-admittime transplant is included.
        [False, False, False, False, False],  # ICD-10 codes do not count.
        [True, False, False, False, True],   # Before the equal-admittime transplant.
    ]
    assert result["prior_window"].tolist() == ["view_admissions_only"] * 3
    assert str(result["prior_window"].dtype) == "string"
    assert result["prior_window_note"].isna().all()
    for field in FIELDS:
        assert availability[field].prior_window == "view_admissions_only"
        assert availability[field].note == "EXTENSION beyond frozen C565."

    # The equally timed admission is the only eligible transplant source.
    _carevue_diagnoses(carevue_view, [
        (11, "585.6"), (13, "99681"), (14, "99681"), (15, "99681"),
    ])
    clear_global_loader()
    assert _load(carevue_view, "mimic", [1])["kidney_tx"].tolist() == [False]

    excluded = _load(carevue_view, "mimic", [14, 140])
    assert excluded.empty
    assert all(str(excluded[field].dtype) == "boolean" for field in FIELDS)


@pytest.mark.parametrize("dbsource", [True, False])
def test_carevue_reports_when_era_cannot_be_guaranteed(carevue_view, dbsource):
    stays = [(1, 100, 10), (11, 100, 11)]
    if dbsource:
        stays = [(*stay, "carevue") for stay in stays]
    _carevue_history(carevue_view, stays, [(11, "V45.12")], dbsource=dbsource)
    availability = {}
    result = _load(carevue_view, "mimic", [1], availability_sink=availability)
    assert result["esrd_dialysis"].tolist() == [True]
    assert result["prior_window"].tolist() == ["view_admissions_only"]
    assert result["prior_window_note"].tolist() == ["prior_window CareVue-era not guaranteed"]
    assert availability["esrd_dialysis"].prior_window_note == result["prior_window_note"].iloc[0]


@pytest.mark.parametrize("present", [(), ("admissions",), ("diagnoses_icd",)])
def test_carevue_renal_history_requires_both_history_tables(carevue_view, present):
    _write(carevue_view, "icustays", [(1, "carevue")], ["ICUSTAY_ID", "DBSOURCE"])
    required = ("admissions", "diagnoses_icd")
    for table in present:
        _write(carevue_view, table, [(101,)], ["HADM_ID"])
    missing = tuple(table for table in required if table not in present)
    _assert_absent(carevue_view, "mimic", "required_source_table_missing", missing)


def test_carevue_refuses_history_outside_view(tmp_path):
    root = tmp_path / "unrestricted_mimic"
    root.mkdir()
    _carevue_history(root, [(1, 100, 10, "carevue")], [(10, "585.6")])
    _assert_absent(root, "mimic", "carevue_view_required")


def _declare_carevue_view(source, view):
    """Link the view's small tables and declare their outputs on both roots."""
    tables = []
    for table in ("icustays", "admissions", "diagnoses_icd", "patients"):
        path = source / f"{table}.parquet"
        (view / path.name).symlink_to(path)
        tables.append({"table": table, "outputs": [
            {"relative_path": path.name, "size_bytes": path.stat().st_size},
        ]})
    manifest = source / "conversion_manifest.json"
    manifest.write_text(json.dumps({"tables": tables}), encoding="utf-8")
    (view / manifest.name).write_text(json.dumps({
        "tables": tables,
        "derived_from": {
            "path": str(manifest),
            "sha256": hashlib.sha256(manifest.read_bytes()).hexdigest(),
        },
    }), encoding="utf-8")


@pytest.mark.parametrize("database", ["miiv", "mimic"], ids=["miiv", "carevue"])
def test_native_export_publishes_outcome_vent_settings_and_renal_history(tmp_path, database):
    from easyicu.api.extraction import extract_database

    carevue = database == "mimic"
    root = tmp_path / "miiv"
    source = tmp_path / "mimicview" / "raw" if carevue else root
    source.parent.mkdir(parents=True, exist_ok=True)
    mortality_source(source, database, upper=carevue)
    stays = pd.read_parquet(source / "icustays.parquet")
    patients = pd.read_parquet(source / "patients.parquet")
    if carevue:
        stays = stays.assign(ROW_ID=[1, 2, 3, 4], DBSOURCE="carevue", LOS=1.0)
        patients = patients.assign(DOD=["2020-01-02", "2020-01-03", None, None])
    else:
        stays = stays.assign(los=1.0)
        patients = patients.assign(dod=["2020-01-02", "2020-01-03", None, None])
    stays.to_parquet(source / "icustays.parquet", index=False)
    patients.to_parquet(source / "patients.parquet", index=False)
    start = pd.Timestamp("2020-01-01 00:35")
    chart = pd.DataFrame({
        "icustay_id" if carevue else "stay_id": [1, 2],
        "charttime": [start, start],
        "itemid": [190 if carevue else 223835] * 2,
        "valuenum": [0.45, 60.0],
        "value": ["0.45", "60"],
        "valueuom": ["%", "%"],
    })
    if carevue:
        admissions = pd.read_parquet(source / "admissions.parquet")
        admissions.assign(ROW_ID=[1, 2, 3, 4]).to_parquet(source / "admissions.parquet", index=False)
        _write(source, "diagnoses_icd", [(1, 101, 11, 1, "V45.12")],
               ["ROW_ID", "SUBJECT_ID", "HADM_ID", "SEQ_NUM", "ICD9_CODE"])
        root = tmp_path / "b6" / "mimic_cv_bkt_s123456789"
        root.mkdir(parents=True)
        _declare_carevue_view(source, root)
        bucket = root / "chartevents_bucket"
        part = bucket / "bucket_id=0" / "part.parquet"
        part.parent.mkdir(parents=True)
        chart.assign(error=0).rename(columns=str.upper).to_parquet(part, index=False)
        write_contract(bucket, key="ITEMID")
    else:
        _write(source, "diagnoses_icd", [(101, 11, 1, "N18.6", 10)],
               ["subject_id", "hadm_id", "seq_num", "icd_code", "icd_version"])
        chart.to_parquet(source / "chartevents.parquet", index=False)

    output = tmp_path / "output"
    modules = ["outcome", "vent_settings", "c565_renal_history"]
    result = extract_database(
        database=database, data_path=root, output_dir=output, modules=modules,
        patient_ids=[1, 2], batch_size=1, resource_budget_mb=2048,
        native_export_v2=True, verbose=False,
    )
    assert not {name: module["errors"] for name, module in result["modules"].items() if module.get("errors")}
    manifest = json.loads((output / "_manifest.json").read_text(encoding="utf-8"))
    assert {member["module"] for member in manifest["files"]} == set(modules)
    outcome = pd.read_parquet(output / "outcome.parquet").set_index("stay_id")
    assert outcome["icu_death"].to_dict() == {1: True, 2: False}
    vent = pd.read_parquet(output / "vent_settings.parquet").set_index(["stay_id", "charttime"])
    assert vent["fio2_set"].to_dict() == {(1, 0.0): 45.0, (2, 0.0): 60.0}
    renal = pd.read_parquet(output / "c565_renal_history.parquet").set_index("stay_id")
    assert renal.loc[[1, 2], list(FIELDS)].values.tolist() == [
        [True, False, False, False, True], [False, False, False, False, False],
    ]
    if carevue:
        assert renal["prior_window"].tolist() == ["view_admissions_only"] * 2
        assert renal["prior_window_note"].tolist() == ["prior_window CareVue-era not guaranteed"] * 2


def test_missing_miiv_diagnosis_table_is_not_a_negative(tmp_path):
    _write(tmp_path, "icustays", [(1, 10, 101)], ["stay_id", "subject_id", "hadm_id"])
    _write(tmp_path, "admissions", [(101, 10, pd.Timestamp("2023-05-01"))],
           ["hadm_id", "subject_id", "admittime"])
    _assert_absent(tmp_path, "miiv", "required_source_table_missing", ("diagnoses_icd",))


def _assert_absent(root, database, reason, missing=()):
    availability = {}
    result = _load(root, database, [1], availability_sink=availability)
    assert result.empty
    assert all(str(result[column].dtype) == "boolean" for column in FIELDS)
    for field in FIELDS:
        record = availability[field]
        assert record.reason_code == reason
        assert record.evaluation_status == "not_evaluable"
        assert record.missing_tables == missing


def test_flags_have_a_stay_level_opt_in_module_only():
    assert EXTRACT_MODULES["c565_renal_history"] == list(FIELDS)
    assert "c565_renal_history" not in EXTRACT_MODULE_ORDER
    assert not any(set(FIELDS).intersection(concepts) for module, concepts in EXTRACT_MODULES.items()
                   if module != "c565_renal_history")


def test_static_module_can_publish_without_longitudinal_stay_bounds(tmp_path):
    from easyicu.api.extraction import extract_database
    from easyicu.research_agent.intake.export_package import open_export_package

    source = tmp_path / "source"
    source.mkdir()
    _write(source, "icustays", [(1, 10, 101), (2, 20, 102)], ["stay_id", "subject_id", "hadm_id"])
    start = pd.Timestamp("2023-05-01")
    _write(source, "admissions", [(101, 10, start), (102, 20, start)],
           ["hadm_id", "subject_id", "admittime"])
    _write(source, "diagnoses_icd", [(101, "N18.6", 10)], ["hadm_id", "icd_code", "icd_version"])
    output = tmp_path / "output"
    module = "c565_renal_history"
    extract_database(
        database="miiv", data_path=source, output_dir=output, modules=[module],
        patient_ids=[1, 2], batch_size=1, resource_budget_mb=2048,
        native_export_v2=True, verbose=False,
    )
    frame = pd.read_parquet(output / f"{module}.parquet").set_index("stay_id")
    assert frame[list(FIELDS)].values.tolist() == [
        [True, False, False, False, True], [False, False, False, False, False],
    ]
    with open_export_package(output) as package:
        assert set(FIELDS) <= set(package.concept_index)
