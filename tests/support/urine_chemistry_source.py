"""Invented urine chemistry and SICdb precondition tables, never patient data."""

from pathlib import Path

import pandas as pd

from easyicu.config import load_src_cfg
from easyicu.datasource import ICUDataSource
from tests.support.bucket_storage import write_fixture_parquet


def urine_chemistry_source(root: Path, database: str, *, bucketed: bool):
    """Write urine/serum controls, invalid units, bounds and ICU-window controls."""
    mimic = database in {"miiv", "mimic", "mimic_demo"}
    specs = [
        ("urine_na", 51100, "urinary sodium", 40., 400., "mmol/L", 50983, "sodium", 140.),
        ("urine_crea", 51082, "urinary creatinine", 120., 1000., "mg/dL", 50912, "creatinine", 1.),
    ]
    if mimic:
        specs += [
            ("urine_osm", 51093, None, 600., 2000., "mOsm/kg", 50964, None, 290.),
            ("urine_urea", 51104, None, 900., 5000., "mg/dL", 51006, None, 20.),
        ]
    rows = []
    expected = {}
    for name, item, label, value, maximum, unit, serum_item, serum_label, serum in specs:
        expected[name] = {0.: value, 1.: 0., 2.: maximum}
        for hour, result, recorded_unit in [
            (.25, value, unit), (1.25, 0., unit), (2.25, maximum, unit),
            (3.25, -1., unit), (4.25, maximum + 1, unit), (5.25, None, unit),
            (6.25, value, "mOsm/L" if name == "urine_osm" else "mg/24h"),
            (-25., value, unit), (73., value, unit),
        ]:
            rows.append((hour, item if mimic else label, result, recorded_unit))
        # These are deliberately in the same hour as the urine measurement.
        rows.append((.25, serum_item if mimic else serum_label, serum, unit))
    frame = pd.DataFrame(rows, columns=["hour", "itemid" if mimic else "labname", "value", "unit"])
    if mimic:
        admission = pd.Timestamp("2150-01-01")
        stay = "stay_id" if database == "miiv" else "icustay_id"
        pd.DataFrame({stay: [1], "subject_id": [11], "hadm_id": [21], "intime": [admission],
                      "outtime": [admission + pd.Timedelta(hours=48)]}).to_parquet(root / "icustays.parquet")
        frame["charttime"] = admission + pd.to_timedelta(frame.pop("hour"), unit="h")
        frame["storetime"] = frame.charttime
        frame["valuenum"] = frame.value
        frame["value"] = frame.value.astype(str)
        frame = frame.rename(columns={"unit": "valueuom"})
        frame["subject_id"], frame["hadm_id"] = 11, 21
        table = "labevents"
    else:
        pd.DataFrame({"patientunitstayid": [1], "uniquepid": [11], "patienthealthsystemstayid": [21],
                      "hospitaladmitoffset": [0], "unitdischargeoffset": [48 * 60],
                      "hospitaldischargeoffset": [48 * 60]}).to_parquet(root / "patient.parquet")
        frame["labresultoffset"] = frame.pop("hour") * 60
        frame["labresultrevisedoffset"] = frame.labresultoffset
        frame["labresulttext"] = frame.value.astype(str)
        frame = frame.rename(columns={"value": "labresult", "unit": "labmeasurenameinterface"})
        frame["labmeasurenamesystem"] = frame.labmeasurenameinterface
        frame["patientunitstayid"] = 1
        table = "lab"
    path = root / f"{table}.parquet"
    if bucketed:
        path = root / f"{table}_bucket" / "bucket_id=0" / "part.parquet"
        path.parent.mkdir(parents=True)
    write_fixture_parquet(frame, path, index=False)
    return ICUDataSource(load_src_cfg(database), base_path=root, enable_cache=False), expected


def sic_precondition_source(root: Path):
    """Reference identifiers are invented: only the reference labels define Yes/No."""
    pd.DataFrame({"CaseID": range(1, 8), "PatientID": range(11, 18), "ICUOffset": 1800,
                  "TimeOfStay": 180000}).to_parquet(root / "cases.parquet")
    pd.DataFrame({
        "CaseID": [1, 2, 3, 4, 5, 5, 6, 7],
        "FieldID": [3104, 3104, 3104, 3104, 3104, 3104, 3099, 3104],
        "RefID": [90001, 90002, 90003, None, 90001, 90002, 90001, 90004],
    }).to_parquet(root / "data_ref.parquet")
    pd.DataFrame({"ReferenceGlobalID": [90001, 90002, 90003, 90004, 90004, None],
                  "ReferenceValue": [" Yes ", "nO", "Unknown", "Yes", "No", "Yes"]}).to_parquet(
        root / "d_references.parquet")
    return ICUDataSource(load_src_cfg("sic"), base_path=root, enable_cache=False)


def urine_module_source(root: Path, database: str):
    """Small complete converted schemas with urine-only observation hours."""
    from easyicu.synthetic_sources import generate_source
    from tests.support.renal_export import write_renal_source

    if database in {"aumc", "hirid", "sic"}:
        generate_source(root, database, stays=3)
        if database == "sic":
            path = root / "data_ref.parquet"
            frame = pd.read_parquet(path)
            frame.loc[frame.CaseID.eq(3), "RefID"] = 90003
            frame.to_parquet(path, index=False)
        return
    write_renal_source(root, "eicu" if database == "eicu" else "miiv", 3,
                       hours=6, urine_rows_per_hour=1)
    if database == "eicu":
        path = root / "patient.parquet"
        frame = pd.read_parquet(path).assign(unitdischargelocation="Home", unitvisitnumber=1,
                                           admissionheight=175.)
    else:
        path = root / "admissions.parquet"
        frame = pd.read_parquet(path).assign(deathtime=pd.NaT)
    frame.to_parquet(path, index=False)
    if database != "eicu":
        pd.DataFrame({"subject_id": [4, 5, 6], "dod": pd.NaT,
                      "dob": pd.Timestamp("2135-01-01"), "gender": "M",
                      "anchor_age": 65, "anchor_year": 2200}).to_parquet(root / "patients.parquet", index=False)
        path = root / "icustays.parquet"
        pd.read_parquet(path).assign(dbsource="metavision").to_parquet(path, index=False)
    control = root.parent / "urine-controls"
    control.mkdir()
    urine_chemistry_source(control, database, bucketed=False)
    table = "lab" if database == "eicu" else "labevents"
    labs = pd.read_parquet(control / f"{table}.parquet")
    if database != "eicu":
        labs["subject_id"], labs["hadm_id"] = 4, 7
        for column in ("charttime", "storetime"):
            labs[column] = labs[column] - pd.Timestamp("2150-01-01") + pd.Timestamp("2200-01-01")
    path = root / f"{table}.parquet"
    pd.concat([pd.read_parquet(path), labs], ignore_index=True).to_parquet(path, index=False)
    if database == "mimic":
        for path in root.glob("*.parquet"):
            frame = pd.read_parquet(path).rename(columns={"stay_id": "icustay_id"})
            frame.to_parquet(path, index=False)
        types = {"integer": "int64", "numeric": "float64", "double": "float64",
                 "character": "object", "logical": "bool", "POSIXct": "datetime64[ns]"}
        for name, config in load_src_cfg(database).tables.items():
            path = root / f"{name}.parquet"
            if not path.exists():
                pd.DataFrame({column: pd.Series(dtype=types.get(str(kind), "object"))
                              for column, kind in (config.columns or {}).items()}).to_parquet(path, index=False)
