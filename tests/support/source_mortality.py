"""Small invented ICU source tables for source semantics contracts."""
from __future__ import annotations

import pandas as pd


def source(root, database, *, upper=False):
    """Four invented stays: ICU death, later death, survivor, unknown status.

    ``upper`` writes MIMIC-III's upper-case column headers.
    """
    root.mkdir()
    if database == "hirid":
        tables = {"general": {"patientid": [1, 2, 3, 4],
            "age": [45, 65, 75, 90],
            "admissiontime": ["2020-01-01"] * 4,
            "discharge_status": ["dead", "alive", "alive", None]},
            "observations": {"patientid": [1, 1, 1, 2], "variableid": [110, 200, 999, 110],
                "datetime": pd.to_datetime(["2020-01-02", "2020-01-03", "2020-02-01", "2020-01-02"]),
                "value": [70., 80., 1., 65.]}}
    elif database == "aumc":
        tables = {"admissions": {"admissionid": [1, 2, 3, 4], "patientid": [11, 12, 13, 14],
            "admittedat": [100000000] * 4, "dischargedat": [200000000] * 4,
            "dateofdeath": [200000000, 210000000, None, None],
            "agegroup": ["18-39", "60-69", "70-79", "80+"],
            "weightgroup": ["59-", "60-69", "70-79", "100-109"],
            "destination": ["Overleden", "Verpleegafdeling", "Huis", None]}}
    elif database == "sic":
        tables = {"cases": {"CaseID": [1, 2, 3, 4], "PatientID": [11, 12, 13, 14],
            "AgeOnAdmission": [45, 65, 75, 90],
            "ICUOffset": [3600] * 4, "TimeOfStay": [90000] * 4,
            "OffsetOfDeath": [90000., 90001., None, None],
            "HospitalDischargeType": [2028, 2028, 2027, None]}}
    elif database == "eicu":
        tables = {"patient": {"patientunitstayid": [1, 2, 3, 4],
            "age": ["45", "65", "75", "> 89"],
            "unitdischargestatus": ["Expired", "Alive", "Alive", None],
            "unitdischargeoffset": [1440] * 4,
            "hospitaldischargestatus": ["Expired", "Expired", "Alive", None]}}
    else:
        identity = "stay_id" if database == "miiv" else "icustay_id"
        tables = {"icustays": {identity: [1, 2, 3, 4], "hadm_id": [11, 12, 13, 14], "subject_id": [101, 102, 103, 104],
            "intime": ["2020-01-01"] * 4, "outtime": ["2020-01-02"] * 4},
            "admissions": {"hadm_id": [11, 12, 13, 14], "subject_id": [101, 102, 103, 104],
                "admittime": ["2020-01-01"] * 4, "dischtime": ["2020-01-03"] * 4,
                "hospital_expire_flag": [1, 1, 0, None],
                "deathtime": ["2020-01-02", "2020-01-03", None, None]},
            "patients": {"subject_id": [101, 102, 103, 104],
                **({"anchor_age": [45, 65, 75, 90], "anchor_year": [2020] * 4} if database == "miiv" else
                   {"dob": ["1975-01-01", "1955-01-01", "1945-01-01", "1930-01-01"]})}}
    for name, rows in tables.items():
        path = root / f"{name}.parquet"
        if database == "hirid" and name == "observations":
            path = root / "observations" / "1.parquet"
            path.parent.mkdir()
        frame = pd.DataFrame(rows)
        if upper:
            frame.columns = [column.upper() for column in frame.columns]
        frame.to_parquet(path, index=False)
    return root
