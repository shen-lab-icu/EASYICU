"""Tiny invented raw tables standing in for both public demo downloads.

All source table schemas come from the packaged configuration. Unobserved
tables have headers only; the two invented ICU stays have demographics,
outcomes and bedside observations. No downloaded or patient data is used.
"""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Any


def synthetic_demo_download(source: Any, paths: Any, job: Any) -> None:
    from easyicu.config import load_src_cfg

    cfg = load_src_cfg(source.database)
    rows = _mimic_rows() if source.database == "miiv" else _eicu_rows()
    paths.raw.mkdir(parents=True, exist_ok=True)
    for name, table in cfg.tables.items():
        columns = list(table.columns or {})
        if not columns:
            continue
        with (paths.raw / f"{name}.csv").open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=columns)
            writer.writeheader()
            writer.writerows(rows.get(name, []))


def _mimic_rows() -> dict[str, list[dict]]:
    rows: dict[str, list[dict]] = {name: [] for name in ("patients", "admissions", "icustays", "chartevents")}
    for index in range(1, 3):
        common = {"subject_id": index, "hadm_id": index + 10}
        rows["patients"].append({
            "subject_id": index, "gender": "F" if index == 1 else "M",
            "anchor_age": 40 + index, "anchor_year": 2100,
            "anchor_year_group": "2097 - 2099",
        })
        rows["admissions"].append({
            **common, "admittime": "2100-01-01 00:00:00", "dischtime": "2100-01-05 00:00:00",
            "admission_type": "URGENT", "admission_location": "EMERGENCY ROOM",
            "discharge_location": "HOME", "hospital_expire_flag": 0,
        })
        rows["icustays"].append({
            **common, "stay_id": index + 100, "first_careunit": "Medical Intensive Care Unit (MICU)",
            "last_careunit": "Medical Intensive Care Unit (MICU)", "los": 2,
            "intime": "2100-01-01 01:00:00", "outtime": "2100-01-03 01:00:00",
        })
        for item, value, unit in ((220045, 80, "bpm"), (220181, 75, "mmHg"), (220277, 97, "%"), (223762, 37, "C")):
            rows["chartevents"].append({
                **common, "stay_id": index + 100, "charttime": "2100-01-01 02:00:00",
                "storetime": "2100-01-01 02:00:00", "itemid": item,
                "value": str(value), "valuenum": value, "valueuom": unit, "warning": 0,
            })
    return rows


def _eicu_rows() -> dict[str, list[dict]]:
    rows: dict[str, list[dict]] = {name: [] for name in ("patient", "apachepatientresult", "vitalperiodic")}
    for index in range(1, 3):
        rows["patient"].append({
            "patientunitstayid": index + 100, "patienthealthsystemstayid": index + 10,
            "uniquepid": f"synthetic-{index}", "gender": "Female" if index == 1 else "Male",
            "age": str(40 + index), "hospitalid": 1, "wardid": 1, "unittype": "MICU",
            "admissionheight": 170, "admissionweight": 70, "dischargeweight": 70,
            "hospitaladmitoffset": -60, "hospitaldischargeoffset": 5760,
            "unitdischargeoffset": 2880, "unitvisitnumber": 1, "unitstaytype": "admit",
            "unitdischargestatus": "Alive", "hospitaldischargestatus": "Alive",
            "unitadmitsource": "Emergency Department", "hospitaladmitsource": "Emergency Department",
            "unitadmittime24": "01:00:00", "unitdischargetime24": "01:00:00",
            "hospitaladmittime24": "00:00:00", "hospitaldischargetime24": "00:00:00",
            "hospitaldischargeyear": 2100,
        })
        rows["apachepatientresult"].append({
            "apachepatientresultsid": index, "patientunitstayid": index + 100,
            "apacheversion": "IVa", "actualicumortality": "ALIVE",
            "actualhospitalmortality": "ALIVE", "actualiculos": 2, "actualhospitallos": 4,
        })
        rows["vitalperiodic"].append({
            "vitalperiodicid": index, "patientunitstayid": index + 100, "observationoffset": 60,
            "heartrate": 80, "systemicmean": 75, "sao2": 97, "temperature": 37,
        })
    return rows


def isolated_demo_paths(root: Path, source: Any):
    from easyicu.webserver.demo_source_contracts import DemoSourcePaths

    root = root / source.database
    return DemoSourcePaths(
        root=root, archive=root / source.archive_filename, raw=root / "raw", export=root / "export",
        extracted_marker=root / "raw" / ".extracted.json",
        converted_marker=root / "raw" / ".converted.json",
        prepared_marker=root / "export" / ".prepared.json",
    )
