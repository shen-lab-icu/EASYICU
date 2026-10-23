"""Bounded, on-disk raw-source fixtures for renal/SOFA-2 parity checks.

No source data or source paths are accepted. Each batch contains at most 256
invented stays; the dense intake/output table is never assembled in memory.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from easyicu.config import load_src_cfg


def write_renal_source(
    root: Path, database: str, stays: int, *, hours: int = 48, urine_rows_per_hour: int = 4,
) -> dict[str, int]:
    """Write long-tailed LOS and irregular renal observations to source parquet.

    ``hours`` is the median LOS, not a fixed stay duration. Some stays have no
    urine or one row; a minority have thousands. This is a parity fixture, not
    a runtime model or a claim about the distribution of clinical source data.
    """
    if database not in {"eicu", "miiv"} or stays < 1 or hours < 1 or urine_rows_per_hour < 1:
        raise ValueError("Expected eicu/miiv and positive fixture dimensions")
    root.mkdir(parents=True, exist_ok=True)
    config = load_src_cfg(database)
    writers = {}
    counts: dict[str, int] = {}
    rng = np.random.default_rng(7301)

    def emit(name, frame):
        table = pa.Table.from_pandas(frame, preserve_index=False)
        if name not in writers:
            writers[name] = pq.ParquetWriter(root / f"{name}.parquet", table.schema)
        writers[name].write_table(table)
        counts[name] = counts.get(name, 0) + len(frame)

    try:
        for first in range(1, stays + 1, 256):
            ids = np.arange(first, min(stays + 1, first + 256), dtype=np.int64)
            n = len(ids)
            weight = 60.0 + ids % 31
            los = np.clip(np.ceil(rng.lognormal(np.log(hours), 1.35, n)), 1, 24 * 365).astype(np.int64)
            los[ids % 29 == 0] = min(hours * 32, 24 * 365)
            if database == "eicu":
                emit("patient", pd.DataFrame({
                    "patientunitstayid": ids, "patienthealthsystemstayid": ids + stays,
                    "uniquepid": ids.astype(str), "unitadmitoffset": 0.0,
                    "hospitaladmitoffset": -120.0, "unitdischargeoffset": los.astype(float) * 60,
                    "hospitaldischargeoffset": (los + 24).astype(float) * 60,
                    "admissionweight": weight, "age": "65", "gender": "Male",
                    "unitdischargestatus": "Alive", "hospitaldischargestatus": "Alive",
                }))
            else:
                emit("icustays", pd.DataFrame({
                    "stay_id": ids, "subject_id": ids + stays, "hadm_id": ids + 2 * stays,
                    "intime": pd.Timestamp("2200-01-01"),
                    "outtime": pd.Timestamp("2200-01-01") + pd.to_timedelta(los, unit="h"),
                    "los": los / 24, "first_careunit": "MICU", "last_careunit": "MICU",
                }))
                emit("admissions", pd.DataFrame({
                    "subject_id": ids + stays, "hadm_id": ids + 2 * stays,
                    "admittime": pd.Timestamp("2199-12-31"),
                    "dischtime": pd.Timestamp("2200-01-01") + pd.to_timedelta(los + 24, unit="h"), "hospital_expire_flag": 0,
                }))

            per_stay = np.maximum(1, np.rint(los * urine_rows_per_hour * rng.uniform(0.65, 1.35, n)).astype(int))
            per_stay[ids % 17 == 0] = 0
            per_stay[ids % 13 == 0] = 1
            clocks = [np.sort(rng.uniform(0, length, count)) for length, count in zip(los, per_stay)]
            clocks = [clock[~((clock > 12) & (clock < 60))] if stay % 23 == 0 else clock for stay, clock in zip(ids, clocks)]
            uid = np.repeat(ids, [len(clock) for clock in clocks])
            hour = np.concatenate(clocks)
            volume = np.where((uid % 7 == 0) & (hour > 12), 8.0, 70.0) / urine_rows_per_hour
            if database == "eicu":
                emit("intakeoutput", pd.DataFrame({
                    "patientunitstayid": uid, "intakeoutputoffset": hour * 60,
                    "cellpath": "flowsheet|Flowsheet Cell Labels|I&O|Output (ml)|Urine",
                    "celllabel": "Urine", "cellvaluenumeric": volume,
                    "intakeoutputentryoffset": hour * 60, "cellvaluetext": volume.astype(str),
                }))
                dialysis = ids % 19 == 0
                rid = ids[dialysis]
                rrt_hours = np.minimum(24.0, los[dialysis] - 1.0)
                emit("treatment", pd.DataFrame({
                    "patientunitstayid": rid, "treatmentoffset": rrt_hours * 60,
                    "treatmentstring": np.full(len(rid), "renal|dialysis|hemodialysis", dtype=object),
                }))
            else:
                emit("outputevents", pd.DataFrame({
                    "stay_id": uid, "subject_id": uid + stays, "hadm_id": uid + 2 * stays,
                    "charttime": pd.Timestamp("2200-01-01") + pd.to_timedelta(hour, unit="h"),
                    "itemid": 226559, "value": volume, "valueuom": "mL",
                }))

            lab_names = ["creatinine", "platelets x 1000", "total bilirubin", "potassium", "pH", "bicarbonate", "paO2"]
            lab_ids = [50912, 51265, 50885, 50971, 50820, 50882, 50821]
            lab_values = [1.0, 200.0, 1.0, 4.5, 7.4, 24.0, 100.0]
            lab_hours = [np.repeat(np.arange(0, length, 12), len(lab_names)) for length in los]
            lid = np.repeat(ids, [len(clock) for clock in lab_hours])
            lh = np.concatenate(lab_hours)
            vals = np.tile(lab_values, len(lid) // len(lab_names))
            if database == "eicu":
                emit("lab", pd.DataFrame({
                    "patientunitstayid": lid, "labresultoffset": lh.astype(float) * 60,
                    "labname": np.tile(lab_names, len(lid) // len(lab_names)), "labresult": vals,
                    "labmeasurenameinterface": np.tile(["mg/dL", "K/uL", "mg/dL", "mmol/L", "", "mmol/L", "mmHg"], len(lid) // len(lab_names)),
                }))
                emit("vitalperiodic", pd.DataFrame({
                    "patientunitstayid": np.repeat(ids, los),
                    "observationoffset": np.concatenate([np.arange(length) for length in los]).astype(float) * 60,
                    "systemicmean": 75.0, "sao2": 98.0,
                }))
            else:
                emit("labevents", pd.DataFrame({
                    "subject_id": lid + stays, "hadm_id": lid + 2 * stays,
                    "charttime": pd.Timestamp("2200-01-01") + pd.to_timedelta(lh, unit="h"),
                    "itemid": np.tile(lab_ids, len(lid) // len(lab_ids)), "valuenum": vals,
                    "valueuom": np.tile(["mg/dL", "K/uL", "mg/dL", "mmol/L", "", "mmol/L", "mmHg"], len(lid) // len(lab_ids)),
                }))
                chart_ids = [226512, 220052, 220739, 223900, 223901, 223835]
                cid = np.repeat(ids, len(chart_ids))
                emit("chartevents", pd.DataFrame({
                    "stay_id": cid, "subject_id": cid + stays, "hadm_id": cid + 2 * stays,
                    "charttime": pd.Timestamp("2200-01-01"),
                    "itemid": np.tile(chart_ids, n),
                    "valuenum": np.column_stack([weight, np.full((n, 5), [75, 4, 5, 6, 21])]).ravel(),
                    "valueuom": np.tile(["kg", "mmHg", "", "", "", "%"], n),
                }))
                dialysis = ids % 19 == 0
                rid = ids[dialysis]
                rrt_hours = np.minimum(24.0, los[dialysis] - 1.0)
                emit("procedureevents", pd.DataFrame({
                    "stay_id": rid, "subject_id": rid + stays, "hadm_id": rid + 2 * stays,
                    "starttime": pd.Timestamp("2200-01-01") + pd.to_timedelta(rrt_hours, unit="h"),
                    "endtime": pd.Timestamp("2200-01-01") + pd.to_timedelta(rrt_hours + 4, unit="h"),
                    "itemid": np.full(len(rid), 225441), "value": np.ones(len(rid)),
                    "valueuom": np.full(len(rid), "hour", dtype=object),
                }))
    finally:
        for writer in writers.values():
            writer.close()

    # Explicit empty source tables prevent missing-file fallbacks. Use the raw
    # config's declared types so zero-row tables remain source-shaped as well.
    types = {"integer": "int64", "numeric": "float64", "double": "float64", "character": "object", "logical": "bool", "POSIXct": "datetime64[ns]"}
    for name, table in config.tables.items():
        if name in writers:
            continue
        frame = pd.DataFrame({col: pd.Series(dtype=types.get(str(kind), "object")) for col, kind in (table.columns or {}).items()})
        frame.to_parquet(root / f"{name}.parquet", index=False)
        counts[name] = 0
    return counts
