"""Physical converted-table construction for the three non-demo ICU schemas."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa

from easyicu.synthetic_source_values import (
    CORE, SEED, binned, category_values, population, raw_values,
)

DATA = Path(__file__).parent / "data"
DATABASES = ("hirid", "aumc", "sic")


def definitions(database):
    configs = json.loads((DATA / "data-sources.json").read_text(encoding="utf-8"))
    concepts = json.loads((DATA / "concept-dict.json").read_text(encoding="utf-8"))
    return next(row for row in configs if row["name"] == database), concepts


def _demographics(database, p):
    death_hours = np.where(p["dead"], p["length"], np.where(p["late"], p["length"] + 240, np.nan))
    if database == "hirid":
        return {"patientid": p["stay"],
                "admissiontime": (pd.Timestamp("2020-01-01") + pd.to_timedelta(p["origin"], unit="ms")).strftime("%Y-%m-%d %H:%M:%S"),
                "age": np.minimum(90, np.round(p["age"] / 5) * 5),
                "sex": np.where(p["male"], "M", "F"),
                "discharge_status": np.where(p["dead"], "dead", "alive")}
    if database == "aumc":
        return {"admissionid": p["stay"], "patientid": p["patient"],
                "admittedat": p["origin"], "dischargedat": p["origin"] + p["length"] * 3_600_000,
                "dateofdeath": p["origin"] + death_hours * 3_600_000,
                "admissioncount": np.where(np.r_[False, p["patient"][1:] == p["patient"][:-1]], 2, 1),
                "lengthofstay": p["length"] * 60, "gender": np.where(p["male"], "Man", "Vrouw"),
                "agegroup": binned(p["age"], [18, 40, 50, 60, 70, 80, 200], ["18-39", "40-49", "50-59", "60-69", "70-79", "80+"]),
                "weightgroup": binned(p["weight"], [0, 60, 70, 80, 90, 100, 110, 500], ["59-", "60-69", "70-79", "80-89", "90-99", "100-109", "110+"]),
                "heightgroup": "170-179", "specialty": "Cardiologie", "location": "IC",
                "urgency": 1, "destination": np.where(p["dead"], "Overleden", "Verpleegafdeling"),
                "admissionyeargroup": "2010-2016"}
    origin = (6 + p["stay"] % 4 * 3) * 3600
    return {"CaseID": p["stay"], "PatientID": p["patient"], "ICUOffset": origin,
            "TimeOfStay": origin + p["length"] * 3600,
            "OffsetOfDeath": origin + death_hours * 3600,
            "HospitalDischargeType": np.where(p["dead"], 2028, 2026),
            "EstimatedSurvivalObservationTime": 3077, "Sex": np.where(p["male"], 735, 736),
            "WeightOnAdmission": np.round(p["weight"] / 5) * 5000, "HeightOnAdmission": 174.,
            "AgeOnAdmission": np.minimum(90, np.round(p["age"] / 5) * 5), "saps3": 30 + p["severity"] * 10,
            "AdmissionYear": 2020, "OffsetAfterFirstAdmission": np.where(
                np.r_[False, p["patient"][1:] == p["patient"][:-1]], 240 * 3600, 0)}


def _events(database, config, concepts, p):
    rng = np.random.default_rng(SEED + 1)
    rows = {name: [] for name in config["tables"]}
    seen = set()
    for name, definition in sorted(concepts.items(), key=lambda pair: pair[0] not in CORE):
        for source in definition.get("sources", {}).get(database, []):
            table = source.get("table")
            if table in {None, "general", "admissions", "cases"}:
                continue
            ids = source.get("ids", [None])
            item = ids[0] if isinstance(ids, list) else ids
            # One real channel per mapping, including aliases sharing a channel.
            key = (table, source.get("sub_var"), item, source.get("regex"))
            if key in seen:
                continue
            seen.add(key)
            hours = np.arange(49) if name in {"map", "urine"} else np.array([0, 6, 12, 24, 48])
            if name not in CORE:
                hours = np.array([0, 6])
            index = np.repeat(np.arange(len(p["stay"])), len(hours))
            hour = np.tile(hours, len(p["stay"]))
            if database == "hirid" and name in {"map", "hr"}:
                index = np.r_[index, np.arange(len(p["stay"]))]
                hour = np.r_[hour, p["length"]]
            count = len(index)
            defaults = config["tables"][table].get("defaults", {})
            time_col = source.get("index_var", defaults.get("index_var"))
            value_col = source.get("val_var", source.get("value_var", defaults.get("val_var", "value")))
            value = raw_values(name, definition, source, p, index, hour, rng)
            if database == "hirid" and name == "weight":
                value = np.round(value / 5) * 5
            if table in {"listitems", "processitems", "procedureorderitems"}:
                value = category_values(name, source, count, database)
            if database == "hirid" and name in {"mech_vent", "vent_mode", "vent_breath_seq"}:
                value = np.ones(count)
            record = {value_col: value}
            if source.get("sub_var"):
                record[source["sub_var"]] = item if item is not None else {
                    "ecmo": "ECMO", "mech_circ_support": "IABP", "sedated_gcs": "GCS pre sedation",
                }[name]
                if item is None:
                    record["itemid"] = 900000
            if database == "hirid":
                time = pd.Timestamp("2020-01-01") + pd.to_timedelta(p["origin"][index], unit="ms") + pd.to_timedelta(hour, unit="h")
                record.update(patientid=p["stay"][index], **{time_col: time})
                if table == "observations":
                    record.update(entertime=time, status=8, type="N", stringvalue=None)
                else:
                    unit = "IU" if name == "adh_rate" else "mg"
                    record.update(enteredentryat=time, doseunit=unit, infusionid=p["stay"][index] * 100000 + int(item),
                                  givendose=np.full(count, .1), route="i.v.", typeid=1, recordstatus=0)
            elif database == "aumc":
                time = p["origin"][index] + hour * 3_600_000
                record.update(admissionid=p["stay"][index], **{time_col: time})
                record.update(registeredat=time, updatedat=time, stop=time + 3_600_000,
                              duration=60, tag="+", unit={"map": "mmHg", "lact": "mmol/l", "hgb": "mmol/l",
                              "crea": "umol/l", "k": "mmol/l", "urine": "ml", "etco2": "Geen"}.get(name, ""), registeredby="synthetic",
                              dose=.1, doseunit="mg", doserateunit="uur", doserateperkg=False,
                              doseunitid="10", doserateunitid="4", iscontinuous=True,
                              orderid=np.arange(count) + 1, islabresult=True,
                              fluidin=50., solutionadministered=50.)
            else:
                time = (6 + p["stay"][index] % 4 * 3) * 3600 + hour * 3600
                record.update(CaseID=p["stay"][index], PatientID=p["patient"][index],
                              id=np.arange(count) + 1, Offset=time, OffsetEnd=time + 3600,
                              OffsetDrugEnd=time + 3600, IsSingleDose=False, GivenState=1,
                              Amount=.00036, AmountPerMinute=.000006, Cnt=1)
                if table == "data_range":
                    record["Data"] = "synthetic ventilation"
            rows[table].append(pd.DataFrame(record, index=np.arange(count)))
    return rows


def _physical_table(database, name, spec, frame):
    """Use physical column names, including SICdb's case-sensitive CSV headers."""
    arrays, names = [], []
    for column in spec["cols"].values():
        col, kind = column["name"], column["spec"]
        series = frame[col] if col in frame else pd.Series([None] * len(frame))
        arrow_type = {"col_integer": pa.int64(), "col_double": pa.float64(),
                      "col_character": pa.string(), "col_logical": pa.bool_(),
                      "col_datetime": pa.timestamp("ns")}[kind]
        if database == "hirid" and name == "general" and col == "admissiontime":
            arrow_type = pa.string()  # CSV converter preserves this string column.
        if database == "aumc" and col in {"doseunitid", "doserateunitid"}:
            arrow_type = pa.string()  # DataConverter.MIXED_TYPE_COLUMNS
        if pa.types.is_string(arrow_type):
            series = series.map(lambda value: str(value) if pd.notna(value) else None)
        elif pa.types.is_integer(arrow_type) and series.isna().any() and len(series):
            arrow_type = pa.float64()  # pandas CSV converter's nullable integers
        arrays.append(pa.array(series, type=arrow_type))
        names.append(col)
    return pa.Table.from_arrays(arrays, names=names)


def source_tables(database: str, stays: int):
    """Yield converted layout paths and small Arrow tables, with no filesystem reads of data."""
    from easyicu.io.data_converter import DataConverter, PARTITIONING_CONFIG

    config, concepts = definitions(database)
    p = population(stays)
    rows = _events(database, config, concepts, p)
    main = {"hirid": "general", "aumc": "admissions", "sic": "cases"}[database]
    rows[main] = [pd.DataFrame(_demographics(database, p))]
    for name, spec in config["tables"].items():
        if database == "aumc" and name == "vital":
            continue  # Optional derived table, not an AmsterdamUMCdb CSV release table.
        frame = pd.concat(rows[name], ignore_index=True) if rows[name] else pd.DataFrame()
        if database == "sic" and "id" in frame:
            frame["id"] = np.arange(1, len(frame) + 1)
        table = _physical_table(database, name, spec, frame)
        if database == "hirid" and name in {"observations", "pharma"}:
            yield f"{name}/1.parquet", table
        elif database == "aumc" and name in PARTITIONING_CONFIG["aumc"]:
            partition = PARTITIONING_CONFIG["aumc"][name]
            bins = np.searchsorted(partition["breaks"], frame[partition["col"]], side="left")
            for number in range(len(partition["breaks"]) + 1):
                selected = np.flatnonzero(bins == number)
                yield f"{name}/{number + 1}.parquet", table.take(selected).sort_by([
                    (partition["col"], "ascending"), ("admissionid", "ascending"),
                ])
        else:
            stem = Path(spec["files"]).stem
            converted = DataConverter.TABLE_NAME_MAP.get(stem, stem)
            yield f"{converted}.parquet", table
