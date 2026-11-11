"""Leakage-free ICU-stay weight with its source and time (S740).

The published ``weight`` concept is an id_tbl median over every row of the
best channel of a stay, ICU episode unclipped (concept/__init__.py id_tbl
reduction), so a fallback channel can mix measurements taken after a
decision time. This owner derives three separate stay-level fields instead:

``weight_admit_strict``          kg, one reading, never a median
``weight_admit_strict_source``   admission | admission_lbs | pre_icu_last | charted_first | none
``weight_admit_strict_time_h``   hours from ICU admission of that reading (0 for static admission fields)

Selection, per stay, over readings in [1, 500] kg:

1. an admission channel: the best-ranked one present (admission before
   admission_lbs), its earliest reading;
2. otherwise the last fallback reading before ICU admission (time < 0);
3. otherwise the first fallback reading at or after ICU admission;
4. otherwise no weight (source ``none``; weight and time missing).

Ties at one time keep the better-ranked fallback channel, then the smaller
value, so the result is deterministic. Callers that condition on a decision
time p keep only rows with ``weight_admit_strict_time_h <= p``.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Optional

import numpy as np
import pandas as pd

CALLBACK = "weight_admit_strict_fields"
WEIGHT, SOURCE, TIME = "weight_admit_strict", "weight_admit_strict_source", "weight_admit_strict_time_h"
FIELDS = (WEIGHT, SOURCE, TIME)
SOURCES = ("admission", "admission_lbs", "pre_icu_last", "charted_first", "none")
MIN_KG, MAX_KG = 1.0, 500.0
LB_TO_KG = 0.45359237
HOUR_NS = 3_600 * 10**9
#: AmsterdamUMCdb admissions.weightgroup midpoints, as the published weight concept maps them.
AUMC_WEIGHTGROUP_KG = {"59-": 50.0, "60-69": 65.0, "70-79": 75.0, "80-89": 85.0,
                       "90-99": 95.0, "100-109": 105.0, "110+": 120.0}
ID_COLUMNS = {"miiv": "stay_id", "aumc": "admissionid", "hirid": "patientid",
              "sic": "CaseID", "eicu": "patientunitstayid"}
_OBS = ["sid", "channel", "rank", "time_h", "kg"]


def select_strict_weight(obs: pd.DataFrame, stays: pd.Series) -> pd.DataFrame:
    """One row per stay in ``stays`` from candidate readings ``obs`` (columns ``_OBS``)."""
    obs = obs.loc[:, _OBS].copy()
    obs["kg"] = pd.to_numeric(obs["kg"], errors="coerce")
    obs["time_h"] = pd.to_numeric(obs["time_h"], errors="coerce")
    obs = obs.loc[obs["kg"].between(MIN_KG, MAX_KG) & np.isfinite(obs["time_h"])]
    admission = obs.loc[obs["channel"].isin(("admission", "admission_lbs"))].assign(
        _r=lambda f: f["channel"].map({"admission": 0, "admission_lbs": 1}))
    admission = admission.sort_values(["sid", "_r", "time_h", "kg"], kind="mergesort").drop_duplicates("sid")
    admission["source"] = admission["channel"]
    fallback = obs.loc[obs["channel"].eq("fallback") & ~obs["sid"].isin(admission["sid"])]
    before = fallback.loc[fallback["time_h"] < 0].sort_values(
        ["sid", "time_h", "rank", "kg"], ascending=[True, False, True, True], kind="mergesort").drop_duplicates("sid")
    before["source"] = "pre_icu_last"
    after = fallback.loc[(fallback["time_h"] >= 0) & ~fallback["sid"].isin(before["sid"])].sort_values(
        ["sid", "time_h", "rank", "kg"], kind="mergesort").drop_duplicates("sid")
    after["source"] = "charted_first"
    chosen = pd.concat([part[["sid", "kg", "source", "time_h"]] for part in (admission, before, after)],
                       ignore_index=True)
    cohort = pd.DataFrame({"sid": pd.unique(pd.Series(stays).dropna())})
    out = cohort.merge(chosen, on="sid", how="left", validate="one_to_one")
    out["source"] = out["source"].fillna("none")
    return pd.DataFrame({
        "sid": out["sid"].to_numpy(),
        WEIGHT: out["kg"].astype("float64").to_numpy(),
        SOURCE: out["source"].astype("string").to_numpy(),
        TIME: out["time_h"].astype("float64").to_numpy(),
    })


def _selected_ids(patient_ids, id_column: str) -> Optional[list]:
    if patient_ids is None:
        return None
    if isinstance(patient_ids, Mapping):
        keys = [key for key in patient_ids if str(key).lower() == id_column.lower()]
        if len(keys) != 1:
            raise ValueError(f"weight_admit_strict needs one {id_column} identifier list")
        values = patient_ids[keys[0]]
        if values is None or isinstance(values, (str, bytes, Mapping)):
            raise ValueError(f"weight_admit_strict needs one {id_column} identifier list")
        return list(values)
    return list(patient_ids)


def _load(data_source, table, columns, id_column, ids, extra=()):
    from ..datasource import FilterOp, FilterSpec

    filters = list(extra)
    if ids is not None:
        filters.append(FilterSpec(column=id_column, op=FilterOp.IN, value=ids))
    frame = data_source.load_table(table, columns=list(columns), filters=filters or None, verbose=False)
    frame = frame.data if hasattr(frame, "data") else frame
    if ids is not None:
        frame = frame.loc[frame[id_column].isin(ids)]
    return frame


def _item(column, values):
    from ..datasource import FilterOp, FilterSpec

    return FilterSpec(column=column, op=FilterOp.IN, value=list(values))


def _hours(times, origins) -> np.ndarray:
    delta = pd.to_datetime(times, errors="coerce") - pd.to_datetime(origins, errors="coerce")
    return delta.to_numpy(dtype="timedelta64[ns]").astype("int64") / HOUR_NS


def _observations(database: str, data_source, ids) -> tuple[pd.DataFrame, pd.Series]:
    def obs(sid, channel, rank, time_h, kg):
        return pd.DataFrame({"sid": np.asarray(sid), "channel": channel, "rank": rank,
                             "time_h": np.asarray(time_h, dtype="float64"), "kg": np.asarray(kg, dtype="float64")})

    if database == "miiv":
        stays = _load(data_source, "icustays", ["stay_id", "intime"], "stay_id", ids)
        rows = _load(data_source, "chartevents", ["stay_id", "charttime", "itemid", "valuenum"], "stay_id", ids,
                     [_item("itemid", (226512, 226531, 224639))])
        rows = rows.merge(stays, on="stay_id", how="inner")
        time_h = _hours(rows["charttime"], rows["intime"]) if len(rows) else np.array([], dtype=float)
        value = pd.to_numeric(rows["valuenum"], errors="coerce").to_numpy(dtype=float)
        item = rows["itemid"].to_numpy()
        channel = np.select([item == 226512, item == 226531], ["admission", "admission_lbs"], "fallback")
        kg = np.where(item == 226531, value * LB_TO_KG, value)
        return obs(rows["stay_id"], channel, 0, time_h, kg), stays["stay_id"]
    if database == "aumc":
        stays = _load(data_source, "admissions", ["admissionid", "admittedat", "weightgroup"], "admissionid", ids)
        group = stays["weightgroup"].astype("string").str.strip().map(AUMC_WEIGHTGROUP_KG)
        static = obs(stays["admissionid"], "admission", 0, np.zeros(len(stays)), group)
        rows = _load(data_source, "numericitems", ["admissionid", "measuredat", "itemid", "value"], "admissionid", ids,
                     [_item("itemid", (20081, 6638, 6637, 10442))])
        rows = rows.merge(stays[["admissionid", "admittedat"]], on="admissionid", how="inner")
        # load_table returns AmsterdamUMCdb times in minutes (source ms / 60000).
        time_h = (pd.to_numeric(rows["measuredat"], errors="coerce")
                  - pd.to_numeric(rows["admittedat"], errors="coerce")) / 60.0
        return pd.concat([static, obs(rows["admissionid"], "fallback", 0, time_h, rows["value"])],
                         ignore_index=True), stays["admissionid"]
    if database == "hirid":
        stays = _load(data_source, "general", ["patientid", "admissiontime"], "patientid", ids)
        rows = _load(data_source, "observations", ["patientid", "datetime", "variableid", "value"], "patientid", ids,
                     [_item("variableid", (10000400,))])
        rows = rows.merge(stays, on="patientid", how="inner")
        time_h = _hours(rows["datetime"], rows["admissiontime"]) if len(rows) else np.array([], dtype=float)
        return obs(rows["patientid"], "fallback", 0, time_h, rows["value"]), stays["patientid"]
    if database == "sic":
        stays = _load(data_source, "cases", ["CaseID", "WeightOnAdmission"], "CaseID", ids)
        kg = pd.to_numeric(stays["WeightOnAdmission"], errors="coerce") / 1000.0
        return obs(stays["CaseID"], "admission", 0, np.zeros(len(stays)), kg), stays["CaseID"]
    if database == "eicu":
        stays = _load(data_source, "patient", ["patientunitstayid", "admissionweight"], "patientunitstayid", ids)
        static = obs(stays["patientunitstayid"], "admission", 0, np.zeros(len(stays)), stays["admissionweight"])
        rows = _load(data_source, "intakeoutput", ["patientunitstayid", "intakeoutputoffset", "celllabel", "cellvaluenumeric"],
                     "patientunitstayid", ids, [_item("celllabel", ("Bodyweight (kg)", "Bodyweight (lb)"))])
        pounds = rows["celllabel"].astype("string").eq("Bodyweight (lb)").fillna(False).to_numpy()
        value = pd.to_numeric(rows["cellvaluenumeric"], errors="coerce").to_numpy(dtype=float)
        fallback = obs(rows["patientunitstayid"], "fallback", pounds.astype(int),
                       pd.to_numeric(rows["intakeoutputoffset"], errors="coerce") / 60.0,
                       np.where(pounds, value * LB_TO_KG, value))
        return pd.concat([static, fallback], ignore_index=True), stays["patientunitstayid"]
    raise ValueError(f"weight_admit_strict has no reader for database {database!r}")


_CACHE: dict = {}


def strict_weight_fields(data_source, patient_ids) -> tuple[pd.DataFrame, str]:
    """Per-stay strict weight fields and the database's stay identifier column.

    The three concepts share one derivation; the last result is cached for the
    same data source and patient scope so a module load reads the sources once.
    """
    database = str(getattr(getattr(data_source, "config", None), "name", "") or "").lower()
    if database not in ID_COLUMNS:
        raise ValueError(f"weight_admit_strict has no reader for database {database!r}")
    id_column = ID_COLUMNS[database]
    ids = _selected_ids(patient_ids, id_column)
    key = (id(data_source), database, None if ids is None else hash(tuple(sorted(ids))))
    cached = _CACHE.get("last")
    if cached is not None and cached[0] == key:
        return cached[1].copy(), id_column
    obs, stays = _observations(database, data_source, ids)
    derived = select_strict_weight(obs, stays).rename(columns={"sid": id_column})
    _CACHE["last"] = (key, derived)
    return derived.copy(), id_column


__all__ = ["CALLBACK", "FIELDS", "SOURCES", "select_strict_weight", "strict_weight_fields"]
