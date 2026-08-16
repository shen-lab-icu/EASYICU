"""Composite ICU outcome endpoints.

Trial-style endpoints that the concept layer did not previously expose:

* ``mort_28d`` / ``mort_90d`` / ``mort_365d`` — fixed-horizon mortality
  measured from **ICU admission** (intime). Requires post-discharge
  death follow-up, available in MIMIC-III/IV (``patients.dod``),
  SICdb (``cases.OffsetOfDeath``) and AmsterdamUMCdb
  (``admissions.dateofdeath``). eICU and HiRID carry only in-hospital
  mortality (use the existing ``death`` concept) -> these horizons are
  returned empty for them, by design, not as an error.
``icu_free_days_28``, ``vent_free_days_28``, and ``icu_readmission`` are
intentionally unavailable until their trajectory/follow-up evidence contracts
can be satisfied. A current-extract stay LOS, hospital mortality flag, or
within-extract sort order is not sufficient evidence for those endpoints.

Like :mod:`easyicu.comorbidity`, this is a per-database loader because
the death-time and stay schemas differ; the heavy lifting reuses the
same ``ICUDataSource.load_table`` access path.
"""

from __future__ import annotations

from typing import Optional

import numpy as np
import pandas as pd

from easyicu.outcome_availability import FOLLOWUP_OUTCOME_DATABASES

from .comorbidity import _lower_cols
from ..databases.profiles import normalize_database_key


def _raw_table(database: str, data_path: object, table: str) -> pd.DataFrame:
    """Read a table's raw parquet, bypassing the datasource time-column
    transforms. SICdb/AmsterdamUMCdb store death/stay times as integer
    second/millisecond offsets that ``ICUDataSource.load_table`` rebases
    to 0 — we need the untouched values for horizon mortality.
    """
    import glob
    import os

    root = data_path
    if root is None:
        root = os.environ.get("EASYICU_DATA_PATH", "")
        try:
            from ..io.data_paths import find_database_path

            root = find_database_path(root, database)
        except Exception:
            root = os.path.join(root, database)
    candidates = [
        os.path.join(root, f"{table}.parquet"),
        *glob.glob(os.path.join(root, "*", f"{table}.parquet")),
    ]
    for path in candidates:
        if os.path.exists(path):
            return pd.read_parquet(path)
    raise FileNotFoundError(f"{table}.parquet not found under {root}")


_HORIZONS = {"mort_28d": 28, "mort_90d": 90, "mort_365d": 365}


def _patient_values(patient_ids):
    if patient_ids is None:
        return None
    if isinstance(patient_ids, dict):
        if not patient_ids:
            return []
        patient_ids = next(iter(patient_ids.values()))
    return list(patient_ids)


def _mimic_stay_death_days(database, data_path) -> pd.DataFrame:
    """MIMIC-III/IV: stay_id/icustay_id + days_from_icu_admit_to_death + los."""
    icu = _lower_cols(_raw_table(database, data_path, "icustays"))
    stay_col = "stay_id" if "stay_id" in icu.columns else "icustay_id"
    icu = icu[["subject_id", "hadm_id", stay_col, "intime", "los"]].copy()
    icu["intime"] = pd.to_datetime(icu["intime"], errors="coerce")
    pat = _lower_cols(_raw_table(database, data_path, "patients"))[
        ["subject_id", "dod"]
    ].copy()
    pat["dod"] = pd.to_datetime(pat["dod"], errors="coerce")
    df = icu.merge(pat, on="subject_id", how="left")
    # MIMIC patients.dod is a DATE, not a death timestamp.  Keep the endpoint
    # at the source-supported calendar-day resolution; subtracting an exact
    # ICU intime from a midnight DATE makes same-day deaths negative and then
    # incorrectly censors them below.
    df["days_to_death"] = (
        df["dod"] - df["intime"].dt.normalize()
    ).dt.total_seconds() / 86400.0
    df["los_days"] = pd.to_numeric(df["los"], errors="coerce")
    return df.rename(columns={stay_col: "_stay", "intime": "_intime"})[
        ["_stay", "hadm_id", "_intime", "days_to_death", "los_days"]
    ]


def _sic_stay_death_days(database, data_path) -> pd.DataFrame:
    cases = _raw_table(database, data_path, "cases")
    cmap = {c.lower(): c for c in cases.columns}
    cid = cmap["caseid"]
    df = pd.DataFrame({"_stay": cases[cid].values})
    off_death = pd.to_numeric(cases[cmap["offsetofdeath"]], errors="coerce")
    icu_off = pd.to_numeric(cases[cmap["icuoffset"]], errors="coerce").fillna(0)
    df["days_to_death"] = (off_death - icu_off) / 86400.0  # SICdb offsets are seconds
    time_of_stay = pd.to_numeric(cases[cmap["timeofstay"]], errors="coerce")
    los_seconds = time_of_stay - icu_off
    df["los_days"] = los_seconds.where(los_seconds >= 0) / 86400.0
    df["followup_days"] = np.nan
    followup_column = cmap.get("estimatedsurvivalobservationtime")
    if followup_column is not None:
        followup = cases[followup_column]
        numeric = pd.to_numeric(followup, errors="coerce")
        followup_days = pd.Series(np.nan, index=cases.index, dtype="float64")
        followup_days.loc[numeric.eq(3076)] = 183.0
        followup_days.loc[numeric.eq(3077)] = 365.0
        text = followup.astype(str).str.strip().str.lower()
        followup_days.loc[text.str.contains("6 month", na=False)] = 183.0
        followup_days.loc[text.str.contains("1 year", na=False)] = 365.0
        df["followup_days"] = followup_days.to_numpy()
    df["hadm_id"] = np.nan
    return df


def _aumc_stay_death_days(database, data_path) -> pd.DataFrame:
    adm = _lower_cols(_raw_table(database, data_path, "admissions"))
    df = pd.DataFrame({"_stay": adm["admissionid"].values})
    # admittedat is the 0 reference; dateofdeath / dischargedat are ms offsets.
    dod = pd.to_numeric(adm["dateofdeath"], errors="coerce")
    admit = pd.to_numeric(adm.get("admittedat", 0), errors="coerce").fillna(0)
    df["days_to_death"] = (dod - admit) / 86400000.0  # ms -> days
    if "dischargedat" in adm.columns:
        df["los_days"] = (
            pd.to_numeric(adm["dischargedat"], errors="coerce") - admit
        ) / 86400000.0
    elif "lengthofstay" in adm.columns:
        df["los_days"] = pd.to_numeric(adm["lengthofstay"], errors="coerce") / 24.0
    else:
        df["los_days"] = np.nan
    df["hadm_id"] = np.nan
    return df


_STAY_OUT_COL = {
    "miiv": "stay_id",
    "miiv_demo": "stay_id",
    "mimic": "icustay_id",
    "mimic_demo": "icustay_id",
    "sic": "CaseID",
    "sic_demo": "CaseID",
    "aumc": "admissionid",
}


def load_outcomes(
    database: str,
    data_path: object = None,
    *,
    patient_ids=None,
    max_patients: Optional[int] = None,
    verbose: bool = False,
) -> pd.DataFrame:
    """Load per-ICU-stay composite outcome endpoints for a database.

    Returns a DataFrame keyed by the database ICU stay id with
    ``mort_28d``/``mort_90d``/``mort_365d`` (nullable boolean). DBs
    without death follow-up return an empty frame.
    """
    db = normalize_database_key(database)
    database = db
    patient_values = _patient_values(patient_ids)
    if patient_values == []:
        return pd.DataFrame()
    if db not in FOLLOWUP_OUTCOME_DATABASES:
        if verbose:
            print(
                f"[outcomes] {database}: no post-discharge death follow-up — "
                "fixed-horizon mortality N/A (use 'death' for in-hospital)"
            )
        return pd.DataFrame()

    if db in ("miiv", "miiv_demo", "mimic", "mimic_demo"):
        base = _mimic_stay_death_days(database, data_path)
    elif db in ("sic", "sic_demo"):
        base = _sic_stay_death_days(database, data_path)
    elif db == "aumc":
        base = _aumc_stay_death_days(database, data_path)
    else:  # pragma: no cover - guarded by _FOLLOWUP_DATABASES
        return pd.DataFrame()

    out = pd.DataFrame({_STAY_OUT_COL[db]: base["_stay"].values})
    dtd = base["days_to_death"].values
    has_death = ~pd.isna(dtd)
    followup_days = pd.to_numeric(
        base.get("followup_days", pd.Series(np.nan, index=base.index)),
        errors="coerce",
    ).to_numpy()
    for name, horizon in _HORIZONS.items():
        died_by = has_death & (dtd <= horizon) & (dtd >= 0)
        known_alive = (has_death & (dtd > horizon)) | (
            ~has_death & (followup_days >= horizon)
        )
        endpoint = pd.array([pd.NA] * len(base), dtype="boolean")
        endpoint[died_by] = True
        endpoint[known_alive] = False
        out[name] = endpoint

    if patient_values is not None:
        stay_col = _STAY_OUT_COL[db]
        out = out[out[stay_col].isin(patient_values)]
    return out.reset_index(drop=True)


__all__ = ["load_outcomes"]
