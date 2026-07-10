"""Composite ICU outcome endpoints.

Trial-style endpoints that the concept layer did not previously expose:

* ``mort_28d`` / ``mort_90d`` / ``mort_365d`` — fixed-horizon mortality
  measured from **ICU admission** (intime). Requires post-discharge
  death follow-up, available in MIMIC-III/IV (``patients.dod``),
  SICdb (``cases.OffsetOfDeath``) and AmsterdamUMCdb
  (``admissions.dateofdeath``). eICU and HiRID carry only in-hospital
  mortality (use the existing ``death`` concept) -> these horizons are
  returned empty for them, by design, not as an error.
* ``icu_free_days_28`` — 0 if dead within 28 days, else
  ``28 - min(icu_los_days, 28)``. A standard ventilator-trial-style
  "free days" composite that penalises death.
* ``icu_readmission`` — whether this ICU stay is a re-admission within
  the same hospitalisation (MIMIC only, where ``icustays`` groups
  cleanly by ``hadm_id``).

Like :mod:`easyicu.comorbidity`, this is a per-database loader because
the death-time and stay schemas differ; the heavy lifting reuses the
same ``ICUDataSource.load_table`` access path.
"""

from __future__ import annotations

from typing import Optional

import numpy as np
import pandas as pd

from .comorbidity import _lower_cols


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


# DBs with post-discharge death follow-up usable for fixed-horizon mortality.
_FOLLOWUP_DATABASES = {
    "miiv",
    "miiv_demo",
    "mimic",
    "mimic_demo",
    "sic",
    "sic_demo",
    "aumc",
}
_HORIZONS = {"mort_28d": 28, "mort_90d": 90, "mort_365d": 365}


def _patient_values(patient_ids):
    if patient_ids is None:
        return None
    if isinstance(patient_ids, dict):
        if not patient_ids:
            return []
        patient_ids = next(iter(patient_ids.values()))
    return list(patient_ids)


def _eicu_vent_free_days(database, data_path, patient_ids, verbose) -> pd.DataFrame:
    """eICU ventilator-free days to day 28 from native actualventdays.

    VFD28 = 0 if the patient died in hospital, else 28 - min(vent_days, 28).
    """
    apr = _raw_table(database, data_path, "apachepatientresult")
    apr.columns = [c.lower() for c in apr.columns]
    apr = apr[apr["apacheversion"] == "IVa"].copy()
    vent = pd.to_numeric(apr["actualventdays"], errors="coerce")
    vent = vent.where(vent >= 0)  # -1 sentinel -> NaN
    died = apr["actualhospitalmortality"].astype(str).str.upper().eq("EXPIRED")
    vfd = np.where(
        died,
        0.0,
        np.where(vent.isna(), np.nan, 28.0 - np.clip(vent, 0, 28)),
    )
    out = pd.DataFrame(
        {
            "patientunitstayid": apr["patientunitstayid"].values,
            "vent_free_days_28": vfd,
        }
    )
    # one row per stay (apachepatientresult can have dup IVa rows)
    out = out.groupby("patientunitstayid", as_index=False)["vent_free_days_28"].min()
    patient_values = _patient_values(patient_ids)
    if patient_values is not None:
        out = out[out["patientunitstayid"].isin(patient_values)]
    return out.reset_index(drop=True)


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
    df["los_days"] = pd.to_numeric(cases[cmap["timeofstay"]], errors="coerce") / 86400.0
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
    ``mort_28d``/``mort_90d``/``mort_365d`` (nullable boolean),
    ``icu_free_days_28`` (float), and ``icu_readmission`` (nullable
    boolean, MIMIC only). DBs without death follow-up return an empty
    frame.
    """
    db = database.lower()
    patient_values = _patient_values(patient_ids)
    if patient_values == []:
        return pd.DataFrame()
    if db in ("eicu", "eicu_demo"):
        # eICU has no post-discharge follow-up (horizon mortality N/A) but
        # DOES carry native ventilator days -> a clean ventilator-free-days
        # endpoint, which MIMIC cannot support (its mech_vent concept is
        # too fragmented: median ~0.02 vent-days/stay).
        return _eicu_vent_free_days(database, data_path, patient_values, verbose)
    if db not in _FOLLOWUP_DATABASES:
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
    for name, horizon in _HORIZONS.items():
        # dead within horizon -> True; known-alive (no death, or death after
        # horizon) -> False. We cannot see beyond the data window, but death
        # follow-up in these DBs covers >= 1 year, so 28/90d are reliable.
        died_by = has_death & (dtd <= horizon) & (dtd >= 0)
        out[name] = pd.array(died_by, dtype="boolean")

    los = pd.to_numeric(base["los_days"], errors="coerce").values
    dead28 = out["mort_28d"].fillna(False).to_numpy(dtype=bool)
    los_capped = np.clip(los, 0, 28)
    icu_free = np.where(
        dead28,
        0.0,
        np.where(np.isnan(los), np.nan, 28.0 - los_capped),
    )
    out["icu_free_days_28"] = icu_free

    # ICU readmission: only well-defined where icustays groups by hadm_id.
    if db in ("miiv", "miiv_demo", "mimic", "mimic_demo") and "hadm_id" in base.columns:
        readmit = (
            base.dropna(subset=["hadm_id"])
            .sort_values(["hadm_id", "_intime"], kind="mergesort", na_position="last")
            .assign(_n=lambda d: d.groupby("hadm_id").cumcount())
        )
        readmit_map = dict(zip(readmit["_stay"], readmit["_n"] > 0))
        out["icu_readmission"] = pd.array(
            [readmit_map.get(s, pd.NA) for s in base["_stay"].values], dtype="boolean"
        )

    if patient_values is not None:
        stay_col = _STAY_OUT_COL[db]
        out = out[out[stay_col].isin(patient_values)]
    return out.reset_index(drop=True)


__all__ = ["load_outcomes"]
