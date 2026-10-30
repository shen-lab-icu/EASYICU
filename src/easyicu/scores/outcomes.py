"""Composite ICU outcome endpoints.

Trial-style endpoints that the concept layer did not previously expose:

* ``mort_28d`` / ``mort_90d`` / ``mort_365d`` — fixed-horizon mortality
  measured from **ICU admission** (intime). A horizon needs post-discharge
  follow-up of survivors as well as dates of death: MIMIC-IV
  (``patients.dod``, censored one year after the last hospital discharge)
  and SICdb (``cases.OffsetOfDeath`` with each case's survival observation
  time) carry both. MIMIC-III and AmsterdamUMCdb record dates of death but
  no follow-up of survivors, so a horizon there would be defined for the
  dead only; eICU and HiRID carry only in-hospital mortality (use the
  existing ``death`` concept). These horizons are returned empty for them,
  by design, not as an error.
* ``followup_days_28d`` / ``followup_days_90d`` /
  ``followup_days_365d`` — the event-or-administrative-censoring time paired
  with each fixed-horizon endpoint. Unknown follow-up remains missing.
* ``icu_free_days_28`` (MIMIC-IV) — from this stay's ICU admission, 0 if
  the patient died by day 28, else 28 minus the days spent in any ICU stay of
  the patient (this one, later ones, other admissions) within those 28 days.
  Unknown day-28 survival or an ICU stay without a discharge time in the
  window leaves it missing.
* ``icu_readmission`` (MIMIC-IV, eICU) — the outcome: after this stay's ICU
  episode the patient returned to an ICU in the same hospitalisation.  A
  patient who died in that episode was not at risk, so it is missing for them.
* ``icu_stay_is_readmission`` (MIMIC-IV, eICU) — a stay attribute, not an
  outcome: this stay belongs to a later ICU episode of its hospitalisation.
  The repeat-stay sensitivity analysis restricts on it.

Both readmission columns read every ICU stay of the source's own stay table,
whatever the extract's cohort.  Stays of one hospitalisation chain into one
ICU episode when the next begins within ``ICU_EPISODE_TRANSFER_GAP_MINUTES``
of the previous discharge (a direct transfer).  ``vent_free_days_28`` stays
unavailable until ventilation episodes are validated.

Like :mod:`easyicu.comorbidity`, this is a per-database loader because
the death-time and stay schemas differ; the heavy lifting reuses the
same ``ICUDataSource.load_table`` access path.
"""

from __future__ import annotations

import logging
from typing import Optional

import numpy as np
import pandas as pd

from easyicu.outcome_availability import (
    FIXED_HORIZON_MORTALITY_ENDPOINTS,
    FOLLOWUP_OUTCOME_DATABASES,
    ICU_FREE_DAY_DATABASES,
    ICU_READMISSION_DATABASES,
)

from .comorbidity import lower_cols
from ..databases.profiles import get_database_profile, normalize_database_key
from ..io.identity import require_unique_keys

LOGGER = logging.getLogger(__name__)


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
    # Schema-first pre-check: verify the prepared-table schema and record
    # source identity before reading any rows. Fail closed with source
    # identity attached instead of bare-reading a wrong/mixed extract.
    try:
        from ..databases.detection import peek_table_columns
        from pathlib import Path as _Path

        _root = _Path(root) if root else None
        if _root is not None and _root.is_dir():
            _columns = peek_table_columns(_root, table)
            LOGGER.info(
                "outcomes source identity: database=%s table=%s root=%s columns=%d",
                database,
                table,
                str(_root),
                len(_columns),
            )
    except Exception as exc:
        # peek raises DatabaseDetectionError with source identity when the
        # schema exists but is unreadable; missing tables fall through to
        # the FileNotFoundError below. Log and continue so the error at the
        # read site keeps its source context.
        LOGGER.warning(
            "outcomes schema pre-check degraded: database=%s table=%s root=%s failure=%s",
            database,
            table,
            str(root),
            type(exc).__name__,
        )
    candidates = [
        os.path.join(root, f"{table}.parquet"),
        *glob.glob(os.path.join(root, "*", f"{table}.parquet")),
    ]
    for path in candidates:
        if os.path.exists(path):
            return pd.read_parquet(path)
    raise FileNotFoundError(f"{table}.parquet not found under {root}")


# The event/follow-up pairing is owned by ``easyicu.outcome_availability`` so
# the materializer and the research-agent survival projection agree on it.
_HORIZONS = {
    endpoint.event_concept: endpoint.horizon_days
    for endpoint in FIXED_HORIZON_MORTALITY_ENDPOINTS.values()
}


def _patient_values(patient_ids):
    if patient_ids is None:
        return None
    if isinstance(patient_ids, dict):
        if not patient_ids:
            return []
        patient_ids = next(iter(patient_ids.values()))
    return list(patient_ids)


def _parse_source_dates(values: pd.Series) -> pd.Series:
    """Parse date/timestamp columns without requiring pandas 2-only formats."""
    parsed = pd.to_datetime(values, errors="coerce")
    # pandas 2 infers one format for the whole vector. Retry unmatched non-null
    # values individually; the common homogeneous source column stays vectorized.
    retry = parsed.isna() & values.notna()
    if retry.any():
        parsed.loc[retry] = values.loc[retry].map(
            lambda value: pd.to_datetime(value, errors="coerce")
        )
    return parsed


def _mimic_stay_death_days(database, data_path) -> pd.DataFrame:
    """MIMIC-IV: stay_id + days_from_icu_admit_to_death + los."""
    icu = lower_cols(_raw_table(database, data_path, "icustays"))
    stay_col = "stay_id" if "stay_id" in icu.columns else "icustay_id"
    # Mortality needs no discharge time; without one ICU-free days are unknown.
    if "outtime" not in icu.columns:
        icu["outtime"] = None
    icu = icu[["subject_id", "hadm_id", stay_col, "intime", "outtime", "los"]].copy()
    require_unique_keys(icu, [stay_col], table="icustays")
    icu["intime"] = _parse_source_dates(icu["intime"])
    icu["outtime"] = pd.to_datetime(_parse_source_dates(icu["outtime"]), errors="coerce")
    if icu["intime"].isna().any():
        raise ValueError("ICU mortality requires valid admission time origins")
    pat = lower_cols(_raw_table(database, data_path, "patients"))[
        ["subject_id", "dod"]
    ].copy()
    require_unique_keys(pat, ["subject_id"], table="patients")
    pat["_dod_absent"] = pat["dod"].isna() | pat["dod"].astype("string").str.strip().eq("")
    pat["dod"] = _parse_source_dates(pat["dod"])
    if (pat["dod"].isna() & ~pat["_dod_absent"]).any():
        raise ValueError("patients.dod contains non-empty unparseable death dates")
    adm = lower_cols(_raw_table(database, data_path, "admissions"))[
        ["hadm_id", "deathtime"]
    ].copy()
    require_unique_keys(adm, ["hadm_id"], table="admissions")
    deathtime_absent = adm["deathtime"].isna() | adm["deathtime"].astype(
        "string"
    ).str.strip().eq("")
    adm["deathtime"] = _parse_source_dates(adm["deathtime"])
    if (adm["deathtime"].isna() & ~deathtime_absent).any():
        raise ValueError(
            "admissions.deathtime contains non-empty unparseable death times"
        )
    df = icu.merge(pat, on="subject_id", how="left", validate="many_to_one", indicator=True)
    df = df.merge(adm, on="hadm_id", how="left", validate="many_to_one")
    # MIMIC patients.dod is a DATE, not a death timestamp; admissions.deathtime
    # records the time of a death in that hospital admission.  Time a death by
    # its recorded time when the stay's own admission records one at or after
    # ICU admission: a calendar day counts a death 25 hours after admission as
    # one day, inside a 24-hour landmark, and one 2 hours after a late-evening
    # admission as one day too.  Otherwise keep the source-supported
    # calendar-day resolution of the date of death; subtracting an exact ICU
    # intime from a midnight DATE makes same-day deaths negative and then
    # incorrectly censors them below.
    death_date = df["dod"].fillna(df["deathtime"].dt.normalize())
    calendar_days = (
        death_date - df["intime"].dt.normalize()
    ).dt.total_seconds() / 86400.0
    recorded_days = (df["deathtime"] - df["intime"]).dt.total_seconds() / 86400.0
    df["days_to_death"] = recorded_days.where(recorded_days >= 0, calendar_days)
    df["los_days"] = pd.to_numeric(df["los"], errors="coerce")
    # MIMIC-IV v3.1 documents a one-year post-discharge censoring window and
    # states that a null patients.dod means the patient survived at least one
    # year after the last hospital discharge.  ICU admission precedes that
    # discharge, so 365 days is a conservative lower bound on follow-up from
    # ICU admission for null-DOD rows.  Do not extrapolate beyond that bound.
    # https://physionet.org/content/mimiciv/3.1/
    documented_absence = (
        (database in {"miiv", "miiv_demo"})
        & df["_merge"].eq("both")
        & df["_dod_absent"].eq(True)
    )
    # Do not borrow the MIMIC-IV 2.0+ contract for failed joins or date
    # parsing failures. Those do not establish follow-up.
    df["followup_days"] = np.where(documented_absence, 365.0, np.nan)
    return df.rename(columns={stay_col: "_stay", "intime": "_intime", "outtime": "_outtime"})[
        [
            "_stay",
            "subject_id",
            "hadm_id",
            "_intime",
            "_outtime",
            "days_to_death",
            "los_days",
            "followup_days",
        ]
    ]


def _sic_stay_death_days(database, data_path) -> pd.DataFrame:
    cases = _raw_table(database, data_path, "cases").reset_index(drop=True)
    cmap = {c.lower(): c for c in cases.columns}
    cid = cmap["caseid"]
    df = pd.DataFrame({"_stay": cases[cid].values})
    raw_death = cases[cmap["offsetofdeath"]]
    off_death = pd.to_numeric(raw_death, errors="coerce")
    icu_off = pd.to_numeric(cases[cmap["icuoffset"]], errors="coerce")
    # SICdb's offset is the first transfer to intensive OR intermediate care.
    # A missing/non-finite origin cannot establish any time relative to it.
    valid_origin = pd.Series(np.isfinite(icu_off) & (icu_off >= 0), index=cases.index)
    valid_death = pd.Series(np.isfinite(off_death), index=cases.index)
    unusable_death = raw_death.notna() & ~valid_death
    icu_off = icu_off.where(valid_origin)
    off_death = off_death.where(valid_death)
    df["days_to_death"] = (off_death - icu_off) / 86400.0  # SICdb offsets are seconds
    time_of_stay = pd.to_numeric(cases[cmap["timeofstay"]], errors="coerce")
    los_seconds = time_of_stay.where(np.isfinite(time_of_stay)) - icu_off
    df["los_days"] = los_seconds.where(los_seconds >= 0) / 86400.0
    df["followup_days"] = np.nan
    followup_column = cmap.get("estimatedsurvivalobservationtime")
    if followup_column is not None:
        followup = cases[followup_column]
        numeric = pd.to_numeric(followup, errors="coerce")
        followup_days = pd.Series(np.nan, index=cases.index, dtype="float64")
        followup_days.loc[numeric.eq(3076)] = 180.0
        followup_days.loc[numeric.eq(3077)] = 365.0
        text = followup.astype(str).str.strip().str.lower()
        followup_days.loc[text.str.contains("6 month", na=False)] = 180.0
        followup_days.loc[text.str.contains("1 year", na=False)] = 365.0
        # SICdb's registry death offset is capped at one year from primary
        # admission. Conservatively treat the 6-month/one-year follow-up
        # category as beginning there too, then move its bound to the first
        # intensive-or-intermediate-care transfer. The source describes an
        # estimated category, so 180 days is a conservative operational
        # assumption, not proof of exact individual follow-up.
        source_bound = followup_days - icu_off / 86400.0
        df["followup_days"] = source_bound.where(
            valid_origin & ~unusable_death & (source_bound >= 0)
        ).to_numpy()
    df["hadm_id"] = np.nan
    return df


_STAY_OUT_COL = {
    "miiv": "stay_id",
    "miiv_demo": "stay_id",
    "sic": "CaseID",
    "sic_demo": "CaseID",
}


def _load_followup_outcomes(
    database: str,
    data_path: object = None,
    *,
    patient_ids=None,
    max_patients: Optional[int] = None,
    verbose: bool = False,
) -> pd.DataFrame:
    """Load per-ICU-stay composite outcome endpoints for a database.

    Returns a DataFrame keyed by the database ICU stay id with
    ``mort_28d``/``mort_90d``/``mort_365d`` (nullable boolean) and the paired
    ``followup_days_<horizon>d`` event/censoring times. DBs without death
    follow-up return an empty frame.
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

    if db in ("miiv", "miiv_demo"):
        base = _mimic_stay_death_days(database, data_path)
    elif db in ("sic", "sic_demo"):
        base = _sic_stay_death_days(database, data_path)
    else:  # pragma: no cover - guarded by FOLLOWUP_OUTCOME_DATABASES
        return pd.DataFrame()

    out = pd.DataFrame({_STAY_OUT_COL[db]: base["_stay"].values})
    dtd = base["days_to_death"].values
    has_death = ~pd.isna(dtd)
    followup_days = pd.to_numeric(
        base.get("followup_days", pd.Series(np.nan, index=base.index)),
        errors="coerce",
    ).to_numpy()
    # A survivor reaches a horizon only through follow-up.  Without follow-up
    # for any stay that records no death, every survivor would be missing and
    # the horizon would be defined for the dead only.
    if (~has_death).any() and not np.isfinite(followup_days[~has_death]).any():
        raise ValueError(
            f"{database}: fixed-horizon mortality has no follow-up for any "
            "stay without a recorded death"
        )
    for name, horizon in _HORIZONS.items():
        died_by = has_death & (dtd <= horizon) & (dtd >= 0)
        known_alive = (has_death & (dtd > horizon)) | (
            ~has_death & (followup_days >= horizon)
        )
        endpoint = pd.array([pd.NA] * len(base), dtype="boolean")
        endpoint[died_by] = True
        endpoint[known_alive] = False
        out[name] = endpoint
        followup_name = FIXED_HORIZON_MORTALITY_ENDPOINTS[name].followup_concept
        followup_time = np.full(len(base), np.nan, dtype="float64")
        followup_time[died_by] = dtd[died_by]
        followup_time[known_alive] = float(horizon)
        out[followup_name] = followup_time

    if db in ICU_FREE_DAY_DATABASES:
        out["icu_free_days_28"] = _icu_free_days_28(base, out["mort_28d"])

    if patient_values is not None:
        stay_col = _STAY_OUT_COL[db]
        out = out[out[stay_col].isin(patient_values)]
    return out.reset_index(drop=True)


_ICU_FREE_DAY_HORIZON = pd.Timedelta(days=28)


def _icu_free_days_28(stays: pd.DataFrame, mort_28d) -> np.ndarray:
    """ICU-free days to day 28 for each row of ``stays`` (every ICU stay).

    ``stays`` carries ``subject_id``, ``_intime`` and ``_outtime``; each row's
    window is its own ICU admission to 28 days later, and the ICU time in it is
    the union of the patient's ICU stays clipped to that window.
    """
    index = pd.DataFrame({
        "_i": np.arange(len(stays)),
        "subject_id": stays["subject_id"].to_numpy(),
        "_t0": stays["_intime"].to_numpy(),
    })
    other = pd.DataFrame({
        "subject_id": stays["subject_id"].to_numpy(),
        "_s": stays["_intime"].to_numpy(),
        "_e": stays["_outtime"].to_numpy(),
    })
    pairs = index.merge(other, on="subject_id")
    window_end = pairs["_t0"] + _ICU_FREE_DAY_HORIZON
    pairs = pairs[(pairs["_s"] < window_end) & (pairs["_e"].isna() | (pairs["_e"] > pairs["_t0"]))]
    # A stay in the window without a discharge time has unknown ICU time.
    unknown = np.zeros(len(stays), dtype=bool)
    unknown[pairs.loc[pairs["_e"].isna(), "_i"].to_numpy()] = True
    pairs = pairs[pairs["_e"].notna()]
    clip_start = pairs[["_s", "_t0"]].max(axis=1)
    clip_end = pd.concat([pairs["_e"], pairs["_t0"] + _ICU_FREE_DAY_HORIZON], axis=1).min(axis=1)
    pieces = pd.DataFrame({"_i": pairs["_i"], "_cs": clip_start, "_ce": clip_end})
    pieces = pieces.sort_values(["_i", "_cs"], kind="mergesort")
    # Union of overlapping pieces: count only time past the furthest end so far.
    covered = pieces.groupby("_i")["_ce"].cummax().groupby(pieces["_i"]).shift()
    begin = pieces["_cs"].where(covered.isna() | (pieces["_cs"] >= covered), covered)
    seconds = (pieces["_ce"] - begin).dt.total_seconds().clip(lower=0)
    icu_days = np.zeros(len(stays), dtype="float64")
    summed = seconds.groupby(pieces["_i"]).sum() / 86400.0
    icu_days[summed.index.to_numpy()] = summed.to_numpy()
    dead = pd.Series(pd.array(mort_28d, dtype="boolean"))
    died = dead.fillna(False).to_numpy(dtype=bool)
    survival_unknown = dead.isna().to_numpy(dtype=bool)
    free = np.clip(28.0 - icu_days, 0.0, 28.0)
    return np.where(died, 0.0, np.where(survival_unknown | unknown, np.nan, free))


# Two ICU stays of one hospitalisation make one ICU episode when the second
# begins within this many minutes of the first's discharge: a direct transfer
# between units, or eICU's step-down continuation, not a return to the ICU.
# On the full sources (2026-10-28) eICU stays discharged to another ICU or a
# step-down bed are followed by the next unit stay after a median of 0
# minutes, and 13% of MIMIC-IV's later stays in one admission begin within an
# hour of the previous discharge, almost all within minutes.
ICU_EPISODE_TRANSFER_GAP_MINUTES = 60.0

ICU_SEQUENCE_OUTPUTS = frozenset({"icu_readmission", "icu_stay_is_readmission"})


def _icu_stay_sequence(database: str, data_path) -> pd.DataFrame:
    """Every ICU stay of the source on one minute clock per hospitalisation.

    Columns: ``_stay`` (native stay id), ``_hosp``, ``_order`` (the source's
    order within the hospitalisation), ``_start`` and ``_end`` in minutes.
    """
    db = normalize_database_key(database)
    profile = get_database_profile(db.removesuffix("_demo"))
    stays = _raw_table(db, data_path, profile.stay_table)
    if db.startswith("miiv"):
        stays = lower_cols(stays)
        needed = (profile.stay_id_col, "hadm_id", "intime", "outtime")
    else:
        needed = (profile.stay_id_col, "patienthealthsystemstayid", "unitvisitnumber",
                  "hospitaladmitoffset", "unitdischargeoffset")
    missing = sorted(set(needed) - set(stays.columns))
    if missing:
        raise ValueError("icu_readmission_source_field_missing:" + ",".join(missing))
    require_unique_keys(stays, [profile.stay_id_col], table=profile.stay_table)
    if db.startswith("miiv"):
        epoch = pd.Timestamp(0)
        start = (_parse_source_dates(stays["intime"]) - epoch).dt.total_seconds() / 60
        end = (_parse_source_dates(stays["outtime"]) - epoch).dt.total_seconds() / 60
        hosp, order = stays["hadm_id"], start
    else:
        # eICU offsets are minutes from unit admission; hospitaladmitoffset is
        # the (negative) offset of hospital admission, so -offset puts every
        # unit stay of a hospitalisation on that hospitalisation's clock.
        start = -pd.to_numeric(stays["hospitaladmitoffset"], errors="coerce")
        end = start + pd.to_numeric(stays["unitdischargeoffset"], errors="coerce")
        hosp = stays["patienthealthsystemstayid"]
        order = pd.to_numeric(stays["unitvisitnumber"], errors="coerce")
    return pd.DataFrame({
        "_stay": stays[profile.stay_id_col].to_numpy(),
        "_hosp": hosp.to_numpy(),
        "_order": pd.to_numeric(order, errors="coerce").to_numpy(),
        "_start": start.to_numpy(dtype="float64"),
        "_end": end.to_numpy(dtype="float64"),
    })


def _icu_episodes(seq: pd.DataFrame) -> pd.DataFrame:
    """Number each stay's ICU episode (0, 1, ...) within its hospitalisation.

    A hospitalisation with a stay lacking a time or an order, two stays sharing
    an order, or an order that disagrees with admission time gets no episode
    numbers: breaking the tie by stay id would decide a readmission by an
    identifier.
    """
    seq = seq.sort_values(["_hosp", "_order", "_start"], kind="mergesort").reset_index(drop=True)
    hosp = seq["_hosp"]
    bad = seq[["_hosp", "_order", "_start", "_end"]].isna().any(axis=1)
    bad |= seq.duplicated(["_hosp", "_order"], keep=False)
    bad |= seq["_start"] < seq.groupby("_hosp")["_start"].shift()
    bad |= seq["_end"] < seq["_start"]
    bad_hosp = bad.groupby(hosp).transform("any").fillna(True).astype(bool) | hosp.isna()
    furthest_end = seq.groupby("_hosp")["_end"].cummax().groupby(hosp).shift()
    gap = seq["_start"] - furthest_end
    new_episode = furthest_end.isna() | (gap > ICU_EPISODE_TRANSFER_GAP_MINUTES)
    episode = new_episode.astype("int64").groupby(hosp).cumsum() - 1
    seq["_episode"] = episode.astype("float64").where(~bad_hosp)
    seq["_last_episode"] = seq.groupby("_hosp")["_episode"].transform("max").where(~bad_hosp)
    return seq


def _load_icu_readmission(database, data_path=None, *, patient_ids=None) -> pd.DataFrame:
    """``icu_readmission`` and ``icu_stay_is_readmission`` per native ICU stay."""
    db = normalize_database_key(database)
    patient_values = _patient_values(patient_ids)
    if db not in ICU_READMISSION_DATABASES or patient_values == []:
        return pd.DataFrame()
    from .icu_mortality import load_icu_death

    identifier = get_database_profile(db.removesuffix("_demo")).stay_id_col
    seq = _icu_episodes(_icu_stay_sequence(db, data_path))
    death = load_icu_death(db, data_path).set_index(identifier)["icu_death"]
    seq["_died"] = pd.array(seq["_stay"].map(death), dtype="boolean")
    # The episode ends at its last stay; did the patient die in that stay?
    last = seq.groupby(["_hosp", "_episode"]).cumcount(ascending=False).eq(0) & seq["_episode"].notna()
    ending = seq.loc[last, ["_hosp", "_episode", "_died"]].rename(columns={"_died": "_episode_died"})
    seq = seq.merge(ending, on=["_hosp", "_episode"], how="left", validate="many_to_one")
    known = seq["_episode"].notna().to_numpy()
    returned = known & (seq["_episode"] < seq["_last_episode"]).to_numpy()
    survived = known & seq["_episode_died"].eq(False).fillna(False).to_numpy(dtype=bool)
    readmission = pd.array([pd.NA] * len(seq), dtype="boolean")
    readmission[survived] = False
    readmission[returned] = True
    later_episode = pd.array([pd.NA] * len(seq), dtype="boolean")
    later_episode[known] = (seq["_episode"] > 0).to_numpy()[known]
    out = pd.DataFrame({
        identifier: seq["_stay"].to_numpy(),
        "icu_readmission": readmission,
        "icu_stay_is_readmission": later_episode,
    })
    if patient_values is not None:
        out = out[out[identifier].isin(patient_values)]
    return out.reset_index(drop=True)


def load_outcomes(database, data_path=None, *, patient_ids=None, max_patients=None,
                  verbose=False, output_concepts=None):
    """Load explicitly named endpoints without changing generic death meaning.

    Without ``output_concepts`` this returns the follow-up endpoints only, as
    before; the ICU death and ICU stay sequence columns load when named.
    """
    requested = set(output_concepts or ())
    options = dict(patient_ids=patient_ids, max_patients=max_patients)
    frames = []
    if "icu_death" in requested:
        from .icu_mortality import load_icu_death

        frames.append(load_icu_death(database, data_path, **options))
    if not requested or requested - {"icu_death"} - ICU_SEQUENCE_OUTPUTS:
        frames.append(_load_followup_outcomes(database, data_path, verbose=verbose, **options))
    if requested & ICU_SEQUENCE_OUTPUTS:
        frames.append(_load_icu_readmission(database, data_path, patient_ids=patient_ids))
    frames = [frame for frame in frames if not frame.empty]
    if not frames:
        return pd.DataFrame()
    result = frames[0]
    for frame in frames[1:]:
        identity = [column for column in frame.columns if column in result.columns]
        if len(identity) != 1:
            raise ValueError("outcome_frames_share_no_single_stay_identifier")
        attrs = dict(result.attrs)
        result = result.merge(frame, on=identity, how="left", validate="one_to_one")
        result.attrs.update(attrs)
    return result


__all__ = ["load_outcomes"]
