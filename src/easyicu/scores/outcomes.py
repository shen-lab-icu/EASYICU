"""Composite ICU outcome endpoints.

Trial-style endpoints that the concept layer did not previously expose:

* ``mort_28d`` / ``mort_90d`` / ``mort_365d`` — fixed-horizon mortality
  measured from **ICU admission** (intime), all-cause, including deaths
  after hospital discharge (so a stay can be ``mort_28d`` without the
  in-hospital ``death``). A horizon needs post-discharge follow-up of
  survivors as well as dates of death. A survivor without a recorded death
  is censored at its source's follow-up bound, and a horizon beyond that
  bound stays missing:

  - MIMIC-IV ``patients.dod``: one year after the last hospital discharge;
  - MIMIC-III ``patients.dod`` (Social Security linkage): 90 days after the
    hospital discharge for MetaVision stays, four years for CareVue stays;
  - AmsterdamUMCdb ``admissions.dateofdeath`` (registry linkage, deaths
    recorded years after discharge in both admission-year groups): one year
    from ICU admission; a death in the ICU is timed at ``dischargedat``;
  - SICdb ``cases.OffsetOfDeath`` with each case's survival observation
    category (six months or one year from the primary admission).

  eICU and HiRID carry only in-hospital mortality (use the existing
  ``death`` concept). These horizons are returned empty for them, by
  design, not as an error.
* ``followup_days_28d`` / ``followup_days_90d`` /
  ``followup_days_365d`` — the event-or-administrative-censoring time paired
  with each fixed-horizon endpoint. Unknown follow-up remains missing.
* ``icu_free_days_28`` (MIMIC-IV) — from this stay's ICU admission, 0 if
  the patient died by day 28, else 28 minus the days spent in any ICU stay of
  the patient (this one, later ones, other admissions) within those 28 days.
  Unknown day-28 survival or an ICU stay without a discharge time in the
  window leaves it missing.
* ``icu_readmission`` (MIMIC-IV, MIMIC-III, eICU) — the outcome: after this stay's ICU
  episode the patient returned to an ICU in the same hospitalisation.  A
  patient who died in that episode was not at risk, so it is missing for them.
* ``icu_stay_is_readmission`` (MIMIC-IV, MIMIC-III, eICU) — a stay attribute, not an
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

from ..utils.thresholds import ge, gt, le, lt
from .comorbidity import lower_cols
from ..databases.profiles import get_database_profile, normalize_database_key
from ..io.identity import require_unique_keys
from ..source_death_times import (
    aumc_registry_death_after_discharge, sic_discharge_codes, sic_recorded_death,
)

LOGGER = logging.getLogger(__name__)


def _raw_table(database: str, data_path: object, table: str) -> pd.DataFrame:
    """Read a table's raw parquet, bypassing the datasource time-column
    transforms. SICdb/AmsterdamUMCdb store death/stay times as integer
    second/millisecond offsets that ``ICUDataSource.load_table`` rebases
    to 0 — we need the untouched values for horizon mortality.
    """
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
    # The table's file set comes from the storage policy owner, validated
    # against the installation inventory (single file, flat directory or
    # bucket contract); a recursive listing here once read hidden staging
    # files and bucket shards in physical order (final review U01).
    from ..storage_layout import restore_frame_order
    from .comorbidity import _build_datasource

    files = _build_datasource(database, root).resolve_target_files(table) if root else ()
    if not files:
        raise FileNotFoundError(f"{table}.parquet not found under {root}")
    if len(files) == 1:
        return pd.read_parquet(files[0])
    return restore_frame_order(
        pd.concat([pd.read_parquet(part) for part in files], ignore_index=True)
    )


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
    """MIMIC-IV/III: ICU stay + days_from_icu_admit_to_death + los + follow-up."""
    icu = lower_cols(_raw_table(database, data_path, "icustays"))
    stay_col = "stay_id" if "stay_id" in icu.columns else "icustay_id"
    # Mortality needs no discharge time; without one ICU-free days are unknown.
    if "outtime" not in icu.columns:
        icu["outtime"] = None
    if "dbsource" not in icu.columns:
        icu["dbsource"] = None
    icu = icu[
        ["subject_id", "hadm_id", stay_col, "intime", "outtime", "los", "dbsource"]
    ].copy()
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
    adm = lower_cols(_raw_table(database, data_path, "admissions"))
    if "dischtime" not in adm.columns:
        adm["dischtime"] = None
    adm = adm[["hadm_id", "deathtime", "dischtime"]].copy()
    require_unique_keys(adm, ["hadm_id"], table="admissions")
    deathtime_absent = adm["deathtime"].isna() | adm["deathtime"].astype(
        "string"
    ).str.strip().eq("")
    adm["deathtime"] = _parse_source_dates(adm["deathtime"])
    if (adm["deathtime"].isna() & ~deathtime_absent).any():
        raise ValueError(
            "admissions.deathtime contains non-empty unparseable death times"
        )
    adm["dischtime"] = pd.to_datetime(_parse_source_dates(adm["dischtime"]), errors="coerce")
    df = icu.merge(pat, on="subject_id", how="left", validate="many_to_one", indicator=True)
    df = df.merge(adm, on="hadm_id", how="left", validate="many_to_one")
    icu_source = df.pop("dbsource")
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
    if database in {"mimic", "mimic_demo"}:
        # MIMIC-III patients.dod holds Social Security deaths for at least
        # four years after the hospital discharge in both CareVue and
        # MetaVision.  Rounds 1-3 bounded MetaVision at 90 days, but 1,762
        # MetaVision stays die 91-365 days after discharge and out-of-hospital
        # deaths reach 1,734 days, so every MetaVision mort_365d was missing
        # (2026-11-04 audit round 4, R4-018; reverts R3-08 for MIMIC-III).
        dbsource = icu_source.reindex(df.index).astype("string").str.strip().str.lower()
        bound_days = pd.Series(4 * 365.0, index=df.index)
        discharge_days = (df["dischtime"] - df["intime"]).dt.total_seconds() / 86400.0
        documented = (
            df["_merge"].eq("both")
            & df["_dod_absent"].eq(True)
            & dbsource.isin(["carevue", "metavision", "both"]).fillna(False)
            & (discharge_days >= 0)
        ).astype(bool)
        df["followup_days"] = (discharge_days + bound_days).where(documented)
        # The bound applies to every stay of the group, dead or alive
        # (2026-11-04 audit round 3, R3-08).
        # Without a usable discharge time the bound is at least the source's
        # linkage window from ICU admission.
        df["source_bound_days"] = (
            discharge_days.where(discharge_days >= 0, 0.0).fillna(0.0) + bound_days
        ).where(dbsource.isin(["carevue", "metavision", "both"]).fillna(False))
    else:
        df["source_bound_days"] = np.nan
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
            "source_bound_days",
        ]
    ]


#: How a SICdb survivor's follow-up category is set when
#: ``EstimatedSurvivalObservationTime`` (ESOT) is empty. SICdb 1.0.8 records
#: ESOT for deaths only. ``reconstruct_by_admission_year`` (default) uses the
#: regularity that decides every recorded ESOT in 1.0.6 and every death's in
#: 1.0.8 (2026-11-06 SIC 1.0.8 review, report section 9.3): admission year
#: 2021 is "6 Months", 2013-2020 "1 Year"; any other or missing year has an
#: unknown category, so its survivors get no follow-up: every horizon and the
#: follow-up time are NA, flagged and warned (no evidence supports even six
#: months for an admission year SICdb has not shown).
#: ``six_month_minimum`` gives every empty ESOT that minimum. Both flag such
#: survivors ``mort_<h>_assumption_based``; neither is documented by SICdb
#: (Documentation p.4: "either 1-year or 6-month").
SIC_FOLLOWUP_POLICIES = ("reconstruct_by_admission_year", "six_month_minimum")
SIC_FOLLOWUP_POLICY = "reconstruct_by_admission_year"
SIC_SIX_MONTH_ADMISSION_YEARS = frozenset({2021})
SIC_ONE_YEAR_ADMISSION_YEARS = frozenset(range(2013, 2021))


def sic_followup_from_admission_year(admission_year: pd.Series, *, policy: str = SIC_FOLLOWUP_POLICY) -> pd.Series:
    """Follow-up days for SICdb cases whose ESOT is empty (NaN: unknown).

    ``reconstruct_by_admission_year``: 2021 gives 180 days, 2013-2020 365;
    any other or missing year is NaN. ``six_month_minimum``: 180 for all.
    """
    if policy not in SIC_FOLLOWUP_POLICIES:
        raise ValueError(f"unknown SICdb follow-up policy {policy!r}")
    if policy == "six_month_minimum":
        return pd.Series(180.0, index=admission_year.index, dtype="float64")
    year = pd.to_numeric(admission_year, errors="coerce")
    days = pd.Series(np.nan, index=admission_year.index, dtype="float64")
    days.loc[year.isin(SIC_SIX_MONTH_ADMISSION_YEARS)] = 180.0
    days.loc[year.isin(SIC_ONE_YEAR_ADMISSION_YEARS)] = 365.0
    return days


def _sic_stay_death_days(database, data_path, *, followup_policy: str = SIC_FOLLOWUP_POLICY) -> pd.DataFrame:
    cases = _raw_table(database, data_path, "cases").reset_index(drop=True)
    codes = sic_discharge_codes(_raw_table(database, data_path, "d_references"))
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
    time_of_stay = pd.to_numeric(cases[cmap["timeofstay"]], errors="coerce")
    # A recorded death in hospital or at ICU discharge without OffsetOfDeath is
    # timed at ICU discharge; one recorded a little before the ICU transfer
    # dies at its start (2026-11-04 audit round 3, R3-15).
    disposition = (cases[cmap["hospitaldischargetype"]] if "hospitaldischargetype" in cmap
                   else pd.Series(np.nan, index=cases.index))
    recorded = sic_recorded_death(
        disposition,
        cases[cmap["dischargestate"]] if "dischargestate" in cmap else pd.Series(np.nan, index=cases.index),
        codes,
    )
    # A discharge code that is not a known death or survival code cannot be
    # read as alive: a code whose d_references label is blank or unknown, and
    # a non-empty code with no reference row at all (lead-r13 delta2 D2-SIC).
    # Without a recorded death such a case has no follow-up.
    disposition_text = disposition.astype("string").str.strip()
    disposition_present = disposition.notna() & disposition_text.ne("").fillna(False)
    disposition_code = pd.to_numeric(disposition, errors="coerce")
    unknown_disposition = (
        disposition_present & ~disposition_code.isin(codes.known) & ~recorded
    ).astype(bool)
    untimed = recorded & off_death.isna() & ~unusable_death & np.isfinite(time_of_stay)
    off_death = off_death.where(~untimed, time_of_stay)
    days_to_death = (off_death - icu_off) / 86400.0  # SICdb offsets are seconds
    df["days_to_death"] = days_to_death.where(~(recorded & days_to_death.lt(0)), 0.0)
    los_seconds = time_of_stay.where(np.isfinite(time_of_stay)) - icu_off
    df["los_days"] = los_seconds.where(los_seconds >= 0) / 86400.0
    df["followup_days"] = np.nan
    df["followup_assumed"] = False
    df["followup_category_unknown"] = False
    # A survivor whose discharge code is not a known death or survival code:
    # its follow-up is unknown for a stated reason (NA and flagged), not
    # missing from the source.
    df["followup_disposition_unknown"] = unknown_disposition.to_numpy(dtype=bool)
    followup_column = cmap.get("estimatedsurvivalobservationtime")
    if followup_column is not None:
        followup = cases[followup_column]
        numeric = pd.to_numeric(followup, errors="coerce")
        followup_days = numeric.map(codes.followup_days).astype("float64")
        text = followup.astype(str).str.strip().str.lower()
        followup_days.loc[text.str.contains("6 month", na=False)] = 180.0
        followup_days.loc[text.str.contains("1 year", na=False)] = 365.0
        # SICdb 1.0.8 leaves ESOT empty for survivors and for 33 recorded
        # deaths without OffsetOfDeath: set every empty category by the
        # follow-up policy (it bounds the deaths' evaluable horizons too) and
        # flag the survivors it decides; death status is unchanged.
        empty = followup.isna()
        year_column = cmap.get("admissionyear")
        years = cases[year_column] if year_column is not None else pd.Series(np.nan, index=cases.index)
        rebuilt = sic_followup_from_admission_year(years, policy=followup_policy)
        followup_days = followup_days.where(~empty, rebuilt)
        df["followup_assumed"] = (empty & ~recorded).to_numpy()
        # An empty category whose admission year is outside 2013-2021 has no
        # year-dependent ceiling. Every horizon and follow-up time is then NA
        # and flagged, for recorded deaths as for survivors: the fact and time
        # of a death are kept (days_to_death), but without the category
        # ceiling no fixed-horizon endpoint is defined (lead-r13 delta2 D2-SIC).
        unknown = empty & rebuilt.isna()
        df["followup_category_unknown"] = unknown.to_numpy()
        if unknown.any():
            LOGGER.warning(
                "sicdb_followup_category_unknown: %d stays (%d with a recorded death) have "
                "an empty follow-up category and an admission year outside 2013-2021; "
                "their 28/90/365-day mortality and follow-up are NA",
                int(unknown.sum()), int((unknown & (recorded | days_to_death.notna())).sum()),
            )
        # Source (round-12 citations): d_references labels "6 Months" and
        # "1 Year" (ReferenceName EstimatedSurvivalObservationTime); the
        # SICdb wiki says "For technical reasons observation time is only 6
        # month for some patients" (sicdb.com, Mortality) and PhysioNet that
        # OffsetOfDeath "is capped at one year" from admission and includes
        # out-of-hospital deaths from governmental data (SICdb 1.0.8). The
        # 2023 descriptor paper measures the year from hospital discharge
        # instead; the admission start used here is the shorter reading.
        # SICdb's registry death offset is capped at one year from primary
        # admission. Conservatively treat the 6-month/one-year follow-up
        # category as beginning there too, then move its bound to the first
        # intensive-or-intermediate-care transfer. The source describes an
        # estimated category, so 180 days is a conservative operational
        # assumption, not proof of exact individual follow-up.
        source_bound = followup_days - icu_off / 86400.0
        df["followup_days"] = source_bound.where(
            valid_origin & ~unusable_death & (source_bound >= 0) & ~unknown_disposition
        ).to_numpy()
        df["source_bound_days"] = source_bound.where(valid_origin & (source_bound >= 0)).to_numpy()
    # Moving the bound to the ICU transfer leaves it a few hours short of a
    # horizon equal to the category (median transfer 2 h after admission).
    # Accept a shortfall under one day: censoring every such survivor would
    # define the one-year horizon for the dead only. No source documents this
    # grace, so survivors who need it are flagged ``mort_<h>_assumption_based``.
    df["followup_grace_days"] = 1.0
    df["hadm_id"] = np.nan
    return df


def _aumc_stay_death_days(database, data_path) -> pd.DataFrame:
    adm = lower_cols(_raw_table(database, data_path, "admissions")).reset_index(drop=True)
    require_unique_keys(adm, ["admissionid"], table="admissions")
    df = pd.DataFrame({"_stay": adm["admissionid"].values})
    # admittedat is the 0 reference; dateofdeath / dischargedat are ms offsets.
    admit = pd.to_numeric(adm.get("admittedat", 0), errors="coerce").fillna(0)
    discharge = pd.to_numeric(adm.get("dischargedat"), errors="coerce")
    dod = pd.to_numeric(adm["dateofdeath"], errors="coerce")
    # A death in the ICU (destination "Overleden") is timed at the ICU
    # discharge: dateofdeath is a registry date, often hours before it, and
    # some such deaths carry none.  This is the ``death`` concept's rule.
    destination = adm.get("destination", pd.Series(pd.NA, index=adm.index))
    in_icu = destination.astype("string").str.strip().str.lower().eq("overleden").fillna(False)
    dod = aumc_registry_death_after_discharge(dod, discharge, day=86400000.0)
    dod = dod.where(~(in_icu & discharge.notna()), discharge)
    df["days_to_death"] = (dod - admit) / 86400000.0  # ms -> days
    df["los_days"] = (discharge - admit) / 86400000.0
    # dateofdeath links registry deaths up to years after discharge in both
    # admission-year groups (2003-2009 and 2010-2016); survivors are taken as
    # followed for one year from ICU admission and no further.
    df["followup_days"] = np.where(dod.isna(), 365.0, np.nan)
    # The source documents dateofdeath only "if applicable (non-null)" and
    # states no follow-up horizon (AmsterdamUMCdb wiki, admissions table), so
    # that year is an assumption (round-12 ruling): survivors with no valid
    # dateofdeath after the death-date checks above are flagged, including a
    # raw date that fails those checks.
    df["followup_assumed"] = dod.isna().to_numpy()
    df["hadm_id"] = np.nan
    return df


_STAY_OUT_COL = {
    "miiv": "stay_id",
    "miiv_demo": "stay_id",
    "mimic": "icustay_id",
    "mimic_demo": "icustay_id",
    "aumc": "admissionid",
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

    if db in ("miiv", "miiv_demo", "mimic", "mimic_demo"):
        base = _mimic_stay_death_days(database, data_path)
    elif db in ("sic", "sic_demo"):
        base = _sic_stay_death_days(database, data_path)
    elif db == "aumc":
        base = _aumc_stay_death_days(database, data_path)
    else:  # pragma: no cover - guarded by FOLLOWUP_OUTCOME_DATABASES
        return pd.DataFrame()

    out = pd.DataFrame({_STAY_OUT_COL[db]: base["_stay"].values})
    dtd = base["days_to_death"].values
    has_death = ~pd.isna(dtd)
    followup_days = pd.to_numeric(
        base.get("followup_days", pd.Series(np.nan, index=base.index)),
        errors="coerce",
    ).to_numpy()
    grace_days = pd.to_numeric(
        base.get("followup_grace_days", pd.Series(0.0, index=base.index)),
        errors="coerce",
    ).fillna(0.0).to_numpy()
    # Stays whose follow-up the source states is unknown (SICdb: a follow-up
    # category that cannot be rebuilt, or an unregistered discharge code).
    # Their horizons are NA and flagged, below; they are not a source defect.
    def _flag(column: str) -> np.ndarray:
        return (
            base.get(column, pd.Series(False, index=base.index))
            .astype("boolean").fillna(False).to_numpy(dtype=bool)
        )

    category_unknown = _flag("followup_category_unknown")
    disposition_unknown = _flag("followup_disposition_unknown") & ~has_death
    followup_unknown = category_unknown | disposition_unknown
    # A survivor reaches a horizon only through follow-up.  When no stay that
    # records no death has follow-up, and the source states no reason for
    # that, the follow-up mechanism itself is missing (no category column, no
    # discharge times): every survivor would be missing and the horizon would
    # be defined for the dead only.  A cohort whose survivors are all
    # explicitly unknown publishes NA and flags instead: refusing it would
    # fail an extraction depending on which stays it holds.
    expected = ~has_death & ~followup_unknown
    if expected.any() and not np.isfinite(followup_days[expected]).any():
        raise ValueError(
            f"{database}: fixed-horizon mortality has no follow-up for any "
            "stay without a recorded death"
        )
    # A source whose follow-up of a stay ends before a horizon records deaths
    # up to that bound only.  The horizon is then missing for the stay whether
    # it died or not: keeping its deaths while its survivors are missing
    # defined MIMIC-III MetaVision one-year mortality for the dead (84.5%;
    # 2026-11-04 audit round 3, R3-08).
    source_bound = pd.to_numeric(
        base.get("source_bound_days", pd.Series(np.nan, index=base.index)),
        errors="coerce",
    ).to_numpy()
    followup_assumed = (
        base.get("followup_assumed", pd.Series(False, index=base.index))
        .astype("boolean").fillna(False).to_numpy(dtype=bool)
    )
    # SICdb stays whose follow-up category could not be rebuilt (recorded
    # deaths included) and survivors with an unregistered discharge code:
    # every horizon stays NA and is flagged as resting on that unknown
    # follow-up; days_to_death is unchanged (lead-r13 D2-SIC).
    for name, horizon in _HORIZONS.items():
        unreachable = (
            np.isfinite(source_bound) & lt(source_bound + grace_days, horizon)
        ) | followup_unknown
        died_by = has_death & le(dtd, horizon) & (dtd >= 0) & ~unreachable
        known_alive = ((has_death & gt(dtd, horizon)) | (
            ~has_death & ge(followup_days + grace_days, horizon)
        )) & ~unreachable
        endpoint = pd.array([pd.NA] * len(base), dtype="boolean")
        endpoint[died_by] = True
        endpoint[known_alive] = False
        out[name] = endpoint
        # A survivor known only through an undocumented follow-up assumption:
        # an assumed horizon (AUMC) or a horizon reached by the grace day
        # (SICdb). Recorded deaths and documented follow-up are not assumed.
        assumed = known_alive & ~has_death & (
            followup_assumed | ~ge(followup_days, horizon)
        )
        flag = pd.array([pd.NA] * len(base), dtype="boolean")
        flag[died_by | known_alive] = False
        flag[assumed | followup_unknown] = True
        out[f"{name}_assumption_based"] = flag
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
    # MIMIC-III icustays has MIMIC-IV's shape (hadm_id, intime, outtime) in
    # upper case; both read the same hospitalisation clock.
    mimic_family = db.startswith(("miiv", "mimic"))
    if mimic_family:
        stays = lower_cols(stays)
        needed = (profile.stay_id_col, "hadm_id", "intime", "outtime")
    else:
        needed = (profile.stay_id_col, "patienthealthsystemstayid", "unitvisitnumber",
                  "hospitaladmitoffset", "unitdischargeoffset")
    missing = sorted(set(needed) - set(stays.columns))
    if missing:
        raise ValueError("icu_readmission_source_field_missing:" + ",".join(missing))
    require_unique_keys(stays, [profile.stay_id_col], table=profile.stay_table)
    if mimic_family:
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


def _load_icu_readmission(database, data_path=None, *, patient_ids=None, source=None) -> pd.DataFrame:
    """``icu_readmission`` and ``icu_stay_is_readmission`` per native ICU stay."""
    db = normalize_database_key(database)
    patient_values = _patient_values(patient_ids)
    if db not in ICU_READMISSION_DATABASES or patient_values == []:
        return pd.DataFrame()
    from .icu_mortality import load_icu_death

    identifier = get_database_profile(db.removesuffix("_demo")).stay_id_col
    seq = _icu_episodes(_icu_stay_sequence(db, data_path))
    death = load_icu_death(db, data_path, source=source).set_index(identifier)["icu_death"]
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
                  verbose=False, output_concepts=None, source=None):
    """Load explicitly named endpoints without changing generic death meaning.

    Without ``output_concepts`` this returns the follow-up endpoints only, as
    before; the ICU death and ICU stay sequence columns load when named.
    """
    requested = set(output_concepts or ())
    options = dict(patient_ids=patient_ids, max_patients=max_patients)
    frames = []
    if "icu_death" in requested:
        from .icu_mortality import load_icu_death

        frames.append(load_icu_death(database, data_path, source=source, **options))
    if not requested or requested - {"icu_death"} - ICU_SEQUENCE_OUTPUTS:
        frames.append(_load_followup_outcomes(database, data_path, verbose=verbose, **options))
    if requested & ICU_SEQUENCE_OUTPUTS:
        frames.append(_load_icu_readmission(database, data_path, patient_ids=patient_ids, source=source))
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
