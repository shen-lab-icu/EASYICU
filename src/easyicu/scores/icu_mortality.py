"""Native ICU discharge mortality, separate from the dictionary's ``death``.

Status and timing have separate authority. In particular HiRID monitoring
items only time a status from ``general``; Amsterdam's ICU destination does
not include the dictionary callback's 72-hour post-discharge deaths.
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd

from easyicu.databases.profiles import get_database_profile, normalize_database_key
from easyicu.io.identity import require_unique_keys

ICU_DEATH_DATABASES = frozenset({
    "mimic", "mimic_demo", "miiv", "miiv_demo", "eicu", "eicu_demo", "hirid", "aumc", "sic",
})


def _columns(frame, *names):
    missing = set(names) - set(frame)
    if missing:
        raise ValueError("icu_death_source_field_missing:" + ",".join(sorted(missing)))


def _status(values, mapping):
    return values.astype("string").str.strip().str.casefold().map(mapping).astype("boolean")


def _hirid_time(root, frame):
    from easyicu.concept.callback_apply import hirid_observation_read_exprs
    from easyicu.databases.research_capabilities import HIRID_TIMING_ITEMS
    from easyicu.runtime import duckdb_spill

    expressions = hirid_observation_read_exprs(Path(root))
    if expressions is None:
        path = Path(root) / "observations.parquet"
        if not path.is_file():
            return pd.Series(float("nan"), index=frame.index)
        expression = "read_parquet('" + str(path).replace("'", "''") + "')"
    else:
        expression = expressions[1]
    # Aggregate on disk; no observation table is brought into Python memory.
    with duckdb_spill.connect() as conn:
        conn.execute("SET memory_limit = '1GB'")
        conn.execute("SET threads = 1")
        conn.register("icu_stays", frame[["patientid"]])
        items = ",".join(str(item.item_id) for item in HIRID_TIMING_ITEMS)
        times = conn.execute(f"""SELECT o.patientid, max(o.datetime) AS last_time
            FROM {expression} o JOIN icu_stays s USING (patientid)
            WHERE o.variableid IN ({items}) GROUP BY o.patientid""").fetchdf()
    last = pd.to_datetime(frame.patientid.map(times.set_index("patientid").last_time))
    start = pd.to_datetime(frame.admissiontime, errors="coerce")
    return ((last - start).dt.total_seconds() / 3600).where(last >= start)


def load_icu_death(database, data_path, *, patient_ids=None, max_patients=None):
    """Return one nullable status per native ICU id and an explicitly typed time.

    Unknown status stays missing. Negative/pre-ICU and post-ICU recorded deaths
    are not ICU deaths. ICU exit can time a discharge status but is never
    presented as a recorded death timestamp.
    """
    from .comorbidity import lower_cols
    from .outcomes import _patient_values, _raw_table

    db = normalize_database_key(database)
    if db not in ICU_DEATH_DATABASES:
        raise ValueError("icu_death_source_unavailable")
    profile = get_database_profile("miiv" if db == "miiv_demo" else db)
    identifier = profile.stay_id_col
    base = db.removesuffix("_demo")

    def read(table):
        raw = _raw_table(db, data_path, table)
        # MIMIC-III ships upper-case headers; like the data source, only that
        # database is lowercased (SICdb's mixed case is its schema).
        return lower_cols(raw) if base == "mimic" else raw

    frame = read(profile.stay_table)
    _columns(frame, identifier)
    require_unique_keys(frame, [identifier], table="icu_death")
    patient_values = _patient_values(patient_ids)
    if patient_values is not None:
        frame = frame.loc[frame[identifier].isin(patient_values)]
    if max_patients is not None:
        frame = frame.head(max_patients)
    frame = frame.reset_index(drop=True)
    kind = "unavailable"
    times = pd.Series(float("nan"), index=frame.index)
    if base == "hirid":
        _columns(frame, "discharge_status", "admissiontime")
        status = _status(frame.discharge_status, {"dead": True, "alive": False})
        times = _hirid_time(data_path, frame)
        kind = "last_observation_proxy"
    elif base == "aumc":
        _columns(frame, "destination", "admittedat", "dischargedat")
        destination = frame.destination.astype("string").str.strip().str.casefold()
        status = (destination == "overleden").astype("boolean").mask(destination == "")
        start, end = (pd.to_numeric(frame[c], errors="coerce") for c in ("admittedat", "dischargedat"))
        status = status.mask(start.isna() | end.isna() | (end < start))
        times, kind = (end - start) / 3_600_000, "icu_discharge_status"
    elif base == "eicu":
        _columns(frame, "unitdischargestatus", "unitdischargeoffset")
        status = _status(frame.unitdischargestatus, {"expired": True, "alive": False})
        times = pd.to_numeric(frame.unitdischargeoffset, errors="coerce") / 60
        kind = "icu_discharge_status"
    elif base == "sic":
        _columns(frame, "OffsetOfDeath", "ICUOffset", "TimeOfStay", "HospitalDischargeType")
        event, start, end = (pd.to_numeric(frame[c], errors="coerce")
                             for c in ("OffsetOfDeath", "ICUOffset", "TimeOfStay"))
        status = ((event >= start) & (event <= end)).astype("boolean")
        known_survivor = pd.to_numeric(frame.HospitalDischargeType, errors="coerce") == 2027
        status = status.mask((event.isna() & ~known_survivor) | start.isna() | end.isna() | (end < start))
        times, kind = (event - start) / 3600, "recorded"
    else:
        _columns(frame, "hadm_id", "intime", "outtime")
        admissions = read("admissions")
        _columns(admissions, "hadm_id", "deathtime", "hospital_expire_flag")
        require_unique_keys(admissions, ["hadm_id"], table="icu_death_admissions")
        frame = frame.merge(admissions[["hadm_id", "deathtime", "hospital_expire_flag"]],
                            on="hadm_id", how="left", validate="many_to_one")
        event, start, end = (pd.to_datetime(frame[c], errors="coerce")
                             for c in ("deathtime", "intime", "outtime"))
        status = ((event >= start) & (event <= end)).astype("boolean")
        status = status.mask((event.isna() & (frame.hospital_expire_flag != 0))
                             | start.isna() | end.isna() | (end < start))
        times, kind = (event - start).dt.total_seconds() / 3600, "recorded"
    out = pd.DataFrame({identifier: frame[identifier], "icu_death": status,
                        "icu_death_time_hours": times.where(status.fillna(False))})
    out.attrs.update(outcome_meaning="icu_death", event_time_kind=kind,
                     analysis_unit="icu_stay", source_identifier=identifier)
    return out
