"""Opt-in stay identity and first invasive ventilation episode producers."""

from __future__ import annotations

import numpy as np
import pandas as pd

from ..config import DATABASE_ID_CONFIG
from ..datasource import storage_source
from ..io.identity import require_unique_keys
from ..storage_binding import in_binding_unit
from ..storage_layout import record_table_unavailable, refuse_broken_presence, table_decision
from ..table.duration import resolve_dur_var_hours
from ..utils.thresholds import gt
from .sicdb_sources import COVERAGE_ATTR
from .stay_tables import stay_table_frame

# The stay key/table come from the same registry as cohort discovery. The
# other columns are the source's person, hospital admission (if any) and
# ranking fields, in precedence order. eICU offsets count back to admission.
_IDENTITY_FIELDS = {
    "miiv": ("subject_id", "hadm_id", ("intime", "hadm_id")),
    "mimic": ("subject_id", "hadm_id", ("intime",)),
    "aumc": ("patientid", None, ("admissioncount", "admittedat")),
    "sic": ("PatientID", None, ("OffsetAfterFirstAdmission", "AdmissionYear")),
    "eicu": (
        "uniquepid", "patienthealthsystemstayid",
        ("hospitaldischargeyear", "patienthealthsystemstayid", "unitvisitnumber", "hospitaladmitoffset"),
    ),
    "hirid": ("patientid", None, ()),
}


def _stay_table(database, data_path, columns):
    source = storage_source(database, data_path)
    table = DATABASE_ID_CONFIG[database]["table"]
    return stay_table_frame(
        source, table, ("support_export", tuple(columns)),
        lambda: source.load_table(table, columns=columns, verbose=False).data.loc[:, columns],
    )


def _selected_stays(rows, patient_ids):
    if patient_ids is None:
        return rows
    if not patient_ids:
        return rows.iloc[:0]
    key, members = next(iter(patient_ids.items()))
    return rows.loc[rows[key].isin(members)]


def _sort_stays(rows, order):
    return rows.sort_values(order, ascending=[c != "hospitaladmitoffset" for c in order], kind="stable")


def _valid_ordinal(values):
    return (np.isfinite(values) & values.ge(1) & values.mod(1).eq(0)).fillna(False)


def _first_hospital_stay(rows, *, person, hospital, order, valid):
    groups = [rows[person], rows[hospital]]
    rankable = valid.groupby(groups, sort=False).transform("all").eq(True)
    ranked = _sort_stays(rows.loc[rankable], order)
    first = ranked.groupby([person, hospital], sort=False).cumcount().eq(0)
    return first.astype("boolean").reindex(rows.index)


def _rank_identity(rows, *, database, stay, person, hospital, order):
    rows = rows.copy()
    if "intime" in order:
        rows["intime"] = pd.to_datetime(rows["intime"], errors="coerce")
    for column in order:
        if column != "intime":
            rows[column] = pd.to_numeric(rows[column], errors="coerce")

    first_hospital = pd.Series(pd.NA, index=rows.index, dtype="boolean")
    if database == "hirid":
        rows["stay_order"] = pd.Series(1, index=rows.index, dtype="Int64")
        rows["is_first_icu_stay"] = pd.Series(True, index=rows.index, dtype="boolean")
        rows["is_first_icu_stay_of_hospital_admission"] = first_hospital
        return _sort_stays(rows, [stay])

    sort_order = [*order, stay]
    valid = rows[person].notna() & rows[order[0]].notna()
    proven_later = pd.Series(False, index=rows.index)
    if database == "aumc":
        count = rows["admissioncount"]
        valid_count = _valid_ordinal(count)
        tied = rows.duplicated([person, "admissioncount"], keep=False)
        unresolved = rows.duplicated([person, "admissioncount", "admittedat"], keep=False)
        valid &= valid_count & ~unresolved & (~tied | np.isfinite(rows["admittedat"]))
        proven_later = valid_count & count.gt(1)
    elif database == "sic":
        offset = rows["OffsetAfterFirstAdmission"]
        valid &= np.isfinite(offset) & offset.ge(0)
        proven_later = np.isfinite(offset) & offset.gt(0)
    elif database == "eicu":
        groups = [rows[person], rows[hospital]]
        visit = rows["unitvisitnumber"]
        valid_visit = _valid_ordinal(visit)
        tied_visit = rows.duplicated([person, hospital, "unitvisitnumber"], keep=False)
        # Use one clock throughout an admission. Offsets must order every
        # sibling when visit numbers fail to establish a complete order.
        by_visit = (valid_visit & ~tied_visit).groupby(groups, sort=False).transform("all").eq(True)
        rows["_visit_order"] = visit.where(by_visit)
        hospital_order = ["_visit_order", "hospitaladmitoffset", stay]
        sort_order = [*order[:2], *hospital_order]
        tied_offset = rows.duplicated([person, hospital, "hospitaladmitoffset"], keep=False)
        valid_offset = np.isfinite(rows["hospitaladmitoffset"]).fillna(False) & ~tied_offset
        hospital_valid = by_visit | valid_offset
        first_hospital = _first_hospital_stay(
            rows, person=person, hospital=hospital, order=hospital_order, valid=hospital_valid,
        )
        later_visit = valid_visit & visit.gt(1)
        # An offset clock that starts with a known return visit cannot rank the admission.
        contradicted = (
            (~by_visit & first_hospital.fillna(False) & later_visit)
            .groupby(groups, sort=False).transform("any").eq(True)
        )
        hospital_valid &= ~contradicted
        first_hospital = first_hospital.mask(contradicted).mask(later_visit, False)
        proven_later = first_hospital.eq(False).fillna(False) & ~contradicted
        year = rows["hospitaldischargeyear"]
        same_year = year.groupby(groups, sort=False).transform("nunique").eq(1)
        valid &= rows[hospital].notna() & hospital_valid & np.isfinite(year) & same_year
    elif hospital:
        first_hospital = _first_hospital_stay(
            rows, person=person, hospital=hospital, order=["intime", stay], valid=rows["intime"].notna(),
        )
        proven_later = first_hospital.eq(False).fillna(False)

    # A sibling with an unknown position can precede any ranked stay.
    rankable = valid.fillna(False).groupby(rows[person], sort=False).transform("all").eq(True)
    rows = _sort_stays(rows, sort_order)
    ranked = rows.loc[rankable]
    positions = ranked.groupby(person, sort=False).cumcount()
    if database == "aumc":
        # Preserve the source ordinal, including gaps in a partial source.
        # If ordinals tie, advance in source time order.
        difference = ranked["admissioncount"] - positions
        ordinal = difference.groupby(ranked[person], sort=False).cummax() + positions
    else:
        ordinal = positions + 1
    first = positions.eq(0).astype("boolean").reindex(rows.index)
    if database == "aumc":
        first &= rows["admissioncount"].eq(1)
    elif database == "sic":
        at_first_admission = rows["OffsetAfterFirstAdmission"].eq(0).fillna(False)
        zero_count = at_first_admission.groupby(rows[person], sort=False).transform("sum")
        first = at_first_admission.astype("boolean").mask(at_first_admission & zero_count.gt(1))
    elif database == "eicu":
        # The encounter key orders otherwise indistinguishable admissions in
        # the same discharge year. Only their hospital firsts are candidates.
        admissions = ranked.groupby([person, order[0]], sort=False)[hospital].transform("nunique")
        earliest = ranked.groupby(person, sort=False)[order[0]].transform("min")
        candidates = admissions.gt(1) & ranked[order[0]].eq(earliest) & first_hospital.loc[ranked.index]
        first.loc[candidates.index[candidates]] = pd.NA
    first = first.where(rankable).mask(proven_later.fillna(False), False)
    # A sort tie-breaker cannot establish the first stay that anchors person ranks.
    unverified_first = ordinal.eq(1) & ~first.loc[ranked.index].fillna(False)
    unverified_people = ranked.loc[unverified_first, person]
    rows["stay_order"] = ordinal.mask(ranked[person].isin(unverified_people)).astype("Int64")
    rows["is_first_icu_stay"] = first
    rows["is_first_icu_stay_of_hospital_admission"] = first_hospital
    return rows


@in_binding_unit
def load_stay_identity(*, database, data_path, patient_ids=None):
    """Rank within source history before projecting the requested cohort."""
    family = database.removesuffix("_demo")
    person, hospital, order = _IDENTITY_FIELDS[family]
    config = DATABASE_ID_CONFIG[database]
    stay = config["id_col"]
    columns = list(dict.fromkeys(c for c in (stay, person, hospital, *order) if c))
    all_rows = _stay_table(database, data_path, columns)
    require_unique_keys(all_rows, [stay], table=config["table"])
    rows = _rank_identity(
        all_rows, database=family, stay=stay, person=person, hospital=hospital, order=order,
    )
    evidence = ",".join(f"{config['table']}.{c}" for c in (*order, stay))
    if family == "eicu":
        evidence += (
            ";hospitaladmitoffset_descending;offset_order_when_visit_numbers_incomplete_or_tied"
            ";same_year_encounter_order_unverified"
        )
    if family == "hirid":
        evidence += ";stay_key_only"
    rows = rows.assign(
        _person_id=rows[person].astype("string") if family == "eicu" else rows[person].astype("Int64").astype("string"),
        _hospital_admission_id=(
            rows[hospital].astype("Int64").astype("string") if hospital
            else pd.Series(pd.NA, index=rows.index, dtype="string")
        ),
        person_linkage_available=rows[person].notna() & (family != "hirid"),
        order_evidence=pd.Series(evidence, index=rows.index, dtype="string"),
    )
    rows = _selected_stays(rows, patient_ids)
    return rows.rename(columns={
        "_person_id": "person_id", "_hospital_admission_id": "hospital_admission_id",
    })[[stay, "person_id", "hospital_admission_id", "person_linkage_available", "stay_order",
        "is_first_icu_stay", "is_first_icu_stay_of_hospital_admission", "order_evidence"]]


def _positive_concept_rows(table, concept, stay, *, value):
    frame = table.data
    if frame.empty:
        return pd.DataFrame(columns=[stay, "time", concept])
    times = table.index_column
    if not pd.api.types.is_numeric_dtype(frame[times]):
        raise ValueError(f"{concept} did not return a numeric ICU-hour clock")
    mask = frame[concept].eq(value) if value is not None else frame[concept].notna()
    return frame.loc[mask, [stay, times, concept]].rename(columns={times: "time"})


def _invasive_windows(table, stay, *, sic):
    frame = table.data
    points = _positive_concept_rows(table, "mech_vent", stay, value="invasive")
    if points.empty:
        return pd.DataFrame({stay: points[stay], "start": pd.Series(dtype=float), "end": pd.Series(dtype=float)})
    if sic:
        # SIC's producer supplies exact overlap seconds before publishing
        # hourly points. Intersect with its surviving invasive-hour keys.
        receipt = frame.attrs[COVERAGE_ATTR]
        coverage = pd.DataFrame({stay: receipt[stay], "time": receipt["hour"],
                                 "start": receipt["overlap_start_seconds"],
                                 "end": receipt["overlap_stop_seconds"]})
        coverage[["start", "end"]] /= 3600.0
        return coverage.merge(points[[stay, "time"]].drop_duplicates(), on=[stay, "time"])[[stay, "start", "end"]]
    duration = resolve_dur_var_hours(frame, column=table.dur_var, concept="mech_vent")
    return pd.DataFrame({stay: points[stay], "start": points.time, "end": points.time + duration.loc[points.index]})


def _first_episode(windows, stay):
    """Union touching or overlapping windows. The source owners bridge gaps upstream."""
    windows = windows.loc[np.isfinite(windows.start) & np.isfinite(windows.end) & (windows.start < windows.end)]
    windows = windows.sort_values([stay, "start", "end"], kind="stable")
    previous_end = windows.groupby(stay, sort=False).end.cummax().groupby(windows[stay], sort=False).shift()
    starts_episode = previous_end.isna() | gt(windows.start - previous_end, 0.0)
    first = starts_episode.groupby(windows[stay], sort=False).cumsum().eq(1)
    return windows.loc[first].groupby(stay, sort=False, as_index=False).agg(
        imv_start_hours=("start", "min"), imv_end_hours=("end", "max"),
    )


def _episode_evidence(first, loaded, *, database, stay):
    evidence = pd.Series("mech_vent", index=first.index, dtype="string")
    # Extubation can terminate a MIMIC window. AUMC start/stop rows only
    # corroborate a boundary, since the window's invasive level owns pairing.
    for concept, column in (("vent_start", "imv_start_hours"), ("vent_end", "imv_end_hours")):
        if concept not in loaded:
            continue
        points = _positive_concept_rows(loaded[concept], concept, stay, value=True)
        if points.empty:
            continue
        matches = first[list(dict.fromkeys([stay, "imv_start_hours", column]))]
        matches = matches.merge(points[[stay, "time"]], on=stay)
        if concept == "vent_end" and database.startswith(("miiv", "mimic")):
            valid_stop = gt(matches.time - matches.imv_start_hours, 0.0) & ~gt(matches.time - matches[column], 0)
            stop = matches.loc[valid_stop].groupby(stay).time.min()
            next_end = first[stay].map(stop)
            first[column] = np.minimum(first[column], next_end.fillna(first[column]))
            found = next_end.notna()
        else:
            found = first[stay].isin(matches.loc[np.isclose(matches.time, matches[column], rtol=0, atol=1e-6), stay])
        evidence.loc[found] += "|" + concept

    if database == "sic":
        rules = _positive_concept_rows(loaded["mech_vent_rule"], "mech_vent_rule", stay, value=None)
        if not rules.empty:
            rules = rules.merge(first, on=stay)
            rules = rules.loc[rules.time.lt(rules.imv_end_hours) & (rules.time + 1).gt(rules.imv_start_hours)]
            labels = rules.groupby(stay).mech_vent_rule.agg(lambda values: "|".join(sorted(set(values))))
            labels = first[stay].map(labels)
            evidence.loc[labels.notna()] += "|" + labels.dropna()
    return evidence


def _load_episode_batch(load_concepts, *, database, stay, selected, kwargs, source_available=True):
    family = database.removesuffix("_demo")
    selected = selected[[stay]].reset_index(drop=True)
    selected = selected.assign(
        imv_start_hours=np.nan, imv_end_hours=np.nan, imv_duration_hours=np.nan,
        episode_status=pd.Series("source_unavailable", index=selected.index, dtype="string"),
        episode_evidence=pd.Series(pd.NA, index=selected.index, dtype="string"),
    )
    if family == "eicu":
        # Point/device and shared setting sources include NIV. Generic
        # respiratorycare start/end markers do not identify the interface.
        selected["episode_status"] = selected.episode_status.replace("source_unavailable", "not_invasive_specific")
        return selected
    names = {
        "miiv": ["mech_vent", "vent_start", "vent_end"],
        "mimic": ["mech_vent", "vent_end"],
        "aumc": ["mech_vent", "vent_start", "vent_end"],
        "hirid": ["mech_vent"],
        "sic": ["mech_vent", "mech_vent_rule"],
    }.get(family)
    if not names or selected.empty or not source_available:
        return selected

    availability = {}
    loaded = load_concepts(**{
        **kwargs, "database": database, "concepts": names, "interval": None, "aggregate": False,
        "merge": False, "r_compatible": False, "chunk_size": 0,
        "patient_ids": {stay: selected[stay].tolist()},
        "availability_sink": availability,
    })
    ventilation = loaded["mech_vent"]
    if availability["mech_vent"].reason not in {"source_unavailable", "unmapped"}:
        selected["episode_status"] = pd.Series("no_invasive_episode", index=selected.index, dtype="string")
    windows = _invasive_windows(ventilation, stay, sic=family == "sic")
    first = _first_episode(windows.loc[windows[stay].isin(selected[stay])], stay)
    if first.empty:
        return selected
    first["episode_evidence"] = _episode_evidence(first, loaded, database=database, stay=stay)
    first = first.set_index(stay)
    for column in first:
        selected[column] = selected[stay].map(first[column]).astype(selected[column].dtype)
    selected["imv_duration_hours"] = selected.imv_end_hours - selected.imv_start_hours
    selected.loc[selected.imv_start_hours.notna(), "episode_status"] = "observed_episode"
    return selected


@in_binding_unit
def load_imv_first_episode(load_concepts, *, database, data_path, patient_ids=None, batch_size=None, **kwargs):
    """One episode record per selected source stay, with a nullable hour clock."""
    config = DATABASE_ID_CONFIG[database]
    stay = config["id_col"]
    selector = next(iter(patient_ids), stay) if patient_ids else stay
    rows = _stay_table(database, data_path, list(dict.fromkeys([stay, selector])))
    require_unique_keys(rows, [stay], table=config["table"])
    selected = _selected_stays(rows, patient_ids)
    if database == "miiv" and not selected.empty:
        # All three MIIV inputs use procedureevents. The compatibility
        # reader can return an empty schema on a missing MIIV table, which
        # would obscure the distinction between no evidence and no source.
        table = "procedureevents"
        source = storage_source(database, data_path)
        presence = table_decision(source, table)
        refuse_broken_presence(table, presence)
        if presence.loader is None:
            record_table_unavailable(source, table)
            return _load_episode_batch(
                load_concepts, database=database, stay=stay, selected=selected,
                kwargs=kwargs, source_available=False,
            )
    # Extract the small episode records before combining loader batches.
    # This preserves per-batch native window units and SIC interval receipts.
    size = batch_size or max(1, len(selected))
    return pd.concat([
        _load_episode_batch(
            load_concepts, database=database, stay=stay, selected=selected.iloc[start:start + size],
            kwargs={**kwargs, "data_path": data_path},
        )
        for start in range(0, max(1, len(selected)), size)
    ], ignore_index=True)
