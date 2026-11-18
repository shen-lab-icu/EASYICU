"""Source-defined producers for the opt-in thesis support exports."""

from __future__ import annotations

from collections.abc import Mapping

import numpy as np
import pandas as pd

from ..config import DATABASE_ID_CONFIG
from ..concept_output_sources import SPECIMEN_STATUS_LEVELS, SPECIMEN_STATUS_OUTPUT_COLUMNS
from ..datasource import storage_source
from ..io.identity import require_unique_keys
from ..storage_binding import in_binding_unit
from ..storage_layout import record_table_unavailable, refuse_broken_presence, table_decision
from ..table.duration import resolve_dur_var_hours
from ..utils.thresholds import gt
from .errors import ConceptTableReadError
from .sicdb_sources import COVERAGE_ATTR
from .stay_tables import stay_table_frame

# The stay key/table come from the same registry as cohort discovery. The
# other columns are the source's person, hospital admission (if any) and
# ranking fields, in precedence order. eICU offsets count back to admission.
_IDENTITY_FIELDS = {
    "miiv": ("subject_id", "hadm_id", ("intime",)),
    "mimic": ("subject_id", "hadm_id", ("intime",)),
    "aumc": ("patientid", None, ("admissioncount", "admittedat")),
    "sic": ("PatientID", None, ("OffsetAfterFirstAdmission", "AdmissionYear", "ICUOffset")),
    "eicu": (
        "uniquepid", "patienthealthsystemstayid",
        ("hospitaldischargeyear", "unitvisitnumber", "hospitaladmitoffset"),
    ),
    "hirid": ("patientid", None, ()),
}


def _normalise_stay_filters(database, patient_ids):
    if patient_ids is None:
        return None
    config = DATABASE_ID_CONFIG[database]
    if not isinstance(patient_ids, Mapping):
        patient_ids = {config["id_col"]: patient_ids}
    for key, members in patient_ids.items():
        if not pd.api.types.is_list_like(members):
            raise ConceptTableReadError(
                database=database, table=config["table"], stage="patient_ids",
                detail=f"filter {key!r} requires a sequence of IDs",
            )
    return patient_ids


def _stay_table(database, data_path, columns, *, filters, optional=()):
    source = storage_source(database, data_path)
    table = DATABASE_ID_CONFIG[database]["table"]
    requested = list(dict.fromkeys([*columns, *(filters or {})]))

    def read():
        rows = source.load_table(table, columns=requested, verbose=False).data
        return rows.loc[:, rows.columns.intersection(requested)]

    rows = stay_table_frame(
        source, table, ("support_export", tuple(requested)), read,
    )
    for key in filters or {}:
        if key not in rows:
            raise ConceptTableReadError(
                database=database, table=table, stage="patient_ids",
                detail=f"filter column {key!r} is absent from the stay table",
            )
    return rows.loc[:, [key for key in requested if key not in optional or key in rows]]


def _selected_stays(rows, patient_ids):
    if patient_ids is None:
        return rows
    selected = pd.Series(bool(patient_ids), index=rows.index)
    for key, members in patient_ids.items():
        selected &= rows[key].isin(members)
    return rows.loc[selected]


def select_support_cohort(database, data_path, patient_ids):
    """Resolve a support export's cohort to one stay-key partition before batching."""
    filters = _normalise_stay_filters(database, patient_ids)
    config = DATABASE_ID_CONFIG[database]
    stay = config["id_col"]
    if filters is None:
        return None
    if not filters or list(filters) == [stay]:
        return {stay: list(filters[stay]) if filters else []}
    rows = _stay_table(database, data_path, [stay], filters=filters)
    require_unique_keys(rows, [stay], table=config["table"])
    return {stay: _selected_stays(rows, filters)[stay].tolist()}


def _sort_stays(rows, order):
    return rows.sort_values(list(order), ascending=[c != "hospitaladmitoffset" for c in order], kind="stable")


def _valid_ordinal(values):
    return (np.isfinite(values) & values.ge(1) & values.mod(1).eq(0)).fillna(False)


def _all_within(mask, groups):
    return mask.fillna(False).groupby(groups, sort=False).transform("all").eq(True)


def _first_by_clock(clock, groups):
    present = _all_within(clock.notna(), groups)
    earliest = clock.eq(clock.groupby(groups, sort=False).transform("min"))
    candidates = earliest.groupby(groups, sort=False).transform("sum")
    return earliest.astype("boolean").where(present & (~earliest | candidates.eq(1)))


def _clock_reverses(rows, groups, column, *, ties=None):
    """Check known clock readings in the order established by another source field."""
    known = rows.loc[np.isfinite(rows[column])]
    group_keys = [known[key] for key in groups]
    clock = known[column]
    prior_max = clock.groupby(group_keys, sort=False).cummax().groupby(group_keys, sort=False).shift()
    reversed_clock = clock.lt(prior_max)
    if ties is not None:
        tied_value = known.groupby([*groups, ties], sort=False)[column].transform("nunique")
        reversed_clock |= tied_value.gt(1)
    return reversed_clock.reindex(rows.index, fill_value=False)


def _rank_identity(rows, *, database, stay, person, hospital, order):
    rows = rows.copy()
    first_hospital = pd.Series(pd.NA, index=rows.index, dtype="boolean")
    if database == "hirid":
        # The source key identifies an ICU stay and provides no longitudinal person clock.
        return _sort_stays(rows.assign(
            stay_order=pd.Series(pd.NA, index=rows.index, dtype="Int64"),
            is_first_icu_stay=first_hospital,
            is_first_icu_stay_of_hospital_admission=first_hospital,
        ), [stay])

    if "intime" in order:
        rows["intime"] = pd.to_datetime(rows["intime"], errors="coerce")
    for column in order:
        if column != "intime":
            rows[column] = pd.to_numeric(rows[column], errors="coerce")

    sort_order = [*order, stay]
    valid = rows[person].notna() & rows[order[0]].notna()
    complete = pd.Series(True, index=rows.index)
    proven_later = pd.Series(False, index=rows.index)
    if database == "aumc":
        count = rows["admissioncount"]
        valid_count = _valid_ordinal(count)
        tied = rows.duplicated([person, "admissioncount"], keep=False)
        unresolved = rows.duplicated([person, "admissioncount", "admittedat"], keep=False)
        valid &= valid_count & ~unresolved & (~tied | np.isfinite(rows["admittedat"]))
        valid &= ~_clock_reverses(_sort_stays(rows, order), [person], "admittedat").reindex(rows.index)
        proven_later = valid_count & count.gt(1)
    elif database == "sic":
        offset = rows["OffsetAfterFirstAdmission"]
        icu_offset = rows["ICUOffset"]
        malformed_icu_offset = icu_offset.notna() & (~np.isfinite(icu_offset) | icu_offset.lt(0))
        rows["ICUOffset"] = icu_offset.mask(malformed_icu_offset)
        valid &= np.isfinite(offset) & offset.ge(0)
        sort_order = ["OffsetAfterFirstAdmission", "ICUOffset", stay]
        rows["_icu_start"] = offset + rows["ICUOffset"]
        sorted_cases = _sort_stays(rows, sort_order)
        inconsistent_year = _clock_reverses(sorted_cases, [person], "AdmissionYear", ties="OffsetAfterFirstAdmission")
        inconsistent_icu_start = _clock_reverses(sorted_cases, [person], "_icu_start")
        valid &= ~malformed_icu_offset & ~(inconsistent_year | inconsistent_icu_start).reindex(rows.index)
        # On a shared admission clock every sibling needs a distinct ICU clock.
        tied_admission = rows.duplicated([person, "OffsetAfterFirstAdmission"], keep=False)
        tied_icu = rows["_icu_start"].notna() & rows.duplicated([person, "_icu_start"], keep=False)
        complete &= ~tied_icu & (~tied_admission | rows["ICUOffset"].notna())
        proven_later = np.isfinite(offset) & offset.gt(0)
    elif database == "eicu":
        groups = [rows[person], rows[hospital]]
        visit = rows["unitvisitnumber"]
        valid_visit = _valid_ordinal(visit)
        tied_visit = rows.duplicated([person, hospital, "unitvisitnumber"], keep=False)
        # Use one clock throughout an admission. Offsets must order every
        # sibling when visit numbers fail to establish a complete order.
        by_visit = _all_within(valid_visit & ~tied_visit, groups)
        rows["_visit_order"] = visit.where(by_visit)
        sort_order = ["hospitaldischargeyear", hospital, "_visit_order", "hospitaladmitoffset", stay]
        offset = rows["hospitaladmitoffset"]
        hospital_clock = -offset.where(np.isfinite(offset))
        first_hospital = _first_by_clock(visit.where(by_visit), groups).where(
            by_visit, _first_by_clock(hospital_clock, groups),
        )
        later_visit = valid_visit & visit.gt(1)
        all_later = _all_within(later_visit, groups)
        known_clocks = _sort_stays(rows.loc[valid_visit & np.isfinite(offset)], ["hospitaladmitoffset", "unitvisitnumber"])
        reversed_visit = _clock_reverses(known_clocks, [person, hospital], "unitvisitnumber")
        starts_with_return = ~by_visit & first_hospital.fillna(False) & later_visit & ~all_later
        contradicted = ~_all_within(~(reversed_visit.reindex(rows.index, fill_value=False) | starts_with_return), groups)
        first_hospital = first_hospital.mask(contradicted).mask(later_visit | all_later, False)
        proven_later = (first_hospital.eq(False).fillna(False) & ~contradicted) | all_later
        year = rows["hospitaldischargeyear"]
        same_year = year.groupby(groups, sort=False).transform("nunique").eq(1)
        valid_offset = hospital_clock.notna() & ~rows.duplicated([person, hospital, "hospitaladmitoffset"], keep=False)
        hospital_valid = (by_visit | valid_offset) & ~contradicted
        valid &= rows[hospital].notna() & hospital_valid & np.isfinite(year) & same_year
        admission_count = rows.groupby([person, "hospitaldischargeyear"], sort=False)[hospital].transform("nunique")
        hospital_positions = _sort_stays(rows, sort_order).groupby([person, hospital], sort=False).cumcount() + 1
        complete &= admission_count.eq(1) & (~valid_visit | visit.le(hospital_positions))
    elif hospital:
        first_hospital = _first_by_clock(rows["intime"], [rows[person], rows[hospital]])
        complete &= ~rows.duplicated([person, "intime"], keep=False)
        proven_later = first_hospital.eq(False).fillna(False)

    # A sibling with an unknown position can precede any ranked stay.
    rankable = _all_within(valid, rows[person])
    rows = _sort_stays(rows, sort_order)
    ranked = rows.loc[rankable]
    positions = ranked.groupby(person, sort=False).cumcount()
    ordinal = positions + 1
    first = positions.eq(0).astype("boolean").reindex(rows.index)
    if database == "aumc":
        first &= rows["admissioncount"].eq(1)
        complete &= rows["admissioncount"].le(ordinal)
    elif database == "sic":
        offset = rows["OffsetAfterFirstAdmission"]
        first_offset = offset.groupby(rows[person], sort=False).transform("min")
        first_ties = (offset.eq(first_offset) & ~complete).groupby(rows[person], sort=False).transform("any")
        first = first.mask(offset.eq(0) & first_ties)
    elif database == "eicu":
        # The encounter key orders otherwise indistinguishable admissions in
        # the same discharge year. Only their hospital firsts are candidates.
        earliest = ranked.groupby(person, sort=False)[order[0]].transform("min")
        candidates = admission_count.loc[ranked.index].gt(1) & ranked[order[0]].eq(earliest) & first_hospital.loc[ranked.index]
        first.loc[candidates.index[candidates]] = pd.NA
    else:
        first = _first_by_clock(rows["intime"], rows[person])
    first = first.where(rankable).mask(proven_later.fillna(False), False)
    complete &= ~ordinal.eq(1).reindex(rows.index, fill_value=False) | first.fillna(False)
    rows["stay_order"] = ordinal.where(_all_within(complete, rows[person])).astype("Int64")
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
    filters = _normalise_stay_filters(database, patient_ids)
    columns = [c for c in (stay, person, hospital, *order) if c]
    optional = ("ICUOffset",) if family == "sic" else ()
    all_rows = _stay_table(database, data_path, columns, filters=filters, optional=optional)
    require_unique_keys(all_rows, [stay], table=config["table"])
    selected = _selected_stays(all_rows, filters) if filters is not None else None
    if optional and "ICUOffset" not in all_rows:
        all_rows = all_rows.assign(ICUOffset=np.nan)
    rows = _rank_identity(
        all_rows, database=family, stay=stay, person=person, hospital=hospital, order=order,
    )
    evidence = ",".join(f"{config['table']}.{c}" for c in (order or (stay,)))
    if family == "eicu":
        evidence += (
            ";hospitaladmitoffset_descending;offset_order_when_visit_numbers_incomplete_or_tied"
            ";same_year_encounter_order_unverified;encounter_grouping=patient.patienthealthsystemstayid"
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
    if selected is not None:
        rows = rows.loc[rows[stay].isin(selected[stay])]
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


def _episode_evidence(first, loaded, *, family, stay):
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
        if concept == "vent_end" and family in {"miiv", "mimic"}:
            valid_stop = gt(matches.time - matches.imv_start_hours, 0.0) & ~gt(matches.time - matches[column], 0)
            stop = matches.loc[valid_stop].groupby(stay).time.min()
            next_end = first[stay].map(stop)
            first[column] = np.minimum(first[column], next_end.fillna(first[column]))
            found = next_end.notna()
        else:
            found = first[stay].isin(matches.loc[np.isclose(matches.time, matches[column], rtol=0, atol=1e-6), stay])
        evidence.loc[found] += "|" + concept

    if family == "sic":
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
    first["episode_evidence"] = _episode_evidence(first, loaded, family=family, stay=stay)
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
    filters = _normalise_stay_filters(database, patient_ids)
    rows = _stay_table(database, data_path, [stay], filters=filters)
    require_unique_keys(rows, [stay], table=config["table"])
    selected = _selected_stays(rows, filters)
    if database.removesuffix("_demo") == "miiv" and not selected.empty:
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


def load_pao2_arterial_hourly_min(load_concepts, *, native_charttime_axis, **kwargs):
    """Reduce admitted arterial specimens by hour, keeping the winning specimen's status."""
    loaded = load_concepts(**{
        **kwargs, "merge": False, "interval": None, "aggregate": False, "r_compatible": False,
    })
    stay = DATABASE_ID_CONFIG[kwargs["database"]]["id_col"]
    status, = SPECIMEN_STATUS_OUTPUT_COLUMNS["po2"]
    columns = [stay, "charttime", "po2", status]
    if isinstance(loaded, Mapping):
        if not loaded:
            return pd.DataFrame(columns=columns)
        loaded = loaded["po2"]
    if isinstance(loaded, pd.DataFrame):
        loaded = native_charttime_axis(loaded)
    else:
        loaded = loaded.data.rename(columns={loaded.index_column: "charttime"})
    if loaded.empty:
        return loaded.reindex(columns=columns)

    if status not in loaded:
        loaded = loaded.assign(**{status: pd.Series(pd.NA, index=loaded.index, dtype="string")})
    observed = loaded.loc[loaded.po2.notna(), columns].copy()
    if observed.empty:
        return observed
    if not pd.api.types.is_numeric_dtype(observed.charttime) or not np.isfinite(observed.charttime).fillna(False).all():
        raise ValueError("po2 did not return a finite numeric ICU-hour clock")

    # Equal minima use the earliest measurement and then the weakest evidence
    # among identical-time readings. The value and status always travel together.
    observed["_status_order"] = pd.Categorical(observed[status], categories=SPECIMEN_STATUS_LEVELS, ordered=True)
    observed = observed.sort_values(["po2", "charttime", "_status_order"], na_position="first")
    observed["charttime"] = np.floor(observed.charttime)
    keys = [stay, "charttime"]
    return observed.drop_duplicates(keys).sort_values(keys).loc[:, columns]
