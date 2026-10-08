"""Opt-in, event-preserving MIIV arterial specimen and FiO2 pairing.

No hourly aggregation, unknown-type inference, source release, or clinical I/O
occurs on import. The existing v2 exact-stay/partial-clock loader is reused.
All returned events, pairs, specimen states and counts are private artefacts.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import numbers
from pathlib import Path
import re
from typing import Sequence

import duckdb
import numpy as np
import pandas as pd

from . import source_events as scoped
from ..resources import load_dictionary, package_path
from ..utils.callback_utils import percent_as_numeric

LAB_ITEMS = (50821, 52033, 50816)
CHART_ITEMS = (223835,)
CHART_LOOKBACK_HOURS = 4.0
NAIVE_CLOCK = re.compile(r"\d{4}-\d{2}-\d{2}[ T]\d{2}:\d{2}:\d{2}(?:\.\d+)?")


@dataclass
class ArterialBloodGasResult:
    events: pd.DataFrame
    po2: pd.DataFrame
    pairs: pd.DataFrame
    clock_context: pd.DataFrame
    receipt: dict


def _specimen_audit(conn, lab_paths):
    """Metadata/counts only for specimens anchored by allowed PO2 events.

    This deliberately checks matching specimen identities/type-row completeness
    before Python-side grouping. Unallowed clinical values are never selected.
    """
    conn.execute("CREATE TEMP TABLE lab_assignments AS SELECT * FROM assigned")
    conn.execute("""CREATE TEMP VIEW po2_specimens AS
        SELECT DISTINCT specimen_id FROM lab_assignments
        WHERE identity_status='identity_allowed' AND itemid=50821
          AND specimen_id IS NOT NULL""")
    return conn.execute(f"""WITH raw_keys AS (
        SELECT specimen_id, subject_id, hadm_id, itemid
        FROM {scoped._relation(lab_paths)}
        WHERE itemid IN (50821,52033,50816)
          AND specimen_id IN (SELECT specimen_id FROM po2_specimens)
    ), totals AS (
        SELECT specimen_id,
            count(DISTINCT subject_id)>1 OR count(DISTINCT hadm_id)>1
                OR count(*) FILTER(WHERE subject_id IS NULL)>0 AS identity_conflict,
            count(*) FILTER(WHERE itemid=52033) AS total_type_rows,
            count(*) FILTER(WHERE itemid=50816) AS total_fio2_rows
        FROM raw_keys GROUP BY specimen_id
    ), returned AS (
        SELECT specimen_id,
            count(*) FILTER(WHERE itemid=52033) AS returned_type_rows,
            count(*) FILTER(WHERE itemid=50816) AS returned_fio2_rows
        FROM lab_assignments WHERE identity_status='identity_allowed'
        GROUP BY specimen_id
    ) SELECT t.*, t.total_type_rows-coalesce(r.returned_type_rows,0) AS unreturned_type_rows,
        t.total_fio2_rows-coalesce(r.returned_fio2_rows,0) AS unreturned_fio2_rows
        FROM totals t LEFT JOIN returned r USING(specimen_id)
        ORDER BY specimen_id""").fetchdf()


def _parse_storetime(value):
    if pd.isna(value):
        return pd.NaT, "MISSING"
    if not NAIVE_CLOCK.fullmatch(str(value)):
        return pd.NaT, "INVALID"
    parsed = pd.to_datetime(value, errors="coerce")
    return (pd.NaT, "INVALID") if pd.isna(parsed) else (parsed, "KNOWN")


def _decorate(events, hashes, dictionary):
    events = events.copy()
    events["event_key"] = (
        events.source_table
        + "|"
        + events.source_file
        + "|"
        + events.source_row_number.astype(str)
    )
    events["source_file_sha256"] = events.source_file.map(hashes)
    events["icu_relative_hours"] = (
        events.event_time - events.intime
    ).dt.total_seconds() / 3600
    events["temporal_position"] = np.select(
        [
            events.event_time.lt(events.intime),
            events.outtime.isna(),
            events.event_time.eq(events.outtime),
            events.event_time.gt(events.outtime),
        ],
        ["before_intime", "outtime_unknown", "at_outtime", "after_outtime"],
        default="within_icu",
    )
    events["assignment_rule"] = np.where(
        events.source_table.eq("labevents"), scoped.LAB_ASSIGNMENT, "native_stay_id"
    )
    source = events.raw_valuenum.astype(object).where(
        events.raw_valuenum.notna(), events.raw_value
    )
    events["numeric_source"] = np.where(
        events.raw_valuenum.notna(), "valuenum", "value"
    )
    events["numeric_value"] = pd.to_numeric(source, errors="coerce")
    events["converted_value"] = events.numeric_value.copy()
    events["dictionary_in_range"] = False
    events["unit_status"] = "not_numeric_specimen_text"
    for item, name in [(50821, "po2"), (50816, "fio2_lab"), (223835, "fio2_chart")]:
        at = events.itemid.eq(item)
        if item != 50821:
            events.loc[at, "converted_value"] = percent_as_numeric(source.loc[at])
        low, high = dictionary[name].minimum, dictionary[name].maximum
        values = events.loc[at, "converted_value"]
        events.loc[at, "dictionary_in_range"] = np.isfinite(values) & values.between(
            low, high
        )
        units = (
            events.loc[at, "raw_valueuom"]
            .fillna("")
            .astype(str)
            .str.strip()
            .str.lower()
        )
        expected = {"mmhg", "mm hg"} if item == 50821 else {"%", "percent"}
        events.loc[at, "unit_status"] = np.where(
            units.eq(""),
            "missing_assumed_dictionary",
            np.where(units.isin(expected), "declared_match", "mismatch"),
        )
    events["specimen_text"] = events.raw_value.map(
        lambda v: "" if pd.isna(v) else str(v).strip().upper()
    )
    parsed = [_parse_storetime(v) for v in events.raw_storetime]
    events["parsed_storetime"] = pd.Series(
        [v[0] for v in parsed], index=events.index, dtype="datetime64[ns]"
    )
    events["storetime_status"] = [v[1] for v in parsed]
    bad_order = events.parsed_storetime.lt(events.event_time)
    events.loc[bad_order, "storetime_status"] = "STORE_BEFORE_CHART"
    return events


def _availability(required):
    states = set(required.storetime_status)
    if states != {"KNOWN"}:
        return pd.NaT, ";".join(sorted(states - {"KNOWN"}))
    return required.parsed_storetime.max(), "KNOWN"


def _pair_tables(events, audit):
    """One PO2 ledger row per physical event; no MAX or duplicate collapse."""
    lab = events.loc[events.source_table.eq("labevents")]
    chart = events.loc[events.source_table.eq("chartevents") & events.itemid.eq(223835)]
    audits = audit.set_index("specimen_id").to_dict("index") if len(audit) else {}
    # Index once; each anchor looks up only its specimen and its own stay.
    specimen_groups = dict(
        tuple(lab.groupby(["specimen_id", "subject_id", "hadm_id"], sort=False))
    )
    chart_groups = {}
    for key, group in chart.groupby(["subject_id", "hadm_id", "stay_id"], sort=False):
        ordered = group.sort_values("event_time", kind="stable")
        times = ordered.event_time.to_numpy()
        valid = ordered.dictionary_in_range & ordered.unit_status.ne("mismatch")
        chart_groups[key] = (ordered, times, times[valid.to_numpy()])
    summaries, pair_rows = [], []
    po2_columns = [
        "po2_event_key",
        "subject_id",
        "hadm_id",
        "stay_id",
        "specimen_id",
        "specimen_state",
        "arterial_certified",
        "type_tokens",
        "type_event_keys",
        "empty_type_rows",
        "unreturned_type_rows",
        "unreturned_fio2_rows",
        "po2_distinct_valid_values",
        "po2_charttime",
        "po2_storetime",
    ]
    pair_columns = [
        "po2_event_key",
        "fio2_event_key",
        "pairing_route",
        "pair_status",
        "subject_id",
        "hadm_id",
        "stay_id",
        "specimen_state",
        "arterial_certified",
        "po2_charttime",
        "po2_storetime",
        "fio2_charttime",
        "fio2_storetime",
        "fio2_assigned_stay_id",
        "same_assigned_stay",
        "fio2_minus_po2_hours",
        "po2_value",
        "fio2_percent",
        "pafi_mmhg",
        "fio2_candidate_count",
        "fio2_distinct_valid_values",
        "unreturned_fio2_rows",
        "unit_assumption",
        "available_at",
        "availability_status",
    ]
    for _, anchor in lab.loc[lab.itemid.eq(50821)].iterrows():
        sid = anchor.specimen_id
        related = (
            specimen_groups.get((sid, anchor.subject_id, anchor.hadm_id), lab.iloc[:0])
            if pd.notna(sid)
            else lab.iloc[:0]
        )
        types = related.loc[related.itemid.eq(52033)]
        info = audits.get(sid, {})
        tokens = sorted(set(types.specimen_text) - {""})
        if pd.isna(sid):
            state = "NO_SPECIMEN_ID"
        elif info.get("identity_conflict", False):
            state = "IDENTITY_CONFLICT"
        elif info.get("unreturned_type_rows", 0):
            state = "INCOMPLETE_ALLOWED_TYPE_EVIDENCE"
        elif types.empty:
            state = "TYPE_MISSING"
        elif not tokens:
            state = "TYPE_EMPTY"
        elif len(tokens) > 1:
            state = "CONFLICTING_TYPES"
        else:
            state = "DIRECT_ARTERIAL" if tokens == ["ART."] else "OTHER_RECORDED_TYPE"
        arterial = state == "DIRECT_ARTERIAL"
        pvals = related.loc[
            related.itemid.eq(50821) & related.dictionary_in_range, "converted_value"
        ]
        summaries.append(
            dict(
                zip(
                    po2_columns,
                    [
                        anchor.event_key,
                        anchor.subject_id,
                        anchor.hadm_id,
                        anchor.stay_id,
                        sid,
                        state,
                        arterial,
                        json.dumps(tokens),
                        json.dumps(sorted(types.event_key.tolist())),
                        int(types.specimen_text.eq("").sum()),
                        int(info.get("unreturned_type_rows", 0)),
                        int(info.get("unreturned_fio2_rows", 0)),
                        pvals.nunique(),
                        anchor.raw_charttime,
                        anchor.raw_storetime,
                    ],
                )
            )
        )
        same_specimen = related.loc[related.itemid.eq(50816)]
        latest = chart.iloc[:0]
        group = chart_groups.get((anchor.subject_id, anchor.hadm_id, anchor.stay_id))
        if group is not None:
            ordered, times, valid_times = group
            position = (
                np.searchsorted(
                    valid_times, anchor.event_time.to_datetime64(), side="right"
                )
                - 1
            )
            if position >= 0:
                latest_time = valid_times[position]
                if (
                    latest_time
                    >= (
                        anchor.event_time - pd.Timedelta(hours=CHART_LOOKBACK_HOURS)
                    ).to_datetime64()
                ):
                    left = np.searchsorted(times, latest_time, side="left")
                    right = np.searchsorted(times, latest_time, side="right")
                    latest = ordered.iloc[left:right]
        # Latest valid chart time, retaining all physical ties (also invalid
        # companions at that time), without repeatedly scanning the full table.
        for route, candidates in [
            ("SAME_SPECIMEN_LAB", same_specimen),
            ("PRIOR_CHART_4H", latest),
        ]:
            count = len(candidates)
            distinct = candidates.loc[
                candidates.dictionary_in_range, "converted_value"
            ].nunique()
            rows = [row for _, row in candidates.iterrows()] or [None]
            for other in rows:
                present = other is not None
                numeric = (
                    present
                    and bool(anchor.dictionary_in_range)
                    and bool(other.dictionary_in_range)
                )
                units = (
                    present
                    and anchor.unit_status != "mismatch"
                    and other.unit_status != "mismatch"
                )
                ratio = (
                    float(100 * anchor.converted_value / other.converted_value)
                    if arterial and numeric and units
                    else np.nan
                )
                status = (
                    "NO_MATCH"
                    if not present
                    else "SPECIMEN_NOT_CERTIFIED"
                    if not arterial
                    else "INVALID_NUMERIC_OR_UNIT"
                    if not numeric or not units
                    else "OBSERVED_PHYSICAL_PAIR"
                )
                required = pd.concat(
                    [
                        anchor.to_frame().T,
                        types,
                        other.to_frame().T if present else events.iloc[:0],
                    ],
                    ignore_index=True,
                )
                available, available_status = (
                    _availability(required)
                    if present and arterial
                    else (pd.NaT, "PAIR_NOT_CERTIFIED")
                )
                pair_rows.append(
                    dict(
                        zip(
                            pair_columns,
                            [
                                anchor.event_key,
                                other.event_key if present else None,
                                route,
                                status,
                                anchor.subject_id,
                                anchor.hadm_id,
                                anchor.stay_id,
                                state,
                                arterial,
                                anchor.raw_charttime,
                                anchor.raw_storetime,
                                other.raw_charttime if present else pd.NaT,
                                other.raw_storetime if present else pd.NaT,
                                other.stay_id if present else pd.NA,
                                bool(other.stay_id == anchor.stay_id)
                                if present
                                else None,
                                (other.event_time - anchor.event_time).total_seconds()
                                / 3600
                                if present
                                else np.nan,
                                anchor.converted_value,
                                other.converted_value if present else np.nan,
                                ratio,
                                count,
                                distinct,
                                int(info.get("unreturned_fio2_rows", 0))
                                if route == "SAME_SPECIMEN_LAB"
                                else 0,
                                "|".join([anchor.unit_status, other.unit_status])
                                if present
                                else anchor.unit_status,
                                available,
                                available_status,
                            ],
                        )
                    )
                )
    return pd.DataFrame(summaries, columns=po2_columns), pd.DataFrame(
        pair_rows, columns=pair_columns
    )


def extract_miiv_arterial_blood_gas(data_source, *, allowed_stay_ids: Sequence[int]):
    """Extract private event/pair ledgers for an explicit allowed MIIV roster.

    This is a retrospective source view. ``available_at`` is unknown if any
    required storetime is missing/invalid or precedes charttime. It does not
    authorize real extraction, generate hourly features, or advance a release.
    """
    if data_source.config.name != "miiv":
        raise scoped.SourceEventContractError("only_miiv_supported")
    allowed = list(allowed_stay_ids) if allowed_stay_ids is not None else []
    if not allowed or any(
        isinstance(v, (bool, np.bool_))
        or not isinstance(v, numbers.Integral)
        or v <= 0
        or v > np.iinfo(np.int64).max
        for v in allowed
    ):
        raise scoped.SourceEventContractError(
            "explicit_positive_integer_stay_ids_required"
        )
    if len(set(allowed)) != len(allowed):
        raise scoped.SourceEventContractError("duplicate_requested_stay_id")
    allowed = sorted(map(int, allowed))
    dictionary = load_dictionary()
    with package_path("concept-dict.json") as path:
        dictionary_sha = scoped._sha(path)
    for name, table, item in [
        ("po2", "labevents", 50821),
        ("fio2_lab", "labevents", 50816),
        ("fio2_chart", "chartevents", 223835),
    ]:
        sources = dictionary[name].for_data_source(data_source.config)
        expected_callback = (
            None if name == "po2" else "transform_fun(percent_as_numeric)"
        )
        if (
            dictionary[name].callback is not None
            or dictionary[name].minimum is None
            or dictionary[name].maximum is None
            or len(sources) != 1
            or sources[0].table != table
            or list(sources[0].ids) != [item]
            or sources[0].callback != expected_callback
        ):
            raise scoped.SourceEventContractError(
                "arterial_api_dictionary_mapping_changed"
            )
    clocks = [
        p.resolve()
        for p in [
            data_source.base_path / "icustays.parquet",
            data_source.base_path / "icu/icustays.parquet",
        ]
        if p.is_file()
    ]
    if len(clocks) != 1:
        raise scoped.SourceEventContractError("one_icustays_parquet_required")
    lab_paths = scoped._files(data_source, "labevents", list(LAB_ITEMS))
    chart_paths = scoped._files(data_source, "chartevents", list(CHART_ITEMS))
    scoped._schemas(lab_paths, {"specimen_id", "labevent_id"}, set())
    files = (
        [("icustays", p) for p in clocks]
        + [("labevents", p) for p in lab_paths]
        + [("chartevents", p) for p in chart_paths]
    )
    hashes = {str(p): scoped._sha(p) for _, p in files}
    conn = duckdb.connect(":memory:", config={"threads": 1, "temp_directory": ""})
    try:
        scoped._prepare_clocks(conn, clocks, allowed)
        context = conn.execute("""SELECT subject_id,hadm_id,stay_id,intime,outtime,outtime_status,
            context_stay_count,context_unknown_outtime_count FROM clocks
            WHERE stay_id IN (SELECT stay_id FROM requested) ORDER BY stay_id""").fetchdf()
        labs, lab_counts = scoped._read_scoped_events(
            conn, lab_paths, "labevents", list(LAB_ITEMS)
        )
        audit = _specimen_audit(conn, lab_paths)
        charts, chart_counts = scoped._read_scoped_events(
            conn, chart_paths, "chartevents", list(CHART_ITEMS)
        )
    finally:
        conn.close()
    events = _decorate(pd.concat([labs, charts], ignore_index=True), hashes, dictionary)
    po2, pairs = _pair_tables(events, audit)
    for _, path in files:
        if hashes[str(path)] != scoped._sha(path):
            raise scoped.SourceEventContractError("source_changed_during_extraction")
    with package_path("concept-dict.json") as path:
        if dictionary_sha != scoped._sha(path):
            raise scoped.SourceEventContractError(
                "dictionary_changed_during_extraction"
            )
    receipt = dict(
        schema="miiv_arterial_blood_gas_v1",
        api_source_sha256=scoped._sha(Path(__file__)),
        source_filter_applied_before_python=True,
        lab_assignment_rule=scoped.LAB_ASSIGNMENT,
        partial_clock_rule="reuse source_events v2 identifiable completions",
        source_loader_sha256=scoped._sha(Path(scoped.__file__)),
        dictionary_sha256=dictionary_sha,
        source_files=[
            dict(table=t, path=str(p), sha256=hashes[str(p)]) for t, p in files
        ],
        source_filter_counts=lab_counts + chart_counts,
        event_rows=len(events),
        po2_rows=len(po2),
        pair_rows=len(pairs),
        allowed_stay_sha256=hashlib.sha256(
            json.dumps(allowed, separators=(",", ":")).encode()
        ).hexdigest(),
        chart_pairing="same subject/hadm/assigned stay; most recent valid chart within inclusive preceding 4h; all physical ties",
        lab_pairing="same nonnull specimen ID; identity conflict blocks arterial certification; all physical pairs",
        specimen_audit_scope="metadata of three lab items for allowed PO2 specimens, including unreturned type-row counts; no unallowed clinical-value fetch",
        official_comparison="4h is a fixed MIT-LCP comparison algorithm, not physiological validity; no replication of MAX or subject-only joins",
        numeric_policy="existing dictionary PO2/FiO2 bounds and percent callback; no hourly aggregation; missing units explicitly assumed dictionary, mismatches not converted",
        availability_policy="all required PO2/type/FiO2 storetimes known and >= their charttimes; otherwise unknown",
        privacy="all ledgers and counts private; no inference of specimen type; no release or extraction authorization",
    )
    return ArterialBloodGasResult(events, po2, pairs, context, receipt)
