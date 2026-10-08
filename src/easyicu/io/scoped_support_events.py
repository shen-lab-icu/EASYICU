"""Opt-in MIIV ventilation procedure records, not real-time support truth.

Only native-stay 225792/225794 records are fetched. Unknown clocks, final
statuses, duplicates and overlapping intervals remain explicit. No clinical
I/O on import, imputation, interval merging, hourly grid or release operation.
"""

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

ITEMS = {225792: "invasive", 225794: "noninvasive"}
NAIVE_CLOCK = re.compile(r"\d{4}-\d{2}-\d{2}[ T]\d{2}:\d{2}:\d{2}(?:\.\d+)?")
AFFIRMATIVE_FINAL_STATUSES = {"finishedrunning", "stopped", "paused"}
OPTIONAL_COLUMNS = ("orderid", "linkorderid", "continueinnextdept")


@dataclass
class ScopedSupportResult:
    events: pd.DataFrame
    clock_context: pd.DataFrame
    receipt: dict


def _parse_clock(value):
    if pd.isna(value):
        return pd.NaT, "missing"
    if not NAIVE_CLOCK.fullmatch(str(value)):
        return pd.NaT, "invalid"
    stamp = pd.to_datetime(value, errors="coerce")
    return (pd.NaT, "invalid") if pd.isna(stamp) else (stamp, "known")


def _procedure_files(source):
    # procedureevents is an ICU table, unlike the old chart/lab helper's default.
    bucket = source.resolve_bucket_directory("procedureevents")
    if bucket is not None:
        paths = source.get_bucket_files_for_ids(bucket, list(ITEMS), duckdb)[2]
    else:
        flat = source.resolve_flat_parquet_directory("procedureevents")
        paths = (
            sorted(flat.glob("*.parquet"))
            if flat is not None
            else [
                p
                for p in (
                    source.base_path / "procedureevents.parquet",
                    source.base_path / "icu/procedureevents.parquet",
                )
                if p.is_file()
            ]
        )
    paths = sorted({Path(p).resolve() for p in paths})
    if not paths:
        raise scoped.SourceEventContractError("missing_parquet_source:procedureevents")
    return paths


def _prepare_native_clocks(conn, paths, allowed):
    # Native stay assignment needs only requested clocks, not other admissions
    # or sibling-stay exit ordering. Missing raw exit is distinct from bad text.
    scoped._schemas(
        paths,
        {"subject_id", "hadm_id", "stay_id", "intime", "outtime"},
        {"intime", "outtime"},
    )
    conn.register(
        "requested", pd.DataFrame({"stay_id": pd.Series(allowed, dtype="int64")})
    )
    conn.execute(f"""CREATE TEMP TABLE targets AS
        SELECT subject_id, hadm_id, stay_id,
            {scoped._clock_sql("intime")} AS intime,
            {scoped._clock_sql("outtime")} AS outtime,
            outtime IS NULL AS outtime_missing
        FROM {scoped._relation(paths)}
        WHERE stay_id IN (SELECT stay_id FROM requested)""")
    checks = {
        "missing_requested_stay": "SELECT count(*) FROM requested WHERE stay_id NOT IN (SELECT stay_id FROM targets)",
        "duplicate_stay_id": "SELECT count(*) FROM (SELECT stay_id FROM targets GROUP BY stay_id HAVING count(*) != 1)",
        "invalid_identity_or_clock": "SELECT count(*) FROM targets WHERE subject_id IS NULL OR hadm_id IS NULL OR intime IS NULL OR (outtime IS NULL AND NOT outtime_missing) OR intime > outtime",
        "hospital_subject_conflict": "SELECT count(*) FROM (SELECT hadm_id FROM targets GROUP BY hadm_id HAVING count(DISTINCT subject_id) != 1)",
    }
    counts = {k: int(conn.execute(q).fetchone()[0]) for k, q in checks.items()}
    if any(counts.values()):
        raise scoped.SourceEventContractError(
            "invalid_requested_native_clock_context", counts=counts
        )
    return conn.execute("""SELECT subject_id,hadm_id,stay_id,intime,outtime,
        CASE WHEN outtime_missing THEN 'unknown' ELSE 'known' END AS outtime_status
        FROM targets ORDER BY stay_id""").fetchdf()


def _decorate(events, hashes, dictionary_sha):
    out = events.copy()
    out["source_table"] = "procedureevents"
    out["source_file_sha256"] = out.source_file.map(hashes)
    out["event_key"] = (
        "procedureevents|" + out.source_file + "|" + out.source_row_number.astype(str)
    )
    out["support_type"] = out.source_item_id.map(ITEMS)
    out["dictionary_sha256"] = dictionary_sha
    out["assignment_rule"] = "native_stay_id"
    for name in ("starttime", "endtime", "storetime"):
        parsed = [_parse_clock(v) for v in out[f"raw_{name}"]]
        out[name] = pd.Series(pd.to_datetime([x[0] for x in parsed]), index=out.index)
        out[f"{name}_status"] = [x[1] for x in parsed]
        out[f"{name}_hours"] = (out[name] - out.intime).dt.total_seconds() / 3600
    out["duration_hours"] = (out.endtime - out.starttime).dt.total_seconds() / 3600
    out["interval_status"] = np.select(
        [
            out.starttime_status.eq("invalid") | out.endtime_status.eq("invalid"),
            out.starttime.isna(),
            out.endtime.isna(),
            out.duration_hours.lt(0),
            out.duration_hours.eq(0),
        ],
        ["invalid_clock", "missing_start", "missing_end", "reversed", "zero_length"],
        default="positive_duration",
    )
    out["store_minus_start_hours"] = (
        out.storetime - out.starttime
    ).dt.total_seconds() / 3600
    out["store_minus_end_hours"] = (
        out.storetime - out.endtime
    ).dt.total_seconds() / 3600
    out["record_clock_status"] = np.select(
        [
            out.starttime.isna() | out.storetime.isna(),
            out.storetime.lt(out.starttime),
            out.endtime.isna(),
            out.storetime.lt(out.endtime),
        ],
        [
            "unknown_start_or_store",
            "store_before_start",
            "end_unknown",
            "store_before_end",
        ],
        default="store_at_or_after_end",
    )
    status = out.raw_statusdescription.map(
        lambda x: "" if pd.isna(x) else str(x).strip().lower()
    )
    out["ultimate_status_class"] = np.select(
        [
            status.eq(""),
            status.isin(AFFIRMATIVE_FINAL_STATUSES),
            status.isin({"cancelled", "canceled", "rewritten"}),
        ],
        ["missing", "affirmative_terminal_record", "cancelled_or_rewritten"],
        default="unrecognized",
    )
    out["numeric_duration"] = pd.to_numeric(out.raw_value, errors="coerce")
    units = out.raw_valueuom.map(lambda x: "" if pd.isna(x) else str(x).strip().lower())
    multipliers = units.map(
        {
            "min": 1 / 60,
            "minute": 1 / 60,
            "minutes": 1 / 60,
            "hour": 1.0,
            "hours": 1.0,
            "day": 24.0,
            "days": 24.0,
        }
    )
    out["duration_value_hours"] = (out.numeric_duration * multipliers).where(
        np.isfinite(out.numeric_duration)
    )
    out["duration_unit_status"] = np.select(
        [units.eq(""), multipliers.notna()],
        ["missing", "recognized_duration_unit"],
        default="not_duration_unit",
    )
    out["duration_value_minus_interval_hours"] = (
        out.duration_value_hours - out.duration_hours
    )
    # Start claim is deliberately independent of endtime and ultimate status.
    out["recorded_start_claim_at"] = out.storetime.where(
        out.starttime.notna() & out.storetime.notna() & out.storetime.ge(out.starttime)
    )
    out["completed_interval_record_at"] = out.storetime.where(
        out.interval_status.eq("positive_duration") & out.storetime.ge(out.endtime)
    )
    out["start_position_in_icu"] = np.select(
        [
            out.starttime.isna(),
            out.starttime.lt(out.intime),
            out.outtime.isna(),
            out.starttime.eq(out.outtime),
            out.starttime.gt(out.outtime),
        ],
        ["unknown", "before_intime", "outtime_unknown", "at_outtime", "after_outtime"],
        default="within_icu",
    )
    return out


def extract_miiv_support_events(
    data_source, *, allowed_stay_ids: Sequence[int]
) -> ScopedSupportResult:
    """Return private native procedure records after exact SQL identity filtering.

    Does not certify historical record versions, ongoing support or absence of
    ventilation. Invalid event clocks are retained; invalid identity clocks fail.
    No physical-page isolation claim is made. No other clinical table is read.
    """
    if data_source.config.name != "miiv":
        raise scoped.SourceEventContractError("only_miiv_supported")
    ids = list(allowed_stay_ids)
    if not ids or any(
        isinstance(x, bool) or not isinstance(x, numbers.Integral) or x <= 0
        for x in ids
    ):
        raise scoped.SourceEventContractError("positive_integer_stay_ids_required")
    allowed = sorted({int(x) for x in ids})
    if len(allowed) != len(ids):
        raise scoped.SourceEventContractError("duplicate_allowed_stay_ids")
    definition = load_dictionary()["mech_vent"]
    specs = definition.for_data_source(data_source.config)
    if (
        len(specs) != 1
        or specs[0].table != "procedureevents"
        or set(specs[0].ids) != set(ITEMS)
        or specs[0].sub_var != "itemid"
        or specs[0].dur_var != "endtime"
    ):
        raise scoped.SourceEventContractError("support_dictionary_mapping_changed")
    with package_path("concept-dict.json") as p:
        dictionary_sha = scoped._sha(p)
    clock_files = [
        p.resolve()
        for p in (
            data_source.base_path / "icustays.parquet",
            data_source.base_path / "icu/icustays.parquet",
        )
        if p.is_file()
    ]
    if len(clock_files) != 1:
        raise scoped.SourceEventContractError("one_icustays_parquet_required")
    event_files = _procedure_files(data_source)
    required = {
        "subject_id",
        "hadm_id",
        "stay_id",
        "itemid",
        "starttime",
        "endtime",
        "storetime",
        "value",
        "valueuom",
        "statusdescription",
    }
    names = scoped._schemas(
        event_files, required, {"starttime", "endtime", "storetime"}
    )
    files = [("icustays", p) for p in clock_files] + [
        ("procedureevents", p) for p in event_files
    ]
    hashes = {str(p): scoped._sha(p) for _, p in files}
    conn = duckdb.connect(":memory:", config={"threads": 1, "temp_directory": ""})
    try:
        context = _prepare_native_clocks(conn, clock_files, allowed)
        optional = ",".join(
            f"e.{x}" if x in names else f"NULL AS {x}" for x in OPTIONAL_COLUMNS
        )
        conn.execute(f"""CREATE VIEW domain AS SELECT
            e.subject_id,e.hadm_id,e.stay_id,e.itemid AS source_item_id,
            e.filename AS source_file,e.file_row_number AS source_row_number,
            e.starttime AS raw_starttime,e.endtime AS raw_endtime,e.storetime AS raw_storetime,
            e.value AS raw_value,e.valueuom AS raw_valueuom,e.statusdescription AS raw_statusdescription,
            {optional}, t.intime,t.outtime,
            CASE WHEN e.subject_id IS DISTINCT FROM t.subject_id OR e.hadm_id IS DISTINCT FROM t.hadm_id
                 THEN 'identity_mismatch' ELSE 'identity_allowed' END AS identity_status
            FROM {scoped._relation(event_files, events=True)} e JOIN targets t ON e.stay_id=t.stay_id
            WHERE e.itemid IN (225792,225794) AND e.stay_id IN (SELECT stay_id FROM requested)""")
        counts = [
            dict(source_item_id=int(i), reason=r, count=int(n))
            for i, r, n in conn.execute(
                "SELECT source_item_id,identity_status,count(*) FROM domain GROUP BY ALL ORDER BY 1,2"
            ).fetchall()
        ]
        # Only requested identity-matched source values cross into Python.
        events = conn.execute(
            "SELECT * EXCLUDE(identity_status) FROM domain WHERE identity_status='identity_allowed'"
        ).fetchdf()
    finally:
        conn.close()
    events = _decorate(events, hashes, dictionary_sha)
    for _, p in files:
        if hashes[str(p)] != scoped._sha(p):
            raise scoped.SourceEventContractError("source_changed_during_extraction")
    with package_path("concept-dict.json") as p:
        if dictionary_sha != scoped._sha(p):
            raise scoped.SourceEventContractError(
                "dictionary_changed_during_extraction"
            )
    receipt = dict(
        schema="miiv_scoped_support_events_v1",
        api_sha256=scoped._sha(Path(__file__)),
        source_loader_sha256=scoped._sha(Path(scoped.__file__)),
        dictionary_sha256=dictionary_sha,
        source_files=[
            dict(table=t, path=str(p), sha256=hashes[str(p)]) for t, p in files
        ],
        allowed_stay_sha256=hashlib.sha256(
            json.dumps(allowed, separators=(",", ":")).encode()
        ).hexdigest(),
        source_filter_counts=counts,
        event_rows=len(events),
        clock_rows=len(context),
        optional_source_columns={x: x in names for x in OPTIONAL_COLUMNS},
        source_filter_applied_before_python=True,
        assignment_rule="native_stay_id",
        time_coordinate="naive source-local deidentified timestamps; relative hours from ICU intime",
        interval_convention="retrospective [starttime,endtime); no merging or hourly filling",
        recorded_start_claim="known start and store>=start; independent of future end/ultimate status; not ongoing support certification",
        completed_interval_record_evidence="positive interval and store>=end; strict store<decision cutoff; single-snapshot working evidence, not version-history certification",
        absence="unknown; documentation not mandatory; no-event stays remain in clock_context",
        privacy="all events, clock ledger and counts private; no release/current/model operation",
    )
    return ScopedSupportResult(events, context, receipt)


def support_window_evidence(
    events: pd.DataFrame, *, window_start, window_end, decision_time
) -> pd.DataFrame:
    """Per-record retrospective and decision-visible evidence for one past window.

    This is a working record-clock rule, not an eligibility certification.
    Returns no negative-support class, including for cancelled/missing records.
    End/status-dependent columns must stay out of start-claim feature histories.
    """
    clocks = [_parse_clock(x)[0] for x in (window_start, window_end, decision_time)]
    start, end, cutoff = clocks
    if any(pd.isna(x) for x in clocks) or not start < end <= cutoff:
        raise scoped.SourceEventContractError("finite_naive_past_window_required")
    out = events[["event_key", "stay_id", "support_type"]].copy()
    positive = events.interval_status.eq("positive_duration")
    left = events.starttime.clip(lower=start)
    right = events.endtime.clip(upper=end)
    out["retrospective_overlap_hours"] = (
        ((right - left).dt.total_seconds() / 3600).clip(lower=0).where(positive)
    )
    out["retrospective_covers_window"] = (
        positive & events.starttime.le(start) & events.endtime.ge(end)
    )
    out["recorded_start_claim_visible"] = events.recorded_start_claim_at.lt(cutoff)
    out["completed_interval_record_visible"] = events.completed_interval_record_at.lt(
        cutoff
    )
    out["completed_window_record_evidence"] = (
        out.retrospective_covers_window
        & out.completed_interval_record_visible
        & events.ultimate_status_class.eq("affirmative_terminal_record")
    )
    out["support_evidence_status"] = np.where(
        out.completed_window_record_evidence,
        "positive_completed_window_record",
        "unknown",
    )
    return out
