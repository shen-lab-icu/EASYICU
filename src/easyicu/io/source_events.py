"""Opt-in MIIV respiratory extraction with exact-stay SQL and private event trace.

This is an explicit candidate API, not a replacement for load_concepts. Lab
assignment inherits the legacy outtime-forward/roll-ends rule; assigning an ICU
identifier does not imply that an event occurred inside that ICU interval.
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import numbers
from pathlib import Path
from typing import Sequence

import duckdb
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from ..datasource import ICUDataSource
from ..resources import load_dictionary, package_path
from ..utils.callback_utils import percent_as_numeric

CONCEPTS = ("spo2", "o2sat", "peep_set", "peep_total", "fio2_chart", "fio2_lab")
LAB_ASSIGNMENT = "legacy_outtime_forward_rollends"
TRACE_COLUMNS = [
    "subject_id", "hadm_id", "stay_id", "concept", "source_table", "source_item_id",
    "source_file", "source_file_sha256", "source_row_number", "event_key",
    "labevent_id", "specimen_id", "raw_charttime", "raw_storetime", "raw_value",
    "raw_valuenum", "raw_valueuom", "intime", "outtime", "assignment_rule",
    "temporal_position", "lab_assignment_status", "callback", "numeric_source", "callback_input",
    "numeric_value", "converted_value", "conversion_status", "bounds_status",
    "icu_relative_time", "hour_bucket", "selected_by_identity",
    "retained_for_aggregation", "exclusion_reason", "aggregate_value", "aggregate_n",
]


@dataclass
class SourceEventResult:
    hourly: pd.DataFrame
    trace: pd.DataFrame
    receipt: dict
    clock_context: pd.DataFrame


class SourceEventContractError(ValueError):
    """Fails closed without including patient identifiers or clinical values."""

    def __init__(self, reason: str, *, counts: dict | None = None):
        self.reason = reason
        self.counts = counts or {}
        super().__init__(reason)


def _sha(path: Path) -> str:
    result = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            result.update(block)
    return result.hexdigest()


def _quote(value: str) -> str:
    return "'" + value.replace("'", "''") + "'"


def _relation(paths: Sequence[Path], *, events: bool = False) -> str:
    args = ", filename=true, file_row_number=true" if events else ""
    return "read_parquet([" + ",".join(_quote(str(p)) for p in paths) + "]" + args + ", union_by_name=true)"


def _files(source: ICUDataSource, table: str, itemids: list[int]) -> list[Path]:
    bucket = source.resolve_bucket_directory(table)
    if bucket is not None:
        paths = source.get_bucket_files_for_ids(bucket, itemids, duckdb)[2]
    else:
        flat = source.resolve_flat_parquet_directory(table)
        if flat is not None:
            paths = sorted(flat.glob("*.parquet"))
        else:
            candidates = [source.base_path / f"{table}.parquet",
                          source.base_path / ("icu" if table == "chartevents" else "hosp") / f"{table}.parquet"]
            paths = [p for p in candidates if p.is_file()]
    paths = sorted({Path(p).resolve() for p in paths})
    if not paths:
        raise SourceEventContractError(f"missing_parquet_source:{table}")
    return paths


def _schemas(paths: list[Path], required: set[str], clocks: set[str]) -> set[str]:
    union = set()
    for path in paths:
        schema = pq.read_schema(path)
        names = set(schema.names)
        if required - names:
            raise SourceEventContractError("missing_required_columns:" + ",".join(sorted(required - names)))
        if {"filename", "file_row_number"} & names:
            raise SourceEventContractError("reserved_trace_columns_in_source")
        for name in required & {"subject_id", "hadm_id", "stay_id", "itemid"}:
            if not pa.types.is_integer(schema.field(name).type):
                raise SourceEventContractError(f"noninteger_source_identity_schema:{name}")
        for name in clocks:
            kind = schema.field(name).type
            # MIIV clocks are naive deidentified local times. Reject a timezone
            # rather than silently drop/convert it or claim they are UTC.
            if getattr(kind, "tz", None) is not None:
                raise SourceEventContractError("timezone_aware_source_clock")
        union |= names
    return union


def _clock_sql(name: str) -> str:
    # Timestamp strings with explicit offsets must not be silently stripped by
    # DuckDB TIMESTAMP casts. Native naive timestamps stringify without offsets.
    return (f"CASE WHEN regexp_full_match(CAST({name} AS VARCHAR), "
            "'[0-9]{4}-[0-9]{2}-[0-9]{2}[ T][0-9]{2}:[0-9]{2}:[0-9]{2}(\\.[0-9]+)?') "
            f"THEN TRY_CAST({name} AS TIMESTAMP) ELSE NULL END")


def _prepare_clocks(conn, paths: list[Path], allowed: list[int]) -> None:
    _schemas(paths, {"subject_id", "hadm_id", "stay_id", "intime", "outtime"}, {"intime", "outtime"})
    conn.register("requested", pd.DataFrame({"stay_id": pd.Series(allowed, dtype="int64")}))
    relation = _relation(paths)
    conn.execute(f"""CREATE VIEW clocks_raw AS
        SELECT subject_id, hadm_id, stay_id,
               {_clock_sql('intime')} AS intime, {_clock_sql('outtime')} AS outtime,
               outtime IS NULL AS outtime_missing
        FROM {relation}
        WHERE hadm_id IN (SELECT hadm_id FROM {relation}
                         WHERE stay_id IN (SELECT stay_id FROM requested))
           OR stay_id IN (SELECT stay_id FROM requested)""")
    checks = {
        "missing_requested_stay": "SELECT count(*) FROM requested WHERE stay_id NOT IN (SELECT stay_id FROM clocks_raw WHERE stay_id IS NOT NULL)",
        "duplicate_stay_id": "SELECT count(*) FROM (SELECT stay_id FROM clocks_raw GROUP BY stay_id HAVING count(*) != 1)",
        "invalid_identity_or_clock": "SELECT count(*) FROM clocks_raw WHERE subject_id IS NULL OR hadm_id IS NULL OR stay_id IS NULL OR intime IS NULL OR (outtime IS NULL AND NOT outtime_missing) OR intime > outtime",
        "hospital_subject_conflict": "SELECT count(*) FROM (SELECT hadm_id FROM clocks_raw GROUP BY hadm_id HAVING count(DISTINCT subject_id) != 1)",
        "ambiguous_outtime": "SELECT count(*) FROM (SELECT subject_id, hadm_id, outtime FROM clocks_raw WHERE outtime IS NOT NULL GROUP BY ALL HAVING count(*) > 1)",
    }
    counts = {key: int(conn.execute(sql).fetchone()[0]) for key, sql in checks.items()}
    if any(counts.values()):
        raise SourceEventContractError("invalid_complete_hospital_clock_context", counts=counts)
    # Only identity/time columns in engine memory; never fetch nonallowed IDs.
    conn.execute("""CREATE TEMP TABLE clocks AS SELECT *,
        CASE WHEN outtime_missing THEN 'unknown' ELSE 'known' END AS outtime_status,
        count(*) OVER (PARTITION BY subject_id, hadm_id) AS context_stay_count,
        count(*) FILTER (WHERE outtime_missing) OVER (PARTITION BY subject_id, hadm_id)
            AS context_unknown_outtime_count
        FROM clocks_raw""")
    conn.execute("CREATE VIEW targets AS SELECT * FROM clocks WHERE stay_id IN (SELECT stay_id FROM requested)")


def _read_scoped_events(conn, paths: list[Path], table: str, itemids: list[int]):
    required = {"subject_id", "hadm_id", "itemid", "charttime", "storetime", "value", "valuenum", "valueuom"}
    if table == "chartevents":
        required.add("stay_id")
    names = _schemas(paths, required, {"charttime", "storetime"})
    optional = ", ".join(f"o.{name}" if name in names else f"NULL::BIGINT AS {name}"
                         for name in ["labevent_id", "specimen_id"])
    items = ",".join(str(i) for i in itemids)
    # Clinical values stay inside the engine until exact assignment AND scope
    # checks succeed. Missing hadm outside this source domain is not quantified.
    if table == "chartevents":
        predicate = "o.stay_id IN (SELECT stay_id FROM requested)"
        assignment = "LEFT JOIN targets a ON o.stay_id = a.stay_id"
        identity_failure = "o.subject_id IS DISTINCT FROM a.subject_id OR o.hadm_id IS DISTINCT FROM a.hadm_id"
        uncertainty_case = ""
        assignment_status = "'native_stay_id'"
    else:
        predicate = "o.hadm_id IN (SELECT hadm_id FROM targets)"
        assignment = """LEFT JOIN LATERAL (
            SELECT min(outtime) FILTER (WHERE outtime >= o.event_time) AS known_future,
                   max(outtime) AS known_max,
                   count(*) FILTER (WHERE outtime_missing) AS unknown_n,
                   min(intime) FILTER (WHERE outtime_missing) AS unknown_lower,
                   count(*) AS context_n
            FROM clocks c WHERE c.subject_id = o.subject_id AND c.hadm_id = o.hadm_id
        ) s ON true
        LEFT JOIN LATERAL (
            SELECT * FROM clocks c
            WHERE c.subject_id = o.subject_id AND c.hadm_id = o.hadm_id
              AND (s.unknown_n = 0
                   OR (s.known_future IS NOT NULL AND c.outtime = s.known_future
                       AND s.unknown_lower > s.known_future)
                   OR (s.known_future IS NULL AND s.unknown_n = 1 AND c.outtime_missing
                       AND (s.known_max IS NULL OR s.unknown_lower > s.known_max)))
            ORDER BY CASE WHEN c.outtime >= o.event_time THEN c.outtime END ASC NULLS LAST,
                     c.outtime DESC
            LIMIT 1
        ) a ON true"""
        identity_failure = "s.context_n = 0"
        uncertainty_case = "WHEN a.stay_id IS NULL THEN 'unknown_outtime_assignment'"
        assignment_status = "CASE WHEN s.unknown_n > 0 THEN 'identified_under_unknown_outtime' ELSE 'complete_clocks' END"
    conn.execute(f"""CREATE OR REPLACE VIEW source_domain AS
        SELECT o.subject_id, o.hadm_id, {('o.stay_id,' if table == 'chartevents' else '')}
               o.itemid, o.filename AS source_file, o.file_row_number AS source_row_number,
               o.charttime AS raw_charttime, o.storetime AS raw_storetime,
               o.value AS raw_value, o.valuenum AS raw_valuenum, o.valueuom AS raw_valueuom,
               {_clock_sql('o.charttime')} AS event_time, {optional}
        FROM {_relation(paths, events=True)} o
        WHERE o.itemid IN ({items}) AND {predicate}""")
    conn.execute(f"""CREATE OR REPLACE VIEW assigned AS
        SELECT o.* EXCLUDE ({('stay_id,' if table == 'chartevents' else '')} event_time),
               o.event_time, a.stay_id, a.intime, a.outtime,
               {assignment_status} AS lab_assignment_status,
               CASE WHEN {identity_failure}
                         THEN 'identity_mismatch'
                    WHEN o.event_time IS NULL THEN 'missing_or_invalid_event_time'
                    {uncertainty_case}
                    WHEN a.stay_id NOT IN (SELECT stay_id FROM requested) THEN 'assigned_stay_not_allowed'
                    ELSE 'identity_allowed' END AS identity_status
        FROM source_domain o {assignment}""")
    counts = [dict(source_table=table, source_item_id=int(item), reason=reason, count=int(count))
              for item, reason, count in conn.execute(
                  "SELECT itemid, identity_status, count(*) FROM assigned GROUP BY ALL ORDER BY 1,2"
              ).fetchall()]
    # This is the ONLY clinical-values fetch into Python. No broad pandas read,
    # post-hoc restriction, ID fallback, or output with a missing caller key.
    frame = conn.execute("SELECT * EXCLUDE(identity_status) FROM assigned WHERE identity_status='identity_allowed'").fetchdf()
    frame["source_table"] = table
    return frame, counts


def _transform(events: pd.DataFrame, specs: list[dict], hashes: dict, interval: float) -> pd.DataFrame:
    parts = []
    for spec in specs:
        part = events.loc[(events.source_table == spec["table"]) & events.itemid.isin(spec["ids"])].copy()
        if part.empty:
            continue
        part["concept"] = spec["concept"]
        part = part.rename(columns={"itemid": "source_item_id"})
        part["source_file_sha256"] = part.source_file.map(hashes)
        part["event_key"] = (part.source_table + "|" + part.source_file + "|" + part.source_row_number.astype(str))
        part["assignment_rule"] = LAB_ASSIGNMENT if spec["table"] == "labevents" else "native_stay_id"
        part["temporal_position"] = np.select(
            [part.event_time < part.intime, part.outtime.isna(), part.event_time == part.outtime, part.event_time > part.outtime],
            ["before_intime", "outtime_unknown", "at_outtime", "after_outtime"], default="within_icu",
        )
        part["callback"] = spec["callback"] or "identity_numeric"
        use_numeric = part.raw_valuenum.notna()
        part["numeric_source"] = np.where(use_numeric, "valuenum", "value")
        callback_input = part.raw_valuenum.astype(object).where(use_numeric, part.raw_value)
        part["callback_input"] = callback_input.map(lambda value: None if pd.isna(value) else str(value)).astype(object)
        part["numeric_value"] = pd.to_numeric(part.callback_input, errors="coerce")
        if spec["callback"]:
            part["converted_value"] = percent_as_numeric(part.callback_input)
        else:
            part["converted_value"] = part.numeric_value
        finite = np.isfinite(part.converted_value.to_numpy(dtype=float))
        within = finite & part.converted_value.between(spec["min"], spec["max"])
        part["conversion_status"] = np.select(
            [part.callback_input.isna(), part.converted_value.isna(), ~finite],
            ["missing_input", "unparseable", "nonfinite"], default="finite",
        )
        part["bounds_status"] = np.where(~finite, "not_evaluable", np.where(within, "within", "outside"))
        part["icu_relative_time"] = (part.event_time - part.intime).dt.total_seconds() / 3600.
        part["hour_bucket"] = np.floor(part.icu_relative_time / interval) * interval
        part["selected_by_identity"] = True
        part["retained_for_aggregation"] = within
        part["exclusion_reason"] = np.where(~finite, part.conversion_status, np.where(within, "", "outside_dictionary_bounds"))
        part["_aggregate_input"] = part.converted_value.where(within)
        parts.append(part)
    if not parts:
        return pd.DataFrame(columns=TRACE_COLUMNS)
    trace = pd.concat(parts, ignore_index=True)
    grouped = trace.groupby(["stay_id", "hour_bucket", "concept"], sort=True)._aggregate_input
    trace["aggregate_value"] = grouped.transform("median")
    trace["aggregate_n"] = grouped.transform("count").astype("int64")
    return trace[TRACE_COLUMNS]


def extract_miiv_respiratory_events(
    data_source: ICUDataSource, *, allowed_stay_ids: Sequence[int],
    concepts: Sequence[str] = CONCEPTS, interval_hours: float = 1.,
) -> SourceEventResult:
    """Extract only explicit MIIV stays, returning private hourly/event artefacts.

    SourceEventContractError aborts on missing keys or ambiguous complete clock
    contexts. No values from unallowed stays cross the SQL/Python boundary.
    This does not provide physical Parquet-page isolation. All returned traces
    and counts are PRIVATE and require independent disclosure review to publish.
    """
    if data_source.config.name != "miiv":
        raise SourceEventContractError("only_miiv_supported")
    allowed = list(allowed_stay_ids) if allowed_stay_ids is not None else []
    if not allowed or any(isinstance(v, (bool, np.bool_)) or not isinstance(v, numbers.Integral)
                          or v <= 0 or v > np.iinfo(np.int64).max for v in allowed):
        raise SourceEventContractError("explicit_positive_integer_stay_ids_required")
    if len(set(allowed)) != len(allowed):
        raise SourceEventContractError("duplicate_requested_stay_id")
    allowed = sorted(map(int, allowed))
    concepts = list(concepts)
    if not concepts or len(set(concepts)) != len(concepts) or set(concepts) - set(CONCEPTS):
        raise SourceEventContractError("unsupported_or_duplicate_concepts")
    if isinstance(interval_hours, bool) or not np.isfinite(interval_hours) or interval_hours <= 0:
        raise SourceEventContractError("invalid_interval_hours")
    with package_path("concept-dict.json") as dictionary_path:
        dictionary_sha = _sha(dictionary_path)
    dictionary = load_dictionary()
    specs = []
    for name in concepts:
        definition = dictionary[name]
        for source in definition.for_data_source(data_source.config):
            if (source.table not in {"chartevents", "labevents"} or source.sub_var != "itemid"
                or source.callback not in {None, "transform_fun(percent_as_numeric)"}
                or definition.callback is not None or definition.minimum is None or definition.maximum is None):
                raise SourceEventContractError("unsupported_dictionary_contract")
            specs.append(dict(concept=name, table=source.table, ids=list(source.ids),
                              callback=source.callback, min=definition.minimum, max=definition.maximum,
                              units=definition.units))
    clock_candidates = [data_source.base_path / "icustays.parquet", data_source.base_path / "icu/icustays.parquet"]
    clock_paths = [p.resolve() for p in clock_candidates if p.is_file()]
    if len(clock_paths) != 1:
        raise SourceEventContractError("one_icustays_parquet_required")
    sources = {table: _files(data_source, table, sorted({i for s in specs if s['table'] == table for i in s['ids']}))
               for table in sorted({s['table'] for s in specs})}
    files = [("icustays", p) for p in clock_paths] + [(table, p) for table, paths in sources.items() for p in paths]
    hashes = {str(p): _sha(p) for _, p in files}
    conn = duckdb.connect(":memory:", config={"threads": 1, "temp_directory": ""})
    try:
        _prepare_clocks(conn, clock_paths, allowed)
        clock_context = conn.execute("""SELECT subject_id, hadm_id, stay_id, intime, outtime,
            outtime_status, context_stay_count, context_unknown_outtime_count
            FROM clocks WHERE stay_id IN (SELECT stay_id FROM requested) ORDER BY stay_id""").fetchdf()
        frames, counts = [], []
        for table, paths in sources.items():
            items = sorted({i for s in specs if s["table"] == table for i in s["ids"]})
            frame, table_counts = _read_scoped_events(conn, paths, table, items)
            frames.append(frame)
            counts.extend(table_counts)
        events = pd.concat(frames, ignore_index=True)
    finally:
        conn.close()
    if not set(events.stay_id) <= set(allowed):
        raise SourceEventContractError("engine_identity_boundary_violation")
    for _, path in files:
        if hashes[str(path)] != _sha(path):
            raise SourceEventContractError("source_changed_during_extraction")
    trace = _transform(events, specs, hashes, float(interval_hours))
    if trace.empty:
        hourly = pd.DataFrame({"stay_id": pd.Series(dtype="int64"), "charttime": pd.Series(dtype="float64"),
                               **{name: pd.Series(dtype="float64") for name in concepts}})
    else:
        # Drop repeated GROUP RESULT rows only. Event rows are never deduplicated.
        aggregate = trace[["stay_id", "hour_bucket", "concept", "aggregate_value"]].drop_duplicates()
        hourly = aggregate.pivot(index=["stay_id", "hour_bucket"], columns="concept", values="aggregate_value").reset_index()
        hourly = hourly.rename(columns={"hour_bucket": "charttime"}).reindex(columns=["stay_id", "charttime", *concepts])
        hourly.columns.name = None
        hourly = hourly.sort_values(["stay_id", "charttime"]).reset_index(drop=True)
    concept_counts = []
    for name in concepts:
        rows = trace.loc[trace.concept.eq(name)]
        concept_counts.append(dict(concept=name, identity_allowed=len(rows), retained=int(rows.retained_for_aggregation.sum()),
                                   retained_zero=int((rows.retained_for_aggregation & rows.converted_value.eq(0)).sum()),
                                   all_null_hours=len(rows.loc[rows.aggregate_n.eq(0), ["stay_id", "hour_bucket"]].drop_duplicates())))
    with package_path("concept-dict.json") as dictionary_path:
        if dictionary_sha != _sha(dictionary_path):
            raise SourceEventContractError("dictionary_changed_during_extraction")
    receipt = dict(
        schema="miiv_scoped_source_events_v2", lab_assignment_rule=LAB_ASSIGNMENT,
        unknown_outtime_rule="identify only if the same unique winner exists for every E>=intime completion; possible tied winner is ambiguous; no imputation or cross-stay non-overlap assumption",
        clock_context_rows=len(clock_context),
        source_filter_applied_before_python=True,
        time_coordinate={"origin": "icu_intime", "unit": "h", "column": "charttime", "timezone": "source_naive_deidentified_local_not_UTC"},
        callback_input_contract="non-null valuenum else value; object-string Series; no fallback from nonfinite valuenum",
        aggregation="event_callback_then_finite_and_dictionary_bounds_then_all_event_median; retain_all_null_observed_hours",
        scope="exact requested stays; SQL uses complete same-hospital identity/clock context; no physical-page isolation claim",
        missing_hadm_contribution="not in exact permitted hospital domain; not quantified",
        clinical_rows_returned=len(events), trace_rows=len(trace), allowed_stay_count=len(allowed),
        allowed_stay_sha256=hashlib.sha256(json.dumps(allowed, separators=(',', ':')).encode()).hexdigest(),
        source_files=[dict(table=table, path=str(path), sha256=hashes[str(path)]) for table, path in files],
        source_filter_counts=counts, concept_counts=concept_counts, concept_sources=specs,
        dictionary_sha256=dictionary_sha, event_identity="source table + absolute source file + zero-based Parquet row; native IDs retained; no event deduplication",
        differences_from_legacy=["no collapse of distinct physical duplicate events", "explicit event-first conversion/bounds/aggregation order", "invalid identity/intime, malformed non-null outtime and duplicate known exits fail closed; native missing outtime uses all-completions unique assignment and an explicit clock ledger", "observed all-null/out-of-range event hours retain keys; legacy prefilters may drop keys; extra missing hours are not new measurements or improved coverage"],
        privacy="private patient trace/counts; publication needs separate disclosure review",
    )
    return SourceEventResult(hourly=hourly, trace=trace, receipt=receipt, clock_context=clock_context)
