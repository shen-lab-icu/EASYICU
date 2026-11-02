"""Opt-in MIIV renal/pressure source events with unmerged dual clocks.

No clinical I/O on import, aggregation, as-of selection or release operation.
All returned rows/counts are private. Reuses frozen v2 exact-stay assignment.
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

CONCEPTS = ("crea", "map", "cvp")
# These are source dictionary labels, not inferred specimen or catheter types.
CHANNELS = {
    ("labevents", 50912): ("crea", "lab_blood_chemistry", "Creatinine"),
    ("labevents", 52546): ("crea", "lab_blood_chemistry", "Creatinine"),
    ("labevents", 52024): ("crea", "lab_whole_blood", "Creatinine, Whole Blood"),
    ("chartevents", 220615): ("crea", "chart_serum", "Creatinine (serum)"),
    ("chartevents", 229761): ("crea", "chart_whole_blood", "Creatinine (whole blood)"),
    ("chartevents", 220052): (
        "map",
        "arterial_bp_mean",
        "Arterial Blood Pressure mean",
    ),
    ("chartevents", 220181): (
        "map",
        "noninvasive_bp_mean",
        "Non Invasive Blood Pressure mean",
    ),
    ("chartevents", 225312): ("map", "art_bp_mean", "ART BP Mean"),
    ("chartevents", 220074): (
        "cvp",
        "central_venous_pressure",
        "Central Venous Pressure",
    ),
}
NAIVE_CLOCK = re.compile(r"\d{4}-\d{2}-\d{2}[ T]\d{2}:\d{2}:\d{2}(?:\.\d+)?")
EXTRA_COLUMNS = [
    "concept",
    "source_channel",
    "item_label",
    "source_item_id",
    "event_key",
    "source_file_sha256",
    "dictionary_sha256",
    "assignment_rule",
    "temporal_position",
    "charttime",
    "storetime",
    "charttime_status",
    "storetime_status",
    "store_delay_hours",
    "clock_order_status",
    "measurement_hours",
    "store_hours",
    "callback",
    "numeric_source",
    "numeric_value",
    "converted_value",
    "conversion_status",
    "bounds_min",
    "bounds_max",
    "bounds_status",
    "expected_unit",
    "unit_status",
    "retained_for_analysis",
    "exclusion_reason",
]


@dataclass
class ScopedMeasurementResult:
    events: pd.DataFrame
    clock_context: pd.DataFrame
    receipt: dict


def _store_clock(value):
    if pd.isna(value):
        return pd.NaT, "missing"
    if not NAIVE_CLOCK.fullmatch(str(value)):
        return pd.NaT, "invalid"
    stamp = pd.to_datetime(value, errors="coerce")
    return (pd.NaT, "invalid") if pd.isna(stamp) else (stamp, "known")


def _specs(data_source, concepts):
    dictionary = load_dictionary()
    specs = []
    for name in concepts:
        definition = dictionary[name]
        sources = definition.for_data_source(data_source.config)
        actual = {(s.table, int(i)) for s in sources for i in s.ids}
        expected = {key for key, val in CHANNELS.items() if val[0] == name}
        if actual != expected or len(actual) != sum(len(s.ids) for s in sources):
            raise scoped.SourceEventContractError(
                "measurement_dictionary_mapping_changed"
            )
        for s in sources:
            if (
                s.callback is not None
                or definition.callback is not None
                or s.sub_var != "itemid"
                or s.index_var not in {None, "charttime"}
                or s.value_var not in {None, "valuenum"}
                or definition.minimum is None
                or definition.maximum is None
            ):
                raise scoped.SourceEventContractError(
                    "unsupported_measurement_dictionary_contract"
                )
            specs.append(
                dict(
                    concept=name,
                    table=s.table,
                    ids=list(s.ids),
                    callback=None,
                    value_var=s.value_var,
                    min=definition.minimum,
                    max=definition.maximum,
                    units=list(definition.units),
                )
            )
    return specs


def _decorate(events, specs, hashes, dictionary_sha):
    parts = []
    for spec in specs:
        part = events.loc[
            (events.source_table == spec["table"]) & events.itemid.isin(spec["ids"])
        ].copy()
        if part.empty:
            continue
        part["concept"] = spec["concept"]
        part["source_channel"] = [
            CHANNELS[(spec["table"], int(i))][1] for i in part.itemid
        ]
        part["item_label"] = [CHANNELS[(spec["table"], int(i))][2] for i in part.itemid]
        part["source_item_id"] = part.itemid.astype("int64")
        part["event_key"] = (
            part.source_table
            + "|"
            + part.source_file
            + "|"
            + part.source_row_number.astype(str)
        )
        part["source_file_sha256"] = part.source_file.map(hashes)
        part["dictionary_sha256"] = dictionary_sha
        part["assignment_rule"] = (
            scoped.LAB_ASSIGNMENT if spec["table"] == "labevents" else "native_stay_id"
        )
        part["temporal_position"] = np.select(
            [
                part.event_time.lt(part.intime),
                part.outtime.isna(),
                part.event_time.eq(part.outtime),
                part.event_time.gt(part.outtime),
            ],
            ["before_intime", "outtime_unknown", "at_outtime", "after_outtime"],
            default="within_icu",
        )
        part["charttime"] = part.event_time
        part["charttime_status"] = "known"
        parsed = [_store_clock(v) for v in part.raw_storetime]
        part["storetime"] = pd.to_datetime([p[0] for p in parsed])
        part["storetime_status"] = [p[1] for p in parsed]
        part["store_delay_hours"] = (
            part.storetime - part.charttime
        ).dt.total_seconds() / 3600
        part["clock_order_status"] = np.select(
            [part.storetime.isna(), part.store_delay_hours.lt(0)],
            ["unknown", "store_before_chart"],
            default="store_at_or_after_chart",
        )
        part["measurement_hours"] = (
            part.charttime - part.intime
        ).dt.total_seconds() / 3600
        part["store_hours"] = (part.storetime - part.intime).dt.total_seconds() / 3600
        part["callback"] = "identity_numeric"
        # Explicit lab value_var=valuenum is respected: no text rescue there.
        use_numeric = part.raw_valuenum.notna() | (spec["value_var"] == "valuenum")
        part["numeric_source"] = np.where(use_numeric, "valuenum", "value")
        raw = part.raw_valuenum.astype(object).where(use_numeric, part.raw_value)
        part["numeric_value"] = pd.to_numeric(raw, errors="coerce").astype(float)
        part["converted_value"] = part.numeric_value
        finite = np.isfinite(part.converted_value.to_numpy())
        part["conversion_status"] = np.select(
            [raw.isna(), part.numeric_value.isna(), ~finite],
            ["missing_input", "unparseable", "nonfinite"],
            default="finite",
        )
        part["bounds_min"], part["bounds_max"] = spec["min"], spec["max"]
        within = finite & part.converted_value.between(spec["min"], spec["max"])
        part["bounds_status"] = np.where(
            ~finite, "not_evaluable", np.where(within, "within", "outside")
        )
        part["expected_unit"] = "|".join(spec["units"])
        accepted_units = {str(u).strip().lower() for u in spec["units"]}
        unit = part.raw_valueuom.map(
            lambda u: None
            if pd.isna(u) or not str(u).strip()
            else str(u).strip().lower()
        )
        part["unit_status"] = np.where(
            unit.isna(),
            "missing_assumed_dictionary",
            np.where(unit.isin(accepted_units), "declared_match", "mismatch"),
        )
        part["retained_for_analysis"] = within & part.unit_status.ne("mismatch")
        part["exclusion_reason"] = np.select(
            [~finite, ~within, part.unit_status.eq("mismatch")],
            [part.conversion_status, "outside_dictionary_bounds", "unit_mismatch"],
            default="",
        )
        parts.append(part)
    if not parts:
        return pd.DataFrame(
            columns=list(events.columns)
            + [c for c in EXTRA_COLUMNS if c not in events.columns]
        )
    return (
        pd.concat(parts, ignore_index=True)
        .sort_values(["source_table", "source_file", "source_row_number", "concept"])
        .reset_index(drop=True)
    )


def extract_miiv_measurement_events(
    data_source, *, allowed_stay_ids: Sequence[int], concepts: Sequence[str] = CONCEPTS
):
    """Return private, unaggregated creatinine/MAP/CVP source ledgers.

    Retained flags concern values/units/bounds only, never availability or ICU
    risk eligibility. Consumers must explicitly define those temporal decisions.
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
    concepts = list(concepts)
    if (
        not concepts
        or len(set(concepts)) != len(concepts)
        or set(concepts) - set(CONCEPTS)
    ):
        raise scoped.SourceEventContractError("unsupported_or_duplicate_concepts")
    with package_path("concept-dict.json") as path:
        dictionary_sha = scoped._sha(path)
    specs = _specs(data_source, concepts)
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
    sources = {
        t: scoped._files(
            data_source,
            t,
            sorted({i for s in specs if s["table"] == t for i in s["ids"]}),
        )
        for t in sorted({s["table"] for s in specs})
    }
    files = [("icustays", p) for p in clocks] + [
        (t, p) for t, paths in sources.items() for p in paths
    ]
    hashes = {str(p): scoped._sha(p) for _, p in files}
    conn = duckdb.connect(":memory:", config={"threads": 1, "temp_directory": ""})
    try:
        scoped._prepare_clocks(conn, clocks, allowed)
        context = conn.execute("""SELECT subject_id,hadm_id,stay_id,intime,outtime,outtime_status,
            context_stay_count,context_unknown_outtime_count FROM clocks
            WHERE stay_id IN (SELECT stay_id FROM requested) ORDER BY stay_id""").fetchdf()
        frames, counts = [], []
        for table, paths in sources.items():
            items = sorted({i for s in specs if s["table"] == table for i in s["ids"]})
            frame, table_counts = scoped._read_scoped_events(conn, paths, table, items)
            frames.append(frame)
            counts.extend(table_counts)
    finally:
        conn.close()
    events = _decorate(
        pd.concat(frames, ignore_index=True), specs, hashes, dictionary_sha
    )
    if not set(events.stay_id) <= set(allowed):
        raise scoped.SourceEventContractError("engine_identity_boundary_violation")
    for _, path in files:
        if hashes[str(path)] != scoped._sha(path):
            raise scoped.SourceEventContractError("source_changed_during_extraction")
    with package_path("concept-dict.json") as path:
        if dictionary_sha != scoped._sha(path):
            raise scoped.SourceEventContractError(
                "dictionary_changed_during_extraction"
            )
    receipt = dict(
        schema="miiv_scoped_measurements_v1",
        api_sha256=scoped._sha(Path(__file__)),
        source_loader_sha256=scoped._sha(Path(scoped.__file__)),
        dictionary_sha256=dictionary_sha,
        specs=specs,
        source_files=[
            dict(table=t, path=str(p), sha256=hashes[str(p)]) for t, p in files
        ],
        source_filter_counts=counts,
        event_rows=len(events),
        clock_rows=len(context),
        allowed_stay_sha256=hashlib.sha256(
            json.dumps(allowed, separators=(",", ":")).encode()
        ).hexdigest(),
        lab_assignment_rule=scoped.LAB_ASSIGNMENT,
        source_filter_applied_before_python=True,
        missing_storetime="unknown; never imputed",
        negative_delay="both clocks and value retained, flagged; consumer defines as-of",
        aggregation="none; no chart/lab deduplication or MAP source pooling",
        units="missing explicitly assumed dictionary; mismatch retained but not eligible",
        availability="no availability or risk-set certification from value-retained flag",
        privacy="all ledgers/counts private; no real extraction authorization or release operation",
    )
    return ScopedMeasurementResult(events, context, receipt)
