"""A wide single-column hourly mean is the exact mean on every route.

Codex delta2 D2-MEAN: the wide-table DuckDB route turned ``mean`` into SQL
``MEAN`` while its exact switch recognised only ``SUM``/``AVG``, so a mean
was DuckDB's floating ``AVG`` (accumulation-order and thread dependent)
instead of the hour's terms collapsed by the order-independent exact mean
the raw path uses. Invented heart rates only.
"""

from __future__ import annotations

import math
from dataclasses import replace

import numpy as np
import pandas as pd
import pytest

from easyicu.concept import ConceptDictionary, ConceptResolver
from easyicu.concept.aggregation_route import force_raw_aggregation, record_duckdb_routes
from easyicu.config import load_src_cfg
from easyicu.datasource import ICUDataSource
from easyicu.resources import load_dictionary
from tests.support.bucket_storage import write_fixture_parquet

pytestmark = pytest.mark.filterwarnings("ignore")

# Hourly heart rates whose naive float accumulation differs from the exactly
# rounded mean (checked below, so the fixture keeps its power).
_HOURS = [
    [154.3, 151.2, 89.9, 150.0, 150.7, 52.0, 115.5, 126.8, 75.6, 129.2],
    [50.4, 119.7, 53.0, 59.6, 140.8, 84.5, 127.9, 96.3],
    [124.4, 158.0, 111.2, 87.2, 60.4, 100.3, 157.8],
]
_STAYS = range(1, 7)


def _write(root, table, frame, layout):
    for column in load_src_cfg("eicu").get_table(table).columns:
        if column not in frame:
            frame[column] = np.nan
    if layout == "file" or table == "patient":
        path = root / f"{table}.parquet"
    elif layout == "flat":
        path = root / table / "part.parquet"
    else:
        path = root / f"{table}_bucket" / "bucket_id=0" / "part.parquet"
        path.parent.mkdir(parents=True)
        (path.parent.parent / "_COMPLETE").write_text("ready")
    path.parent.mkdir(parents=True, exist_ok=True)
    write_fixture_parquet(frame, path, index=False)


def _source(root, layout):
    root.mkdir(parents=True)
    _write(root, "patient", pd.DataFrame({
        "patientunitstayid": list(_STAYS), "patienthealthsystemstayid": list(_STAYS),
        "uniquepid": [f"synthetic-{i}" for i in _STAYS],
        "unitdischargeoffset": 24 * 60, "hospitaladmitoffset": 0,
    }), layout)
    rows = []
    for stay in _STAYS:
        for hour, values in enumerate(_HOURS):
            # Reverse every other stay: the exact mean ignores row order.
            ordered = values if stay % 2 else values[::-1]
            rows += [(stay, hour * 60 + 1 + i, v) for i, v in enumerate(ordered)]
    _write(root, "vitalperiodic", pd.DataFrame(
        rows, columns=["patientunitstayid", "observationoffset", "heartrate"]), layout)
    return ICUDataSource(load_src_cfg("eicu"), base_path=root, enable_cache=False)


def _monitor_hr():
    definition = load_dictionary(source_name="eicu")["hr"]
    monitor = [s for s in definition.sources["eicu"] if s.table == "vitalperiodic"]
    return ConceptDictionary({"hr": replace(definition, sources={"eicu": monitor})})


def _mean_hr(source):
    table = ConceptResolver(_monitor_hr())._load_single_concept(
        "hr", source, aggregator="mean", patient_ids=list(_STAYS),
        interval=pd.Timedelta(hours=1), verbose=False,
    )
    frame = table.data[["patientunitstayid", table.index_column, "hr"]]
    frame.columns = ["stay", "hour", "hr"]
    frame = frame.astype({"hour": float})
    return frame.sort_values(["stay", "hour"], kind="mergesort").reset_index(drop=True)


def _threads(count):
    from easyicu import datasource

    connection = datasource._get_duckdb_connection()
    previous = connection.execute("SELECT current_setting('threads')").fetchone()[0]
    connection.execute(f"SET threads={count}")
    return connection, previous


def test_fixture_separates_float_accumulation_from_the_exact_mean():
    for values in _HOURS:
        naive = 0.0
        for value in values:
            naive += value
        assert naive / len(values) != math.fsum(values) / len(values)


def test_wide_route_mean_is_exact_and_matches_raw_across_threads_and_layouts(tmp_path):
    expected = pd.DataFrame({
        "stay": np.repeat(list(_STAYS), len(_HOURS)),
        "hour": np.tile(np.arange(len(_HOURS), dtype=float), len(_STAYS)),
        "hr": np.tile([math.fsum(v) / len(v) for v in _HOURS], len(_STAYS)),
    })
    results = {}
    for layout in ("file", "flat", "bucket"):
        source = _source(tmp_path / layout, layout)
        for threads in (1, 4):
            connection, previous = _threads(threads)
            try:
                with record_duckdb_routes() as routes:
                    results[(layout, threads, "sql")] = _mean_hr(source)
            finally:
                connection.execute(f"SET threads={previous}")
            if layout != "bucket":
                assert "wide_single_column" in routes, (layout, routes)
        with force_raw_aggregation(), record_duckdb_routes() as raw_routes:
            results[(layout, "raw")] = _mean_hr(source)
        assert not raw_routes
    for key, frame in results.items():
        pd.testing.assert_frame_equal(frame, expected, check_exact=True, check_dtype=False, obj=str(key))
