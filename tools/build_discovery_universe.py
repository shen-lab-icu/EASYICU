#!/usr/bin/env python3
"""Build a multi-aggregate universe parquet for the idea-mined instances.

Mirrors the universe_m3 layout the 9-question bench uses: measurement concepts
get 6 aggregates (max/min/mean/n/first/measured), binary EVENT concepts get a
single 0/1 column (present=1 else 0), durations get a single value (absent=0),
static/outcome concepts get a single column. Built from the full module-grouped
MIIV export with duckdb (low memory over ~98M long-format rows).
"""
from __future__ import annotations

from pathlib import Path

import duckdb
import pyarrow.parquet as pq

EXPORT = Path("/Volumes/外置硬盘/easyicu_fullexport_miiv_20260610")
OUT = Path("research_output/universe_discovery/universe_discovery.parquet")

# concept -> handling
MEASUREMENT = ["na", "lact", "urine24", "uo_24h", "map", "crea", "sofa2",
               "hr", "resp", "spo2", "temp"]
EVENT = ["rrt", "circ_failure", "sep3_sofa2", "heparin"]   # binary 0/1
DURATION_ZERO = ["norepi_dur"]                              # absent -> 0
STATIC = ["age", "sex"]
OUTCOME = ["death", "los_icu"]


def file_for(concept: str) -> Path:
    for f in sorted(EXPORT.glob("*.parquet")):
        if concept in pq.read_schema(f).names:
            return f
    raise SystemExit(f"concept not found in export: {concept}")


def main() -> None:
    OUT.parent.mkdir(parents=True, exist_ok=True)
    con = duckdb.connect()
    base = file_for("age")
    con.execute(
        f"CREATE TABLE u AS SELECT DISTINCT stay_id FROM "
        f"read_parquet('{base.as_posix()}') WHERE stay_id IS NOT NULL"
    )

    def join(sql_select_from: str) -> None:
        con.execute(
            f"CREATE OR REPLACE TABLE u AS SELECT u.*, m.* EXCLUDE (stay_id) "
            f"FROM u LEFT JOIN ({sql_select_from}) m USING (stay_id)"
        )

    has_time = lambda f: "charttime" in pq.read_schema(f).names  # noqa: E731

    for c in MEASUREMENT:
        f = file_for(c)
        t = has_time(f)
        first = f'arg_min("{c}", charttime)' if t else f'any_value("{c}")'
        join(
            f'SELECT stay_id, max("{c}") AS "{c}_max", min("{c}") AS "{c}_min", '
            f'avg("{c}") AS "{c}_mean", count("{c}") AS "{c}_n", '
            f'{first} AS "{c}_first", '
            f'CAST(count("{c}")>0 AS INTEGER) AS "{c}_measured" '
            f'FROM read_parquet(\'{f.as_posix()}\') WHERE stay_id IS NOT NULL '
            f'GROUP BY stay_id'
        )

    for c in EVENT:
        f = file_for(c)
        join(
            f'SELECT stay_id, CAST(count("{c}")>0 AS INTEGER) AS "{c}" '
            f'FROM read_parquet(\'{f.as_posix()}\') WHERE stay_id IS NOT NULL '
            f'GROUP BY stay_id'
        )

    for c in DURATION_ZERO:
        f = file_for(c)
        join(
            f'SELECT stay_id, max("{c}") AS "{c}" '
            f'FROM read_parquet(\'{f.as_posix()}\') WHERE stay_id IS NOT NULL '
            f'GROUP BY stay_id'
        )

    for c in STATIC + OUTCOME:
        f = file_for(c)
        join(
            f'SELECT stay_id, any_value("{c}") AS "{c}" '
            f'FROM read_parquet(\'{f.as_posix()}\') WHERE stay_id IS NOT NULL '
            f'GROUP BY stay_id'
        )

    # event concepts + death are 0 when absent (no event), not missing
    for c in EVENT + ["death"]:
        con.execute(f'UPDATE u SET "{c}" = COALESCE("{c}", 0)')
    for c in DURATION_ZERO:
        con.execute(f'UPDATE u SET "{c}" = COALESCE("{c}", 0)')

    con.execute(f"COPY u TO '{OUT.as_posix()}' (FORMAT parquet)")
    n = con.execute("SELECT count(*) FROM u").fetchone()[0]
    ncol = len(con.execute("SELECT * FROM u LIMIT 0").description)
    print(f"universe built: {OUT}  n_stays={n}  ncols={ncol}")
    # quick non-missing report for the instance variables
    for c in ["na_measured", "lact_measured", "urine24_measured", "rrt",
              "circ_failure", "sep3_sofa2", "heparin", "norepi_dur", "death"]:
        try:
            v = con.execute(f'SELECT avg(CAST("{c}" AS DOUBLE)) FROM u').fetchone()[0]
            print(f"   {c:16s} mean={round(float(v),4)}")
        except Exception as e:  # noqa: BLE001
            print(f"   {c}: {e}")
    con.close()


if __name__ == "__main__":
    main()
