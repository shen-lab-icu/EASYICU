"""DuckDB vs pandas extraction parity census for one database.

Every concept a database routes through a DuckDB aggregation entry point must
publish the same rows, values and dtypes as the pandas ``load_table`` path.
This census loads each concept twice, in separate worker processes so no
resolver cache is shared: once normally (recording which DuckDB routes it
takes) and once inside ``force_raw_aggregation``.  It then compares the two
frames byte for byte.

The report holds counts and concept names only (no stay identifiers or
times), so it may be run on credentialed data and shared as an aggregate.

    PYTHONPATH=src python tools/duckdb_raw_parity_census.py \\
        --database eicu --data-path /path/to/eicu_demo/raw --out /scratch/census_eicu
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import warnings
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import numpy as np
import pandas as pd

STAY_TABLES = {
    "eicu": ("patient", "patientunitstayid"),
    "miiv": ("icustays", "stay_id"),
    "mimic": ("icustays", "icustay_id"),
    "aumc": ("admissions", "admissionid"),
    "hirid": ("general", "patientid"),
    "sic": ("cases", "CaseID"),
}


#: Identifier and time columns rows are matched on; everything else is a value.
KEY_COLUMNS = frozenset(
    {column for _, column in STAY_TABLES.values()}
    | {"charttime", "starttime", "endtime", "start", "stop", "measuredat", "time", "index"}
)


def _same_cell(x: object, y: object) -> bool:
    missing_x, missing_y = bool(pd.isna(x)), bool(pd.isna(y))
    if missing_x or missing_y:
        return missing_x and missing_y
    return bool(x == y)


def compare_frames(left: pd.DataFrame, right: pd.DataFrame) -> Dict[str, object]:
    """Compare two published concept frames; return counts only.

    Rows are matched on every non-value column (identifiers and times);
    ``value_cells_differ`` counts matched cells whose bytes differ (NaN
    equals NaN), and ``dtype_differs`` lists columns whose dtype differs.
    """

    left = left.reset_index(drop=True)
    right = right.reset_index(drop=True)
    report: Dict[str, object] = {
        "rows_left": int(len(left)),
        "rows_right": int(len(right)),
    }
    if list(left.columns) != list(right.columns):
        report["columns_differ"] = sorted(set(left.columns) ^ set(right.columns))
        report["equal"] = False
        return report
    dtype_diff = [
        str(c) for c in left.columns if str(left[c].dtype) != str(right[c].dtype)
    ]
    keys = [c for c in left.columns if c in KEY_COLUMNS]
    values = [c for c in left.columns if c not in keys]
    if keys:
        # Repeated keys (window concepts) are matched in sorted order.
        def _ranked(frame: pd.DataFrame) -> pd.DataFrame:
            frame = frame.sort_values(list(frame.columns), kind="mergesort", na_position="last")
            return frame.assign(__rank=frame.groupby(keys, dropna=False).cumcount())

        merged = _ranked(left).merge(
            _ranked(right), on=[*keys, "__rank"], how="outer",
            suffixes=("__l", "__r"), indicator=True,
        )
    else:
        merged = None
    if merged is None:
        only_left = only_right = 0
        cells = int((~left.eq(right) & ~(left.isna() & right.isna())).to_numpy().sum()) if len(left) == len(right) else 0
    else:
        only_left = int((merged["_merge"] == "left_only").sum())
        only_right = int((merged["_merge"] == "right_only").sum())
        both = merged.loc[merged["_merge"] == "both"]
        cells = 0
        for column in values:
            a = both[f"{column}__l"].to_numpy()
            b = both[f"{column}__r"].to_numpy()
            if a.dtype.kind == "f" and b.dtype.kind == "f":
                a64 = a.astype("float64")
                b64 = b.astype("float64")
                same = (a64.view(np.int64) == b64.view(np.int64)) | (
                    np.isnan(a64) & np.isnan(b64)
                )
            else:
                same = np.array([_same_cell(x, y) for x, y in zip(a, b)], dtype=bool)
            cells += int((~same).sum())
    report.update(
        rows_only_left=only_left,
        rows_only_right=only_right,
        value_cells_differ=cells,
        dtype_differs=dtype_diff,
    )
    report["equal"] = (
        only_left == 0 and only_right == 0 and cells == 0 and not dtype_diff
        and len(left) == len(right)
    )
    return report


def _stay_ids(database: str, data_path: str, max_stays: Optional[int]) -> List[object]:
    table, column = STAY_TABLES[database]
    frame = pd.read_parquet(Path(data_path) / f"{table}.parquet", columns=[column])
    ids = sorted(frame[column].dropna().unique().tolist())
    return ids[:max_stays] if max_stays else ids


def _candidate_concepts(database: str) -> List[str]:
    from easyicu.resources import load_dictionary

    dictionary = load_dictionary(source_name=database)
    names = []
    for name, definition in dictionary.items():
        sources = getattr(definition, "sources", None) or {}
        if database in sources and not getattr(definition, "sub_concepts", None):
            names.append(name)
    return sorted(names)


def _worker(args: argparse.Namespace) -> int:
    warnings.filterwarnings("ignore")
    import logging

    logging.disable(logging.CRITICAL)
    import easyicu
    from easyicu.concept.aggregation_route import (
        force_raw_aggregation,
        record_duckdb_routes,
    )

    out = Path(args.out) / args.worker
    out.mkdir(parents=True, exist_ok=True)
    ids = _stay_ids(args.database, args.data_path, args.max_stays)
    names = args.concepts.split(",") if args.concepts else _candidate_concepts(args.database)
    status: Dict[str, object] = {}
    for name in names:
        try:
            with record_duckdb_routes() as routes:
                if args.worker == "raw":
                    with force_raw_aggregation():
                        frame = easyicu.load_concepts(
                            [name], patient_ids=ids, database=args.database,
                            data_path=args.data_path, merge=True,
                        )
                else:
                    frame = easyicu.load_concepts(
                        [name], patient_ids=ids, database=args.database,
                        data_path=args.data_path, merge=True,
                    )
            frame = frame if isinstance(frame, pd.DataFrame) else pd.DataFrame(frame)
            frame.reset_index(drop=True).to_parquet(out / f"{name}.parquet")
            status[name] = {"routes": dict(routes)}
        except Exception as exc:  # recorded per concept, not hidden
            status[name] = {"error": type(exc).__name__}
    (out / "status.json").write_text(json.dumps(status, indent=1, sort_keys=True))
    return 0


def _run_worker(argv: Sequence[str], worker: str, concepts: Optional[List[str]]) -> None:
    command = [sys.executable, __file__, *argv, "--worker", worker]
    if concepts is not None:
        command += ["--concepts", ",".join(concepts)]
    subprocess.run(command, check=True, env=dict(os.environ))


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--database", required=True, choices=sorted(STAY_TABLES))
    parser.add_argument("--data-path", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--max-stays", type=int, default=None)
    parser.add_argument("--concepts", default=None, help="comma-separated subset")
    parser.add_argument("--worker", choices=["duckdb", "raw"], default=None)
    args = parser.parse_args(argv)
    if args.worker:
        return _worker(args)

    passthrough = [
        "--database", args.database, "--data-path", args.data_path,
        "--out", args.out,
    ] + (["--max-stays", str(args.max_stays)] if args.max_stays else [])
    _run_worker(passthrough, "duckdb", args.concepts.split(",") if args.concepts else None)
    out = Path(args.out)
    normal = json.loads((out / "duckdb" / "status.json").read_text())
    routed = sorted(n for n, s in normal.items() if s.get("routes"))
    _run_worker(passthrough, "raw", routed)
    raw = json.loads((out / "raw" / "status.json").read_text())

    concepts: Dict[str, object] = {}
    for name in routed:
        entry: Dict[str, object] = {"routes": normal[name]["routes"]}
        if "error" in raw.get(name, {}):
            entry.update(equal=False, raw_error=raw[name]["error"])
        else:
            entry.update(compare_frames(
                pd.read_parquet(out / "duckdb" / f"{name}.parquet"),
                pd.read_parquet(out / "raw" / f"{name}.parquet"),
            ))
        concepts[name] = entry
    summary = {
        "database": args.database,
        "concepts_loaded": len(normal),
        "load_errors": sorted(n for n, s in normal.items() if "error" in s),
        "duckdb_routed": len(routed),
        "equal": sum(1 for e in concepts.values() if e.get("equal")),
        "mismatched": sorted(n for n, e in concepts.items() if not e.get("equal")),
        "concepts": concepts,
    }
    (out / "parity_report.json").write_text(json.dumps(summary, indent=1, sort_keys=True))
    print(json.dumps({k: v for k, v in summary.items() if k != "concepts"}, indent=1))
    return census_exit_code(summary)


def census_exit_code(summary: Dict[str, object]) -> int:
    """1 when any routed concept differs or any concept failed to load.

    A concept that raised in the normal run was never compared, so the census
    cannot vouch for it; the routes that did load are still compared and
    reported.
    """

    return 1 if summary.get("mismatched") or summary.get("load_errors") else 0


if __name__ == "__main__":
    raise SystemExit(main())
