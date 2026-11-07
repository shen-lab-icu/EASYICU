#!/usr/bin/env python3
"""lead-r13 reseal table: per table and float column, -0.0 counts in the source vs the bucket (counts only).

Inputs are seal build receipts written by seal_rebuild.py ($WORK/<build>/receipt_<db>.json). The builder's
float-fidelity QC compares every float value with its source row (bound by the source ordinal) and fails
the build on any difference, so a PASS receipt's per-column profile is both the source and the bucket count.
Optional --current JSON gives the -0.0 count of the CURRENT formal bucket per db/table (built before the fix),
for the "before" column:  {"aumc/numericitems": {"value": 36052}, ...}.

  python negzero_table.py receipt_sic.json receipt_eicu.json ... [--current current.json] [--build2 r2.json ...]
Prints a Markdown table. --build2 receipts are checked for an equal per-column profile (counts and sha256).
Exit 0 only when every DB has a PASS receipt (pending allowed with --allow-pending) and every build2 profile
is equal; 1 on any finding; 2 usage.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

DB_LABEL = {
    "sic": "SICdb 1.0.8",
    "eicu": "eICU-CRD",
    "cv": "MIMIC-III CareVue",
    "miiv": "MIMIC-IV",
    "aumc": "AUMCdb",
}


def load(paths):
    out = {}
    for p in paths:
        r = json.loads(Path(p).read_text())
        out[r["db"]] = r
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("receipts", nargs="+")
    ap.add_argument("--current", type=Path)
    ap.add_argument("--build2", nargs="*", default=[])
    ap.add_argument("--allow-pending", action="store_true")
    a = ap.parse_args()
    b1, b2 = load(a.receipts), load(a.build2)
    cur = json.loads(a.current.read_text()) if a.current else {}
    rows = [
        "| DB | table | float column | values | source −0.0 | seal bucket −0.0 | current bucket −0.0 | +0.0 | NaN | build2 profile |",
        "|---|---|---|---|---|---|---|---|---|---|",
    ]
    bad = 0
    for db in ("sic", "eicu", "cv", "miiv", "aumc"):
        r = b1.get(db)
        if r is None:
            rows.append(
                f"| {DB_LABEL[db]} | — | — | — | pending | pending | — | — | — | — |"
            )
            bad += not a.allow_pending
            continue
        bad += r.get("verdict") != "PASS"
        for table, t in sorted(r["tables"].items()):
            prof = t.get("float_fidelity") or {}
            if not prof:
                rows.append(
                    f"| {DB_LABEL[db]} | {table} | (no float column) | — | — | — | — | — | — | — |"
                )
            for col, p in sorted(prof.items()):
                c = (cur.get(f"{db}/{table}") or {}).get(col)
                q = (
                    (((b2.get(db) or {}).get("tables") or {}).get(table) or {})
                    .get("float_fidelity", {})
                    .get(col)
                )
                same = "—" if not b2 else ("equal" if q == p else "DIFFERS")
                bad += same == "DIFFERS"
                seal = (
                    p["neg_zero"]
                    if r.get("verdict") == "PASS"
                    else f"{r.get('verdict')}"
                )
                rows.append(
                    f"| {DB_LABEL[db]} | {table} | {col} | {p['values']:,} | {p['neg_zero']:,} | {seal if isinstance(seal, str) else f'{seal:,}'} | "
                    f"{'—' if c is None else f'{c:,}'} | {p['pos_zero']:,} | {p['nan']:,} | {same} |"
                )
    print("\n".join(rows))
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
