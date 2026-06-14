#!/usr/bin/env python3
"""Freeze the raw source-item catalog of a database for T2/T3 tier triage.

The idea-mining feasibility classifier needs to tell apart two kinds of
"concept not in the EasyICU dictionary":

* **T2 (new concept authorable)** — the construct is NOT curated in the
  concept dictionary, but the *source database actually measures it* (an
  itemid with a matching label exists in ``d_labitems`` / ``d_items``). An
  AI-drafted concept definition + callback could expose it, pending human
  confirmation.
* **T3 (not in this database)** — the construct was never recorded in the
  source tables at all (e.g. neurological outcome scores, quantitative EEG,
  microcirculation), so no dictionary extension can recover it.

This tool reads the catalog tables (``d_labitems``, ``d_items``) of a
converted database and writes a compact, reproducible JSON snapshot the
classifier matches candidate concepts against. It stores only item
metadata (itemid / label / category / table), never patient data.
"""
from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import pyarrow.parquet as pq

DEFAULT_DB = Path("/Volumes/外置硬盘/databases/mimiciv")
DEFAULT_OUT = (
    Path(__file__).resolve().parents[1] / "benchmark" / "source_item_catalog_miiv.json"
)


def _read_first(db: Path, *candidates: str) -> Path | None:
    for rel in candidates:
        p = db / rel
        if p.exists():
            return p
    return None


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--db", type=Path, default=DEFAULT_DB)
    ap.add_argument("--database", default="miiv")
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT)
    args = ap.parse_args()

    lab = _read_first(args.db, "hosp/d_labitems.parquet", "d_labitems.parquet")
    items = _read_first(args.db, "icu/d_items.parquet", "d_items.parquet")
    if lab is None and items is None:
        raise SystemExit(f"no d_labitems / d_items catalog under {args.db}")

    rows: list[dict] = []
    counts: dict[str, int] = {}
    if lab is not None:
        cols = ["itemid", "label", "category", "fluid"]
        avail = [c for c in cols if c in pq.read_schema(lab).names]
        tbl = pq.read_table(lab, columns=avail).to_pylist()
        for r in tbl:
            label = str(r.get("label") or "").strip()
            if not label:
                continue
            rows.append(
                {
                    "itemid": int(r["itemid"]),
                    "label": label,
                    "category": str(r.get("category") or ""),
                    # specimen is the deterministic wrong-fluid guard for the
                    # T2 concept-proposer (e.g. drop "LDH, Ascites" for a blood
                    # concept).
                    "fluid": str(r.get("fluid") or ""),
                    "abbrev": "",
                    "param_type": "",
                    "table": "hosp/labevents",
                }
            )
        counts["hosp/labevents"] = len(tbl)
    if items is not None:
        cols = [
            "itemid",
            "label",
            "abbreviation",
            "category",
            "linksto",
            "param_type",
            "unitname",
        ]
        avail = [c for c in cols if c in pq.read_schema(items).names]
        tbl = pq.read_table(items, columns=avail).to_pylist()
        for r in tbl:
            label = str(r.get("label") or "").strip()
            if not label:
                continue
            rows.append(
                {
                    "itemid": int(r["itemid"]),
                    "label": label,
                    "category": str(r.get("category") or ""),
                    "fluid": "",
                    "abbrev": str(r.get("abbreviation") or "").strip(),
                    # param_type (Numeric/Text/Checkbox/Solution/...) +
                    # unitname drive the deterministic measurability/role gate.
                    "param_type": str(r.get("param_type") or ""),
                    "unitname": str(r.get("unitname") or ""),
                    "table": f"icu/{r.get('linksto') or 'chartevents'}",
                }
            )
        counts["icu/chartevents"] = len(tbl)

    payload = {
        "database": args.database,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "source_paths": {
            "d_labitems": str(lab) if lab else None,
            "d_items": str(items) if items else None,
        },
        "source_table_row_counts": counts,
        "n_items": len(rows),
        "items": rows,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(
        json.dumps(payload, ensure_ascii=False, indent=1), encoding="utf-8"
    )
    print(f"wrote {args.out}  n_items={len(rows)}  tables={counts}")


if __name__ == "__main__":
    main()
