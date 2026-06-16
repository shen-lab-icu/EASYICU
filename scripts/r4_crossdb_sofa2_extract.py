#!/usr/bin/env python
"""R4 + R3/Table1 cross-database SOFA-2 extraction — ONE-SHOT full cohort.

For each database, with a SINGLE full-cohort `load_concepts` call (NO
patient_ids filter, NO chunk loop — that re-reads every source table per
chunk and is the slow path), pull the six SOFA-2 organ-component subscores
plus demographics, then aggregate per stay in pandas (worst = max subscore,
the standard SOFA aggregation).

Outputs, per database:
  * research_output/r4_sofa2_perstay/<db>.parquet
      one row per stay: id, sofa2_<comp> (worst subscore 0-4), present_<comp>,
      sofa2_total (sum of available comps), n_components, age, sex, death.
      This is the source data for the R4 mis-imputation-bias ablation.
  * summary json (Table 1 demographics + SOFA-2 completeness + per-component
    present rate + component value distributions) for R3.

The in-process peak is a flat ~2-3 GB regardless of cohort size (easyicu
streams per source table), so one-shot is both correct and fastest. Forced
fast path: EASYICU_FORCE_INPROCESS_BATCH=1 + batch_size > any cohort.

Usage:
    EASYICU_FORCE_INPROCESS_BATCH=1 python scripts/r4_crossdb_sofa2_extract.py \
        --dbs miiv --out research_output/r4_table1_20260616.json
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import time
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")
logging.disable(logging.WARNING)
os.environ.setdefault("EASYICU_FORCE_INPROCESS_BATCH", "1")
os.environ.setdefault("EASYICU_DISABLE_AUTO_CHUNK", "1")

import sys

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from easyicu.api import load_concepts  # noqa: E402

DB_ROOT = Path("/Volumes/外置硬盘/databases")
# label, subdir, id_table (for exact N), easyicu db key
DBS = {
    "mimic": ("MIMIC-III", "mimiciii", "icustays.parquet", "mimic"),
    "miiv": ("MIMIC-IV", "mimiciv", "icustays.parquet", "miiv"),
    "eicu": ("eICU-CRD", "eicu", "patient.parquet", "eicu"),
    "aumc": ("AmsterdamUMCdb", "aumc", "admissions.parquet", "aumc"),
    "hirid": ("HiRID", "hirid", "general_table.parquet", "hirid"),
    "sic": ("SICdb", "sic", "cases.parquet", "sic"),
}
COMPS = [
    "sofa2_resp",
    "sofa2_coag",
    "sofa2_liver",
    "sofa2_cardio",
    "sofa2_cns",
    "sofa2_renal",
]
BIG_BATCH = 5_000_000


def _summ_numeric(series) -> dict:
    s = pd.Series(series).dropna().astype(float)
    if s.empty:
        return {"n": 0}
    return {
        "n": int(s.size),
        "median": round(float(np.median(s)), 1),
        "q1": round(float(np.percentile(s, 25)), 1),
        "q3": round(float(np.percentile(s, 75)), 1),
        "mean": round(float(s.mean()), 2),
    }


def _full_n(subdir: str, id_table: str):
    try:
        return pq.read_metadata(DB_ROOT / subdir / id_table).num_rows
    except Exception as e:  # noqa: BLE001
        return None, str(e)


def extract_db(db_key: str, perstay_dir: Path) -> dict:
    label, subdir, id_table, db = DBS[db_key]
    dp = str(DB_ROOT / subdir)
    rec: dict = {"label": label, "data_path": dp}
    rec["icu_stays_full"] = _full_n(subdir, id_table)

    # ---- demographics: ONE-SHOT full cohort ----
    print(f"[{label}] demographics (one-shot full)...", flush=True)
    t = time.time()
    demo = load_concepts(["age", "sex", "death"], database=db, data_path=dp,
                         batch_size=BIG_BATCH)
    idcol = demo.columns[0]
    g_demo = demo.groupby(idcol).first(numeric_only=False)
    rec["demo_n"] = int(g_demo.shape[0])
    rec["demo_seconds"] = round(time.time() - t, 1)
    print(f"[{label}]   demo {rec['demo_n']} stays in {rec['demo_seconds']}s",
          flush=True)

    # ---- SOFA-2 components: ONE-SHOT full cohort, NO chunk loop ----
    print(f"[{label}] SOFA-2 components (one-shot full, no loop)...", flush=True)
    t = time.time()
    comp = load_concepts(COMPS, database=db, data_path=dp, batch_size=BIG_BATCH)
    rec["comp_load_seconds"] = round(time.time() - t, 1)
    cidcol = comp.columns[0]
    have = [c for c in COMPS if c in comp.columns]
    print(f"[{label}]   loaded {len(comp)} rows, comps={have} "
          f"in {rec['comp_load_seconds']}s", flush=True)

    # worst (max) subscore per stay = standard SOFA aggregation
    g = comp.groupby(cidcol)[have].max()
    g = g.reindex(g_demo.index)  # align to full demo cohort
    present = g.notna()
    g_total = g.sum(axis=1, min_count=1)
    n_comp = present.sum(axis=1)

    perstay = pd.DataFrame(index=g.index)
    for c in COMPS:
        perstay[c] = g[c] if c in g else np.nan
        perstay[f"present_{c}"] = present[c] if c in present else False
    perstay["sofa2_total"] = g_total
    perstay["n_components"] = n_comp
    perstay["age"] = g_demo["age"] if "age" in g_demo else np.nan
    if "sex" in g_demo:
        perstay["sex"] = g_demo["sex"].astype(str)
    if "death" in g_demo:
        perstay["death"] = (
            g_demo["death"].fillna(0).astype(float).clip(0, 1).astype(int)
        )
    perstay.index.name = "stay_id"

    perstay_dir.mkdir(parents=True, exist_ok=True)
    pq_path = perstay_dir / f"{db}.parquet"
    perstay.reset_index().to_parquet(pq_path, index=False)
    rec["perstay_path"] = str(pq_path)
    rec["perstay_rows"] = int(perstay.shape[0])

    # ---- summary numbers for R3 / Table 1 ----
    total = int(perstay.shape[0])
    rec["sofa2_stays"] = total
    rec["component_present_pct"] = {
        c: (round(100 * float(present[c].sum()) / total, 1)
            if c in present and total else None)
        for c in COMPS
    }
    complete = int((n_comp == 6).sum())
    rec["sofa2_complete_stays"] = complete
    rec["sofa2_complete_pct"] = round(100 * complete / total, 1) if total else None
    rec["sofa2_total_value"] = _summ_numeric(g_total.values)
    rec["component_value"] = {
        c: _summ_numeric(g[c].values) for c in have
    }
    if "age" in g_demo:
        rec["age"] = _summ_numeric(g_demo["age"].values)
    if "sex" in g_demo:
        sx = g_demo["sex"].astype(str).str.lower()
        known = int(sx.isin(["female", "male", "f", "m"]).sum())
        female = int(sx.str.startswith("f").sum())
        rec["sex_female_pct"] = round(100 * female / known, 1) if known else None
    if "death" in g_demo:
        died = int((g_demo["death"].fillna(0).astype(float) == 1).sum())
        rec["mortality_pct"] = round(100 * died / g_demo.shape[0], 1)
    rec["total_seconds"] = round(time.time() - t, 1)
    return rec


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dbs", nargs="+", default=list(DBS.keys()),
                    help="db keys: mimic miiv eicu aumc hirid sic")
    ap.add_argument("--out", default="research_output/r4_table1_20260616.json")
    args = ap.parse_args()

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    perstay_dir = out_path.parent / "r4_sofa2_perstay"

    existing = {}
    if out_path.exists():
        try:
            existing = json.loads(out_path.read_text(encoding="utf-8")).get(
                "databases", {})
        except Exception:  # noqa: BLE001
            existing = {}

    result = {
        "generated": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "method": (
            "ONE-SHOT full-cohort load_concepts per DB (no patient_ids, no "
            "chunk loop). Per-stay worst (max) of each SOFA-2 organ-component "
            "subscore; present = >=1 non-null hourly subscore; complete = all "
            "six present. Per-stay table saved to r4_sofa2_perstay/<db>.parquet."
        ),
        "databases": existing,
    }

    for db_key in args.dbs:
        label = DBS[db_key][0]
        t0 = time.time()
        try:
            rec = extract_db(db_key, perstay_dir)
        except Exception as e:  # noqa: BLE001
            import traceback
            rec = {"label": label, "error": str(e),
                   "traceback": traceback.format_exc()}
        result["databases"][label] = rec
        result["generated"] = time.strftime("%Y-%m-%dT%H:%M:%S")
        out_path.write_text(json.dumps(result, ensure_ascii=False, indent=2),
                            encoding="utf-8")
        print(f"[{label}] DONE in {round(time.time()-t0,1)}s -> "
              f"{ {k: v for k, v in rec.items() if k not in ('component_value','traceback')} }",
              flush=True)

    print(f"\nwrote {out_path}")


if __name__ == "__main__":
    main()
