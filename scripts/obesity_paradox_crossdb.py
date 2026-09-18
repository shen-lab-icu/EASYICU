#!/usr/bin/env python
"""Cross-6-DB transportability of the obesity paradox (Fig 5 source data).

Per DB: per-stay worst BMI (EasyICU bmi concept) -> in-hospital mortality by BMI
category, with Wilson 95% CIs per cell, per-DB overall mortality, and an explicit
extreme-obesity J-uptick flag (obese II-III mortality > obese I, non-overlapping).
Heavily studied, but not across these six harmonized public ICU databases ->
cross-DB transportability is the contribution.
"""
import argparse
import json
import math
import os
import sys
import tempfile
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")
os.environ.setdefault("EASYICU_FORCE_INPROCESS_BATCH", "1")
sys.path.insert(0, "src")
import numpy as np, pandas as pd
from easyicu.api import load_concepts
ROOT = "/Volumes/外置硬盘/databases"
DBS = [("MIMIC-III", "mimiciii", "mimic"), ("MIMIC-IV", "mimiciv", "miiv"),
     ("eICU-CRD", "eicu", "eicu"), ("AmsterdamUMCdb", "aumc", "aumc"),
     ("HiRID", "hirid", "hirid"), ("SICdb", "sic", "sic")]
CATS = [("underweight", 0, 18.5), ("normal", 18.5, 25), ("overweight", 25, 30), ("obese I", 30, 35), ("obese II-III", 35, 80)]

def wilson(k, n, z=1.96):
    if n == 0:
        return (None, None, None)
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (round(100 * p, 1), round(100 * (c - h), 1), round(100 * (c + h), 1))


def _parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__)
    # No dirty-tree default: without --out, write into an ephemeral temp dir.
    ap.add_argument("--out", default=None, help="output directory (default: tempfile.mkdtemp)")
    return ap.parse_args()


def main() -> None:
    args = _parse_args()
    out_dir = Path(args.out) if args.out else Path(tempfile.mkdtemp(prefix="easyicu_obesity_"))
    out_dir.mkdir(parents=True, exist_ok=True)
    out = {}
    csv_rows = []
    for label, sub, db in DBS:
        dp = f"{ROOT}/{sub}"
        try:
            df = load_concepts(["bmi", "death"], database=db, data_path=dp, batch_size=2_000_000)
            idc = df.columns[0]
            g = df.groupby(idc).agg(bmi=("bmi", "max"), death=("death", "max"))
            g = g.dropna(subset=["bmi"])
            g = g[(g.bmi >= 10) & (g.bmi <= 80)]
            g["death"] = g["death"].fillna(0).clip(0, 1)
            rows = []
            for name, lo, hi in CATS:
                m = (g.bmi >= lo) & (g.bmi < hi)
                n = int(m.sum())
                k = int(g.death[m].sum())
                p, loci, hici = wilson(k, n)
                rows.append({"cat": name, "n": n, "deaths": k, "mort": p, "lo": loci, "hi": hici})
                csv_rows.append({"database": label, "bmi_category": name, "n": n, "deaths": k,
                                 "mortality_pct": p, "ci_lo": loci, "ci_hi": hici})
            # J-uptick: obese II-III lower CI > obese I upper CI (non-overlap upward)
            o1 = next(r for r in rows if r["cat"] == "obese I")
            o2 = next(r for r in rows if r["cat"] == "obese II-III")
            uptick = (o2["mort"] is not None and o1["mort"] is not None and o2["lo"] is not None
                      and o1["hi"] is not None and o2["lo"] > o1["hi"])
            ntot = int(g.shape[0])
            dtot = int(g.death.sum())
            out[label] = {"n_with_bmi": ntot, "overall_mort_pct": round(100 * dtot / ntot, 1) if ntot else None,
                        "by_bmi_category": rows, "extreme_obesity_j_uptick": bool(uptick)}
            print(f"[{label}] n={ntot} overall={out[label]['overall_mort_pct']}% J-uptick={uptick}", flush=True)
        except Exception as e:
            out[label] = {"error": str(e)[:160]}
            print(f"[{label}] ERR {str(e)[:120]}", flush=True)
    (out_dir / "obesity_paradox_crossdb.json").write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    pd.DataFrame(csv_rows).to_csv(out_dir / "obesity_paradox_crossdb.csv", index=False)
    print(f"wrote {out_dir}/{{json,csv}}")


if __name__ == "__main__":
    main()
