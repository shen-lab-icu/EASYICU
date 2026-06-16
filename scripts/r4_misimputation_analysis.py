#!/usr/bin/env python
"""R4 analysis: structural-no-source mis-imputation bias on SOFA-2.

Reads the per-stay SOFA-2 tables written by r4_crossdb_sofa2_extract.py and
produces the source data + numbers for Figure 4.

Three results:

  A. WITHIN-COHORT ABLATION (ground truth) on a structurally complete DB
     (default MIMIC-IV). For each organ component c, simulate "structural
     no-source" by removing c entirely, then handle the gap three ways:
       - leave_out   : 5-component partial score (honest handling)
       - zero_impute : treat absent component as 0 / "normal" (silent default;
                       for a SUM this equals leave_out in level but is *claimed*
                       to be a full SOFA-2)
       - mean_impute : fill c with its cohort-mean subscore ("borrow")
     Quantify vs the true 6-component score: severity understatement (ΔSOFA),
     mortality discrimination (AUROC, bootstrap CI), and Sepsis-3 (SOFA>=2)
     mis-classification.

  B. CROSS-DB COMPARABILITY: a database with a real structural gap (SICdb, no
     GCS -> no CNS) vs a complete DB (MIMIC-IV). Naive total-SOFA-2 comparison
     vs component-matched comparison (both restricted to the shared 5
     components) -> shows the apparent severity gap is a structural-imputation
     artifact, not case-mix.

  C. LEAVE-ONE-COMPONENT-OUT across all six components on the complete DB ->
     which structural gap, mis-imputed, hurts most (heatmap source data).

Outputs (research_output/r4_misimputation/):
  - ablation_<db>.csv, leave_one_out_<db>.csv, crossdb_comparability.csv
  - fig4_misimputation.png  (draft; manuscript redraw via nature-figure)
  - r4_summary.json

Usage:
    python scripts/r4_misimputation_analysis.py \
        --perstay-dir research_output/r4_sofa2_perstay \
        --complete-db miiv --gap-db sic
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

COMPS = [
    "sofa2_resp", "sofa2_coag", "sofa2_liver",
    "sofa2_cardio", "sofa2_cns", "sofa2_renal",
]
RNG = np.random.default_rng(20260616)


def _auc(score: np.ndarray, y: np.ndarray) -> float:
    """Mann-Whitney AUROC; ignores NaN scores."""
    m = ~np.isnan(score)
    s, yy = score[m], y[m]
    pos, neg = s[yy == 1], s[yy == 0]
    if pos.size == 0 or neg.size == 0:
        return float("nan")
    order = np.argsort(s, kind="mergesort")
    ranks = np.empty_like(order, dtype=float)
    ranks[order] = np.arange(1, s.size + 1)
    # average ties
    _, inv, cnt = np.unique(s, return_inverse=True, return_counts=True)
    csum = np.cumsum(cnt)
    avg = (csum - cnt / 2.0 + 0.5)
    ranks = avg[inv]
    r_pos = ranks[yy == 1].sum()
    return float((r_pos - pos.size * (pos.size + 1) / 2.0) / (pos.size * neg.size))


def _auc_ci(score, y, n=400):
    score = np.asarray(score, float)
    y = np.asarray(y, float)
    base = _auc(score, y)
    idx = np.arange(score.size)
    boots = []
    for _ in range(n):
        b = RNG.choice(idx, size=idx.size, replace=True)
        boots.append(_auc(score[b], y[b]))
    boots = np.array([x for x in boots if not np.isnan(x)])
    lo, hi = (np.percentile(boots, [2.5, 97.5]) if boots.size else (np.nan, np.nan))
    return round(base, 4), round(float(lo), 4), round(float(hi), 4)


def _load(perstay_dir: Path, db: str) -> pd.DataFrame:
    p = perstay_dir / f"{db}.parquet"
    if not p.exists():
        raise FileNotFoundError(p)
    df = pd.read_parquet(p)
    for c in COMPS:
        if c not in df:
            df[c] = np.nan
    if "death" not in df:
        raise ValueError(f"{db}: no death column")
    return df


def ablation(df: pd.DataFrame, outdir: Path, db: str) -> list[dict]:
    """Component-removal ablation on a complete cohort (ground truth)."""
    d = df.dropna(subset=COMPS).copy()  # complete cases = ground truth
    y = d["death"].to_numpy(float)
    true_total = d[COMPS].sum(axis=1).to_numpy(float)
    a_true, lo_t, hi_t = _auc_ci(true_total, y)
    sep_true = (true_total >= 2)
    rows = [{
        "component_removed": "(none) TRUE 6-comp",
        "strategy": "reference",
        "median_sofa2": float(np.median(true_total)),
        "mean_sofa2": round(float(true_total.mean()), 3),
        "mean_understatement": 0.0,
        "auroc": a_true, "auroc_lo": lo_t, "auroc_hi": hi_t,
        "sep3_pos_rate": round(float(sep_true.mean()), 4),
        "sep3_misclass_pct": 0.0,
    }]
    for c in COMPS:
        others = [x for x in COMPS if x != c]
        base5 = d[others].sum(axis=1).to_numpy(float)
        cmean = float(d[c].mean())
        strategies = {
            "leave_out": base5,                      # honest 5-comp
            "zero_impute": base5,                    # absent=0 (silent default)
            "mean_impute": base5 + round(cmean),     # borrow cohort mean
        }
        for name, tot in strategies.items():
            a, lo, hi = _auc_ci(tot, y)
            sep = (tot >= 2)
            # mis-classification vs true Sepsis-3 (SOFA>=2) status
            misclass = float((sep != sep_true).mean())
            rows.append({
                "component_removed": c,
                "strategy": name,
                "median_sofa2": float(np.median(tot)),
                "mean_sofa2": round(float(tot.mean()), 3),
                "mean_understatement": round(float((true_total - tot).mean()), 3),
                "auroc": a, "auroc_lo": lo, "auroc_hi": hi,
                "sep3_pos_rate": round(float(sep.mean()), 4),
                "sep3_misclass_pct": round(100 * misclass, 2),
            })
    out = pd.DataFrame(rows)
    out.to_csv(outdir / f"ablation_{db}.csv", index=False)
    return rows


def crossdb_comparability(complete: pd.DataFrame, gap: pd.DataFrame,
                          gap_missing: list[str], outdir: Path,
                          complete_db: str, gap_db: str) -> dict:
    shared = [c for c in COMPS if c not in gap_missing]
    rows = []

    def _block(df, label, comps):
        sub = df.dropna(subset=comps)
        tot = sub[comps].sum(axis=1).to_numpy(float)
        y = sub["death"].to_numpy(float)
        a, lo, hi = _auc_ci(tot, y)
        return {
            "cohort": label, "components": "+".join(c.split("_")[1] for c in comps),
            "n": int(sub.shape[0]),
            "median_sofa2": float(np.median(tot)),
            "mean_sofa2": round(float(tot.mean()), 3),
            "auroc": a, "auroc_lo": lo, "auroc_hi": hi,
        }

    # naive: complete 6-comp vs gap 5-comp (its real total)
    rows.append({"comparison": "naive", **_block(complete, complete_db, COMPS)})
    rows.append({"comparison": "naive", **_block(gap, gap_db, shared)})
    # matched: both on shared 5 comps
    rows.append({"comparison": "matched", **_block(complete, complete_db, shared)})
    rows.append({"comparison": "matched", **_block(gap, gap_db, shared)})

    out = pd.DataFrame(rows)
    out.to_csv(outdir / "crossdb_comparability.csv", index=False)

    naive = {r["cohort"]: r for r in rows if r["comparison"] == "naive"}
    matched = {r["cohort"]: r for r in rows if r["comparison"] == "matched"}
    naive_gap = naive[complete_db]["median_sofa2"] - naive[gap_db]["median_sofa2"]
    matched_gap = (matched[complete_db]["median_sofa2"]
                   - matched[gap_db]["median_sofa2"])
    return {
        "gap_missing": gap_missing,
        "shared_components": shared,
        "naive_median_gap": round(naive_gap, 3),
        "matched_median_gap": round(matched_gap, 3),
        "artifact_fraction": (round(1 - matched_gap / naive_gap, 3)
                              if naive_gap else None),
        "rows": rows,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--perstay-dir", default="research_output/r4_sofa2_perstay")
    ap.add_argument("--complete-db", default="miiv")
    ap.add_argument("--gap-db", default="sic")
    ap.add_argument("--gap-missing", nargs="+", default=["sofa2_cns"])
    ap.add_argument("--out", default="research_output/r4_misimputation")
    args = ap.parse_args()

    perstay_dir = Path(args.perstay_dir)
    outdir = Path(args.out)
    outdir.mkdir(parents=True, exist_ok=True)
    summary: dict = {"complete_db": args.complete_db, "gap_db": args.gap_db}

    comp_df = _load(perstay_dir, args.complete_db)
    summary["ablation"] = ablation(comp_df, outdir, args.complete_db)
    print(f"[ablation] {args.complete_db}: {comp_df.shape[0]} stays", flush=True)

    try:
        gap_df = _load(perstay_dir, args.gap_db)
        summary["crossdb"] = crossdb_comparability(
            comp_df, gap_df, args.gap_missing, outdir,
            args.complete_db, args.gap_db)
        print(f"[crossdb] {args.gap_db}: naive gap "
              f"{summary['crossdb']['naive_median_gap']} -> matched "
              f"{summary['crossdb']['matched_median_gap']} "
              f"(artifact {summary['crossdb']['artifact_fraction']})", flush=True)
    except FileNotFoundError:
        summary["crossdb"] = {"skipped": f"{args.gap_db} not extracted yet"}
        print(f"[crossdb] skipped: {args.gap_db} parquet not found", flush=True)

    (outdir / "r4_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"wrote {outdir/'r4_summary.json'}")


if __name__ == "__main__":
    main()
