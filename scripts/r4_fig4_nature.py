#!/usr/bin/env python
"""Figure 4 (publication, npj Digital Medicine) — Python/matplotlib.

Structural-no-source mis-imputation bias on SOFA-2 across harmonized ICU
databases. Code-backed result figure per the figure protocol; source data are
the CSVs written by r4_misimputation_analysis.py.

Core claim (one sentence): treating a structurally-absent SOFA-2 organ
component as imputable missing data silently understates severity, alters
mortality discrimination, and misclassifies Sepsis-3 — and the EasyICU
six-database concept layer shows the correct handling is database-specific
(SICdb's absent CNS being the real-world case).

Layout: hero Panel A (full width) + bottom row B/C/D (asymmetric hero).
  A  Mortality discrimination (AUROC, 95% bootstrap CI) when each organ
     component is unavailable, vs the true 6-component reference. AUROC is
     rank-based, so leave-out / zero-impute / mean-impute are identical here —
     one bar per component is the honest representation.
  B  Severity understatement (mean SOFA-2 points) under the silent default
     (drop / zero-"normal" impute) vs mean-impute.
  C  Sepsis-3 (SOFA>=2) misclassification (%) under the same two handlings.
  D  Real cross-database comparability: MIMIC-IV vs SICdb (no CNS source),
     naive total vs component-matched — the apparent severity gap is half
     structural artifact.

Exports editable SVG + PDF (vector) + 600 dpi TIFF.

Usage:
    python scripts/r4_fig4_nature.py --indir research_output/r4_misimputation \
        --db miiv --gap-db sic --out research_output/r4_misimputation/Figure4
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import Patch

mpl.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
    "svg.fonttype": "none",
    "pdf.fonttype": 42,
    "font.size": 7,
    "axes.titlesize": 8,
    "axes.labelsize": 7,
    "xtick.labelsize": 6.5,
    "ytick.labelsize": 6.5,
    "legend.fontsize": 6,
    "axes.spines.right": False,
    "axes.spines.top": False,
    "axes.linewidth": 0.8,
    "legend.frameon": False,
    "xtick.major.width": 0.8,
    "ytick.major.width": 0.8,
})

# Restrained palette: one neutral, one signal (harmful silent default), one accent.
C_NEUTRAL = "#7A8CA3"   # slate — component unavailable (Panel A)
C_REF = "#222222"        # reference line
C_SILENT = "#C6603D"    # signal/coral — drop / zero "normal" impute (harmful default)
C_MEAN = "#3D7FA6"      # accent/teal — mean-impute
C_NAIVE = "#C6603D"     # naive cross-DB
C_MATCH = "#3D7FA6"     # component-matched
C_LOSS = "#B5453B"      # AUROC loss highlight

COMP_LABEL = {
    "sofa2_resp": "Resp", "sofa2_coag": "Coag", "sofa2_liver": "Liver",
    "sofa2_cardio": "Cardio", "sofa2_cns": "CNS", "sofa2_renal": "Renal",
}


def _paneltag(ax, tag, dx=-0.085, dy=1.04):
    ax.text(dx, dy, tag, transform=ax.transAxes, fontsize=10, fontweight="bold",
            va="top", ha="right")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--indir", default="research_output/r4_misimputation")
    ap.add_argument("--db", default="miiv", help="complete reference DB")
    ap.add_argument("--gap-db", default="sic")
    ap.add_argument("--out", default="research_output/r4_misimputation/Figure4")
    args = ap.parse_args()
    indir = Path(args.indir)

    abl = pd.read_csv(indir / f"ablation_{args.db}.csv")
    ref = abl[abl["strategy"] == "reference"].iloc[0]
    lo = abl[abl["strategy"] == "leave_out"].set_index("component_removed")
    mn = abl[abl["strategy"] == "mean_impute"].set_index("component_removed")
    comps = [c for c in COMP_LABEL if c in lo.index]
    labels = [COMP_LABEL[c] for c in comps]

    # reference complete-case n (from crossdb naive row for the same DB)
    _cd0 = pd.read_csv(indir / "crossdb_comparability.csv")
    try:
        _ref_n = int(_cd0[(_cd0["comparison"] == "naive") &
                          (_cd0["cohort"] == args.db)]["n"].iloc[0])
    except Exception:
        _ref_n = 0

    fig = plt.figure(figsize=(7.2, 5.8))  # ~183 mm wide
    gs = fig.add_gridspec(2, 3, height_ratios=[1.0, 1.0], hspace=0.62, wspace=0.42,
                          left=0.08, right=0.985, top=0.9, bottom=0.1)
    axA = fig.add_subplot(gs[0, :])
    axB = fig.add_subplot(gs[1, 0])
    axC = fig.add_subplot(gs[1, 1])
    axD = fig.add_subplot(gs[1, 2])

    x = np.arange(len(comps))

    # ---------- Panel A: AUROC ----------
    aur = lo.loc[comps, "auroc"].to_numpy()
    aer = np.vstack([aur - lo.loc[comps, "auroc_lo"].to_numpy(),
                     lo.loc[comps, "auroc_hi"].to_numpy() - aur])
    # color: loss (below reference) vs gain
    cols = [C_LOSS if v < ref["auroc"] else C_NEUTRAL for v in aur]
    axA.axhspan(ref["auroc_lo"], ref["auroc_hi"], color="0.85", zorder=0)
    axA.axhline(ref["auroc"], color=C_REF, ls="--", lw=1.0, zorder=1)
    axA.bar(x, aur, 0.62, yerr=aer, capsize=2.5, color=cols,
            edgecolor="white", linewidth=0.4,
            error_kw=dict(lw=0.8, ecolor="0.25"), zorder=2)
    axA.text(len(comps) - 0.5, ref["auroc"] + 0.0015,
             f"true 6-component SOFA-2 = {ref['auroc']:.3f}",
             ha="right", va="bottom", fontsize=6.3, color=C_REF)
    for xi, v in zip(x, aur):
        axA.text(xi, lo.loc[comps[xi], "auroc_hi"] + 0.001, f"{v:.3f}",
                 ha="center", va="bottom", fontsize=5.8, color="0.2")
    axA.set_xticks(x); axA.set_xticklabels(labels)
    axA.set_ylabel("AUROC, in-hospital mortality")
    axA.set_xlabel("Organ component made structurally unavailable")
    ymin = min(lo.loc[comps, "auroc_lo"].min(), ref["auroc_lo"]) - 0.004
    ymax = max(lo.loc[comps, "auroc_hi"].max(), ref["auroc_hi"]) + 0.006
    axA.set_ylim(ymin, ymax)
    axA.set_title(
        f"Mortality discrimination when one SOFA-2 component is structurally "
        f"unavailable ({args.db.upper()} complete cases, n = {_ref_n:,})",
        fontsize=8, pad=8, loc="left")
    _paneltag(axA, "a", dx=-0.045)
    axA.legend(handles=[
        Patch(fc=C_LOSS, label="discrimination lost vs full SOFA-2"),
        Patch(fc=C_NEUTRAL, label="no loss / slight gain"),
    ], loc="upper left", ncol=1, bbox_to_anchor=(0.0, 1.0))

    # ---------- Panel B: severity understatement ----------
    w = 0.4
    us_silent = lo.loc[comps, "mean_understatement"].to_numpy()
    us_mean = mn.loc[comps, "mean_understatement"].to_numpy()
    axB.axhline(0, color="0.4", lw=0.7)
    axB.bar(x - w/2, us_silent, w, color=C_SILENT, label="drop / zero-impute (silent default)")
    axB.bar(x + w/2, us_mean, w, color=C_MEAN, label="mean-impute")
    axB.set_xticks(x); axB.set_xticklabels(labels, rotation=30, ha="right")
    axB.set_ylabel("SOFA-2 understatement (points)")
    _paneltag(axB, 'b')
    axB.set_title("Severity bias", fontsize=8)
    # shared strategy legend lives in the inter-row gap (used by panels b & c)
    fig.legend(handles=[
        Patch(fc=C_SILENT, label="drop / zero-“normal” impute (silent default)"),
        Patch(fc=C_MEAN, label="mean-impute"),
    ], loc="center", ncol=2, bbox_to_anchor=(0.5, 0.475), fontsize=6.3)

    # ---------- Panel C: Sepsis-3 misclassification ----------
    mc_silent = lo.loc[comps, "sep3_misclass_pct"].to_numpy()
    mc_mean = mn.loc[comps, "sep3_misclass_pct"].to_numpy()
    axC.bar(x - w/2, mc_silent, w, color=C_SILENT)
    axC.bar(x + w/2, mc_mean, w, color=C_MEAN)
    axC.set_xticks(x); axC.set_xticklabels(labels, rotation=30, ha="right")
    axC.set_ylabel("Sepsis-3 (SOFA≥2) misclassified (%)")
    _paneltag(axC, 'c')
    axC.set_title("Clinical-threshold flips", fontsize=8)

    # ---------- Panel D: cross-DB comparability ----------
    cd = pd.read_csv(indir / "crossdb_comparability.csv")
    cohorts = [args.db, args.gap_db]
    coh_label = {"miiv": "MIMIC-IV\n(6-comp)", "sic": "SICdb\n(no CNS)"}
    naive = cd[cd["comparison"] == "naive"].set_index("cohort")
    matched = cd[cd["comparison"] == "matched"].set_index("cohort")
    xd = np.arange(len(cohorts))
    nv = [naive.loc[c, "median_sofa2"] for c in cohorts]
    mt = [matched.loc[c, "median_sofa2"] for c in cohorts]
    axD.bar(xd - w/2, nv, w, color=C_NAIVE, label="naive total")
    axD.bar(xd + w/2, mt, w, color=C_MATCH, label="component-matched (5-comp)")
    axD.set_xticks(xd)
    axD.set_xticklabels([coh_label.get(c, c) for c in cohorts])
    axD.set_ylabel("Median SOFA-2")
    naive_gap = nv[0] - nv[1]
    matched_gap = mt[0] - mt[1]
    art = (1 - matched_gap / naive_gap) if naive_gap else float("nan")
    axD.set_ylim(0, max(nv + mt) + 3.8)
    axD.legend(loc="upper left", bbox_to_anchor=(-0.02, 1.02), fontsize=5.8)
    axD.annotate(f"naive gap {naive_gap:.0f} → matched {matched_gap:.0f} pt:\n"
                 f"{art*100:.0f}% of the apparent gap\nis structural CNS artifact",
                 xy=(0.98, 0.74), xycoords="axes fraction",
                 ha="right", va="top", fontsize=5.6, color="0.12")
    _paneltag(axD, 'd')
    axD.set_title("Cross-database comparability", fontsize=8)

    fig.suptitle(
        "Mis-imputing a structurally-absent SOFA-2 organ component distorts severity, "
        "discrimination, and Sepsis-3 classification",
        fontsize=9, fontweight="bold", y=0.975, x=0.5)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out.with_suffix(".svg"), bbox_inches="tight")
    fig.savefig(out.with_suffix(".pdf"), bbox_inches="tight")
    fig.savefig(out.with_suffix(".tiff"), dpi=600, bbox_inches="tight",
                pil_kwargs={"compression": "tiff_lzw"})
    fig.savefig(out.with_suffix(".png"), dpi=300, bbox_inches="tight")
    print(f"wrote {out}.svg/.pdf/.tiff/.png")
    print(f"crossdb: naive_gap={naive_gap}, matched_gap={matched_gap}, artifact={art:.2f}")


if __name__ == "__main__":
    main()
