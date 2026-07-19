"""Legacy, unrouted trajectory-feature clustering experiment.

This module is retained only to reproduce historical artifacts and unit fixtures.
It is intentionally absent from the capability registry and execution dispatch:
its SOFA-shaped feature discovery, default outcome, time horizon, k search, and
minimum-window rules make scientific choices that belong to the planner/coder.
New production code must use agent-declared clustering products and deterministic
rendering/validation, not call this script as a general ICU capability.

Design decisions (deliberate, non-overclaiming):

* trajectory-FEATURE clustering, NOT latent-class growth analysis (LCGA/GBTM). We
  build per-stay features per trajectory family (baseline / last-observed / mean /
  slope / AUC / coverage over the OBSERVED windows only) and cluster the feature
  matrix. Labs/scores are NEVER zero-imputed — trailing-window NA from shorter
  stays is real and is excluded from feature computation, not treated as 0.
* k is selected by the highest mean silhouette over a small grid (default 2..6),
  tie-broken to the smaller k; a GaussianMixture BIC pass is recorded as a
  secondary witness. KMeans is deterministic (random_state=0, n_init=10). A
  pure-numpy Lloyd + silhouette fallback keeps the runner working without sklearn.
* the headline is DESCRIPTIVE: ``adjusted_effect`` is ``None`` and no primary
  odds/hazard ratio is emitted. The outcome-by-cluster contrast reports per-class
  mortality with Wilson intervals, never a causal effect.

Blocks (status ``blocked``) when: no SOFA-2 trajectory columns resolve, the
analysis cohort is empty after the minimum-observed-window rule, or fewer than 2
non-degenerate clusters survive (it never fabricates classes).
"""

from __future__ import annotations

import textwrap

__all__ = ["trajectory_clustering_analysis_code"]


def trajectory_clustering_analysis_code() -> str:
    """Return the legacy reproduction script; not a production entrypoint."""
    return textwrap.dedent(r"""
        import json
        import os
        import re
        from pathlib import Path

        import numpy as np
        import pandas as pd

        out_dir = Path(os.environ["STEP_OUT_DIR"])
        out_dir.mkdir(parents=True, exist_ok=True)
        run_dir = out_dir.parents[2]
        current_step_id = out_dir.parent.name
        cohort_path = Path(os.environ["COHORT_PARQUET"])

        df = pd.read_parquet(cohort_path).copy()
        n_universe = int(len(df))

        # --- research context: outcome + tunables (case-neutral) --------------
        outcome = "death"
        min_windows = 4
        k_lo, k_hi = 2, 6
        try:
            ctx = json.loads((run_dir / "research_context.json").read_text("utf-8"))
            outcome = str(ctx.get("target_outcome") or "death").strip() or "death"
            prefs = ctx.get("user_preferences") or {}
            if isinstance(prefs, dict):
                if prefs.get("min_observed_windows"):
                    min_windows = int(prefs["min_observed_windows"])
                kr = prefs.get("k_range") or prefs.get("n_clusters_range")
                if isinstance(kr, (list, tuple)) and len(kr) == 2:
                    k_lo, k_hi = int(kr[0]), int(kr[1])
        except Exception:
            pass

        def _resolve(colnames, *cands):
            low = {c.lower(): c for c in colnames}
            for cand in cands:
                if not cand:
                    continue
                if cand in colnames:
                    return cand
                if cand.lower() in low:
                    return low[cand.lower()]
            for cand in cands:
                if not cand:
                    continue
                for c in colnames:
                    if cand.lower() in c.lower():
                        return c
            return None

        cols = list(df.columns)
        event_col = _resolve(cols, outcome, "death", "died", "mortality", "hospital_mortality")
        age_col = _resolve(cols, "age")

        def _fail(reason):
            summary = {
                "step": current_step_id,
                "status": "blocked",
                "analysis_family": "phenotyping",
                "interpretation_class": "phenotyping_descriptive",
                "blocking_reason": reason,
                "adjusted_effect": None,
                "primary_estimand": "Blocked: " + reason,
                "n_universe": n_universe,
                "outputs": [],
            }
            (out_dir / "step_summary.json").write_text(
                json.dumps(summary, indent=2, ensure_ascii=False)
            )
            print(json.dumps(summary))

        # --- discover trajectory families case-neutrally ----------------------
        # A window column is ``<family>_h<start>_<end>`` where family is ``sofa2``
        # (total) or ``sofa2_<organ>``. Group columns by family, ordered by window
        # start hour.
        win_re = re.compile(r"^(sofa2(?:_[a-z]+)?)_h(\d+)_(\d+)$", re.IGNORECASE)
        families = {}
        for c in cols:
            m = win_re.match(c)
            if not m:
                continue
            fam = m.group(1).lower()
            start = int(m.group(2))
            families.setdefault(fam, []).append((start, c))
        for fam in families:
            families[fam] = [c for _s, c in sorted(families[fam])]

        if not families or "sofa2" not in families:
            _fail(
                "No SOFA-2 trajectory window columns resolved "
                "(expected '<family>_h<start>_<end>', e.g. sofa2_h0_6)."
            )
            raise SystemExit(0)

        total_windows = families["sofa2"]
        n_total_windows = len(total_windows)

        # --- analysis cohort: adults with enough observed TOTAL-SOFA2 windows --
        work = df.copy()
        if age_col is not None:
            work = work[pd.to_numeric(work[age_col], errors="coerce").fillna(999) >= 18]
        total_mat = work[total_windows].apply(pd.to_numeric, errors="coerce")
        observed_total = total_mat.notna().sum(axis=1)
        keep = observed_total >= int(min_windows)
        work = work[keep.to_numpy()]
        n_analysis = int(len(work))
        if n_analysis < max(20, (k_hi + 1)):
            _fail(
                "Analysis cohort too small after the minimum-observed-window rule "
                f"(n={n_analysis}, need >= {max(20, k_hi + 1)}; "
                f"min_observed_windows={min_windows} of {n_total_windows})."
            )
            raise SystemExit(0)

        pd.DataFrame(
            [
                {"stage": "universe", "n": n_universe},
                {"stage": "adults", "n": int((observed_total.index.size))},
                {"stage": f"min_{min_windows}_observed_total_windows", "n": n_analysis},
            ]
        ).to_csv(out_dir / "cohort_flow.csv", index=False)

        pd.DataFrame(
            [
                {
                    "rule": "trailing-window NA excluded from feature computation, "
                    "never imputed as 0",
                    "min_observed_total_windows": int(min_windows),
                    "n_total_windows": int(n_total_windows),
                    "families": ",".join(sorted(families)),
                }
            ]
        ).to_csv(out_dir / "na_handling.csv", index=False)

        # --- per-stay trajectory FEATURES over OBSERVED windows only ----------
        def _family_features(mat, fam):
            # mat: (n, w) numeric array with NaN for unobserved windows.
            n, w = mat.shape
            idx = np.arange(w, dtype=float)
            feats = {}
            baseline = np.full(n, np.nan)
            last = np.full(n, np.nan)
            mean = np.full(n, np.nan)
            slope = np.zeros(n)
            auc = np.full(n, np.nan)
            variability = np.zeros(n)
            coverage = np.zeros(n)
            for i in range(n):
                row = mat[i]
                obs = np.isfinite(row)
                k = int(obs.sum())
                coverage[i] = k / w if w else 0.0
                if k == 0:
                    continue
                xo = idx[obs]
                yo = row[obs]
                baseline[i] = yo[0]
                last[i] = yo[-1]
                mean[i] = float(np.mean(yo))
                if k >= 2:
                    # OLS slope over observed points; trapezoid AUC normalised by span
                    xm = xo - xo.mean()
                    denom = float(np.sum(xm * xm))
                    slope[i] = float(np.sum(xm * (yo - yo.mean())) / denom) if denom > 0 else 0.0
                    span = xo[-1] - xo[0]
                    # trapezoid integral (version-safe; np.trapz removed in newer numpy)
                    trap = float(np.sum((xo[1:] - xo[:-1]) * (yo[1:] + yo[:-1]) / 2.0))
                    auc[i] = trap / span if span > 0 else float(np.mean(yo))
                    variability[i] = float(np.std(yo))
                else:
                    auc[i] = yo[0]
            feats[f"{fam}_baseline"] = baseline
            feats[f"{fam}_last"] = last
            feats[f"{fam}_mean"] = mean
            feats[f"{fam}_slope"] = slope
            feats[f"{fam}_auc"] = auc
            feats[f"{fam}_variability"] = variability
            feats[f"{fam}_coverage"] = coverage
            return feats

        feature_frame = {}
        for fam in sorted(families):
            mat = work[families[fam]].apply(pd.to_numeric, errors="coerce").to_numpy(dtype=float)
            feature_frame.update(_family_features(mat, fam))
        features = pd.DataFrame(feature_frame, index=work.index)

        # Columns that need >=2 observed points fall back to a documented neutral
        # (slope/variability=0); *_baseline/_last/_mean/_auc may still be NaN when a
        # family was never observed for a stay -> fill with the column median (a
        # transparent, non-zero neutral) for the clustering matrix only.
        feat_cols = [c for c in features.columns if not c.endswith("_coverage")]
        X = features[feat_cols].copy()
        col_median = X.median(numeric_only=True)
        X = X.fillna(col_median).fillna(0.0)
        features.to_csv(out_dir / "trajectory_features.csv", index=False)

        # --- standardise (guard zero-variance) --------------------------------
        Xv = X.to_numpy(dtype=float)
        mu = Xv.mean(axis=0)
        sd = Xv.std(axis=0)
        sd = np.where(sd > 1e-12, sd, 1.0)
        Xs = (Xv - mu) / sd

        # --- cluster: KMeans over k grid, silhouette-selected -----------------
        rng_seed = 0

        def _numpy_kmeans(Xs, k, seed):
            rng = np.random.RandomState(seed)
            n = Xs.shape[0]
            centers = Xs[rng.choice(n, k, replace=False)].copy()
            labels = np.zeros(n, dtype=int)
            for _ in range(100):
                d = ((Xs[:, None, :] - centers[None, :, :]) ** 2).sum(axis=2)
                new = d.argmin(axis=1)
                if np.array_equal(new, labels):
                    labels = new
                    break
                labels = new
                for j in range(k):
                    m = labels == j
                    if m.any():
                        centers[j] = Xs[m].mean(axis=0)
            return labels

        def _silhouette(Xs, labels, seed, sample=4000):
            n = Xs.shape[0]
            if len(set(labels.tolist())) < 2:
                return -1.0
            try:
                from sklearn.metrics import silhouette_score

                ss = sample if n > sample else None
                return float(
                    silhouette_score(Xs, labels, sample_size=ss, random_state=seed)
                )
            except Exception:
                rng = np.random.RandomState(seed)
                idx = rng.choice(n, min(sample, n), replace=False)
                Xsub, lsub = Xs[idx], labels[idx]
                sils = []
                for i in range(len(Xsub)):
                    same = lsub == lsub[i]
                    same[i] = False
                    if not same.any():
                        continue
                    a = np.sqrt(((Xsub[same] - Xsub[i]) ** 2).sum(axis=1)).mean()
                    b = np.inf
                    for lab in set(lsub.tolist()):
                        if lab == lsub[i]:
                            continue
                        other = lsub == lab
                        if other.any():
                            b = min(
                                b,
                                np.sqrt(((Xsub[other] - Xsub[i]) ** 2).sum(axis=1)).mean(),
                            )
                    if np.isfinite(b) and max(a, b) > 0:
                        sils.append((b - a) / max(a, b))
                return float(np.mean(sils)) if sils else -1.0

        def _fit_kmeans(Xs, k, seed):
            try:
                from sklearn.cluster import KMeans

                km = KMeans(n_clusters=k, random_state=seed, n_init=10)
                return km.fit_predict(Xs)
            except Exception:
                return _numpy_kmeans(Xs, k, seed)

        def _bic_gmm(Xs, k, seed):
            try:
                from sklearn.mixture import GaussianMixture

                gm = GaussianMixture(n_components=k, random_state=seed, covariance_type="diag")
                gm.fit(Xs)
                return float(gm.bic(Xs))
            except Exception:
                return float("nan")

        metrics_rows = []
        solutions = {}
        k_hi = min(k_hi, max(2, n_analysis - 1))
        for k in range(max(2, k_lo), k_hi + 1):
            labels = _fit_kmeans(Xs, k, rng_seed)
            sizes = np.bincount(labels, minlength=k)
            # degenerate solution guard: a cluster < 1% or < 20 stays
            degenerate = bool(((sizes < max(20, 0.01 * n_analysis)).any()))
            sil = _silhouette(Xs, labels, rng_seed)
            bic = _bic_gmm(Xs, k, rng_seed)
            solutions[k] = (labels, sizes, degenerate)
            metrics_rows.append(
                {
                    "k": int(k),
                    "silhouette": round(float(sil), 6),
                    "bic": round(float(bic), 4) if np.isfinite(bic) else None,
                    "degenerate": degenerate,
                    "chosen": False,
                }
            )

        # choose the highest-silhouette NON-degenerate k, tie-break to smaller k
        eligible = [m for m in metrics_rows if not m["degenerate"]]
        if not eligible:
            _fail(
                "No non-degenerate clustering solution (every k produced a cluster "
                "smaller than 1% / 20 stays); refusing to fabricate latent classes."
            )
            raise SystemExit(0)
        best = max(eligible, key=lambda m: (m["silhouette"], -m["k"]))
        chosen_k = int(best["k"])
        for m in metrics_rows:
            m["chosen"] = bool(m["k"] == chosen_k)
        labels, sizes, _deg = solutions[chosen_k]

        pd.DataFrame(metrics_rows).to_csv(out_dir / "clustering_metrics.csv", index=False)

        # --- stability: seed-perturbation adjusted-Rand (report before interpret)
        def _adjusted_rand(a, b):
            try:
                from sklearn.metrics import adjusted_rand_score

                return float(adjusted_rand_score(a, b))
            except Exception:
                return float("nan")

        rand_scores = []
        for s in range(1, 11):
            lab_s = _fit_kmeans(Xs, chosen_k, s)
            ar = _adjusted_rand(labels, lab_s)
            if np.isfinite(ar):
                rand_scores.append(ar)
        stability_mean = float(np.mean(rand_scores)) if rand_scores else float("nan")
        stability_sd = float(np.std(rand_scores)) if rand_scores else float("nan")
        pd.DataFrame(
            [
                {
                    "chosen_k": chosen_k,
                    "adjusted_rand_mean": round(stability_mean, 6),
                    "adjusted_rand_sd": round(stability_sd, 6),
                    "n_resamples": len(rand_scores),
                }
            ]
        ).to_csv(out_dir / "cluster_stability.csv", index=False)

        # --- certified outputs -------------------------------------------------
        work = work.copy()
        work["cluster"] = labels
        assignments = pd.DataFrame(
            {
                "stay_id": (
                    work[_resolve(work.columns.tolist(), "stay_id", "icustay_id", "id")]
                    if _resolve(work.columns.tolist(), "stay_id", "icustay_id", "id")
                    else np.arange(n_analysis)
                ),
                "cluster": labels,
                "total_sofa2_coverage": features["sofa2_coverage"].to_numpy(),
            }
        )
        assignments.to_csv(out_dir / "cluster_assignments.csv", index=False)

        size_rows = []
        for c in range(chosen_k):
            n_c = int((labels == c).sum())
            size_rows.append(
                {
                    "cluster": int(c),
                    "n": n_c,
                    "pct": round(100.0 * n_c / n_analysis, 4),
                    "degenerate_flag": bool(n_c < max(20, 0.01 * n_analysis)),
                }
            )
        pd.DataFrame(size_rows).to_csv(out_dir / "cluster_sizes.csv", index=False)

        # LONG cluster-characteristic profile (cluster, feature, mean, median)
        char_rows = []
        for c in range(chosen_k):
            m = labels == c
            for col in feat_cols:
                vals = features.loc[m.nonzero()[0] if False else work.index[m], col]
                vals = pd.to_numeric(vals, errors="coerce").dropna()
                if vals.empty:
                    continue
                char_rows.append(
                    {
                        "cluster": int(c),
                        "feature": col,
                        "mean": round(float(vals.mean()), 6),
                        "median": round(float(vals.median()), 6),
                    }
                )
        pd.DataFrame(char_rows).to_csv(out_dir / "cluster_characteristics.csv", index=False)

        # DESCRIPTIVE outcome-by-cluster with Wilson intervals (never causal)
        def _wilson(k, n, z=1.96):
            if n == 0:
                return (float("nan"), float("nan"))
            p = k / n
            denom = 1 + z * z / n
            centre = (p + z * z / (2 * n)) / denom
            half = (z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n))) / denom
            return (max(0.0, centre - half), min(1.0, centre + half))

        outcome_rows = []
        if event_col is not None:
            y = (pd.to_numeric(work[event_col], errors="coerce").fillna(0) >= 1).astype(int).to_numpy()
            for c in range(chosen_k):
                m = labels == c
                n_c = int(m.sum())
                d_c = int(y[m].sum())
                lo, hi = _wilson(d_c, n_c)
                outcome_rows.append(
                    {
                        "cluster": int(c),
                        "n": n_c,
                        "n_deaths": d_c,
                        "mortality_rate": round(d_c / n_c, 6) if n_c else None,
                        "ci_low": round(lo, 6),
                        "ci_high": round(hi, 6),
                    }
                )
        pd.DataFrame(outcome_rows).to_csv(out_dir / "outcome_by_cluster.csv", index=False)

        chosen_metric = next(m for m in metrics_rows if m["chosen"])
        summary = {
            "step": current_step_id,
            "status": "ok",
            "analysis_family": "phenotyping",
            "interpretation_class": "phenotyping_descriptive",
            "primary_estimand": (
                f"Deterministic trajectory-feature clustering into {chosen_k} latent "
                "classes (silhouette-selected) with seed-stability and a DESCRIPTIVE "
                "outcome-by-cluster contrast (not causal)."
            ),
            "adjusted_effect": None,
            "n_universe": n_universe,
            "n_analysis": n_analysis,
            "n_clusters": chosen_k,
            "chosen_k": chosen_k,
            "silhouette": chosen_metric["silhouette"],
            "bic": chosen_metric["bic"],
            "stability_mean": round(stability_mean, 6),
            "trajectory_families": sorted(families),
            "notes": [
                "Deterministic trajectory-feature clustering (no LLM coder).",
                "KMeans over a silhouette-selected k grid; GMM BIC recorded as a "
                "secondary witness; seed-perturbation adjusted-Rand stability.",
                "Trailing-window NA excluded from feature computation; SOFA-2 windows "
                "are NEVER imputed as 0.",
                "Clusters are DESCRIPTIVE phenotypes; the outcome-by-cluster contrast "
                "is not a causal effect.",
            ],
            "output_files": {
                "cluster_assignments": "cluster_assignments.csv",
                "cluster_sizes": "cluster_sizes.csv",
                "cluster_characteristics": "cluster_characteristics.csv",
                "clustering_metrics": "clustering_metrics.csv",
                "cluster_stability": "cluster_stability.csv",
                "outcome_by_cluster": "outcome_by_cluster.csv",
                "trajectory_features": "trajectory_features.csv",
                "na_handling": "na_handling.csv",
                "cohort_flow": "cohort_flow.csv",
            },
        }
        (out_dir / "step_summary.json").write_text(
            json.dumps(summary, indent=2, ensure_ascii=False, default=str)
        )
        print(
            json.dumps(
                {
                    "n_analysis": n_analysis,
                    "chosen_k": chosen_k,
                    "silhouette": chosen_metric["silhouette"],
                    "stability_mean": round(stability_mean, 6),
                }
            )
        )
        """).strip()
