"""Parked deterministic ordinal dose-response script generator.

This module is retained for historical replay, direct-import compatibility, and
unit fixtures. It is intentionally absent from primary-runner dispatch and does
not own the exposure encoding, model, contrasts, or estimand. New production
primary analyses remain Planner/Coder-owned.

The generated script:

* reads ``COHORT_PARQUET`` + ``STEP_OUT_DIR`` and the run's
  ``research_context.json`` (primary exposure + target outcome + covariates);
* resolves the declared exposure to its ordinal aggregate column case-neutrally
  (peak/``_max`` tier), validating it is a low-cardinality ordered integer grade
  with >= 3 levels (blocks rather than guessing when it is not ordinal);
* fits a covariate-adjusted logistic **trend** model (outcome ~ stage_linear +
  confounders) and reports the odds ratio per +1 stage -- the scale-neutral
  ``adjusted_effect`` (+ ``adjusted_effect_scale='odds_ratio'``) the primary-
  effect extractor already binds as the manuscript headline;
* fits a covariate-adjusted logistic **per-stage** model (outcome ~ C(stage),
  lowest stage as reference) for the forest rows, and checks whether the crude
  per-stage event rates are monotonically non-decreasing;
* writes a secondary continuous-outcome gradient (per-stage median/mean + a
  rank trend) when a continuous secondary outcome column is present;
* writes the family DATA tables (``dose_response`` / ``dose_response_trend`` /
  ``los_gradient`` / ``cohort_flow`` / ``exposure_derivation``) and a
  ``step_summary.json`` with ``analysis_family='association'`` so the existing
  association forest renderer draws the per-stage odds ratios.

It intentionally emits NO figures: the family figure renderer builds the
manuscript figure from these tables in the dedicated figure step.
"""

from __future__ import annotations

import textwrap

__all__ = ["ordinal_dose_response_analysis_code"]


def ordinal_dose_response_analysis_code() -> str:
    """Return a runner script that fits the primary dose-response gradient."""
    return textwrap.dedent(r"""
        import json
        import math
        import os
        from pathlib import Path

        import numpy as np
        import pandas as pd

        out_dir = Path(os.environ["STEP_OUT_DIR"])
        out_dir.mkdir(parents=True, exist_ok=True)
        run_dir = out_dir.parents[2]
        current_step_id = out_dir.parent.name
        cohort_path = Path(os.environ["COHORT_PARQUET"])

        df = pd.read_parquet(cohort_path).copy()
        n_universe = len(df)

        # --- research context: exposure + outcome + adjustment set ------------
        # Every case-specific value is read from research_context.json (written
        # from the benchmark item / study config). The skill body stays
        # exposure/outcome/covariate-agnostic so it is reusable across graded
        # exposures (KDIGO stage, severity tier, quartile, ...).
        exposure = ""
        outcome = "death"
        secondary_outcome = ""
        req_covariates = []
        try:
            ctx = json.loads((run_dir / "research_context.json").read_text("utf-8"))
            exposure = str(ctx.get("primary_exposure") or "").strip()
            outcome = str(ctx.get("target_outcome") or "death").strip() or "death"
            prefs = ctx.get("user_preferences") or {}
            if isinstance(prefs, dict):
                req_covariates = [
                    str(c).strip()
                    for c in (prefs.get("covariates") or [])
                    if str(c).strip()
                ]
                _sec = prefs.get("secondary_outcomes") or prefs.get("secondary_outcome")
                if isinstance(_sec, (list, tuple)):
                    secondary_outcome = str(_sec[0]).strip() if _sec else ""
                elif _sec:
                    secondary_outcome = str(_sec).strip()
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

        # --- resolve the graded exposure to its ordinal aggregate column ------
        # The universe stores a graded concept as an aggregate family
        # (``<concept>_{max,min,mean,n,first,measured}``). The dose is the PEAK
        # tier in the window -> the ``_max`` aggregate. Prefer the composite
        # grade (``aki_stage_max``) over its sub-component grades
        # (``aki_stage_creat_max`` / ``_uo_`` / ``_rrt_``) and over the
        # non-grade aggregates (``_mean`` / ``_n`` / ``_measured``).
        # Case-neutral clinical aliases map a declared concept name to its
        # universe stem when they differ (last-resort, not a value guess).
        _ALIASES = {
            "kdigo": ("aki_stage", "kdigo_stage", "aki"),
            "aki": ("aki_stage", "kdigo_stage"),
        }
        _SUBCOMPONENT_INFIX = ("_creat_", "_uo_", "_rrt_", "_scr_", "_cr_")
        _NONGRADE_SUFFIX = ("_mean", "_n", "_measured", "_first_time", "_last_time")

        def _is_ordinal_grade(series):
            num = pd.to_numeric(series, errors="coerce").dropna()
            if num.empty:
                return False, []
            vals = num.to_numpy()
            # integer-valued, small cardinality, ordered from a low base
            if not np.all(np.isfinite(vals)):
                return False, []
            if np.any(np.abs(vals - np.round(vals)) > 1e-9):
                return False, []
            levels = sorted(int(v) for v in np.unique(np.round(vals)))
            if len(levels) < 3 or len(levels) > 12:
                return False, levels
            if (max(levels) - min(levels)) > 20:
                return False, levels
            return True, levels

        def _resolve_ordinal_exposure(colnames, name):
            if not name:
                return None, []
            nm = str(name).lower().replace(" ", "_")
            stems = [nm]
            for key, aliases in _ALIASES.items():
                if key in nm or nm in key:
                    stems.extend(a.lower() for a in aliases)
            # candidate columns: contain a stem, end in ``_max``, are not a
            # sub-component grade nor a non-grade aggregate.
            cands = []
            for c in colnames:
                cl = c.lower()
                if not cl.endswith("_max"):
                    continue
                if any(sub in cl for sub in _SUBCOMPONENT_INFIX):
                    continue
                if any(cl.endswith(sfx) for sfx in _NONGRADE_SUFFIX):
                    continue
                if any(cl.startswith(s[:4]) or s in cl for s in stems if len(s) >= 3):
                    cands.append(c)
            # prefer the shortest matching name (the composite, not a variant)
            cands.sort(key=lambda c: (len(c), c))
            for c in cands:
                ok, levels = _is_ordinal_grade(df[c])
                if ok:
                    return c, levels
            # fall back to a direct ``<name>_max`` / ``<name>`` resolution
            direct = _resolve(colnames, nm + "_max", nm)
            if direct is not None:
                ok, levels = _is_ordinal_grade(df[direct])
                if ok:
                    return direct, levels
            return None, []

        cols = list(df.columns)
        exp_col, exp_levels = _resolve_ordinal_exposure(cols, exposure)
        event_col = _resolve(
            cols, outcome, "death", "died", "mortality", "hospital_mortality"
        )
        age_col = _resolve(cols, "age")
        exposure_name = exposure or (exp_col or "exposure")

        def _fail(reason):
            summary = {
                "step": current_step_id,
                "status": "blocked",
                "analysis_family": "association",
                "blocking_reason": reason,
                "exposure_name": exposure_name,
                "target_outcome": outcome,
                "adjusted_effect": None,
                "primary_estimand": "Blocked: " + reason,
                "outputs": [],
            }
            (out_dir / "step_summary.json").write_text(
                json.dumps(summary, indent=2, ensure_ascii=False)
            )
            print(json.dumps(summary))

        if exp_col is None:
            _fail(
                "Could not resolve a graded ordinal exposure column for "
                f"'{exposure_name}' (need a low-cardinality ordered integer "
                "grade with >= 3 levels, e.g. an ``aki_stage_max`` aggregate)."
            )
            raise SystemExit(0)
        if event_col is None:
            _fail(f"Missing binary outcome column (outcome='{outcome}').")
            raise SystemExit(0)

        # --- analysis cohort: adults with observed grade + outcome ------------
        work = df.copy()
        if age_col is not None:
            work = work[pd.to_numeric(work[age_col], errors="coerce") >= 18]
        stage_raw = pd.to_numeric(work[exp_col], errors="coerce")
        y_raw = pd.to_numeric(work[event_col], errors="coerce")
        keep = stage_raw.notna().to_numpy() & y_raw.notna().to_numpy()
        work = work[keep]
        stage = np.round(pd.to_numeric(work[exp_col], errors="coerce").to_numpy()).astype(
            float
        )
        Y = (pd.to_numeric(work[event_col], errors="coerce").fillna(0) >= 1).astype(
            float
        ).to_numpy()
        n_analysis = int(len(work))
        levels = sorted(int(v) for v in np.unique(stage))
        if n_analysis == 0 or len(levels) < 3:
            _fail(
                "Too few graded exposure levels after cohort construction "
                f"(n={n_analysis}, levels={levels})."
            )
            raise SystemExit(0)
        ref_level = levels[0]

        # --- confounder design matrix (case-neutral; numeric-coerced) ---------
        # user_preferences.covariates drives the adjustment set when present;
        # otherwise a demographics + severity default is used. The severity
        # term auto-detects a total SOFA score, else the present per-organ SOFA
        # ``_max`` components EXCLUDING the renal component (which shares the
        # exposure's causal pathway for an AKI-grade dose). Names/defaults are
        # not tied to any one study question.
        conf_terms = []
        conf_cols = []
        conf_skipped = []

        def _confounder_values(name):
            col = _resolve(work.columns.tolist(), name)
            if col is None or col == exp_col or col == event_col:
                conf_skipped.append({"covariate": name, "reason": "column_absent"})
                return None, None
            low = str(name).lower()
            colow = str(col).lower()
            if low in ("sex", "gender", "male") or colow in ("sex", "gender"):
                sx = work[col].astype(str).str.lower()
                vals = sx.isin(["m", "male", "1", "1.0"]).astype(float).to_numpy()
                return "sex_M", vals
            num = pd.to_numeric(work[col], errors="coerce")
            if float(num.notna().mean()) <= 0.5:
                conf_skipped.append({"covariate": name, "reason": "coverage_below_50pct"})
                return None, None
            vals = num.fillna(num.median()).to_numpy().astype(float)
            if not np.isfinite(vals).all():
                finite = vals[np.isfinite(vals)]
                fill = float(np.nanmedian(finite)) if finite.size else 0.0
                vals = np.where(np.isfinite(vals), vals, fill)
            return str(col), vals

        def _add_confounder(name):
            term, vals = _confounder_values(name)
            if term is None or vals is None:
                return
            if float(np.nanstd(vals)) == 0.0:
                conf_skipped.append({"covariate": name, "reason": "no_variation"})
                return
            conf_cols.append(vals)
            conf_terms.append(term)

        def _default_severity_terms(colnames):
            total = _resolve(colnames, "sofa2", "sofa_total", "sofa", "apache", "saps")
            if total is not None:
                return [total]
            organs = ("resp", "cardio", "cns", "coag", "liver")  # renal excluded
            found = []
            for organ in organs:
                c = _resolve(colnames, f"sofa_{organ}_max", f"sofa_{organ}")
                if c is not None:
                    found.append(c)
            return found

        if req_covariates:
            adjustment_source = "config"
            adj_names = list(req_covariates)
        else:
            adjustment_source = "default:demographics+severity"
            adj_names = ["age", "sex"] + _default_severity_terms(work.columns.tolist())
        for _c in adj_names:
            _add_confounder(_c)

        if conf_cols:
            C = np.column_stack(conf_cols).astype(float)
            mu = C.mean(axis=0)
            sd = C.std(axis=0)
            sd = np.where(sd > 0, sd, 1.0)
            Cs = (C - mu) / sd
        else:
            Cs = np.empty((n_analysis, 0))

        def _irls_logistic(Xd, yv, max_iter=100):
            n, p = Xd.shape
            beta = np.zeros(p)
            for _ in range(max_iter):
                eta = np.clip(Xd @ beta, -30, 30)
                mu = 1.0 / (1.0 + np.exp(-eta))
                s = np.clip(mu * (1.0 - mu), 1e-9, None)
                z = eta + (yv - mu) / s
                XtW = Xd.T * s
                try:
                    beta_new = np.linalg.solve(XtW @ Xd, XtW @ z)
                except np.linalg.LinAlgError:
                    break
                if np.max(np.abs(beta_new - beta)) < 1e-9:
                    beta = beta_new
                    break
                beta = beta_new
            eta = np.clip(Xd @ beta, -30, 30)
            mu = 1.0 / (1.0 + np.exp(-eta))
            s = np.clip(mu * (1.0 - mu), 1e-9, None)
            try:
                cov = np.linalg.inv((Xd.T * s) @ Xd)
                se = np.sqrt(np.diag(cov))
            except np.linalg.LinAlgError:
                se = np.full(p, np.nan)
            return beta, se

        def _fit_logit(Xd, yv):
            try:
                import statsmodels.api as sm

                res = sm.GLM(yv, Xd, family=sm.families.Binomial()).fit()
                return (
                    np.asarray(res.params, dtype=float),
                    np.asarray(res.bse, dtype=float),
                    "statsmodels.GLM",
                )
            except Exception as exc:
                beta, se = _irls_logistic(Xd, yv)
                return beta, se, f"numpy_irls ({str(exc)[:60]})"

        # --- primary: covariate-adjusted TREND (odds ratio per +1 stage) ------
        stage_lin = stage.astype(float)
        Xtrend = np.column_stack([np.ones(n_analysis), stage_lin, Cs])
        beta_t, se_t, fit_engine = _fit_logit(Xtrend, Y)
        trend_or = float(np.exp(beta_t[1]))
        trend_lo = float(np.exp(beta_t[1] - 1.96 * se_t[1]))
        trend_hi = float(np.exp(beta_t[1] + 1.96 * se_t[1]))
        with np.errstate(divide="ignore", invalid="ignore"):
            ztrend = beta_t[1] / se_t[1] if se_t[1] > 0 else 0.0
        trend_p = math.erfc(abs(float(ztrend)) / math.sqrt(2.0))

        # --- per-stage odds ratios vs the lowest stage (forest rows) ----------
        nonref = [lv for lv in levels if lv != ref_level]
        stage_dummies = np.column_stack(
            [(stage == lv).astype(float) for lv in nonref]
        ) if nonref else np.empty((n_analysis, 0))
        Xcat = np.column_stack([np.ones(n_analysis), stage_dummies, Cs])
        beta_c, se_c, _ = _fit_logit(Xcat, Y)

        per_stage_rows = []
        crude_rates = {}
        for lv in levels:
            mask = stage == lv
            n_lv = int(mask.sum())
            ev_lv = int(Y[mask].sum())
            rate = float(ev_lv / n_lv) if n_lv else float("nan")
            crude_rates[lv] = rate
            row = {
                "stage": int(lv),
                "n": n_lv,
                "n_events": ev_lv,
                "event_rate": rate,
                "is_reference": bool(lv == ref_level),
            }
            if lv == ref_level:
                row.update(
                    {"odds_ratio": 1.0, "or_ci_low": 1.0, "or_ci_high": 1.0, "or_p_value": float("nan")}
                )
            else:
                j = 1 + nonref.index(lv)
                b, s = float(beta_c[j]), float(se_c[j])
                row.update(
                    {
                        "odds_ratio": float(np.exp(b)),
                        "or_ci_low": float(np.exp(b - 1.96 * s)) if np.isfinite(s) else float("nan"),
                        "or_ci_high": float(np.exp(b + 1.96 * s)) if np.isfinite(s) else float("nan"),
                        "or_p_value": (
                            math.erfc(abs(b / s) / math.sqrt(2.0)) if s > 0 else float("nan")
                        ),
                    }
                )
            per_stage_rows.append(row)

        ordered_rates = [crude_rates[lv] for lv in levels if math.isfinite(crude_rates[lv])]
        monotonic = all(
            ordered_rates[i] <= ordered_rates[i + 1] + 1e-9
            for i in range(len(ordered_rates) - 1)
        )
        n_events = int(Y.sum())

        # --- secondary: continuous-outcome gradient (median/mean + rank trend)-
        los_col = None
        if secondary_outcome:
            los_col = _resolve(work.columns.tolist(), secondary_outcome)
        if los_col is None:
            los_col = _resolve(
                work.columns.tolist(), "los_icu", "icu_los", "los", "length_of_stay"
            )
        los_rows = []
        los_trend = None
        if los_col is not None and los_col not in (exp_col, event_col):
            los_vals = pd.to_numeric(work[los_col], errors="coerce").to_numpy()
            los_ok = np.isfinite(los_vals)
            for lv in levels:
                mask = (stage == lv) & los_ok
                vv = los_vals[mask]
                if vv.size:
                    los_rows.append(
                        {
                            "stage": int(lv),
                            "n": int(vv.size),
                            "median": float(np.median(vv)),
                            "mean": float(np.mean(vv)),
                            "q1": float(np.percentile(vv, 25)),
                            "q3": float(np.percentile(vv, 75)),
                        }
                    )
            # Spearman (rank-Pearson) trend of stage vs the continuous outcome
            sv = stage[los_ok]
            lv2 = los_vals[los_ok]
            if sv.size > 2 and np.unique(sv).size > 1 and np.unique(lv2).size > 1:
                rs = pd.Series(sv).rank().to_numpy()
                rl = pd.Series(lv2).rank().to_numpy()
                rho = float(np.corrcoef(rs, rl)[0, 1])
                los_trend = {
                    "outcome": str(los_col),
                    "spearman_rho": rho,
                    "n": int(sv.size),
                }

        # --- family DATA tables -----------------------------------------------
        pd.DataFrame(per_stage_rows).to_csv(out_dir / "dose_response.csv", index=False)
        pd.DataFrame(
            [
                {
                    "contrast_id": "trend_per_stage",
                    "contrast": f"{exposure_name} per +1 stage",
                    "point_estimate": trend_or,
                    "ci_low": trend_lo,
                    "ci_high": trend_hi,
                    "se": float(se_t[1]),
                    "p_value": float(trend_p),
                    "scale": "odds_ratio",
                    "estimator": "adjusted_logistic_trend",
                    "per_stage_monotonic": bool(monotonic),
                }
            ]
        ).to_csv(out_dir / "dose_response_trend.csv", index=False)
        if los_rows:
            pd.DataFrame(los_rows).to_csv(out_dir / "los_gradient.csv", index=False)
        pd.DataFrame(
            [
                {"stage": "universe", "n": int(n_universe)},
                {"stage": "analysis_cohort", "n": int(n_analysis)},
                {"stage": "events", "n": int(n_events)},
            ]
        ).to_csv(out_dir / "cohort_flow.csv", index=False)
        pd.DataFrame(
            [
                {
                    "exposure": exposure_name,
                    "source_column": exp_col,
                    "grade_levels": ",".join(str(lv) for lv in levels),
                    "reference_level": int(ref_level),
                    "rule": "peak (_max) ordinal grade in window",
                }
            ]
        ).to_csv(out_dir / "exposure_derivation.csv", index=False)

        summary = {
            "step": current_step_id,
            "status": "ok",
            "analysis_family": "association",
            "exposure_name": exposure_name,
            "primary_predictor": exposure_name,
            "target_outcome": outcome,
            "primary_estimand": (
                "Covariate-adjusted odds ratio per +1 "
                f"{exposure_name} stage (dose-response trend)."
            ),
            "interpretation_class": "dose_response",
            "adjusted_effect": trend_or,
            "adjusted_effect_scale": "odds_ratio",
            "adjusted_effect_ci_low": trend_lo,
            "adjusted_effect_ci_high": trend_hi,
            "adjusted_effect_se": float(se_t[1]),
            "adjusted_effect_p_value": float(trend_p),
            "per_stage_monotonic": bool(monotonic),
            "grade_levels": [int(lv) for lv in levels],
            "reference_level": int(ref_level),
            "per_stage_effects": per_stage_rows,
            "secondary_outcome_gradient": los_trend,
            "n_universe": int(n_universe),
            "n_analysis_cohort": n_analysis,
            "events_primary": n_events,
            "adjustment_source": adjustment_source,
            "adjustment_covariates": conf_terms,
            "adjustment_skipped": conf_skipped,
            "fit_engine": fit_engine,
            "notes": [
                "Deterministic primary dose-response gradient (no LLM coder).",
                "Headline = covariate-adjusted logistic trend OR per +1 stage; "
                "per-stage ORs (vs lowest stage) are the forest rows.",
                f"Adjustment set ({adjustment_source}): "
                + (", ".join(conf_terms) or "none"),
                (
                    "Per-stage crude event rates are "
                    + ("monotonically non-decreasing." if monotonic else "NOT monotonic.")
                ),
            ],
            "output_files": {
                "dose_response": "dose_response.csv",
                "dose_response_trend": "dose_response_trend.csv",
                **({"los_gradient": "los_gradient.csv"} if los_rows else {}),
                "cohort_flow": "cohort_flow.csv",
                "exposure_derivation": "exposure_derivation.csv",
            },
        }
        (out_dir / "step_summary.json").write_text(
            json.dumps(summary, indent=2, ensure_ascii=False, default=str)
        )
        print(
            json.dumps(
                {
                    "adjusted_effect": trend_or,
                    "adjusted_effect_scale": "odds_ratio",
                    "per_stage_monotonic": bool(monotonic),
                    "events_primary": n_events,
                    "exposure_name": exposure_name,
                }
            )
        )
        """).strip()
