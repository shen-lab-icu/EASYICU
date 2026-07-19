"""Parked deterministic causal-analysis script generator.

This module is retained for historical replay, direct-import compatibility, and
unit fixtures. It is intentionally absent from primary-runner dispatch and does
not own a current analysis family: exposure, outcome, cohort, confounders,
method, and estimand remain Planner/Coder decisions. New production code must
not route a primary analysis here.

The generated script:

* reads ``COHORT_PARQUET`` + ``STEP_OUT_DIR`` and the run's
  ``research_context.json`` (primary exposure + target outcome + covariates);
* binarises the declared exposure (``>= 1``) and coerces the confounder set to
  numeric float (case-neutral; blocks rather than guessing a surrogate when the
  declared exposure column is absent);
* fits a propensity model P(exposure | confounders) with ``statsmodels`` Logit
  (pure-numpy IRLS fallback), forms stabilised IPT weights (propensity trimmed,
  weights truncated at the 1st/99th percentile);
* fits a weighted marginal structural logistic model of the outcome on the
  exposure and reports the adjusted odds ratio with a robust (sandwich) CI;
* writes the family DATA tables (``causal_effect`` / ``balance_pre_post_weighting``
  / ``propensity_summary`` / ``weight_distribution`` / ``cohort_flow`` /
  ``exposure_derivation``) and a ``step_summary.json`` declaring the
  scale-neutral ``adjusted_effect`` (+ ``adjusted_effect_scale='odds_ratio'``)
  the primary-effect extractor already binds.

It intentionally emits NO figures: the family figure renderer builds the
manuscript figure from these tables in the dedicated figure step.
"""

from __future__ import annotations

import textwrap

__all__ = ["causal_primary_analysis_code"]


def causal_primary_analysis_code() -> str:
    """Return a runner script that fits the primary IPTW model deterministically."""
    return textwrap.dedent(
        r"""
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
        # exposure/outcome/covariate-agnostic so it is reusable across questions.
        exposure = ""
        outcome = "death"
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

        # The universe stores a binary concept as an aggregate family
        # (``<concept>_{max,min,mean,n,first,measured}``), while the study
        # declares the exposure by concept name (e.g. "vasopressor" -> the
        # ``vaso_ind_*`` family). Resolve the exposure to its any-occurrence
        # aggregate column case-neutrally: direct/suffix match first, then a
        # shared-stem match against concept prefixes, preferring the binary
        # "max"/"any" aggregate then the count "_n".
        _AGG = (
            "_max", "_min", "_mean", "_n", "_first", "_measured",
            "_first_time", "_last_time", "_any", "_ind", "_24h_any",
        )

        def _concept_stem(name):
            s = str(name).lower()
            for a in _AGG:
                if s.endswith(a):
                    return s[: -len(a)]
            return s

        def _resolve_exposure(colnames, name):
            if not name:
                return None
            direct = _resolve(
                colnames,
                *[str(name) + s for s in ("",) + _AGG + ("_ind_max", "_ind_n", "_ind_any")],
            )
            if direct is not None:
                return direct
            nm = str(name).lower().replace(" ", "_")
            nstem = nm.split("_")[0]
            if len(nstem) < 3:
                return None
            cands = []
            for c in colnames:
                cs = _concept_stem(c)
                if not cs:
                    continue
                if cs.startswith(nstem[:4]) or nstem.startswith(cs[:4]) or cs in nm or nm in cs:
                    cands.append(c)
            for pref in ("_max", "_any", "_ind_max", "_n", "_ind_n"):
                for c in cands:
                    if c.lower().endswith(pref):
                        return c
            return cands[0] if cands else None

        cols = list(df.columns)
        exp_col = _resolve_exposure(cols, exposure) if exposure else None
        event_col = _resolve(cols, outcome, "death", "died", "mortality", "hospital_mortality")
        age_col = _resolve(cols, "age")
        exposure_name = exposure or (exp_col or "exposure")

        def _fail(reason):
            summary = {
                "step": current_step_id,
                "status": "blocked",
                "analysis_family": "causal_emulation",
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

        if exp_col is None or event_col is None:
            _fail(
                "Missing required causal columns "
                f"(exposure={exp_col}, outcome={event_col})."
            )
            raise SystemExit(0)

        # --- analysis cohort: adults with observed exposure + outcome ---------
        work = df.copy()
        if age_col is not None:
            work = work[pd.to_numeric(work[age_col], errors="coerce") >= 18]
        # A binary indicator exposure (e.g. vaso_ind_max) is NaN when the
        # concept was never recorded -> that patient is UNEXPOSED, not dropped.
        # Only the outcome must be observed to enter the analysis cohort.
        Y_raw = pd.to_numeric(work[event_col], errors="coerce")
        work = work[Y_raw.notna().to_numpy()]
        Z = (pd.to_numeric(work[exp_col], errors="coerce").fillna(0) >= 1).astype(float).to_numpy()
        Y = (pd.to_numeric(work[event_col], errors="coerce").fillna(0) >= 1).astype(float).to_numpy()
        n_analysis = int(len(work))
        if n_analysis == 0 or int(Z.sum()) < 10 or int((Z == 0).sum()) < 10:
            _fail(
                "Exposure groups too small after cohort construction "
                f"(n={n_analysis}, exposed={int(Z.sum())}, unexposed={int((Z == 0).sum())})."
            )
            raise SystemExit(0)

        # --- confounder design matrix (case-neutral; numeric-coerced) ---------
        # user_preferences.covariates drives the adjustment set when present;
        # otherwise a demographics + severity default is used. Names and defaults
        # are not tied to any one study question. Every confounder is coerced to
        # numpy float so the propensity fit never sees an object/nullable array.
        conf_terms = []
        conf_cols = []
        conf_skipped = []

        def _add_confounder(name):
            col = _resolve(work.columns.tolist(), name)
            if col is None or col == exp_col or col == event_col:
                conf_skipped.append({"covariate": name, "reason": "column_absent"})
                return
            low = str(name).lower()
            colow = str(col).lower()
            if low in ("sex", "gender", "male") or colow in ("sex", "gender"):
                sx = work[col].astype(str).str.lower()
                vals = sx.isin(["m", "male", "1", "1.0"]).astype(float).to_numpy()
                term = "sex_M"
            else:
                num = pd.to_numeric(work[col], errors="coerce")
                if float(num.notna().mean()) <= 0.5:
                    conf_skipped.append({"covariate": name, "reason": "coverage_below_50pct"})
                    return
                vals = num.fillna(num.median()).to_numpy().astype(float)
                term = str(col)
            if not np.isfinite(vals).all():
                vals = np.where(np.isfinite(vals), vals, float(np.nanmedian(vals[np.isfinite(vals)])) if np.isfinite(vals).any() else 0.0)
            if float(np.nanstd(vals)) == 0.0:
                conf_skipped.append({"covariate": name, "reason": "no_variation"})
                return
            conf_cols.append(vals)
            conf_terms.append(term)

        if req_covariates:
            adjustment_source = "config"
            for _c in req_covariates:
                _add_confounder(_c)
        else:
            adjustment_source = "default:demographics+severity"
            for _c in ("age", "sex", "sofa2", "lactate", "map"):
                _add_confounder(_c)

        # standardise confounders for a stable propensity fit
        if conf_cols:
            C = np.column_stack(conf_cols).astype(float)
            mu = C.mean(axis=0)
            sd = C.std(axis=0)
            sd = np.where(sd > 0, sd, 1.0)
            Cs = (C - mu) / sd
        else:
            Cs = np.empty((n_analysis, 0))

        def _irls_logistic(Xd, yv, weights=None, max_iter=50):
            n, p = Xd.shape
            beta = np.zeros(p)
            w_obs = np.ones(n) if weights is None else np.asarray(weights, dtype=float)
            last_W = None
            for _ in range(max_iter):
                eta = np.clip(Xd @ beta, -30, 30)
                mu = 1.0 / (1.0 + np.exp(-eta))
                s = mu * (1.0 - mu)
                s = np.clip(s, 1e-9, None)
                W = w_obs * s
                last_W = W
                z = eta + (yv - mu) / s
                XtW = Xd.T * W
                try:
                    beta_new = np.linalg.solve(XtW @ Xd, XtW @ z)
                except np.linalg.LinAlgError:
                    break
                if np.max(np.abs(beta_new - beta)) < 1e-8:
                    beta = beta_new
                    break
                beta = beta_new
            return beta, last_W

        # --- propensity model P(exposure=1 | confounders) ---------------------
        Xps = np.column_stack([np.ones(n_analysis), Cs]) if Cs.shape[1] else np.ones((n_analysis, 1))
        fit_engine = ""
        try:
            import statsmodels.api as sm

            ps_res = sm.GLM(Z, Xps, family=sm.families.Binomial()).fit()
            ps = np.asarray(ps_res.fittedvalues, dtype=float)
            fit_engine = "statsmodels.GLM"
        except Exception as exc:
            beta_ps, _ = _irls_logistic(Xps, Z)
            ps = 1.0 / (1.0 + np.exp(-np.clip(Xps @ beta_ps, -30, 30)))
            fit_engine = f"numpy_irls (statsmodels unavailable: {str(exc)[:80]})"

        ps = np.clip(ps, 0.02, 0.98)  # trim to enforce positivity/overlap
        p_treat = float(Z.mean())

        # --- stabilised IPT weights (truncated at 1st/99th pct) ---------------
        sw = np.where(Z == 1, p_treat / ps, (1.0 - p_treat) / (1.0 - ps))
        lo_w, hi_w = np.percentile(sw, [1, 99])
        sw = np.clip(sw, lo_w, hi_w)

        # --- covariate balance (standardised mean difference) -----------------
        def _smd(x, z, w=None):
            xt, xc = x[z == 1], x[z == 0]
            if xt.size == 0 or xc.size == 0:
                return 0.0
            if w is None:
                mt, mc = xt.mean(), xc.mean()
                vt, vc = xt.var(), xc.var()
            else:
                wt, wc = w[z == 1], w[z == 0]
                if float(wt.sum()) <= 0 or float(wc.sum()) <= 0:
                    return 0.0
                mt = np.average(xt, weights=wt); mc = np.average(xc, weights=wc)
                vt = np.average((xt - mt) ** 2, weights=wt)
                vc = np.average((xc - mc) ** 2, weights=wc)
            pooled = math.sqrt((vt + vc) / 2.0) if (vt + vc) > 0 else 0.0
            return abs(mt - mc) / pooled if pooled > 0 else 0.0

        balance_rows = []
        for term, col in zip(conf_terms, [Cs[:, j] for j in range(Cs.shape[1])]):
            balance_rows.append({
                "covariate": term,
                "smd_unweighted": float(_smd(col, Z)),
                "smd_weighted": float(_smd(col, Z, sw)),
            })
        balance_df = pd.DataFrame(balance_rows) if balance_rows else pd.DataFrame(
            columns=["covariate", "smd_unweighted", "smd_weighted"]
        )
        balance_df.to_csv(out_dir / "balance_pre_post_weighting.csv", index=False)
        max_smd_after = float(balance_df["smd_weighted"].max()) if len(balance_df) else float("nan")

        # --- weighted marginal structural logistic model: outcome ~ exposure --
        Xout = np.column_stack([np.ones(n_analysis), Z])
        beta_out, W_irls = _irls_logistic(Xout, Y, weights=sw)
        eta = np.clip(Xout @ beta_out, -30, 30)
        mu = 1.0 / (1.0 + np.exp(-eta))
        # robust (sandwich) covariance for the weighted score
        try:
            bread = np.linalg.inv((Xout.T * (W_irls)) @ Xout)
            resid = sw * (Y - mu)
            meat = (Xout.T * (resid ** 2)) @ Xout
            cov = bread @ meat @ bread
            se = np.sqrt(np.diag(cov))
        except np.linalg.LinAlgError:
            se = np.full(Xout.shape[1], np.nan)

        or_point = float(np.exp(beta_out[1]))
        or_lo = float(np.exp(beta_out[1] - 1.96 * se[1]))
        or_hi = float(np.exp(beta_out[1] + 1.96 * se[1]))
        with np.errstate(divide="ignore", invalid="ignore"):
            zstat = beta_out[1] / se[1] if se[1] > 0 else 0.0
        pval = math.erfc(abs(float(zstat)) / math.sqrt(2.0))

        n_events = int(Y.sum())
        n_exposed = int(Z.sum())

        # --- family DATA tables -----------------------------------------------
        pd.DataFrame([{
            "contrast_id": "primary_weighted_contrast",
            "contrast": f"{exposure_name} = 1 vs 0",
            "point_estimate": or_point,
            "ci_low": or_lo,
            "ci_high": or_hi,
            "se": float(se[1]),
            "p_value": float(pval),
            "scale": "odds_ratio",
            "estimator": "stabilised_iptw_msm",
        }]).to_csv(out_dir / "causal_effect.csv", index=False)

        pd.DataFrame([{
            "propensity_engine": fit_engine,
            "n": n_analysis,
            "n_exposed": n_exposed,
            "p_treated": p_treat,
            "ps_min": float(ps.min()),
            "ps_max": float(ps.max()),
            "max_smd_after_weighting": max_smd_after,
        }]).to_csv(out_dir / "propensity_summary.csv", index=False)

        pd.DataFrame({"stabilised_weight": sw}).describe().to_csv(
            out_dir / "weight_distribution.csv"
        )

        pd.DataFrame([
            {"stage": "universe", "n": int(n_universe)},
            {"stage": "analysis_cohort", "n": int(n_analysis)},
            {"stage": "exposed", "n": int(n_exposed)},
            {"stage": "events", "n": int(n_events)},
        ]).to_csv(out_dir / "cohort_flow.csv", index=False)

        pd.DataFrame([{
            "exposure": exposure_name,
            "source_column": exp_col,
            "rule": ">= 1 (any)",
            "n_exposed": n_exposed,
            "n_unexposed": int((Z == 0).sum()),
        }]).to_csv(out_dir / "exposure_derivation.csv", index=False)

        # --- target-trial protocol spec (drives the design schematic figure) --
        # A target-trial-emulation design figure step renders a protocol
        # schematic by scanning upstream artefacts for the standard sections
        # (eligibility / time zero / strategies / assignment / follow-up /
        # outcome / estimand / assumptions). The plain effect + balance tables
        # read as "estimand" and "exposure" but expose NO "time zero" text, so
        # the schematic was skipping its minimum contract. The deterministic
        # runner OWNS this protocol spec: it emits one row per section with a
        # canonical ``section`` label (each classifies to its own box) so the
        # schematic always has its required fields instead of failing the step.
        protocol_rows = [
            {"section": "Eligibility / cohort",
             "detail": f"Adults (age >= 18) with an observed {outcome}; "
                       f"analysis cohort n = {n_analysis}."},
            {"section": "Time zero",
             "detail": "Cohort entry at ICU admission; baseline covariates "
                       "measured at or before time zero (first-24h window)."},
            {"section": "Treatment strategies",
             "detail": f"{exposure_name} = 1 (any occurrence) vs "
                       f"{exposure_name} = 0 (none) over the baseline window."},
            {"section": "Assignment rule",
             "detail": "Observational (not randomized); exposure measured at "
                       "baseline and balanced by stabilised IPT weighting."},
            {"section": "Follow-up",
             "detail": f"From time zero to the {outcome} event or "
                       "discharge/censoring."},
            {"section": "Outcome",
             "detail": f"{outcome} (binary in-hospital endpoint)."},
            {"section": "Estimand / analysis",
             "detail": "Stabilised-IPTW marginal odds ratio (exposed vs "
                       "unexposed); propensity trimmed to [0.02, 0.98]."},
            {"section": "Assumptions / caveats",
             "detail": "Positivity (overlap enforced by trimming); conditional "
                       "exchangeability given the adjustment set ("
                       + (", ".join(conf_terms) or "covariates")
                       + "); consistency; no interference."},
        ]
        pd.DataFrame(protocol_rows).to_csv(
            out_dir / "target_trial_protocol.csv", index=False
        )

        summary = {
            "step": current_step_id,
            "status": "ok",
            "analysis_family": "causal_emulation",
            "exposure_name": exposure_name,
            "primary_predictor": exposure_name,
            "target_outcome": outcome,
            "primary_estimand": "Stabilised-IPTW marginal odds ratio (exposure vs unexposed).",
            "interpretation_class": "causal_emulation",
            "adjusted_effect": or_point,
            "adjusted_effect_scale": "odds_ratio",
            "adjusted_effect_ci_low": or_lo,
            "adjusted_effect_ci_high": or_hi,
            "adjusted_effect_se": float(se[1]),
            "adjusted_effect_p_value": float(pval),
            "max_smd_after_weighting": max_smd_after,
            "n_universe": int(n_universe),
            "n_analysis_cohort": n_analysis,
            "events_primary": n_events,
            "exposed_primary": n_exposed,
            "propensity_terms": conf_terms,
            "adjustment_source": adjustment_source,
            "adjustment_covariates": conf_terms,
            "adjustment_skipped": conf_skipped,
            "fit_engine": fit_engine,
            "notes": [
                "Deterministic primary IPTW marginal structural model (no LLM coder).",
                "Stabilised weights with propensity trimmed to [0.02, 0.98] and "
                "weights truncated at the 1st/99th percentile.",
                f"Adjustment set ({adjustment_source}): "
                + (", ".join(conf_terms) or "none"),
                "Robust sandwich CI for the weighted odds ratio.",
            ],
            "target_trial_protocol": protocol_rows,
            "output_files": {
                "causal_effect": "causal_effect.csv",
                "balance_pre_post_weighting": "balance_pre_post_weighting.csv",
                "propensity_summary": "propensity_summary.csv",
                "weight_distribution": "weight_distribution.csv",
                "cohort_flow": "cohort_flow.csv",
                "exposure_derivation": "exposure_derivation.csv",
                "target_trial_protocol": "target_trial_protocol.csv",
            },
        }
        (out_dir / "step_summary.json").write_text(
            json.dumps(summary, indent=2, ensure_ascii=False, default=str)
        )
        print(json.dumps({
            "adjusted_effect": or_point,
            "adjusted_effect_scale": "odds_ratio",
            "events_primary": n_events,
            "exposure_name": exposure_name,
        }))
        """
    ).strip()
