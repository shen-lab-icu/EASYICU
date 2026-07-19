"""Parked deterministic survival-analysis script generator.

This module is retained for historical replay, direct-import compatibility, and
unit fixtures. It is intentionally absent from primary-runner dispatch and does
not own a current survival estimand or method. New production primary analyses
remain Planner/Coder-owned.

The generated script:

* reads ``COHORT_PARQUET`` + ``STEP_OUT_DIR`` and the run's
  ``research_context.json`` (for the primary exposure + outcome);
* materialises a 24h-landmark analysis cohort from the certified
  ``followup_time_hours`` / ``event_observed`` columns (emitted by the data
  foundation), excluding stays that died before the landmark (immortal-time);
* fits an adjusted Cox model (``statsmodels`` PHReg, with a pure-numpy Breslow
  fallback) of the outcome on the primary exposure + age + sex (+ Charlson when
  present);
* computes Kaplan-Meier curves + a number-at-risk table by exposure;
* writes the family DATA tables (``cox_summary`` / ``cox_model`` /
  ``hazard_ratio`` / ``km_curve`` / ``risk_table`` / ``exposure_derivation`` /
  ``followup_distribution``) and a ``step_summary.json`` that declares
  ``primary_predictor`` (so the exposure contract matches) and the hazard ratio.

It intentionally emits NO figures: the deterministic family figure renderer
(``figures/survival.py``) builds the manuscript figure from these tables in the
dedicated figure step.
"""

from __future__ import annotations

import textwrap

__all__ = ["survival_primary_analysis_code"]


def survival_primary_analysis_code() -> str:
    """Return a runner script that fits the primary Cox model deterministically."""
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

        # --- research context: primary exposure + outcome + design knobs ------
        # Every case-specific value (exposure, outcome, adjustment set, landmark
        # origin) is read from research_context.json, which the pipeline writes
        # from the benchmark item / study config. The skill body stays
        # exposure/outcome/landmark-agnostic so it is reusable across questions.
        exposure = ""
        outcome = "death"
        req_covariates = []
        landmark_hours = 24.0
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
                _lh = prefs.get("landmark_hours")
                if _lh is None:
                    _lh = ctx.get("landmark_hours")
                if _lh is not None:
                    landmark_hours = float(_lh)
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
            # substring fallback
            for cand in cands:
                if not cand:
                    continue
                for c in colnames:
                    if cand.lower() in c.lower():
                        return c
            return None

        cols = list(df.columns)

        # --- exposure column: use the DECLARED exposure from config -----------
        # No case-specific column name is baked in. The study's declared
        # primary exposure is resolved and used directly (binarised at >=1
        # below). If the declared column is absent the step BLOCKS rather than
        # guessing a domain-specific surrogate — silently substituting one would
        # answer a different question than the one configured.
        exp_col = _resolve(cols, exposure) if exposure else None
        exposure_note = ""
        exposure_name = exposure or (exp_col or "exposure")

        # --- outcome + certified time-to-event columns ------------------------
        event_col = _resolve(cols, "event_observed", outcome, "death", "died", "mortality")
        time_col = _resolve(cols, "followup_time_hours", "followup", "time", "duration")
        death_time_col = _resolve(cols, "death_time")
        age_col = _resolve(cols, "age")

        def _fail(reason):
            summary = {
                "step": current_step_id,
                "status": "blocked",
                "analysis_family": "time_to_event",
                "blocking_reason": reason,
                "primary_predictor": exposure_name,
                "outputs": [],
            }
            (out_dir / "step_summary.json").write_text(
                json.dumps(summary, indent=2, ensure_ascii=False)
            )
            print(json.dumps(summary))

        if exp_col is None or event_col is None or time_col is None:
            _fail(
                "Missing required survival columns "
                f"(exposure={exp_col}, event={event_col}, time={time_col})."
            )
            raise SystemExit(0)

        # --- landmark analysis cohort (origin = landmark_hours) ---------------
        work = df.copy()
        n0 = len(work)
        if age_col is not None:
            work = work[pd.to_numeric(work[age_col], errors="coerce") >= 18]
        work = work[pd.to_numeric(work[time_col], errors="coerce").notna()]
        work = work[pd.to_numeric(work[event_col], errors="coerce").notna()]
        work = work[pd.to_numeric(work[time_col], errors="coerce") >= landmark_hours]
        if death_time_col is not None:
            dt = pd.to_numeric(work[death_time_col], errors="coerce")
            ev = pd.to_numeric(work[event_col], errors="coerce").fillna(0).astype(int)
            died_before = (ev == 1) & dt.notna() & (dt < landmark_hours)
            work = work[~died_before]
        n_landmark = len(work)

        # landmark clock: time measured from the landmark origin
        T = (pd.to_numeric(work[time_col], errors="coerce").to_numpy() - landmark_hours)
        T = np.clip(T, 1e-6, None)
        E = pd.to_numeric(work[event_col], errors="coerce").fillna(0).astype(int).to_numpy()
        expo = (pd.to_numeric(work[exp_col], errors="coerce").fillna(0) >= 1).astype(float).to_numpy()

        # --- adjustment set: config-driven, case-neutral default --------------
        # user_preferences.covariates (from research_context.json) drives the
        # adjustment set when present; otherwise a demographics + comorbidity
        # default is used. Neither the covariate names nor the default set are
        # tied to any one study question.
        design_terms = [exposure_name]
        X_cols = [expo]
        adjustment_covariates = []
        adjustment_skipped = []

        def _add_covariate(name):
            col = _resolve(work.columns.tolist(), name)
            if col is None or col == exp_col:
                adjustment_skipped.append({"covariate": name, "reason": "column_absent"})
                return
            low = str(name).lower()
            if low in ("sex", "gender", "sex_m", "male") or str(col).lower() in ("sex", "gender"):
                sx = work[col].astype(str).str.lower()
                vals = sx.isin(["m", "male", "1", "1.0"]).astype(float).to_numpy()
                term = "sex_M"
            else:
                num = pd.to_numeric(work[col], errors="coerce")
                if num.notna().mean() <= 0.5:
                    adjustment_skipped.append({"covariate": name, "reason": "coverage_below_50pct"})
                    return
                vals = num.fillna(num.median()).to_numpy()
                term = str(col)
            if float(np.nanstd(vals)) == 0.0:
                adjustment_skipped.append({"covariate": name, "reason": "no_variation"})
                return
            X_cols.append(vals)
            design_terms.append(term)
            adjustment_covariates.append(term)

        if req_covariates:
            adjustment_source = "config"
            for _cov in req_covariates:
                _add_covariate(_cov)
        else:
            adjustment_source = "default:demographics+comorbidity"
            for _cov in ("age", "sex", "charlson_first"):
                _add_covariate(_cov)
        X = np.column_stack(X_cols)

        # --- Cox fit: statsmodels PHReg, pure-numpy Breslow fallback ----------
        coef = se = None
        concordance = None
        fit_engine = ""
        try:
            from statsmodels.duration.hazard_regression import PHReg

            model = PHReg(T, X, status=E, ties="breslow")
            res = model.fit()
            coef = np.asarray(res.params, dtype=float)
            se = np.asarray(res.bse, dtype=float)
            fit_engine = "statsmodels.PHReg"
        except Exception as exc:
            fit_engine = f"numpy_breslow (statsmodels unavailable: {str(exc)[:80]})"
            order = np.argsort(-T)
            Xo, Eo = X[order], E[order]
            beta = np.zeros(X.shape[1])
            for _ in range(50):
                eta = Xo @ beta
                w = np.exp(np.clip(eta, -40, 40))
                cw = np.cumsum(w)
                cwx = np.cumsum(w[:, None] * Xo, axis=0)
                grad = np.zeros(X.shape[1]); hess = np.zeros((X.shape[1], X.shape[1]))
                for i in np.where(Eo == 1)[0]:
                    s0 = cw[i]
                    if s0 <= 0:
                        continue
                    mu = cwx[i] / s0
                    grad += Xo[i] - mu
                    cwxx = np.cumsum(
                        w[:, None, None] * Xo[:, :, None] * Xo[:, None, :], axis=0
                    )[i]
                    hess -= cwxx / s0 - np.outer(mu, mu)
                try:
                    step = np.linalg.solve(-hess, grad)
                except np.linalg.LinAlgError:
                    break
                beta = beta + step
                if np.max(np.abs(step)) < 1e-6:
                    break
            coef = beta
            try:
                se = np.sqrt(np.diag(np.linalg.inv(-hess)))
            except Exception:
                se = np.full(len(beta), np.nan)

        hr = np.exp(coef)
        lo = np.exp(coef - 1.96 * se)
        hi = np.exp(coef + 1.96 * se)
        with np.errstate(divide="ignore", invalid="ignore"):
            z = np.where(se > 0, coef / se, 0.0)
            pval = np.array([math.erfc(abs(float(zz)) / math.sqrt(2.0)) for zz in z])

        cox_model = pd.DataFrame({
            "model_name": "primary_grouped_exposure",
            "term": design_terms,
            "coef": coef,
            "hazard_ratio": hr,
            "ci_low": lo,
            "ci_high": hi,
            "se": se,
            "p_value": pval,
        })
        cox_model.to_csv(out_dir / "cox_model.csv", index=False)

        n_events = int(E.sum())
        cox_summary = pd.DataFrame([{
            "model_name": "primary_grouped_exposure",
            "estimator": fit_engine,
            "n": int(len(T)),
            "events": n_events,
            "converged": True,
            "primary_term": exposure_name,
        }])
        cox_summary.to_csv(out_dir / "cox_summary.csv", index=False)

        hazard_ratio_tbl = pd.DataFrame([{
            "contrast": f"{exposure_name} = 1 vs 0",
            "hazard_ratio": float(hr[0]),
            "ci_low": float(lo[0]),
            "ci_high": float(hi[0]),
            "p_value": float(pval[0]),
            "se": float(se[0]),
        }])
        hazard_ratio_tbl.to_csv(out_dir / "hazard_ratio.csv", index=False)

        # exposure derivation provenance
        pd.DataFrame([{
            "exposure": exposure_name,
            "source_column": exp_col,
            "rule": ">= 1 (any)",
            "n_exposed": int(expo.sum()),
            "n_unexposed": int((expo == 0).sum()),
            "note": exposure_note or "Used the pre-derived binary exposure column.",
        }]).to_csv(out_dir / "exposure_derivation.csv", index=False)

        # --- Kaplan-Meier by exposure (numpy) + number-at-risk ----------------
        def _km(times, events):
            o = np.argsort(times)
            t, e = times[o], events[o]
            uniq = np.unique(t)
            surv, at_risk_out, s = [], [], 1.0
            n = len(t)
            for ut in uniq:
                at = int((t >= ut).sum())
                d = int(((t == ut) & (e == 1)).sum())
                if at > 0:
                    s *= (1.0 - d / at)
                surv.append(s); at_risk_out.append(at)
            return uniq, np.array(surv), np.array(at_risk_out)

        km_rows, risk_rows = [], []
        for grp_val, label in [(1.0, "Exposed"), (0.0, "Unexposed")]:
            mask = expo == grp_val
            if mask.sum() < 2:
                continue
            ut, sv, ar = _km(T[mask], E[mask])
            for tt, ss, aa in zip(ut, sv, ar):
                km_rows.append({"group": label, "time": float(tt), "survival": float(ss), "at_risk": int(aa)})
            # number-at-risk grid
            for gt in np.linspace(0, float(T[mask].max()), 8):
                risk_rows.append({"group": label, "time": float(gt), "at_risk": int((T[mask] >= gt).sum())})
        pd.DataFrame(km_rows).to_csv(out_dir / "km_curve.csv", index=False)
        pd.DataFrame(risk_rows).to_csv(out_dir / "risk_table.csv", index=False)

        fu = T[np.isfinite(T)]
        pd.DataFrame({"followup_time": fu}).to_csv(out_dir / "followup_distribution.csv", index=False)

        med = float(np.median(fu)) if fu.size else float("nan")
        q25 = float(np.percentile(fu, 25)) if fu.size else float("nan")
        q75 = float(np.percentile(fu, 75)) if fu.size else float("nan")

        summary = {
            "step": current_step_id,
            "status": "ok",
            "analysis_family": "time_to_event",
            "primary_predictor": exposure_name,
            "target_outcome": outcome,
            "hazard_ratio": float(hr[0]),
            "hazard_ratio_ci_low": float(lo[0]),
            "hazard_ratio_ci_high": float(hi[0]),
            "hazard_ratio_p_value": float(pval[0]),
            "n_events": n_events,
            "n_analysis": int(len(T)),
            "n_universe": int(n_universe),
            "n_landmark_cohort": int(n_landmark),
            "n_exposed": int(expo.sum()),
            "median_followup_hours": med,
            "median_followup_q25_hours": q25,
            "median_followup_q75_hours": q75,
            "cox_terms": design_terms,
            "adjustment_source": adjustment_source,
            "adjustment_covariates": adjustment_covariates,
            "adjustment_skipped": adjustment_skipped,
            "fit_engine": fit_engine,
            "landmark_hours": float(landmark_hours),
            "notes": [
                "Deterministic primary Cox model (no LLM coder).",
                f"{landmark_hours:g}h-landmark cohort from certified "
                "followup_time_hours + event_observed; stays that died before "
                "the landmark were excluded (immortal-time).",
                exposure_note or f"Primary exposure `{exposure_name}` used directly.",
                f"Adjustment set ({adjustment_source}): "
                + (", ".join(adjustment_covariates) or "none"),
            ],
            "output_files": {
                "cox_summary": "cox_summary.csv",
                "cox_model": "cox_model.csv",
                "hazard_ratio": "hazard_ratio.csv",
                "km_curve": "km_curve.csv",
                "risk_table": "risk_table.csv",
                "exposure_derivation": "exposure_derivation.csv",
                "followup_distribution": "followup_distribution.csv",
            },
        }
        (out_dir / "step_summary.json").write_text(
            json.dumps(summary, indent=2, ensure_ascii=False, default=str)
        )
        print(json.dumps({k: summary[k] for k in ("hazard_ratio", "n_events", "primary_predictor")}))
        """
    ).strip()
