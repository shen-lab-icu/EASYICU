"""Deterministic missingness / measurement-process audit runner.

The audit counterpart of :mod:`deterministic_ordinal` /
:mod:`deterministic_survival`: a per-concept measurement-missingness table that
distinguishes **structural no-source** (the concept is not sourced for the stay
at all) from **measurement missingness** (the concept is sourced but was not
measured in the window). It returns a self-contained runner script that computes
the audit WITHOUT an LLM coder call.

Motivation (E3 / M1, 2026-07-08): the missingness/measurement audit is a PURE
COUNT — per concept, how many stays have a measured value vs none, and the
percentages. Yet on real runs the LLM coder for this step reliably exhausted its
retry budget (~27.6 min, IDENTICAL across two runs) and failed with no code,
blocking the whole run on ``execution_complete``. A deterministic runner removes
both the flakiness and the dominant coder round-trip (the runner itself is
<1 % of wall-clock; the coder call is ~60 %).

The generated script:

* reads ``COHORT_PARQUET`` + ``STEP_OUT_DIR`` and the run's
  ``research_context.json``;
* discovers the concepts to audit case-neutrally — every base concept ``X`` that
  carries a paired ``X_measured`` indicator, plus any concepts named in
  ``user_preferences`` — excluding ids / demographics / the outcome;
* for each concept computes ``n_total``, ``measured_one_n`` (measured >= once),
  ``value_missing_n`` (never measured) and their percentages, and the
  structural-vs-measurement split (``value_present_but_measured_zero_n`` and a
  ``missingness_kind`` label) using the ``_measured`` flag as the authoritative
  measurement indicator, never silently imputing;
* writes ``missingness_measurement_audit.csv`` (the schema the deterministic
  missingness figure renderer consumes) + ``cohort_flow.csv`` and a
  ``step_summary.json`` with ``analysis_family='data_quality'`` and
  ``adjusted_effect=None`` (a descriptive audit, never an effect estimate).

It intentionally emits NO figure: the family figure renderer builds the
manuscript figure from ``missingness_measurement_audit.csv`` in the figure step.
"""

from __future__ import annotations

import textwrap

__all__ = ["missingness_measurement_audit_code"]


def missingness_measurement_audit_code() -> str:
    """Return a runner script that computes the per-concept missingness audit."""
    return textwrap.dedent(r"""
        import json
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
        n_total = int(len(df))

        # --- research context: optional explicit concept list ------------------
        req_concepts = []
        try:
            ctx = json.loads((run_dir / "research_context.json").read_text("utf-8"))
            prefs = ctx.get("user_preferences") or {}
            if isinstance(prefs, dict):
                for key in ("audit_concepts", "feature_concepts", "concepts", "features"):
                    vals = prefs.get(key)
                    if isinstance(vals, (list, tuple)):
                        req_concepts.extend(str(v).strip() for v in vals if str(v).strip())
        except Exception:
            pass

        cols = list(df.columns)
        low = {c.lower(): c for c in cols}

        # id / demographic / outcome columns are not measurement concepts.
        _NON_CONCEPT = {
            "stay_id", "hadm_id", "subject_id", "icustay_id", "patient_id", "id",
            "age", "sex", "gender", "adm", "admission_type", "ethnicity", "race",
            "death", "died", "mortality", "hospital_mortality", "hospital_expire_flag",
            "los_icu", "los_hosp", "icu_los", "hospital_los", "length_of_stay",
            "followup_time_hours", "event_observed",
        }
        _SUFFIX_SKIP = ("_measured", "_first_time", "_last_time", "_n", "_time")

        def _is_flag(colname):
            return colname.lower().endswith("_measured")

        # --- discover the concepts to audit (case-neutral) ---------------------
        # Primary source of truth: every base concept X that carries an
        # ``X_measured`` indicator. Add any explicitly requested concept present
        # in the cohort. Never audit ids / demographics / the outcome.
        concepts = []
        seen = set()

        def _add(base):
            b = str(base)
            if not b or b.lower() in _NON_CONCEPT or b in seen:
                return
            if b not in df.columns and (b + "_measured") not in df.columns:
                return
            seen.add(b)
            concepts.append(b)

        for c in cols:
            if _is_flag(c):
                _add(c[: -len("_measured")])
        for name in req_concepts:
            _add(low.get(name.lower(), name))

        # Fallback: if no _measured flags exist at all, audit every non-id,
        # non-aggregate value column so the step still produces a real audit.
        if not concepts:
            for c in cols:
                cl = c.lower()
                if cl in _NON_CONCEPT or any(cl.endswith(s) for s in _SUFFIX_SKIP):
                    continue
                _add(c)

        def _fail(reason):
            summary = {
                "step": current_step_id,
                "status": "blocked",
                "analysis_family": "data_quality",
                "blocking_reason": reason,
                "adjusted_effect": None,
                "primary_estimand": "Blocked: " + reason,
                "n_total": n_total,
                "outputs": [],
            }
            (out_dir / "step_summary.json").write_text(
                json.dumps(summary, indent=2, ensure_ascii=False)
            )
            print(json.dumps(summary))

        if n_total == 0:
            _fail("Analysis cohort is empty; nothing to audit.")
            raise SystemExit(0)
        if not concepts:
            _fail(
                "No auditable concept columns found (no '<concept>_measured' "
                "indicators and no non-id value columns)."
            )
            raise SystemExit(0)

        # --- per-concept measurement audit -------------------------------------
        rows = []
        for base in concepts:
            flag_col = base + "_measured"
            value_col = base if base in df.columns else None
            has_flag = flag_col in df.columns

            if has_flag:
                measured_flag = pd.to_numeric(df[flag_col], errors="coerce").fillna(0)
                measured_mask = measured_flag >= 1
            elif value_col is not None:
                # no explicit indicator -> a non-null value counts as measured.
                measured_mask = df[value_col].notna()
            else:
                continue

            measured_one_n = int(measured_mask.sum())
            value_missing_n = int(n_total - measured_one_n)

            # Structural no-source: the concept is not sourced for ANY stay in
            # this cohort/database (indicator all-zero, or value column entirely
            # absent/NaN). Distinct from measurement missingness (sourced, but
            # not measured for a given stay).
            if value_col is not None:
                value_present = df[value_col].notna()
            else:
                value_present = pd.Series(False, index=df.index)
            structural_no_source = bool(measured_one_n == 0) or (
                value_col is None and not has_flag
            )
            # Rows with a value present but the measurement indicator says zero
            # (a genuine present-but-unmeasured, e.g. a derived source flag).
            if has_flag:
                present_but_zero = int((value_present & ~measured_mask).sum())
            else:
                present_but_zero = 0

            kind = "structural_no_source" if structural_no_source else "measurement_missing"
            rows.append(
                {
                    "concept": base,
                    "variable": base,
                    "label": base.replace("_", " "),
                    "n_total": n_total,
                    "measured_one_n": measured_one_n,
                    "measured_one_pct": round(100.0 * measured_one_n / n_total, 6),
                    "value_missing_n": value_missing_n,
                    "value_missing_pct": round(100.0 * value_missing_n / n_total, 6),
                    # aliases so every downstream resolver (figure renderer /
                    # validator) finds a column it recognises.
                    "measured_n": measured_one_n,
                    "n_nonmissing": measured_one_n,
                    "missing_n": value_missing_n,
                    "missing_pct": round(100.0 * value_missing_n / n_total, 6),
                    "measured_pct": round(100.0 * measured_one_n / n_total, 6),
                    "value_present_but_measured_zero_n": present_but_zero,
                    "missingness_kind": kind,
                    "has_measured_indicator": bool(has_flag),
                }
            )

        audit = pd.DataFrame(rows)
        audit = audit.sort_values("value_missing_pct", ascending=False).reset_index(drop=True)
        audit.to_csv(out_dir / "missingness_measurement_audit.csv", index=False)

        pd.DataFrame(
            [
                {"stage": "universe_or_cohort", "n": n_total},
                {"stage": "concepts_audited", "n": int(len(audit))},
                {
                    "stage": "structural_no_source_concepts",
                    "n": int((audit["missingness_kind"] == "structural_no_source").sum()),
                },
            ]
        ).to_csv(out_dir / "cohort_flow.csv", index=False)

        worst = audit.head(5)[["concept", "value_missing_pct"]].to_dict("records")
        n_structural = int((audit["missingness_kind"] == "structural_no_source").sum())
        summary = {
            "step": current_step_id,
            "status": "ok",
            "analysis_family": "data_quality",
            "interpretation_class": "missingness_measurement_audit",
            "primary_estimand": (
                "Deterministic per-concept measurement-missingness audit "
                "(measured vs missing fraction; structural-no-source vs "
                "measurement-missing distinguished via the '_measured' indicator)."
            ),
            "adjusted_effect": None,
            "n_total": n_total,
            "n_concepts_audited": int(len(audit)),
            "n_structural_no_source": n_structural,
            "worst_measured_concepts": worst,
            "notes": [
                "Deterministic missingness audit (no LLM coder).",
                "measured_one_n uses the '<concept>_measured' indicator when present, "
                "else a non-null value counts as measured.",
                "Labs/vitals are NEVER imputed to 0; missing means unmeasured.",
                "structural_no_source = concept sourced for no stay in this cohort; "
                "measurement_missing = sourced but unmeasured for a given stay.",
            ],
            "output_files": {
                "missingness_measurement_audit": "missingness_measurement_audit.csv",
                "cohort_flow": "cohort_flow.csv",
            },
        }
        (out_dir / "step_summary.json").write_text(
            json.dumps(summary, indent=2, ensure_ascii=False, default=str)
        )
        print(
            json.dumps(
                {
                    "n_total": n_total,
                    "n_concepts_audited": int(len(audit)),
                    "n_structural_no_source": n_structural,
                }
            )
        )
        """).strip()
