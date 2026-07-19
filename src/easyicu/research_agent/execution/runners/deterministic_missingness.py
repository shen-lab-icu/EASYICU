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
  ``missingness_kind`` label). ``_measured`` is the authoritative availability
  signal except for a narrowly detected binary event-status encoding where the
  complete 0/1 value, positive flag, and event-count signal agree exactly;
* writes ``missingness_measurement_audit.csv`` (the schema the deterministic
  missingness figure renderer consumes), ``analytic_denominators.csv`` from the
  current step's declared inputs, ``cohort_flow.csv``, and a
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
        requested_inputs = []
        requested_outputs = []
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

        # The plan owns the complete-case contract.  Reading the current
        # step's declared inputs lets the deterministic audit emit the requested
        # analytic denominator instead of returning only a compact concept
        # count and falsely satisfying a richer step.
        try:
            plan = json.loads((run_dir / "analysis_plan.json").read_text("utf-8"))
            for planned_step in plan.get("steps") or []:
                if str(planned_step.get("step_id") or "") == current_step_id:
                    requested_inputs = [
                        str(value).strip()
                        for value in (planned_step.get("inputs") or [])
                        if str(value).strip()
                    ]
                    requested_outputs = [
                        str(value).strip()
                        for value in (planned_step.get("expected_outputs") or [])
                        if str(value).strip()
                    ]
                    break
        except Exception:
            pass

        cols = list(df.columns)
        low = {c.lower(): c for c in cols}

        # IDs are never audit variables. Demographics and outcomes are excluded
        # only from broad discovery; if the current step explicitly declares
        # them, their direct value availability belongs in that step's audit.
        _IDENTIFIER_COLUMNS = {
            "stay_id", "hadm_id", "subject_id", "icustay_id", "patient_id", "id",
        }
        _NON_CONCEPT = {
            *_IDENTIFIER_COLUMNS,
            "age", "sex", "gender", "adm", "admission_type", "ethnicity", "race",
            "death", "died", "mortality", "hospital_mortality", "hospital_expire_flag",
            "los_icu", "los_hosp", "icu_los", "hospital_los", "length_of_stay",
            "followup_time_hours", "event_observed",
        }
        _SUFFIX_SKIP = ("_measured", "_first_time", "_last_time", "_n", "_time")

        def _is_flag(colname):
            return colname.lower().endswith("_measured")

        def _representative_value_column(base):
            '''Resolve one value aggregate paired with ``<base>_measured``.

            Wide ICU exports commonly pair ``crea_measured`` with
            ``crea_first`` or ``aki_stage_measured`` with ``aki_stage_max``;
            requiring an exact bare ``base`` silently loses the value/flag
            discordance audit.  The closed aggregate suffix list is structural
            and case-neutral.
            '''
            suffixes = ("", "_first", "_max", "_last", "_mean", "_min")
            base_lower = base.lower()
            for requested in requested_inputs:
                resolved = (
                    requested
                    if requested in df.columns
                    else low.get(requested.lower())
                )
                if resolved is None:
                    continue
                if resolved.lower() in {
                    base_lower + suffix for suffix in suffixes
                }:
                    return resolved

            candidates = [
                base,
                base + "_first",
                base + "_max",
                base + "_last",
                base + "_mean",
                base + "_min",
            ]
            for candidate in candidates:
                if candidate in df.columns:
                    return candidate
                matched = low.get(candidate.lower())
                if matched is not None:
                    return matched
            return None

        # --- discover the concepts to audit (case-neutral) ---------------------
        # Primary source of truth: every base concept X that carries an
        # ``X_measured`` indicator. Add any explicitly requested concept present
        # in the cohort. Never audit ids / demographics / the outcome.
        concepts = []
        seen = set()

        def _add(base, *, declared=False):
            b = str(base)
            if (
                not b
                or b.lower() in _IDENTIFIER_COLUMNS
                or (not declared and b.lower() in _NON_CONCEPT)
                or b in seen
            ):
                return
            if (
                b not in df.columns
                and (b + "_measured") not in df.columns
                and _representative_value_column(b) is None
            ):
                return
            seen.add(b)
            concepts.append(b)

        _FAMILY_SUFFIXES = (
            "_measured", "_first_time", "_last_time", "_first", "_last",
            "_max", "_mean", "_min", "_n",
        )

        def _family_base(column):
            text = str(column)
            lowered = text.lower()
            for suffix in _FAMILY_SUFFIXES:
                if lowered.endswith(suffix) and len(text) > len(suffix):
                    return text[: -len(suffix)]
            return text

        if requested_inputs:
            # The current plan is the audit scope. Collapse aggregate-family
            # members (X_max/X_measured/...) to one concept row, but retain
            # explicitly declared direct variables such as age or outcome.
            for requested in requested_inputs:
                resolved = (
                    requested
                    if requested in df.columns
                    else low.get(requested.lower(), requested)
                )
                _add(_family_base(resolved), declared=True)
        else:
            # Backward-compatible discovery for legacy plans with no input
            # contract: audit every paired measurement concept.
            for c in cols:
                if _is_flag(c):
                    _add(c[: -len("_measured")])
            for name in req_concepts:
                _add(low.get(name.lower(), name), declared=True)

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
            value_col = _representative_value_column(base)
            has_flag = flag_col in df.columns

            if has_flag:
                raw_measured_flag = pd.to_numeric(df[flag_col], errors="coerce")
                measured_flag = raw_measured_flag.fillna(0)
                measured_mask = measured_flag >= 1
            elif value_col is not None:
                # no explicit indicator -> a non-null value counts as measured.
                measured_mask = df[value_col].notna()
            else:
                continue

            # Structural no-source: the concept is not sourced for ANY stay in
            # this cohort/database (indicator all-zero, or value column entirely
            # absent/NaN). Distinct from measurement missingness (sourced, but
            # not measured for a given stay).
            if value_col is not None:
                value_present = df[value_col].notna()
            else:
                value_present = pd.Series(False, index=df.index)
            value_present_n = int(value_present.sum())

            # Some wide exports use ``X_measured`` for an event-presence flag,
            # not measurement availability: a fully observed binary X is 0/1
            # and the paired flag is exactly 1 where X == 1. Treating negative
            # states as unmeasured would turn a complete binary status into
            # near-total missingness (for example, a rare procedure). Keep this
            # inference narrow and case-neutral: both binary states must occur,
            # every value and flag must be present, and an independently emitted
            # ``X_n > 0`` event-count signal must exactly agree with the value and
            # flag. Partial continuous labs/vitals and an unknown bad binary flag
            # cannot enter this branch.
            indicator_semantics = "measurement_availability"
            raw_indicator_one_n = int(measured_mask.sum())
            count_candidate = base + "_n"
            count_col = (
                count_candidate
                if count_candidate in df.columns
                else low.get(count_candidate.lower())
            )
            if (
                has_flag
                and value_col is not None
                and count_col is not None
                and value_present_n == n_total
            ):
                numeric_value = pd.to_numeric(df[value_col], errors="coerce")
                event_count = pd.to_numeric(df[count_col], errors="coerce")
                value_levels = set(numeric_value.dropna().unique().tolist())
                flag_levels = set(measured_flag.dropna().unique().tolist())
                is_complete_binary_status = (
                    numeric_value.notna().all()
                    and raw_measured_flag.notna().all()
                    and event_count.notna().all()
                    and bool(event_count.ge(0).all())
                    and value_levels == {0, 1}
                    and flag_levels.issubset({0, 1})
                    and bool((measured_mask == numeric_value.eq(1)).all())
                    and bool((measured_mask == event_count.gt(0)).all())
                )
                if is_complete_binary_status:
                    indicator_semantics = "binary_event_presence"
                    measured_mask = value_present

            measured_one_n = int(measured_mask.sum())
            value_missing_n = int(n_total - measured_one_n)
            structural_no_source = bool(
                measured_one_n == 0 and value_present_n == 0
            )
            # Rows with a value present but the measurement indicator says zero
            # (a genuine present-but-unmeasured, e.g. a derived source flag).
            if has_flag:
                present_but_zero = int((value_present & ~measured_mask).sum())
                measured_but_missing = int((measured_mask & ~value_present).sum())
            else:
                present_but_zero = 0
                measured_but_missing = 0

            if structural_no_source:
                kind = "structural_no_source"
            elif indicator_semantics == "binary_event_presence":
                kind = "binary_event_status_complete"
            elif present_but_zero or measured_but_missing:
                kind = "measurement_flag_conflict"
            else:
                kind = "measurement_missing"
            rows.append(
                {
                    "concept": base,
                    "variable": base,
                    "label": base.replace("_", " "),
                    "value_column": value_col or "",
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
                    "measured_but_value_missing_n": measured_but_missing,
                    "raw_value_missing_n": int(n_total - value_present_n),
                    "raw_indicator_one_n": raw_indicator_one_n,
                    "indicator_semantics": indicator_semantics,
                    "event_count_column": (
                        count_col
                        if indicator_semantics == "binary_event_presence"
                        else ""
                    ),
                    "missingness_kind": kind,
                    "has_measured_indicator": bool(has_flag),
                }
            )

        audit = pd.DataFrame(rows)
        audit = audit.sort_values("value_missing_pct", ascending=False).reset_index(drop=True)
        audit.to_csv(out_dir / "missingness_measurement_audit.csv", index=False)

        # --- declared analytic denominators ----------------------------------
        resolved_inputs = []
        missing_declared_inputs = []
        for requested in requested_inputs:
            resolved = requested if requested in df.columns else low.get(requested.lower())
            if resolved is None:
                # A bare concept name (e.g. ``crea`` present only as
                # ``crea_first``/``crea_measured``) is audited above via
                # ``_representative_value_column``; resolve the analytic
                # denominator the SAME way so a legitimately-audited concept is
                # not spuriously flagged as a missing declared input (which would
                # block an otherwise-complete missingness audit).
                resolved = _representative_value_column(
                    _family_base(low.get(requested.lower(), requested))
                )
            if resolved is None:
                missing_declared_inputs.append(requested)
                continue
            if resolved not in resolved_inputs:
                resolved_inputs.append(resolved)

        denominator_rows = []
        for column in resolved_inputs:
            observed_n = int(df[column].notna().sum())
            denominator_rows.append(
                {
                    "analysis_set": "observed:" + column,
                    "required_variables": column,
                    "n_total": n_total,
                    "n_complete": observed_n,
                    "n_excluded_missing": int(n_total - observed_n),
                    "complete_pct": round(100.0 * observed_n / n_total, 6),
                }
            )
        expects_analytic_denominator = any(
            "analytic_denominator" in value.lower()
            for value in requested_outputs
        )
        denominator_error = None
        if missing_declared_inputs:
            denominator_error = (
                "Declared analytic inputs are absent from the cohort: "
                + ", ".join(missing_declared_inputs)
            )
        elif expects_analytic_denominator and not requested_inputs:
            denominator_error = (
                "The analytic-denominator contract declares no input variables."
            )

        if resolved_inputs and denominator_error is None:
            complete_mask = df[resolved_inputs].notna().all(axis=1)
            complete_n = int(complete_mask.sum())
        else:
            complete_n = None
        denominator_rows.insert(
            0,
            {
                "analysis_set": "all_requested_inputs",
                "required_variables": "|".join(requested_inputs),
                "n_total": n_total,
                "n_complete": complete_n,
                "n_excluded_missing": (
                    int(n_total - complete_n) if complete_n is not None else None
                ),
                "complete_pct": (
                    round(100.0 * complete_n / n_total, 6)
                    if complete_n is not None
                    else None
                ),
            },
        )
        pd.DataFrame(denominator_rows).to_csv(
            out_dir / "analytic_denominators.csv", index=False
        )

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
        n_binary_event_status = int(
            (audit["indicator_semantics"] == "binary_event_presence").sum()
        )
        summary = {
            "step": current_step_id,
            "status": "blocked" if denominator_error else "ok",
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
            "n_binary_event_status": n_binary_event_status,
            "all_requested_inputs_complete_n": complete_n,
            "requested_input_count": len(requested_inputs),
            "resolved_input_count": len(resolved_inputs),
            "missing_declared_inputs": missing_declared_inputs,
            "blocking_reason": denominator_error,
            "worst_measured_concepts": worst,
            "notes": [
                "Deterministic missingness audit (no LLM coder).",
                "measured_one_n uses the '<concept>_measured' availability indicator "
                "when present, except for an exact complete-binary event-status "
                "encoding; otherwise a non-null value counts as measured.",
                "Labs/vitals are NEVER imputed to 0; missing means unmeasured.",
                "structural_no_source = concept sourced for no stay in this cohort; "
                "measurement_missing = sourced but unmeasured for a given stay.",
            ],
            "output_files": {
                "missingness_measurement_audit": "missingness_measurement_audit.csv",
                "analytic_denominators": "analytic_denominators.csv",
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
