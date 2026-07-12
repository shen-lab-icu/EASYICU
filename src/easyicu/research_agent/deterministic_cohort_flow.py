"""Deterministic primary-cohort attrition runner.

The pipeline materialises the locked analysis cohort before executing planned
steps.  A cohort-flow step that reads only ``COHORT_PARQUET`` therefore sees the
*already filtered* cohort and can incorrectly relabel it as the study universe.
This runner replays the locked CTAS predicates against ``run_dir/cohort.parquet``
and verifies that the final denominator matches ``cohort_analysis.parquet``.

It emits both the canonical renderer inputs (``cohort_flow.csv`` and
``attrition.csv``) and planner-facing aliases (``cohort_attrition.csv`` and
``cohort_denominators.csv``).  No case-specific variable, threshold, or score is
hard-coded: every criterion comes from ``cohort_locked.json``.
"""

from __future__ import annotations

import textwrap

__all__ = ["primary_cohort_flow_code"]


def primary_cohort_flow_code() -> str:
    """Return a self-contained script that replays locked cohort predicates."""

    return textwrap.dedent(r"""
        import json
        import math
        import os
        from pathlib import Path

        import pandas as pd

        from easyicu.research_agent.cohort_schema import (
            CohortDefinition,
            build_cohort,
            register_cohort_concept_ids,
        )

        out_dir = Path(os.environ["STEP_OUT_DIR"])
        out_dir.mkdir(parents=True, exist_ok=True)
        run_dir = out_dir.parents[2]
        current_step_id = out_dir.parent.name
        universe_path = run_dir / "cohort.parquet"
        analysis_path = run_dir / "cohort_analysis.parquet"
        lock_path = run_dir / "cohort_locked.json"

        def _write_blocked(reason):
            summary = {
                "step": current_step_id,
                "status": "blocked",
                "analysis_family": "cohort",
                "blocking_reason": reason,
                "adjusted_effect": None,
                "outputs": [],
            }
            (out_dir / "step_summary.json").write_text(
                json.dumps(summary, indent=2, ensure_ascii=False),
                encoding="utf-8",
            )
            print(json.dumps(summary))

        if not universe_path.exists():
            _write_blocked("Raw study-universe parquet is missing.")
            raise SystemExit(0)
        if not lock_path.exists():
            _write_blocked("Locked cohort definition is missing.")
            raise SystemExit(0)

        try:
            universe = pd.read_parquet(universe_path).reset_index(drop=True)
            lock_payload = json.loads(lock_path.read_text(encoding="utf-8"))
            register_cohort_concept_ids(universe.columns)
            definition = CohortDefinition.from_dict(lock_payload.get("cohort") or {})
        except Exception as exc:
            _write_blocked(
                "Could not load the universe and locked CTAS definition: "
                f"{type(exc).__name__}: {exc}"
            )
            raise SystemExit(0)

        n_universe = int(len(universe))
        if n_universe == 0:
            _write_blocked("Raw study universe is empty.")
            raise SystemExit(0)

        def _criterion_text(pred):
            value = pred.value
            if pred.op in ("missing", "not_missing"):
                rhs = ""
            elif isinstance(value, (list, tuple)):
                rhs = " " + ", ".join(str(item) for item in value)
            else:
                rhs = " " + str(value)
            return f"{pred.concept_id} [{pred.aggregation}] {pred.op}{rhs}".strip()

        def _criterion_id(kind, order, pred):
            concept = str(pred.concept_id).strip().lower().replace(" ", "_")
            return f"{kind}_{order:02d}_{concept}"

        current = universe.copy()
        stages = [
            {
                "stage": "universe",
                "n": n_universe,
                "percent_of_universe": 100.0,
                "n_removed_from_prior_stage": 0,
                "criterion": "All supplied study-universe records",
                "criterion_id": "universe",
                "kind": "universe",
                "n_at_start": n_universe,
            }
        ]
        criteria_through = []

        ordered_predicates = [
            ("include", pred) for pred in definition.inclusion
        ] + [("exclude", pred) for pred in definition.exclusion]
        for order, (kind, pred) in enumerate(ordered_predicates, start=1):
            n_before = int(len(current))
            one = CohortDefinition(
                name=f"criterion_{order}",
                inclusion=(pred,) if kind == "include" else (),
                exclusion=(pred,) if kind == "exclude" else (),
            )
            current = build_cohort(one, current).reset_index(drop=True)
            n_after = int(len(current))
            removed = int(n_before - n_after)
            text = _criterion_text(pred)
            criterion_id = _criterion_id(kind, order, pred)
            criteria_through.append(text)
            stages.append(
                {
                    "stage": criterion_id,
                    "n": n_after,
                    "percent_of_universe": round(100.0 * n_after / n_universe, 6),
                    "n_removed_from_prior_stage": removed,
                    "criterion": text,
                    "criterion_id": criterion_id,
                    "kind": kind,
                    "n_at_start": n_before,
                }
            )

        n_final = int(len(current))
        expected_analysis_n = None
        if analysis_path.exists():
            try:
                expected_analysis_n = int(len(pd.read_parquet(analysis_path)))
            except Exception:
                expected_analysis_n = None

        provenance_path = run_dir / "cohort_analysis_provenance.json"
        provenance_analysis_n = None
        if provenance_path.exists():
            try:
                provenance = json.loads(provenance_path.read_text(encoding="utf-8"))
                if provenance.get("n_analysis_cohort") is not None:
                    provenance_analysis_n = int(provenance["n_analysis_cohort"])
            except Exception:
                provenance_analysis_n = None

        stay_id_col = "stay_id" if "stay_id" in universe.columns else None
        missing_stay_id_n = (
            int(universe[stay_id_col].isna().sum()) if stay_id_col else None
        )
        duplicate_stay_id_rows = (
            int(universe[stay_id_col].duplicated(keep=False).sum())
            if stay_id_col
            else None
        )

        blocking_reasons = []
        if expected_analysis_n is None:
            blocking_reasons.append("Materialised analysis cohort is unavailable.")
        elif expected_analysis_n != n_final:
            blocking_reasons.append(
                "Locked-predicate replay does not match cohort_analysis.parquet "
                f"({n_final} != {expected_analysis_n})."
            )
        if provenance_analysis_n is not None and provenance_analysis_n != n_final:
            blocking_reasons.append(
                "Locked-predicate replay does not match recorded provenance "
                f"({n_final} != {provenance_analysis_n})."
            )
        if stay_id_col is None:
            blocking_reasons.append("stay_id is absent; one-row-per-stay cannot be verified.")
        elif missing_stay_id_n or duplicate_stay_id_rows:
            blocking_reasons.append(
                "stay_id integrity failed "
                f"(missing={missing_stay_id_n}, duplicate_rows={duplicate_stay_id_rows})."
            )

        flow = pd.DataFrame(stages)[
            [
                "stage",
                "n",
                "percent_of_universe",
                "n_removed_from_prior_stage",
                "criterion",
            ]
        ]
        flow.to_csv(out_dir / "cohort_flow.csv", index=False)

        attrition_rows = [
            {
                "attrition_category": "universe",
                "n": n_universe,
                "percent_of_universe": 100.0,
                "status": "denominator",
                "reason": "All supplied study-universe records",
                "partition_role": "denominator_only",
            }
        ]
        for row in stages[1:]:
            attrition_rows.append(
                {
                    "attrition_category": row["criterion_id"],
                    "n": int(row["n_removed_from_prior_stage"]),
                    "percent_of_universe": round(
                        100.0 * row["n_removed_from_prior_stage"] / n_universe, 6
                    ),
                    "status": "excluded",
                    "reason": row["criterion"],
                    "partition_role": "partition_category",
                }
            )
        attrition_rows.append(
            {
                "attrition_category": "primary_analysis_cohort",
                "n": n_final,
                "percent_of_universe": round(100.0 * n_final / n_universe, 6),
                "status": "retained",
                "reason": "All locked CTAS inclusion/exclusion predicates",
                "partition_role": "partition_category",
            }
        )
        pd.DataFrame(attrition_rows).to_csv(out_dir / "attrition.csv", index=False)

        detailed_attrition = []
        for order, row in enumerate(stages):
            n_start = int(row["n_at_start"])
            n_remaining = int(row["n"])
            removed = int(row["n_removed_from_prior_stage"])
            detailed_attrition.append(
                {
                    "criterion_order": order,
                    "criterion_id": row["criterion_id"],
                    "criterion_label": row["criterion"],
                    "population_before": stages[order - 1]["stage"] if order else "none",
                    "n_at_start_rows": n_start,
                    "n_remaining_rows": n_remaining,
                    "n_excluded_rows": removed,
                    "excluded_fraction_of_start": (
                        removed / n_start if n_start else 0.0
                    ),
                    "excluded_percentage_of_start": (
                        100.0 * removed / n_start if n_start else 0.0
                    ),
                    "criterion_definition": row["criterion"],
                    "status": "denominator_recorded" if order == 0 else "applied",
                }
            )
        pd.DataFrame(detailed_attrition).to_csv(
            out_dir / "cohort_attrition.csv", index=False
        )

        denominator_rows = []
        criteria_so_far = []
        for order, row in enumerate(stages):
            if order:
                criteria_so_far.append(str(row["criterion"]))
            denominator_rows.append(
                {
                    "denominator_id": row["criterion_id"],
                    "denominator_label": str(row["stage"]).replace("_", " ").title(),
                    "criteria_through": " | ".join(criteria_so_far) or "none",
                    "n_rows": int(row["n"]),
                    "n_unique_stay_ids": (
                        int(current[stay_id_col].nunique(dropna=True))
                        if order == len(stages) - 1 and stay_id_col
                        else None
                    ),
                    "missing_stay_id_n": missing_stay_id_n,
                    "duplicate_stay_id_rows": duplicate_stay_id_rows,
                    "fraction_of_universe_rows": float(row["n"]) / n_universe,
                    "percentage_of_universe_rows": float(row["percent_of_universe"]),
                    "one_row_per_stay_id": not bool(
                        (missing_stay_id_n or 0) or (duplicate_stay_id_rows or 0)
                    ),
                    "is_primary_cohort": order == len(stages) - 1,
                    "status": "eligible_denominator",
                    "notes": "Deterministic replay of the locked CTAS cohort definition.",
                }
            )
        pd.DataFrame(denominator_rows).to_csv(
            out_dir / "cohort_denominators.csv", index=False
        )

        summary = {
            "step": current_step_id,
            "status": "blocked" if blocking_reasons else "ok",
            "analysis_family": "cohort",
            "interpretation_class": "locked_primary_cohort_flow",
            "primary_estimand": (
                f"Locked cohort attrition from {n_universe:,} universe records "
                f"to {n_final:,} analysis records."
            ),
            "adjusted_effect": None,
            "n_universe": n_universe,
            "n_analysis": n_final,
            "expected_analysis_n": expected_analysis_n,
            "provenance_analysis_n": provenance_analysis_n,
            "missing_stay_id_n": missing_stay_id_n,
            "duplicate_stay_id_rows": duplicate_stay_id_rows,
            "blocking_reason": " ".join(blocking_reasons) or None,
            "output_files": {
                "cohort_flow": "cohort_flow.csv",
                "attrition": "attrition.csv",
                "cohort_attrition": "cohort_attrition.csv",
                "cohort_denominators": "cohort_denominators.csv",
            },
        }
        (out_dir / "step_summary.json").write_text(
            json.dumps(summary, indent=2, ensure_ascii=False, default=str),
            encoding="utf-8",
        )
        print(json.dumps(summary))
        """).strip()
