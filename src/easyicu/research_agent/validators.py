"""Deterministic verification gates between LLM output and downstream use.

The validators are the *trust* layer. Their job is to inspect agent
output (plans, scripts, run results) and decide whether it is safe to
let it influence the manuscript scaffold.

There are three of them:

1. :class:`CohortAuditor` — checks that the cohort frame matches its
   declared descriptor (size, id columns present, target outcome
   present, no surprise NaN-only columns).
2. :class:`ConceptUsageAuditor` — static analysis of the generated
   script: flags forbidden patterns like
   ``df['sofa'].mean()`` or  ``df['lact'].mean()`` (skewed lab without
   median fallback).
3. :class:`StatisticalValidator` — runs after the script and inspects
   the artefacts produced. Re-derives outcome incidence from the
   cohort, cross-checks the script's ``step_summary.json`` against
   the cohort, flags out-of-monotonic SOFA strata.

Each validator returns a list of :class:`ValidationFinding` objects
that the pipeline appends to the manifest. Severity ``error`` blocks
manuscript generation; ``warning`` is surfaced but does not block.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd

from .schema import (
    AggregationRule,
    AnalysisStep,
    ConceptDescriptor,
    EvidenceRecord,
    ResearchContext,
    ValidationFinding,
    VariableRole,
)


# ---------------------------------------------------------------------------
# CohortAuditor
# ---------------------------------------------------------------------------


class CohortAuditor:
    """Confirm the dataframe matches the descriptor it claims to."""

    name = "cohort_auditor"

    def audit(
        self,
        *,
        context: ResearchContext,
        cohort_path: Path,
    ) -> List[ValidationFinding]:
        findings: List[ValidationFinding] = []
        try:
            df = pd.read_parquet(cohort_path)
        except Exception as exc:
            return [ValidationFinding(
                validator=self.name, severity="error",
                message=f"Could not read cohort parquet: {exc}",
            )]

        # Row count
        if context.cohort.n_stays != int(len(df)):
            findings.append(ValidationFinding(
                validator=self.name, severity="error",
                message=(
                    f"Row count mismatch: descriptor says n_stays={context.cohort.n_stays:,} "
                    f"but cohort parquet has {len(df):,} rows."
                ),
            ))

        # Required id columns
        for col in context.cohort.id_columns:
            if col not in df.columns:
                findings.append(ValidationFinding(
                    validator=self.name, severity="error",
                    message=f"Declared id column '{col}' missing from cohort.",
                ))

        # Target outcome present and binary if labelled binary
        outcome = context.target_outcome
        if outcome:
            if outcome not in df.columns:
                findings.append(ValidationFinding(
                    validator=self.name, severity="error",
                    message=f"Target outcome '{outcome}' missing from cohort.",
                ))
            else:
                v = context.variable(outcome)
                if v and v.role == VariableRole.OUTCOME:
                    s = df[outcome].dropna()
                    if not s.empty and set(s.unique()) - {0, 1, True, False, 0.0, 1.0}:
                        findings.append(ValidationFinding(
                            validator=self.name, severity="warning",
                            message=(
                                f"Target outcome '{outcome}' has non-binary values "
                                f"({sorted(set(s.unique()))[:5]}…); confirm this is intended."
                            ),
                        ))

        # NaN-only columns
        for col in df.columns:
            if df[col].isna().all():
                findings.append(ValidationFinding(
                    validator=self.name, severity="warning",
                    message=f"Column '{col}' is entirely missing in the cohort.",
                ))

        # High-missing flag for any declared variable
        for v in context.variables:
            if v.missingness and v.missingness.fraction_missing > 0.5:
                findings.append(ValidationFinding(
                    validator=self.name, severity="warning",
                    message=(
                        f"Variable '{v.name}' has {v.missingness.fraction_missing:.0%} "
                        "missingness; downstream associations are at risk of selection bias."
                    ),
                    detail={"fraction_missing": v.missingness.fraction_missing},
                ))

        return findings


# ---------------------------------------------------------------------------
# ConceptUsageAuditor
# ---------------------------------------------------------------------------


_FORBIDDEN_AGG_PATTERNS_BY_KIND = {
    # role + agg method => human-readable message
    ("ordinal_score", "mean"): "Taking mean() of an ordinal SOFA component is a category error; aggregate by max within window.",
    ("ordinal_score", "std"):  "std() of an ordinal SOFA component is meaningless; report a level distribution instead.",
    ("composite_score", "mean"): "Total SOFA is a sum of 0–4 components; treat as ordinal/integer-count, not continuous (use max-within-window or report distribution).",
    ("composite_score", "std"):  "std() of a composite ordinal score is misleading; prefer median (IQR) or distribution table.",
    ("ordinal_score_gcs", "mean"): "GCS is ordinal; report worst (min) or representative (last/first), not mean.",
}


class ConceptUsageAuditor:
    """Static analysis of generated scripts for ICU rule violations."""

    name = "concept_usage_auditor"

    def audit(
        self,
        *,
        context: ResearchContext,
        script_text: str,
        step: Optional[AnalysisStep] = None,
    ) -> List[ValidationFinding]:
        findings: List[ValidationFinding] = []

        var_by_name = {v.name: v for v in context.variables}

        # Find calls of the form df['col'].FN() or df.col.FN() for FN in {mean, std}
        pat_bracket = re.compile(r"""\[(['"])(?P<col>[^'"]+)\1\]\s*\.\s*(?P<fn>mean|std)\s*\(""")
        pat_attr = re.compile(r"""\.(?P<col>[a-zA-Z_][a-zA-Z0-9_]*)\s*\.\s*(?P<fn>mean|std)\s*\(""")

        def _check(col: str, fn: str) -> None:
            v = var_by_name.get(col)
            if v is None:
                return
            role_key = v.role.value
            key = (role_key, fn)
            if key in _FORBIDDEN_AGG_PATTERNS_BY_KIND:
                findings.append(ValidationFinding(
                    validator=self.name, severity="error",
                    message=_FORBIDDEN_AGG_PATTERNS_BY_KIND[key],
                    detail={"column": col, "function": fn, "step_id": step.step_id if step else None},
                ))
                return
            # GCS special case
            if v.name.lower() == "gcs" and fn == "mean":
                findings.append(ValidationFinding(
                    validator=self.name, severity="error",
                    message=_FORBIDDEN_AGG_PATTERNS_BY_KIND[("ordinal_score_gcs", "mean")],
                    detail={"column": col, "function": fn},
                ))
                return
            # Skewed lab + mean (no median nearby) → warning
            if v.role == VariableRole.LAB and fn == "mean":
                # If 'median' appears anywhere in the script, downgrade to info.
                if "median" not in script_text:
                    findings.append(ValidationFinding(
                        validator=self.name, severity="warning",
                        message=(
                            f"Lab variable '{col}' summarised by mean() with no median() in "
                            "the same script. Right-skewed labs are conventionally reported "
                            "as median (IQR)."
                        ),
                        detail={"column": col, "function": fn},
                    ))

        for m in pat_bracket.finditer(script_text):
            _check(m.group("col"), m.group("fn"))
        for m in pat_attr.finditer(script_text):
            _check(m.group("col"), m.group("fn"))

        # Imputation-without-flag pattern
        if re.search(r"\.fillna\s*\(\s*0\s*\)", script_text):
            findings.append(ValidationFinding(
                validator=self.name, severity="warning",
                message=(
                    "Detected fillna(0) — silent imputation to zero is rarely correct for "
                    "ICU variables. Use a missing-indicator or document the imputation explicitly."
                ),
            ))

        return findings


# ---------------------------------------------------------------------------
# StatisticalValidator
# ---------------------------------------------------------------------------


class StatisticalValidator:
    """Cross-check the artefacts a script produced against the cohort."""

    name = "statistical_validator"

    def audit(
        self,
        *,
        context: ResearchContext,
        cohort_path: Path,
        step: AnalysisStep,
        out_dir: Path,
        step_summary: Dict[str, Any],
    ) -> List[ValidationFinding]:
        findings: List[ValidationFinding] = []
        outcome = context.target_outcome

        # 1. Recompute outcome incidence and compare with reported.
        if outcome:
            try:
                df = pd.read_parquet(cohort_path)
                if outcome in df.columns:
                    truth = float(df[outcome].dropna().astype(int).mean())
                    reported = step_summary.get("outcome_rate")
                    if reported is not None:
                        diff = abs(float(reported) - truth)
                        if diff > 1e-3:
                            findings.append(ValidationFinding(
                                validator=self.name, severity="error",
                                message=(
                                    f"Reported outcome rate {reported:.4f} disagrees with "
                                    f"cohort recompute {truth:.4f} (Δ={diff:.4f})."
                                ),
                                detail={"reported": reported, "truth": truth},
                            ))
            except Exception as exc:
                findings.append(ValidationFinding(
                    validator=self.name, severity="warning",
                    message=f"Could not recompute outcome rate: {exc}",
                ))

        # 2. SOFA-zero anomaly cross-check.
        sofa_csv = out_dir / "sofa_strata.csv"
        if sofa_csv.exists():
            try:
                strata = pd.read_csv(sofa_csv)
                if {"outcome_rate", "n"}.issubset(strata.columns):
                    # the score column is whichever isn't outcome_rate or n
                    score_cols = [c for c in strata.columns if c not in {"outcome_rate", "n"}]
                    if score_cols:
                        sc = score_cols[0]
                        try:
                            r0 = float(strata.loc[strata[sc] == 0, "outcome_rate"].iloc[0])
                            r1 = float(strata.loc[strata[sc] == 1, "outcome_rate"].iloc[0])
                            if r0 > r1:
                                findings.append(ValidationFinding(
                                    validator=self.name, severity="warning",
                                    message=(
                                        f"{sc}==0 outcome rate ({r0:.3f}) exceeds {sc}==1 "
                                        f"({r1:.3f}). This is non-monotonic and is a known "
                                        "signature of component-level missingness rather than "
                                        "absent organ dysfunction. Verify component "
                                        "availability before interpreting clinically."
                                    ),
                                    detail={"score": sc, "rate_at_zero": r0, "rate_at_one": r1},
                                ))
                        except (IndexError, KeyError):
                            pass
            except Exception as exc:
                findings.append(ValidationFinding(
                    validator=self.name, severity="warning",
                    message=f"Could not parse sofa_strata.csv: {exc}",
                ))

        # 3. Primary-association OR cross-check (T1.6).
        #    The mock pipeline writes ``primary_association.csv`` with one
        #    row per coefficient (variable, coef, odds_ratio, ...). The
        #    step_summary records ``primary_or`` for the predictor. Re-
        #    derive the OR from the table and flag if the two disagree by
        #    more than 1e-3 — that mirrors the outcome-rate check above.
        pa_csv = out_dir / "primary_association.csv"
        if pa_csv.exists():
            try:
                pa = pd.read_csv(pa_csv)
                reported = step_summary.get("primary_or")
                predictor = step_summary.get("predictor")
                if reported is not None and predictor and "variable" in pa.columns and "odds_ratio" in pa.columns:
                    match = pa.loc[pa["variable"] == predictor, "odds_ratio"]
                    if not match.empty:
                        recomputed = float(match.iloc[0])
                        diff = abs(float(reported) - recomputed)
                        if diff > 1e-3:
                            findings.append(ValidationFinding(
                                validator=self.name, severity="error",
                                message=(
                                    f"Reported primary OR {reported:.4f} disagrees "
                                    f"with recompute from {pa_csv.name} ({recomputed:.4f}, "
                                    f"Δ={diff:.4f})."
                                ),
                                detail={"reported": reported, "recomputed": recomputed,
                                        "predictor": predictor},
                            ))
            except Exception as exc:
                findings.append(ValidationFinding(
                    validator=self.name, severity="warning",
                    message=f"Could not parse primary_association.csv: {exc}",
                ))

        # 4. Sanity: the script must have produced some artefact.
        if not any(out_dir.iterdir()):
            findings.append(ValidationFinding(
                validator=self.name, severity="error",
                message=f"Step '{step.step_id}' produced no output artefacts.",
            ))

        # 5. Codex-grade train/test performance metrics (T1.8). Whenever a
        #    step writes ``model_performance_train_test.csv``, re-validate
        #    AUC ∈ [0.5, 1.0], Brier ∈ [0, 0.5] and calibration slope
        #    ∈ [0.5, 2.0]. Out-of-range values are *errors* — they
        #    indicate that the held-out test produced sign-flipped or
        #    grossly mis-calibrated predictions.
        perf_csv = out_dir / "model_performance_train_test.csv"
        if perf_csv.exists():
            try:
                perf = pd.read_csv(perf_csv)
                for _, row in perf.iterrows():
                    model = str(row.get("model", "?"))
                    auc = row.get("auc", float("nan"))
                    brier = row.get("brier", float("nan"))
                    cal_slope = row.get("calibration_slope", float("nan"))
                    if pd.notna(auc) and not (0.5 <= float(auc) <= 1.0):
                        findings.append(ValidationFinding(
                            validator=self.name, severity="error",
                            message=(
                                f"Model '{model}' held-out AUC {auc:.3f} outside "
                                "the plausible discriminative range [0.5, 1.0]."
                            ),
                            detail={"model": model, "auc": float(auc)},
                        ))
                    if pd.notna(brier) and not (0.0 <= float(brier) <= 0.5):
                        findings.append(ValidationFinding(
                            validator=self.name, severity="warning",
                            message=(
                                f"Model '{model}' Brier score {brier:.3f} outside "
                                "the plausible range [0, 0.5]."
                            ),
                            detail={"model": model, "brier": float(brier)},
                        ))
                    if pd.notna(cal_slope) and not (0.5 <= float(cal_slope) <= 2.0):
                        findings.append(ValidationFinding(
                            validator=self.name, severity="warning",
                            message=(
                                f"Model '{model}' calibration slope {cal_slope:.3f} "
                                "outside the well-calibrated range [0.5, 2.0]."
                            ),
                            detail={"model": model, "calibration_slope": float(cal_slope)},
                        ))
            except Exception as exc:
                findings.append(ValidationFinding(
                    validator=self.name, severity="warning",
                    message=f"Could not parse {perf_csv.name}: {exc}",
                ))

        # 6. T1.8 score-0 elevated-mortality data-quality signal. When
        #    a step's summary reports a sofa2_zero_rate that exceeds the
        #    overall mortality_rate, surface this as a *data_quality_signal*
        #    finding rather than an error — codex's central scientific
        #    claim was exactly this kind of finding.
        try:
            zero_rate = step_summary.get("sofa2_zero_rate")
            overall = step_summary.get("mortality_rate")
            if zero_rate is not None and overall is not None:
                z, o = float(zero_rate), float(overall)
                if z == z and o == o and z > o + 1e-6:
                    findings.append(ValidationFinding(
                        validator=self.name, severity="warning",
                        message=(
                            "Data-quality signal: SOFA-2 score==0 stratum "
                            f"mortality {z:.3f} exceeds overall mortality "
                            f"{o:.3f}. Consistent with a component-availability "
                            "artefact in upstream concept construction."
                        ),
                        detail={
                            "sofa2_zero_rate": z, "overall_rate": o,
                            "category": "data_quality_signal",
                        },
                    ))
        except (TypeError, ValueError):
            pass

        return findings


__all__ = [
    "CohortAuditor",
    "ConceptUsageAuditor",
    "StatisticalValidator",
]
