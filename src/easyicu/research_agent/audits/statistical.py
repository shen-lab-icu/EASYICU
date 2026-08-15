"""StatisticalValidator and StatisticalGuard."""

from __future__ import annotations

import json
import math
import re
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Set

import pandas as pd

from ..contracts.ordered_stratified import ordered_stratified_numeric_findings
from ..schema import (
    AnalysisStep,
    ResearchContext,
    ValidationFinding,
)
from ..trajectory.contract import trajectory_phenotyping_artifact_findings

# ---------------------------------------------------------------------------
# CohortAuditor
# ---------------------------------------------------------------------------


# Patient-level identifier column names. Their presence means a cohort can
# be reasoned about at the patient level (within-patient non-independence,
# first-stay selection). Stay-level keys (stay_id, icustay_id) and

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

        primary_exposure = step_summary.get("primary_exposure")
        if isinstance(primary_exposure, Mapping):
            status = (
                str(
                    primary_exposure.get("reconciliation_status")
                    or primary_exposure.get("status")
                    or ""
                )
                .strip()
                .lower()
            )
            cohort_n = self._finite_nonnegative_count(step_summary.get("cohort_n"))
            missing_n = self._finite_nonnegative_count(
                primary_exposure.get("missing_n")
            )
            counts = primary_exposure.get("counts")
            usable_group_n = 0.0
            counted_total_n = 0.0
            if isinstance(counts, Mapping):
                for label, value in counts.items():
                    normalized_label = str(label).strip().lower()
                    numeric = self._finite_nonnegative_count(value)
                    if numeric is not None:
                        counted_total_n += numeric
                    if any(
                        token in normalized_label
                        for token in ("unavailable", "missing", "unknown")
                    ):
                        continue
                    if numeric is not None:
                        usable_group_n += numeric
            explicit_usable_counts = [
                self._finite_nonnegative_count(primary_exposure.get(key))
                for key in (
                    "available_n",
                    "nonmissing_n",
                    "observed_n",
                    "reconciled_n",
                    "usable_n",
                )
                if key in primary_exposure
            ]
            explicit_no_usable = bool(explicit_usable_counts) and all(
                value is not None and value <= 0 for value in explicit_usable_counts
            )
            all_missing_by_cohort = (
                cohort_n is not None
                and cohort_n > 0
                and missing_n is not None
                and missing_n >= cohort_n
                and usable_group_n <= 0
            )
            all_missing_by_counts = counted_total_n > 0 and usable_group_n <= 0
            if (
                status
                in {
                    "unavailable",
                    "failed",
                    "error",
                    "not_available",
                    "fail_closed",
                    "failed_closed",
                    "fail-closed",
                    "failed-closed",
                }
                or all_missing_by_cohort
                or all_missing_by_counts
                or explicit_no_usable
            ):
                findings.append(
                    ValidationFinding(
                        validator=self.name,
                        severity="error",
                        message=(
                            "The completed step declares a primary exposure but "
                            "no cohort row has a usable reconciled exposure value. "
                            "Repair the metadata/value binding or fail the step; "
                            "do not publish an all-unavailable primary-exposure result."
                        ),
                        detail={
                            "step_id": step.step_id,
                            "reconciliation_status": status or None,
                            "cohort_n": cohort_n,
                            "missing_n": missing_n,
                            "usable_group_n": usable_group_n,
                        },
                    )
                )

        # Controlled ordered-group summaries remain agent-authored, but every
        # denominator, interval, descriptive statistic, and trend test is
        # independently replayed from the locked cohort before publication.
        findings.extend(
            ordered_stratified_numeric_findings(
                cohort_path=cohort_path,
                step=step,
                out_dir=out_dir,
                step_summary=step_summary,
            )
        )
        findings.extend(
            trajectory_phenotyping_artifact_findings(
                context=context,
                cohort_path=cohort_path,
                step=step,
                out_dir=out_dir,
                step_summary=step_summary,
            )
        )

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
                            findings.append(
                                ValidationFinding(
                                    validator=self.name,
                                    severity="error",
                                    message=(
                                        f"Reported outcome rate {reported:.4f} disagrees with "
                                        f"cohort recompute {truth:.4f} (Δ={diff:.4f})."
                                    ),
                                    detail={"reported": reported, "truth": truth},
                                )
                            )
            except Exception as exc:
                findings.append(
                    ValidationFinding(
                        validator=self.name,
                        severity="warning",
                        message=f"Could not recompute outcome rate: {exc}",
                    )
                )

        # 2. Primary-association OR cross-check (T1.6).
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
                if (
                    reported is not None
                    and predictor
                    and "variable" in pa.columns
                    and "odds_ratio" in pa.columns
                ):
                    match = pa.loc[pa["variable"] == predictor, "odds_ratio"]
                    if not match.empty:
                        recomputed = float(match.iloc[0])
                        diff = abs(float(reported) - recomputed)
                        if diff > 1e-3:
                            findings.append(
                                ValidationFinding(
                                    validator=self.name,
                                    severity="error",
                                    message=(
                                        f"Reported primary OR {reported:.4f} disagrees "
                                        f"with recompute from {pa_csv.name} ({recomputed:.4f}, "
                                        f"Δ={diff:.4f})."
                                    ),
                                    detail={
                                        "reported": reported,
                                        "recomputed": recomputed,
                                        "predictor": predictor,
                                    },
                                )
                            )
            except Exception as exc:
                findings.append(
                    ValidationFinding(
                        validator=self.name,
                        severity="warning",
                        message=f"Could not parse primary_association.csv: {exc}",
                    )
                )

        # 3. Sanity: the script must have produced some artefact.
        if not any(out_dir.iterdir()):
            findings.append(
                ValidationFinding(
                    validator=self.name,
                    severity="error",
                    message=f"Step '{step.step_id}' produced no output artefacts.",
                )
            )

        # 4. Codex-grade train/test performance metrics (T1.8). Whenever a
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
                        findings.append(
                            ValidationFinding(
                                validator=self.name,
                                severity="error",
                                message=(
                                    f"Model '{model}' held-out AUC {auc:.3f} outside "
                                    "the plausible discriminative range [0.5, 1.0]."
                                ),
                                detail={"model": model, "auc": float(auc)},
                            )
                        )
                    if pd.notna(brier) and not (0.0 <= float(brier) <= 0.5):
                        findings.append(
                            ValidationFinding(
                                validator=self.name,
                                severity="warning",
                                message=(
                                    f"Model '{model}' Brier score {brier:.3f} outside "
                                    "the plausible range [0, 0.5]."
                                ),
                                detail={"model": model, "brier": float(brier)},
                            )
                        )
                    if pd.notna(cal_slope) and not (0.5 <= float(cal_slope) <= 2.0):
                        findings.append(
                            ValidationFinding(
                                validator=self.name,
                                severity="warning",
                                message=(
                                    f"Model '{model}' calibration slope {cal_slope:.3f} "
                                    "outside the well-calibrated range [0.5, 2.0]."
                                ),
                                detail={
                                    "model": model,
                                    "calibration_slope": float(cal_slope),
                                },
                            )
                        )
            except Exception as exc:
                findings.append(
                    ValidationFinding(
                        validator=self.name,
                        severity="warning",
                        message=f"Could not parse {perf_csv.name}: {exc}",
                    )
                )

        # 5. Degenerate-partition disclosure caution (clustering / trajectory).
        #    When a step emits a cluster-size distribution, surface an OBJECTIVE
        #    degeneracy fact — a single group, or one dominant cluster plus a
        #    near-empty (<1% of cohort) pocket — as an advisory caution. This is
        #    the agent-facing mirror of the post-hoc phenotype validity check:
        #    silhouette / ARI computed on such a partition are inflated by
        #    outlier isolation and must NOT be reported as evidence of robust
        #    subphenotypes without disclosing the size imbalance. It is a
        #    WARNING, never a block: a degenerate partition is still a legitimate
        #    (negative) finding to report honestly — the rule layer surfaces the
        #    fact and never imposes k, algorithm, scaling or outlier handling.
        deg = self._degenerate_partition(out_dir, step_summary)
        if deg is not None:
            findings.append(
                ValidationFinding(
                    validator=self.name,
                    severity="warning",
                    message=(
                        f"Degenerate cluster partition ({deg['reason']}). Silhouette "
                        "and resampling ARI on such a partition are inflated by "
                        "outlier isolation, not evidence of separated subphenotypes; "
                        "disclose the cluster sizes and do not present this as a "
                        "robust multi-subphenotype solution."
                    ),
                    detail=deg,
                )
            )

        return findings

    @staticmethod
    def _finite_nonnegative_count(value: Any) -> Optional[float]:
        try:
            numeric = float(value)
        except (TypeError, ValueError):
            return None
        if not math.isfinite(numeric) or numeric < 0:
            return None
        return numeric

    @staticmethod
    def _degenerate_partition(
        out_dir: Path, step_summary: Dict[str, Any]
    ) -> Optional[Dict[str, Any]]:
        """Return a degeneracy descriptor when a cluster-size distribution is
        objectively broken, else ``None``.

        Reuses the post-hoc scorecard threshold (single group, or smallest
        group < 1% of the cohort). Reads ``cluster_sizes.csv`` (columns include
        ``n`` and/or ``pct``) or a ``cluster_sizes`` mapping/list in the step
        summary. Stays silent (``None``) when the step produced no cluster-size
        evidence — absence is not degeneracy.
        """
        ns: List[float] = []
        # Prefer the emitted table; fall back to the step summary.
        try:
            csv_path = out_dir / "cluster_sizes.csv"
            if not csv_path.exists():
                for p in out_dir.glob("*cluster_sizes*.csv"):
                    csv_path = p
                    break
            if csv_path.exists():
                tbl = pd.read_csv(csv_path)
                if "n" in tbl.columns:
                    ns = [float(x) for x in tbl["n"].dropna().tolist()]
                elif "pct" in tbl.columns:
                    ns = [float(x) for x in tbl["pct"].dropna().tolist()]
        except Exception:
            ns = []
        if not ns:
            raw = (step_summary or {}).get("cluster_sizes")
            try:
                if isinstance(raw, dict):
                    ns = [float(v) for v in raw.values()]
                elif isinstance(raw, (list, tuple)):
                    ns = [float(v) for v in raw]
            except Exception:
                ns = []
        ns = [x for x in ns if x is not None and x >= 0]
        if not ns:
            return None
        k = len(ns)
        total = sum(ns)
        if total <= 0:
            return None
        min_frac = min(ns) / total
        if k < 2:
            return {
                "reason": f"single-group solution (k={k})",
                "n_clusters": k,
                "min_cluster_fraction": round(min_frac, 6),
            }
        if min_frac < 0.01:
            return {
                "reason": (
                    f"one dominant cluster plus a near-empty group "
                    f"({min_frac * 100:.2f}% of cohort) across k={k}"
                ),
                "n_clusters": k,
                "min_cluster_fraction": round(min_frac, 6),
            }
        return None
class StatisticalGuard:
    """Broader statistical QA checks beyond per-step numerical consistency."""

    name = "statistical_guard"

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
        family = (
            (context.user_preferences.inferred_analysis_family or "").lower()
            if context.user_preferences
            else ""
        )
        df = pd.read_parquet(cohort_path)

        if (
            family == "prediction_model"
            or (out_dir / "model_performance_train_test.csv").exists()
        ):
            summary_text = json.dumps(step_summary or {}, ensure_ascii=False).lower()

            def _summary_has_any(tokens: Sequence[str]) -> bool:
                return any(token in summary_text for token in tokens)

            perf_csv = out_dir / "model_performance_train_test.csv"
            perf_candidates = [perf_csv] if perf_csv.exists() else []
            if not perf_candidates:
                for candidate in out_dir.glob("*.csv"):
                    name = candidate.name.lower()
                    if any(
                        token in name
                        for token in ("performance", "prediction", "model")
                    ):
                        perf_candidates.append(candidate)
            has_performance_metric = _summary_has_any(
                (
                    "auroc",
                    "auc",
                    "brier",
                    "calibration",
                    "cv_auroc",
                    "held_out_auroc",
                    "cv_brier",
                )
            )
            perf_columns: Set[str] = set()
            for candidate in perf_candidates:
                try:
                    perf_columns.update(
                        str(c).lower() for c in pd.read_csv(candidate, nrows=5).columns
                    )
                except Exception:
                    continue
            has_performance_metric = has_performance_metric or any(
                token in perf_columns
                for token in (
                    "auroc",
                    "auc",
                    "brier",
                    "brier_score",
                    "cv_auroc_mean",
                    "held_out_auroc",
                )
            )
            if not has_performance_metric:
                findings.append(
                    ValidationFinding(
                        validator=self.name,
                        severity="warning",
                        message=(
                            "Prediction-model style analysis did not emit held-out performance artefacts. "
                            "Report train/test (or equivalent validation) performance before publication."
                        ),
                    )
                )
            else:
                has_calibration = (
                    "calibration_slope" in perf_columns
                    or "calibration_intercept" in perf_columns
                    or "brier" in summary_text
                    or "brier_score" in summary_text
                    or "calibration" in summary_text
                )
                if not has_calibration:
                    findings.append(
                        ValidationFinding(
                            validator=self.name,
                            severity="warning",
                            message="Prediction model performance is missing calibration_slope or Brier/calibration metadata.",
                        )
                    )
            has_split_metadata = any(
                k in (step_summary or {})
                for k in (
                    "n_train",
                    "n_test",
                    "split_strategy",
                    "cv_folds",
                    "validation_scheme",
                    "cross_validation",
                )
            ) or _summary_has_any(
                ("5-fold", "cross-validation", "cv_folds", "split_strategy")
            )
            if not has_split_metadata:
                findings.append(
                    ValidationFinding(
                        validator=self.name,
                        severity="warning",
                        message=(
                            "Prediction analysis did not document a train/test split or equivalent validation scheme. "
                            "Guard against leakage by recording split_strategy, n_train, and n_test."
                        ),
                    )
                )
            if context.target_outcome and context.target_outcome in df.columns:
                try:
                    events = int(
                        pd.to_numeric(df[context.target_outcome], errors="coerce")
                        .fillna(0)
                        .astype(int)
                        .sum()
                    )
                except Exception:
                    events = 0
                requested_covariates = len(
                    getattr(context.user_preferences, "covariates", []) or []
                )
                if requested_covariates > 0 and events < max(
                    10, 10 * requested_covariates
                ):
                    findings.append(
                        ValidationFinding(
                            validator=self.name,
                            severity="warning",
                            message=(
                                f"Only {events} events were available for {requested_covariates} requested adjustment covariates. "
                                "This may be an events-per-variable problem for a stable prediction model."
                            ),
                            detail={
                                "events": events,
                                "requested_covariates": requested_covariates,
                            },
                        )
                    )

        if family == "survival" or any(
            term in (step.intent or "").lower()
            for term in ("survival", "cox", "kaplan", "hazard")
        ):
            step_text = json.dumps(step_summary, ensure_ascii=False).lower()
            if "cox" in step_text or "cox" in (step.method or "").lower():
                documented = any(
                    token in step_text
                    for token in ("ph_assumption", "proportional hazards", "schoenfeld")
                )
                documented = documented or any(
                    "ph" in p.name.lower()
                    and p.suffix.lower() in {".csv", ".json", ".txt"}
                    for p in out_dir.iterdir()
                )
                if not documented:
                    findings.append(
                        ValidationFinding(
                            validator=self.name,
                            severity="warning",
                            message="Cox-style survival analysis did not document a proportional-hazards assumption check.",
                        )
                    )

        for csv_path in [p for p in out_dir.iterdir() if p.suffix.lower() == ".csv"]:
            try:
                tab = pd.read_csv(csv_path)
            except Exception:
                continue
            pval_cols = [
                c
                for c in tab.columns
                if c.lower() in {"p", "p_value", "pvalue", "pval"}
            ]
            adjust_cols = [
                c
                for c in tab.columns
                if c.lower() in {"q_value", "adjusted_p", "p_adj", "padj", "fdr"}
            ]
            family_col = next(
                (
                    column
                    for column in ("hypothesis_family_id", "family_id")
                    if column in tab.columns
                ),
                None,
            )
            # A coefficient dump is not itself a hypothesis family. Only a
            # typed, prespecified family can create an actionable multiplicity
            # warning; nuisance and sensitivity rows are never counted merely
            # because they carry p-values.
            if not pval_cols or family_col is None:
                continue
            scoped = tab.copy()
            if "term_role" in scoped.columns:
                roles = scoped["term_role"].map(
                    lambda value: re.sub(
                        r"[^a-z0-9]+", "_", str(value or "").lower()
                    ).strip("_")
                )
                scoped = scoped.loc[
                    ~roles.isin({"intercept", "adjustment", "availability", "nuisance"})
                ]
            if "analysis_role" in scoped.columns:
                analysis_roles = scoped["analysis_role"].map(
                    lambda value: re.sub(
                        r"[^a-z0-9]+", "_", str(value or "").lower()
                    ).strip("_")
                )
                scoped = scoped.loc[~analysis_roles.eq("sensitivity")]
            scoped = scoped.loc[scoped[family_col].notna()].copy()
            scoped[family_col] = scoped[family_col].astype(str).str.strip()
            scoped = scoped.loc[scoped[family_col].ne("")]
            warned = False
            for family_id, family_rows in scoped.groupby(family_col, sort=False):
                finite_p_value_count = sum(
                    int(
                        pd.to_numeric(family_rows[column], errors="coerce")
                        .dropna()
                        .between(0.0, 1.0)
                        .sum()
                    )
                    for column in pval_cols
                )
                finite_adjusted_count = sum(
                    int(
                        pd.to_numeric(family_rows[column], errors="coerce")
                        .dropna()
                        .between(0.0, 1.0)
                        .sum()
                    )
                    for column in adjust_cols
                )
                if (
                    finite_p_value_count <= 1
                    or finite_adjusted_count >= finite_p_value_count
                ):
                    continue
                findings.append(
                    ValidationFinding(
                        validator=self.name,
                        severity="warning",
                        message=(
                            f"Table '{csv_path.name}' contains multiple p-values without an adjusted-p / q-value column. "
                            "If this is a family of simultaneous tests, control multiplicity explicitly."
                        ),
                        detail={
                            "table": csv_path.name,
                            "hypothesis_family_id": str(family_id),
                            "p_value_columns": pval_cols,
                            "finite_p_value_count": finite_p_value_count,
                            "finite_adjusted_p_count": finite_adjusted_count,
                        },
                    )
                )
                warned = True
                break
            if warned:
                break

        return findings
