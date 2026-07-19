"""Methodological-rigor audits: does the analysis METHOD match the study design?

The figure layer makes a survival question *look* like a survival study (a
Kaplan-Meier curve). This module checks the harder question underneath: did the
analysis actually *use* a time-to-event estimator, or did it answer a
time-to-event question with a static odds ratio? A figure renderer will happily
draw a "hazard ratio" forest from whatever table exists; only a method audit
catches a survival question answered with logistic regression.

This is the layer that separates a research agent from a template dispatcher:
it encodes the clinical-epidemiology rigor the canonical benchmark items test
for (immortal-time bias, complete-case bias, discrimination-without-calibration,
causal contrasts without covariate balance, clusters without stability). It is
deterministic and case-neutral -- the checks are keyed on the study-design
family, not on any benchmark variable.

The audit runs in two stages so the decision logic is trivially testable:

* :func:`extract_method_signals` reads the run's :class:`EvidenceStore` into a
  flat :class:`MethodSignals` bag (booleans + a missing fraction);
* :func:`audit_method_appropriateness` turns that bag + the locked family into
  :class:`ValidationFinding` objects with no I/O.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Sequence

from .evidence import EvidenceStore
from .schema import EvidenceRecord, ResearchContext, ValidationFinding
from .planning.study_design import infer_study_design_family

_VALIDATOR = "methodological_rigor"

# Missingness at/above this fraction turns a complete-case analysis into a
# materially different (biased) estimate from the full-cohort target.
_COMPLETE_CASE_BIAS_THRESHOLD = 0.20


@dataclass
class MethodSignals:
    """Flat, testable summary of what methods a run actually produced."""

    family: str
    # time-to-event
    has_hazard_ratio: bool = False
    has_survival_curve: bool = False
    landmark_or_timezero_defined: bool = False
    # association / static
    has_odds_ratio: bool = False
    # prediction
    has_auroc: bool = False
    has_calibration: bool = False
    held_out_reported: bool = False
    # causal
    has_covariate_balance: bool = False
    # phenotyping
    has_cluster_assignment: bool = False
    has_cluster_stability: bool = False
    # cross-cutting data-quality
    missing_fraction: Optional[float] = None
    complete_case_used: bool = False


def _finding(
    severity: str,
    message: str,
    *,
    detail: Optional[dict] = None,
    evidence_ids: Optional[List[str]] = None,
) -> ValidationFinding:
    return ValidationFinding(
        validator=_VALIDATOR,
        severity=severity,
        message=message,
        evidence_ids=list(evidence_ids or []),
        detail=detail,
    )


def audit_method_appropriateness(signals: MethodSignals) -> List[ValidationFinding]:
    """Findings when the analysis method does not match the study design.

    Pure decision logic over :class:`MethodSignals` -- no evidence I/O -- so the
    rules can be unit-tested against constructed inputs.
    """

    findings: List[ValidationFinding] = []
    family = signals.family

    if family == "time_to_event":
        if not (signals.has_hazard_ratio or signals.has_survival_curve):
            findings.append(
                _finding(
                    "error",
                    "Time-to-event question answered without any time-to-event "
                    "estimator: no hazard ratio (Cox) and no survival curve "
                    "(Kaplan-Meier / cumulative incidence) were produced.",
                    detail={"family": family},
                )
            )
        elif signals.has_odds_ratio and not signals.has_hazard_ratio:
            findings.append(
                _finding(
                    "error",
                    "Time-to-event question reported a static odds ratio but no "
                    "hazard ratio: a logistic/OR model ignores censoring and "
                    "follow-up time. Report a Cox/time-to-event estimand.",
                    detail={"family": family},
                )
            )
        if not signals.landmark_or_timezero_defined:
            findings.append(
                _finding(
                    "warning",
                    "Time-to-event analysis does not state an explicit time zero "
                    "/ landmark: exposure defined over a follow-up window without "
                    "landmarking risks immortal-time bias.",
                    detail={"family": family},
                )
            )

    elif family == "prediction":
        if not signals.has_auroc:
            findings.append(
                _finding(
                    "warning",
                    "Prediction study reports no discrimination metric (AUROC / "
                    "PR-AUC).",
                    detail={"family": family},
                )
            )
        if not signals.has_calibration:
            findings.append(
                _finding(
                    "warning",
                    "Prediction study reports discrimination without calibration: "
                    "a well-discriminating model can still be clinically unusable "
                    "if its risks are miscalibrated.",
                    detail={"family": family},
                )
            )
        if not signals.held_out_reported:
            findings.append(
                _finding(
                    "warning",
                    "Prediction study does not report a held-out / external split: "
                    "in-sample performance is optimistic.",
                    detail={"family": family},
                )
            )

    # NOTE: causal_emulation is intentionally NOT handled here. Causal-family
    # rigor (DAG, positivity diagnostic, covariate balance, negative control,
    # E-value) is owned by the more thorough ``causal_audit.run_causal_audit``,
    # which is already wired into the pipeline. Adding a balance check here would
    # double-fire against it. This module owns only the design->method-match axis
    # for the families causal_audit does not cover.

    elif family == "phenotyping":
        if signals.has_cluster_assignment and not signals.has_cluster_stability:
            findings.append(
                _finding(
                    "warning",
                    "Clusters reported without stability evidence (silhouette / "
                    "bootstrap / consensus): a data-driven partition with no "
                    "stability check may be an arbitrary cut.",
                    detail={"family": family},
                )
            )

    # Cross-cutting: a complete-case estimate under material missingness is not
    # the full-cohort estimate. This applies to every family.
    if (
        signals.complete_case_used
        and signals.missing_fraction is not None
        and signals.missing_fraction >= _COMPLETE_CASE_BIAS_THRESHOLD
    ):
        findings.append(
            _finding(
                "warning",
                "Complete-case analysis under material missingness "
                f"({signals.missing_fraction:.0%}): the complete-case estimate is "
                "not the full-cohort estimate. Report a missingness sensitivity "
                "analysis (e.g. multiple imputation or a pattern-mixture bound) "
                "before interpreting it as the target estimand.",
                detail={
                    "family": family,
                    "missing_fraction": signals.missing_fraction,
                },
            )
        )

    return findings


def _record_by_current_name(
    evidence: EvidenceStore,
    name: str,
    *,
    evidence_records: Optional[Sequence[EvidenceRecord]] = None,
) -> Optional[EvidenceRecord]:
    """Resolve ``name`` only inside an explicitly authorised record view."""

    if evidence_records is None:
        return evidence.get(name)
    records_by_id = {record.evidence_id: record for record in evidence_records}
    direct = records_by_id.get(name)
    if direct is not None:
        return direct
    evidence_id = evidence.aliases().get(name)
    return records_by_id.get(evidence_id) if evidence_id is not None else None


def _any_record(
    evidence: EvidenceStore,
    *names: str,
    evidence_records: Optional[Sequence[EvidenceRecord]] = None,
) -> bool:
    for name in names:
        if (
            _record_by_current_name(
                evidence,
                name,
                evidence_records=evidence_records,
            )
            is not None
        ):
            return True
    return False


def _text_signal(
    evidence: EvidenceStore,
    *tokens: str,
    evidence_records: Optional[Sequence[EvidenceRecord]] = None,
) -> bool:
    lowered = tuple(t.lower() for t in tokens)
    records = evidence.records() if evidence_records is None else evidence_records
    for record in records:
        haystack = " ".join(
            [
                str(record.evidence_id or ""),
                str(record.description or ""),
                str(record.relative_path or ""),
            ]
        ).lower()
        if any(tok in haystack for tok in lowered):
            return True
    return False


def _plan_landmark_defined(
    context: ResearchContext,
    evidence: EvidenceStore,
    *,
    evidence_records: Optional[Sequence[EvidenceRecord]] = None,
) -> bool:
    tokens = (
        "landmark",
        "time zero",
        "time-zero",
        "immortal",
        "incident",
        "index date",
    )
    haystack = " ".join(
        [
            str(context.research_question or ""),
            (
                " ".join(str(getattr(n, "note", n)) for n in (context.notes or []))
                if getattr(context, "notes", None)
                else ""
            ),
        ]
    ).lower()
    if any(tok in haystack for tok in tokens):
        return True
    return _text_signal(evidence, *tokens, evidence_records=evidence_records)


def _missing_fraction(
    evidence: EvidenceStore,
    *,
    evidence_records: Optional[Sequence[EvidenceRecord]] = None,
) -> Optional[float]:
    import pandas as pd

    for name in (
        "missingness",
        "missingness_summary",
        "measurement_missingness",
        "cohort_missingness_audit",
        "missingness_measurement_audit",
    ):
        record = _record_by_current_name(
            evidence,
            name,
            evidence_records=evidence_records,
        )
        if record is None:
            continue
        try:
            path = evidence.root / record.relative_path
            frame = (
                pd.read_csv(path) if path.suffix == ".csv" else pd.read_parquet(path)
            )
        except Exception:
            continue
        for col in frame.columns:
            low = str(col).lower()
            if "missing" in low and (
                "pct" in low or "frac" in low or "fraction" in low
            ):
                series = pd.to_numeric(frame[col], errors="coerce").dropna()
                if series.empty:
                    continue
                value = float(series.max())
                # Accept either a 0-1 fraction or a 0-100 percentage.
                return value / 100.0 if value > 1.0 else value
    return None


def extract_method_signals(
    context: ResearchContext,
    evidence: EvidenceStore,
    *,
    evidence_records: Optional[Sequence[EvidenceRecord]] = None,
) -> MethodSignals:
    """Read an authorised evidence view into a :class:`MethodSignals` bag.

    ``evidence_records`` is the digest-verified current-authority snapshot used
    by production writers. ``None`` retains the legacy whole-store behaviour
    for standalone callers that have no execution ledger.
    """

    family = str(infer_study_design_family(context))
    return MethodSignals(
        family=family,
        has_hazard_ratio=_any_record(
            evidence,
            "cox_summary",
            "hazard_ratio",
            "hazard_ratios",
            "cox_model",
            evidence_records=evidence_records,
        )
        or _text_signal(
            evidence,
            "hazard ratio",
            "cox proportional",
            evidence_records=evidence_records,
        ),
        has_survival_curve=_any_record(
            evidence,
            "km_curve",
            "kaplan_meier",
            "survival_curve",
            "survival_curve_points",
            evidence_records=evidence_records,
        )
        or _text_signal(
            evidence,
            "kaplan-meier",
            "kaplan meier",
            "cumulative incidence",
            evidence_records=evidence_records,
        ),
        landmark_or_timezero_defined=_plan_landmark_defined(
            context,
            evidence,
            evidence_records=evidence_records,
        ),
        has_odds_ratio=_any_record(
            evidence,
            "primary_association",
            "adjusted_association",
            "primary_or",
            evidence_records=evidence_records,
        )
        or _text_signal(
            evidence,
            "odds ratio",
            "logistic regression",
            evidence_records=evidence_records,
        ),
        has_auroc=_any_record(
            evidence,
            "auroc",
            "model_performance",
            "roc_curve",
            evidence_records=evidence_records,
        )
        or _text_signal(
            evidence,
            "auroc",
            "auc",
            "discrimination",
            evidence_records=evidence_records,
        ),
        has_calibration=_any_record(
            evidence,
            "calibration_curve",
            "calibration",
            evidence_records=evidence_records,
        )
        or _text_signal(
            evidence,
            "calibration",
            "brier",
            evidence_records=evidence_records,
        ),
        held_out_reported=_any_record(
            evidence,
            "split_strategy",
            evidence_records=evidence_records,
        )
        or _text_signal(
            evidence,
            "held-out",
            "held out",
            "test set",
            "external validation",
            "train/test",
            evidence_records=evidence_records,
        ),
        has_covariate_balance=_any_record(
            evidence,
            "covariate_balance",
            "balance_table",
            evidence_records=evidence_records,
        )
        or _text_signal(
            evidence,
            "covariate balance",
            "standardized mean difference",
            "love plot",
            evidence_records=evidence_records,
        ),
        has_cluster_assignment=_any_record(
            evidence,
            "cluster_characteristics",
            "cluster_mortality",
            "cluster_profiles",
            evidence_records=evidence_records,
        )
        or _text_signal(
            evidence,
            "cluster",
            "phenotype",
            "subphenotype",
            evidence_records=evidence_records,
        ),
        has_cluster_stability=_any_record(
            evidence,
            "silhouette_score",
            "clustering_metrics",
            "cluster_metrics",
            evidence_records=evidence_records,
        )
        or _text_signal(
            evidence,
            "silhouette",
            "bootstrap stability",
            "consensus matrix",
            evidence_records=evidence_records,
        ),
        missing_fraction=_missing_fraction(
            evidence,
            evidence_records=evidence_records,
        ),
        complete_case_used=_text_signal(
            evidence,
            "complete-case",
            "complete case",
            "complete_case",
            evidence_records=evidence_records,
        ),
    )


class MethodologicalRigorAuditor:
    """Evidence-driven method-appropriateness + classic-bias audit."""

    name = _VALIDATOR

    def audit(
        self,
        *,
        context: ResearchContext,
        evidence: EvidenceStore,
        evidence_records: Optional[Sequence[EvidenceRecord]] = None,
    ) -> List[ValidationFinding]:
        signals = extract_method_signals(
            context,
            evidence,
            evidence_records=evidence_records,
        )
        return audit_method_appropriateness(signals)


__all__ = [
    "MethodSignals",
    "MethodologicalRigorAuditor",
    "audit_method_appropriateness",
    "extract_method_signals",
]
