"""Shared helpers for the audit owner modules split out of validators.py."""

from __future__ import annotations

from typing import Any, Dict, List, Sequence

import pandas as pd

from ..schema import (
    ResearchContext,
    ValidationFinding,
)

# ---------------------------------------------------------------------------
# CohortAuditor
# ---------------------------------------------------------------------------


# Patient-level identifier column names. Their presence means a cohort can
# be reasoned about at the patient level (within-patient non-independence,
# first-stay selection). Stay-level keys (stay_id, icustay_id) and

_PATIENT_ID_COLUMNS = (
    "subject_id",
    "patient_id",
    "patientid",
    "person_id",
    "uniquepid",
)
# ICU length-of-stay columns, expressed in DAYS in the EasyICU export.
_LOS_DAY_COLUMNS = ("los_icu", "los", "icu_los", "los_icu_days")


def cohort_hygiene_findings(
    df: pd.DataFrame,
    context: ResearchContext,
) -> List[ValidationFinding]:
    """Impartial, advisory cohort-hygiene flags (``warning``, never blocking).

    These surface standard cohort-hygiene questions — patient-level
    non-independence and short-stay exposure — so they are visible and
    recorded. They deliberately do NOT impose an analytical choice (a
    minimum LoS, first-stay deduplication, ...): per the impartiality rule
    the analyst decides, and the auditor only ensures the question was put.
    Severity is always ``warning`` so the gate never fail-closes on them.

    See ``feedback_rules_must_be_impartial`` and
    ``feedback_coverage_gap_vs_missing_policy``.
    """
    findings: List[ValidationFinding] = []
    cols = {str(c).lower() for c in df.columns}

    # (A) Patient-level non-independence assessability. Only relevant when an
    # outcome model is in scope (a stay-level association treats each ICU
    # stay as independent). If no patient identifier is present, that
    # assumption cannot even be CHECKED from this export — structural
    # no-source — so advise re-extraction rather than penalising the analysis
    # or silently assuming independence.
    outcome = getattr(context, "target_outcome", None)
    has_patient_id = any(pid in cols for pid in _PATIENT_ID_COLUMNS)
    if outcome and not has_patient_id:
        findings.append(
            ValidationFinding(
                validator="cohort_auditor",
                severity="warning",
                message=(
                    "Cohort is keyed at the ICU-stay level with no patient "
                    "identifier; within-patient non-independence and first-stay "
                    "selection cannot be assessed from this export. Re-extract "
                    "with a patient identifier (e.g. subject_id) if repeat ICU "
                    "stays could affect the outcome model."
                ),
                detail={
                    "kind": "cohort_hygiene",
                    "subkind": "patient_independence_unassessable",
                    "structural_no_source": True,
                    "impartial": True,
                },
            )
        )

    # (B) Short-stay exposure. If an ICU LoS column (days) is present, report
    # the fraction of very short stays. Excluding <24h stays is a defensible
    # convention, NOT a requirement, so this records the distribution and
    # leaves the choice to the analyst.
    los_col = next(
        (c for c in df.columns if str(c).lower() in _LOS_DAY_COLUMNS),
        None,
    )
    if los_col is not None:
        los = pd.to_numeric(df[los_col], errors="coerce").dropna()
        if not los.empty:
            frac_short = float((los < 1.0).mean())
            if frac_short > 0:
                findings.append(
                    ValidationFinding(
                        validator="cohort_auditor",
                        severity="warning",
                        message=(
                            f"{frac_short:.0%} of stays have ICU length-of-stay "
                            f"<1 day (column '{los_col}'); consider whether "
                            "incomplete exposure affects the analysis. No "
                            "minimum-LoS filter is imposed — recorded for the "
                            "analyst to judge."
                        ),
                        detail={
                            "kind": "cohort_hygiene",
                            "subkind": "short_stay_exposure",
                            "fraction_los_under_1_day": frac_short,
                            "los_column": los_col,
                            "impartial": True,
                        },
                    )
                )

    # (C) Elapsed-time columns that contradict their own anchor. The export
    # writes ``<concept>_time`` as hours elapsed from the cohort anchor, so a
    # negative value places the event before the anchor that defines the row.
    #
    # The rule is STRUCTURAL, not a named-column rule: it reads the export's
    # own convention off the export. MEASURED on the real MIMIC-IV cohort --
    # 21 elapsed-time columns, and 20 of them have min EXACTLY 0.00. The
    # convention is not in doubt; one column breaks it, in 28 of 94,458 rows.
    #
    # Every one of the nine recorded tasks carries those same 28 rows and NONE
    # was told. Only a survival analysis ever compares the columns, so h1 hit
    # it 20 minutes into its primary step, wrote its own guard, raised, and
    # died with four provider calls unspent -- having discovered a property of
    # the cohort that was fixed before it was ever handed one.
    #
    # Reported, never enforced: 0.03 % of rows is a data-quality fact, not a
    # reason to refuse an analysis, and which rows to drop or censor is the
    # analyst's call. Same severity and impartiality as (A) and (B).
    for column in df.columns:
        if not str(column).lower().endswith("_time"):
            continue
        elapsed = pd.to_numeric(df[column], errors="coerce").dropna()
        if elapsed.empty:
            continue
        negative_n = int((elapsed < 0).sum())
        if not negative_n:
            continue
        findings.append(
            ValidationFinding(
                validator="cohort_auditor",
                severity="warning",
                message=(
                    f"{negative_n} row(s) give column '{column}' a negative "
                    "elapsed time, placing the event before the anchor the row "
                    f"is measured from (minimum {float(elapsed.min()):.4g}). "
                    "Every other elapsed-time column in this export starts at "
                    "0. No row is dropped or censored — recorded for the "
                    "analyst to judge before any time-to-event analysis."
                ),
                detail={
                    "kind": "cohort_hygiene",
                    "subkind": "elapsed_time_precedes_anchor",
                    "column": str(column),
                    "negative_n": negative_n,
                    "minimum": float(elapsed.min()),
                    "n_rows": int(len(df)),
                    "impartial": True,
                },
            )
        )

    return findings
def dedupe_findings(
    findings: Sequence[ValidationFinding],
) -> List[ValidationFinding]:
    """Collapse byte-identical findings within the same authority scope.

    The pilot run on 2026-05-15 surfaced the same
    ``concept_usage_auditor`` message recorded 5 times in a single run
    because step-level audits fire across every step that touches the
    flagged column. The output reads like 5 separate problems when it
    is one. This helper keeps the first occurrence (preserves order),
    records the rolled-up count under ``detail['duplicate_count']``,
    and merges ``evidence_ids`` across the collapsed group so no
    reference is lost.

    Findings that already declare a non-empty ``detail`` are still merged when
    their owner scope matches: ``detail.step_id`` participates in the dedupe
    key.  This prevents the same prose emitted by two independent steps from
    being collapsed under the first step's authority and then incorrectly
    retired when only that first step succeeds.  Other detail remains
    shallow-copied and the duplicate count overwrites only the dedicated key.
    """
    seen: Dict[tuple, int] = {}
    out: List[ValidationFinding] = []
    for f in findings:
        owner_step_id = str((f.detail or {}).get("step_id") or "").strip() or None
        key = (f.validator, f.severity, f.message, owner_step_id)
        if key not in seen:
            seen[key] = len(out)
            out.append(f)
            continue
        idx = seen[key]
        existing = out[idx]
        new_detail: Dict[str, Any] = dict(existing.detail or {})
        new_detail["duplicate_count"] = new_detail.get("duplicate_count", 1) + 1
        merged_evidence = list(existing.evidence_ids)
        for eid in f.evidence_ids:
            if eid not in merged_evidence:
                merged_evidence.append(eid)
        out[idx] = existing.model_copy(
            update={"detail": new_detail, "evidence_ids": merged_evidence},
        )
    return out
