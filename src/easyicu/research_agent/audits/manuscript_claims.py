"""Manuscript numeric-claim auditing.

Extracted from :mod:`easyicu.research_agent.pipeline` so the numeric
plausibility checks (AUROC / Brier / outcome-rate consistency between
the writer-generated manuscript and the deterministic step summaries)
can be reasoned about and tested without pulling the full
``ResearchAgentPipeline`` class into memory.

Public entry point
------------------
* :func:`audit_manuscript_numeric_claims` — given the bound manuscript
  text and a per-step summary registry, returns a list of
  :class:`ValidationFinding` for every numeric claim that conflicts
  with the source evidence.

Internal helpers
----------------
``_first_summary_scalar``, ``_extract_metric_claims`` and
``_extract_percent_claims_near`` keep leading underscores — they have
no callers outside this module.
"""

from __future__ import annotations

import re
from typing import Any, Dict, List, Optional, Sequence

from ..schema import ValidationFinding


def audit_manuscript_numeric_claims(
    bound_manuscript: str,
    *,
    per_step_records: Sequence[Dict[str, Any]] | None = None,
) -> List[ValidationFinding]:
    """Block manuscript claims that drift from registered numeric evidence.

    Evidence binding proves a sentence points somewhere; it does not prove the
    number in the sentence matches the machine-readable result. Small local
    models can cite the right table while rounding from memory or inventing a
    confidence interval. This audit catches the high-risk prediction metrics
    before a run can be treated as manuscript-ready.
    """
    summaries = [
        item.get("step_summary")
        for item in (per_step_records or [])
        if isinstance(item.get("step_summary"), dict)
    ]
    if not summaries:
        return []

    findings: List[ValidationFinding] = []

    auroc = _first_summary_scalar(
        summaries,
        (
            "auroc",
            "statistic:auroc",
            "auc",
            "statistic:auc",
            "held_out_auroc",
            "statistic:held_out_auroc",
            "cv_auroc",
            "statistic:cv_auroc",
            "cv_auroc_mean",
            "statistic:cv_auroc_mean",
            "mean_auroc",
            "auroc_mean",
        ),
    )
    if auroc is not None:
        claimed_aurocs = _extract_metric_claims(bound_manuscript, r"\b(?:AUROC|AUC)\b")
        for claimed in claimed_aurocs:
            # Allow ordinary two-decimal rounding (0.7769 -> 0.78), but not
            # manuscript-friendly drift such as 0.82.
            if abs(claimed - auroc) > 0.015:
                findings.append(
                    ValidationFinding(
                        validator="manuscript_numeric_auditor",
                        severity="error",
                        message=(
                            f"Manuscript AUROC claim {claimed:.3g} does not match "
                            f"registered AUROC {auroc:.3g}."
                        ),
                        detail={
                            "metric": "auroc",
                            "claimed": claimed,
                            "registered": auroc,
                        },
                    )
                )
        ci_low = _first_summary_scalar(
            summaries,
            ("auroc_ci_lower", "statistic:auroc_ci_lower", "auc_ci_lower", "ci_lower_auroc"),
        )
        ci_high = _first_summary_scalar(
            summaries,
            ("auroc_ci_upper", "statistic:auroc_ci_upper", "auc_ci_upper", "ci_upper_auroc"),
        )
        if (ci_low is None or ci_high is None) and re.search(
            r"\b(?:AUROC|AUC)\b.{0,80}\b95\s*%\s*CI\b",
            bound_manuscript,
            flags=re.IGNORECASE | re.DOTALL,
        ):
            findings.append(
                ValidationFinding(
                    validator="manuscript_numeric_auditor",
                    severity="error",
                    message=(
                        "Manuscript reports an AUROC confidence interval, but no "
                        "AUROC CI bounds are registered in step_summary evidence."
                    ),
                    detail={"metric": "auroc_ci"},
                )
            )

    brier = _first_summary_scalar(
        summaries,
        (
            "brier_score",
            "statistic:brier_score",
            "held_out_brier",
            "statistic:held_out_brier",
            "cv_brier_mean",
            "statistic:cv_brier_mean",
            "brier_mean",
        ),
    )
    if brier is not None:
        for claimed in _extract_metric_claims(bound_manuscript, r"\bBrier(?: score)?\b"):
            if abs(claimed - brier) > 0.015:
                findings.append(
                    ValidationFinding(
                        validator="manuscript_numeric_auditor",
                        severity="error",
                        message=(
                            f"Manuscript Brier claim {claimed:.3g} does not match "
                            f"registered Brier score {brier:.3g}."
                        ),
                        detail={
                            "metric": "brier_score",
                            "claimed": claimed,
                            "registered": brier,
                        },
                    )
                )

    baseline = _first_summary_scalar(
        summaries,
        (
            "baseline_prevalence",
            "statistic:baseline_prevalence",
            "outcome_rate",
            "statistic:outcome_rate",
            "event_rate",
            "statistic:event_rate",
        ),
    )
    if baseline is not None:
        for claimed in _extract_percent_claims_near(
            bound_manuscript,
            r"\b(?:baseline prevalence|mortality|death|outcome incidence)\b",
        ):
            if abs(claimed - baseline) > 0.015:
                findings.append(
                    ValidationFinding(
                        validator="manuscript_numeric_auditor",
                        severity="error",
                        message=(
                            f"Manuscript prevalence claim {claimed:.3g} does not match "
                            f"registered baseline prevalence {baseline:.3g}."
                        ),
                        detail={
                            "metric": "baseline_prevalence",
                            "claimed": claimed,
                            "registered": baseline,
                        },
                    )
                )

    return findings


def _first_summary_scalar(
    summaries: Sequence[Dict[str, Any]], keys: Sequence[str]
) -> Optional[float]:
    # Lazy import to avoid a circular dependency with pipeline.py, which
    # imports `audit_manuscript_numeric_claims` from this module.
    from ..pipeline import _first_present_scalar

    for summary in summaries:
        value = _first_present_scalar(summary, keys)
        if value is None:
            continue
        try:
            return float(value)
        except (TypeError, ValueError):
            continue
    return None


def _extract_metric_claims(text: str, metric_pattern: str) -> List[float]:
    claims: List[float] = []
    clean_text = re.sub(r"\[[^\]]+\]\([^)]*\)", "", text or "")
    clean_text = re.sub(r"<!--.*?-->", "", clean_text, flags=re.DOTALL)
    pattern = re.compile(
        metric_pattern + r"[^0-9]{0,40}([01](?:\.\d+)?)",
        flags=re.IGNORECASE,
    )
    for match in pattern.finditer(clean_text):
        try:
            value = float(match.group(1))
        except (TypeError, ValueError):
            continue
        if 0.0 <= value <= 1.0:
            claims.append(value)
    return claims


def _extract_percent_claims_near(text: str, phrase_pattern: str) -> List[float]:
    claims: List[float] = []
    clean_text = re.sub(r"\[[^\]]+\]\([^)]*\)", "", text or "")
    clean_text = re.sub(r"<!--.*?-->", "", clean_text, flags=re.DOTALL)
    pattern = re.compile(
        phrase_pattern + r".{0,80}?([0-9]+(?:\.[0-9]+)?)\s*%",
        flags=re.IGNORECASE | re.DOTALL,
    )
    # A percentage that introduces a confidence/credible interval (e.g.
    # "95% CI", "95% confidence interval") is never a prevalence/mortality
    # value; the interval width and the outcome rate are unrelated. Without
    # this guard the lazy proximity window happily binds the "95%" from a
    # "... odds of ICU death ... and a 95% confidence interval ..." sentence
    # to the "death" phrase and falsely flags a 0.95 prevalence claim.
    ci_trailer = re.compile(
        r"\s*(?:CI\b|confidence\s+interval|credible\s+interval)",
        flags=re.IGNORECASE,
    )
    for match in pattern.finditer(clean_text):
        trailer = clean_text[match.end(): match.end() + 32]
        if ci_trailer.match(trailer):
            continue
        try:
            value = float(match.group(1)) / 100.0
        except (TypeError, ValueError):
            continue
        if 0.0 <= value <= 1.0:
            claims.append(value)
    return claims




__all__ = ["audit_manuscript_numeric_claims"]
