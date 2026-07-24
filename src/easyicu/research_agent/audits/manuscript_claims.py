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

from ..authority.runtime_artifacts import current_successful_step_records
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
    current_records = current_successful_step_records(per_step_records or [])
    summaries = [
        item.get("step_summary")
        for item in current_records
        if isinstance(item.get("step_summary"), dict)
    ]
    if not summaries:
        return []

    findings: List[ValidationFinding] = []

    # A run legitimately registers MORE THAN ONE AUROC: the primary model
    # step and any sensitivity / feature-eligibility / audit step each report
    # their own discrimination. The writer may cite any of them (primary model
    # AUROC in Results, an audit-step AUROC in a robustness sentence). So a
    # manuscript AUROC is a hallucination ONLY when it matches NO registered
    # per-step value within rounding tolerance — not when it differs from the
    # first one we happened to read. Collapsing to a single "the" AUROC made
    # the auditor flag a correctly-cited primary AUROC (0.868) against an
    # arbitrarily-picked audit-step AUROC (0.812). Match-any preserves the
    # hallucination catch (0.766 rounded to 0.7 still matches nothing) while
    # killing that false positive.
    registered_aurocs = _all_summary_scalars(
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
    if registered_aurocs:
        claimed_aurocs = _extract_metric_claims(bound_manuscript, r"\b(?:AUROC|AUC)\b")
        for claimed in claimed_aurocs:
            # Allow ordinary two-decimal rounding (0.7769 -> 0.78), but not
            # manuscript-friendly drift such as 0.82. Compare against the
            # NEAREST registered value across every step.
            nearest = min(registered_aurocs, key=lambda r: abs(claimed - r))
            if abs(claimed - nearest) > 0.015:
                findings.append(
                    ValidationFinding(
                        validator="manuscript_numeric_auditor",
                        severity="error",
                        message=(
                            f"Manuscript AUROC claim {claimed:.3g} does not match "
                            f"any registered AUROC (nearest {nearest:.3g})."
                        ),
                        detail={
                            "metric": "auroc",
                            "claimed": claimed,
                            "registered": nearest,
                            "registered_all": sorted(set(registered_aurocs)),
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

    registered_briers = _all_summary_scalars(
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
    if registered_briers:
        for claimed in _extract_metric_claims(bound_manuscript, r"\bBrier(?: score)?\b"):
            nearest = min(registered_briers, key=lambda r: abs(claimed - r))
            if abs(claimed - nearest) > 0.015:
                findings.append(
                    ValidationFinding(
                        validator="manuscript_numeric_auditor",
                        severity="error",
                        message=(
                            f"Manuscript Brier claim {claimed:.3g} does not match "
                            f"any registered Brier score (nearest {nearest:.3g})."
                        ),
                        detail={
                            "metric": "brier_score",
                            "claimed": claimed,
                            "registered": nearest,
                            "registered_all": sorted(set(registered_briers)),
                        },
                    )
                )

    registered_baselines = _all_summary_scalars(
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
    if registered_baselines:
        for claimed in _extract_percent_claims_near(
            bound_manuscript,
            r"\b(?:baseline prevalence|mortality|death|outcome incidence)\b",
            skip_stratified_context=True,
        ):
            nearest = min(registered_baselines, key=lambda r: abs(claimed - r))
            if abs(claimed - nearest) > 0.015:
                findings.append(
                    ValidationFinding(
                        validator="manuscript_numeric_auditor",
                        severity="error",
                        message=(
                            f"Manuscript prevalence claim {claimed:.3g} does not match "
                            f"any registered baseline prevalence (nearest {nearest:.3g})."
                        ),
                        detail={
                            "metric": "baseline_prevalence",
                            "claimed": claimed,
                            "registered": nearest,
                            "registered_all": sorted(set(registered_baselines)),
                        },
                    )
                )

    return findings


def _first_summary_scalar(
    summaries: Sequence[Dict[str, Any]], keys: Sequence[str]
) -> Optional[float]:
    from ..scalar_utils import _first_present_scalar

    for summary in summaries:
        value = _first_present_scalar(summary, keys)
        if value is None:
            continue
        try:
            return float(value)
        except (TypeError, ValueError):
            continue
    return None


def _all_summary_scalars(
    summaries: Sequence[Dict[str, Any]], keys: Sequence[str]
) -> List[float]:
    """Every per-step value present under ``keys``, one per summary.

    Unlike :func:`_first_summary_scalar` (which stops at the first step that
    carries the metric), this returns the value from *each* step so the
    auditor can accept a manuscript number that matches any registered step,
    not only the first. ``_first_present_scalar`` resolves the canonical key
    within a step, so a step that registers both ``auroc`` and ``test_auroc``
    contributes its primary value once.
    """
    from ..scalar_utils import _first_present_scalar

    values: List[float] = []
    for summary in summaries:
        value = _first_present_scalar(summary, keys)
        if value is None:
            continue
        try:
            values.append(float(value))
        except (TypeError, ValueError):
            continue
    return values


# A number that sits between the metric keyword and the value behind a
# difference/tolerance cue is a *delta between two estimates*, not a reported
# point estimate — e.g. "AUROC estimates differing by less than 0.001". Binding
# such a delta to the registered AUROC produces a spurious mismatch (the same
# false-positive class the CI guard in ``_extract_percent_claims_near`` handles).
_METRIC_DELTA_CUE = re.compile(
    r"\b(?:estimates?\s+)?differ(?:ed|ing)?\s+by"
    r"(?:\s+(?:less\s+than|under|up\s+to|no\s+more\s+than|at\s+most))?\s*$|"
    r"\b(?:difference|delta)\s+(?:was|of|under|below|less\s+than|"
    r"no\s+more\s+than|at\s+most)\s*$|±\s*$|\+/-\s*$",
    flags=re.IGNORECASE,
)


def _strip_manuscript_noise(text: str) -> str:
    """Remove markdown links, HTML comments, and binder footnote-definition
    lines before metric extraction.

    Footnote-definition lines (``[^claim_2]: value=...; field=metrics.auroc;
    evidence=statistic_step_summary_1c8c8ff2; display=0.831``) are auto-appended
    machine provenance, not author prose. They contain metric field NAMES
    ("metrics.auroc") followed by content-addressed step IDs whose leading digit
    can be parsed as a spurious bare ``0``/``1`` AUROC/Brier claim — an
    intermittent false positive that fires only when the sha happens to start
    with a digit. The critic already skips these lines (see agents/core.py); the
    numeric auditor must too.
    """
    clean = re.sub(r"\[[^\]]+\]\([^)]*\)", "", text or "")
    clean = re.sub(r"<!--.*?-->", "", clean, flags=re.DOTALL)
    clean = re.sub(r"(?m)^\s*\[\^[^\]]+\]:.*$", "", clean)
    return clean


def _extract_metric_claims(text: str, metric_pattern: str) -> List[float]:
    claims: List[float] = []
    clean_text = _strip_manuscript_noise(text)
    # Require a DECIMAL point: a real AUROC/Brier point estimate is always
    # reported with decimals (0.83, not 1). A bare integer near the metric word
    # is a figure/table number, an enumeration, or a step-id digit, never a
    # reported metric — excluding it removes that whole false-positive class
    # while still catching an implausible "1.0"/"0.0" claim written with decimals.
    pattern = re.compile(
        metric_pattern + r"([^0-9]{0,40})([01]\.\d+)",
        flags=re.IGNORECASE,
    )
    for match in pattern.finditer(clean_text):
        # A cue is authoritative only when it sits between the metric name and
        # the number ("AUROC estimates differing by less than 0.001"). Ordinary
        # prose elsewhere in the clause ("Within the cohort, AUROC was 0.92")
        # must not disable numeric verification.
        if _METRIC_DELTA_CUE.search(match.group(1)):
            continue
        try:
            value = float(match.group(2))
        except (TypeError, ValueError):
            continue
        if 0.0 <= value <= 1.0:
            claims.append(value)
    return claims


def _extract_percent_claims_near(
    text: str,
    phrase_pattern: str,
    *,
    skip_stratified_context: bool = False,
) -> List[float]:
    claims: List[float] = []
    clean_text = _strip_manuscript_noise(text)
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
        if skip_stratified_context:
            window = clean_text[max(0, match.start() - 96): match.end() + 96]
            if _looks_like_stratified_rate_context(window):
                continue
        try:
            value = float(match.group(1)) / 100.0
        except (TypeError, ValueError):
            continue
        if 0.0 <= value <= 1.0:
            claims.append(value)
    return claims


def _looks_like_stratified_rate_context(window: str) -> bool:
    """Return true when a percent is clearly subgroup-specific, not baseline.

    The baseline-prevalence audit should catch statements like "overall
    cohort mortality was 5.6%" when the registered event rate is 9.4%. It
    should not treat "the SOFA-2=0 stratum had mortality 5.6%" as a baseline
    claim. Keep this lexical and case-neutral: any stratum/subgroup/bin/level
    language can describe a legitimate within-group rate.
    """

    text = (window or "").lower()
    if not re.search(
        r"\b(?:stratum|strata|subgroup|sub-group|category|level|bin|"
        r"quartile|tertile|decile)\b",
        text,
    ):
        return False
    return True




__all__ = ["audit_manuscript_numeric_claims"]
