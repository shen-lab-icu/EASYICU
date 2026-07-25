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
from typing import Any, Dict, List, Mapping, NamedTuple, Optional, Sequence, Tuple

from ..authority.runtime_artifacts import current_successful_step_records
from ..schema import ValidationFinding


_AUROC_SUMMARY_KEYS = (
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
)

_AUROC_CI_LOWER_KEYS = (
    "auroc_ci_lower",
    "statistic:auroc_ci_lower",
    "auc_ci_lower",
    "ci_lower_auroc",
)

_AUROC_CI_UPPER_KEYS = (
    "auroc_ci_upper",
    "statistic:auroc_ci_upper",
    "auc_ci_upper",
    "ci_upper_auroc",
)

_BRIER_SUMMARY_KEYS = (
    "brier_score",
    "statistic:brier_score",
    "held_out_brier",
    "statistic:held_out_brier",
    "cv_brier_mean",
    "statistic:cv_brier_mean",
    "brier_mean",
)

_PREVALENCE_SUMMARY_KEYS = (
    "baseline_prevalence",
    "statistic:baseline_prevalence",
    "outcome_rate",
    "statistic:outcome_rate",
    "event_rate",
    "statistic:event_rate",
)


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
    # Parallel to ``summaries``: which step each one came from, so a claim the
    # binder resolved to a step can be compared against that step alone.
    summary_owners = [
        str(item.get("step_id") or "")
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
    registered_aurocs = _all_summary_scalars(summaries, _AUROC_SUMMARY_KEYS)
    if registered_aurocs:
        claimed_aurocs = _extract_metric_claims_with_footnote(
            bound_manuscript, r"\b(?:AUROC|AUC)\b"
        )
        footnote_steps = footnote_step_ids(bound_manuscript)
        for claimed, footnote_id in claimed_aurocs:
            # Scope the comparison to the step the binder resolved this number
            # to, when it resolved one. Match-any answers "is this AUROC
            # registered anywhere in the run?", which passes a sentence that
            # attributes the sensitivity model's 0.85 to the primary model.
            # It stays as the fallback for an unbound claim, where there is no
            # step coordinate to scope by — and where the untraced-numeric
            # finding already fires.
            scope = _scoped_registered_values(
                summaries=summaries,
                summary_owners=summary_owners,
                keys=_AUROC_SUMMARY_KEYS,
                footnote_steps=footnote_steps,
                footnote_id=footnote_id,
            )
            if scope.cited_step_lacks_metric:
                findings.append(
                    _cited_step_lacks_metric_finding(
                        metric="auroc",
                        label="AUROC",
                        claimed=claimed,
                        cited_step=scope.cited_step,
                    )
                )
                continue
            scoped, step_id = scope.values, scope.step_id
            comparison = scoped or registered_aurocs
            # Allow ordinary two-decimal rounding (0.7769 -> 0.78), but not
            # manuscript-friendly drift such as 0.82.
            nearest = min(comparison, key=lambda r: abs(claimed - r))
            if abs(claimed - nearest) > _METRIC_TOLERANCE:
                findings.append(
                    ValidationFinding(
                        validator="manuscript_numeric_auditor",
                        severity="error",
                        message=(
                            f"Manuscript AUROC claim {claimed:.3g} does not match "
                            + (
                                f"the AUROC registered by step {step_id!r} "
                                f"(nearest {nearest:.3g})."
                                if scoped
                                else f"any registered AUROC (nearest {nearest:.3g})."
                            )
                        ),
                        detail={
                            "metric": "auroc",
                            "claimed": claimed,
                            "registered": nearest,
                            "registered_all": sorted(set(registered_aurocs)),
                            "scoped_to_step": step_id if scoped else None,
                        },
                    )
                )
        # Scoped the same way as the point estimate: a manuscript that reports
        # the primary model's CI is not covered by a *different* step having
        # registered CI bounds. When the sentence carries a resolvable step,
        # that step must own the bounds; otherwise any registered pair counts.
        ci_steps = {
            step_id
            for step_id in (
                footnote_steps.get(fid or "")
                for _, fid in _extract_metric_claims_with_footnote(
                    bound_manuscript, r"\b(?:AUROC|AUC)\b"
                )
            )
            if step_id
        }
        # No `or list(summaries)` fallback: when the sentence names steps, a
        # *different* step owning CI bounds does not cover it. Falling back
        # there is the same borrowing the point-estimate scope closed.
        if ci_steps:
            ci_summaries = [
                summary
                for summary, owner in zip(summaries, summary_owners)
                if owner in ci_steps
            ]
        else:
            ci_summaries = list(summaries)
        # Bounds are resolved as a PAIR from one summary. Reading the lower
        # bound with one scan and the upper with another lets a two-model
        # manuscript take 0.71 from the primary step and 0.84 from the
        # sensitivity step and call the result an interval.
        ci_pairs = _summary_ci_pairs(
            ci_summaries, _AUROC_CI_LOWER_KEYS, _AUROC_CI_UPPER_KEYS
        )
        ci_low = ci_pairs[0][0] if ci_pairs else None
        ci_high = ci_pairs[0][1] if ci_pairs else None

        # Then check the interval the manuscript actually printed against the
        # registered one. Establishing that *a* CI exists says nothing about
        # whether 0.71-0.84 is that CI.
        if ci_pairs:
            owned_pairs = _summary_ci_pairs_by_owner(
                summaries, summary_owners, _AUROC_CI_LOWER_KEYS, _AUROC_CI_UPPER_KEYS
            )
            for low, high, footnote_id in _extract_ci_claims(
                bound_manuscript, r"\b(?:AUROC|AUC)\b"
            ):
                cited_step = footnote_steps.get(footnote_id or "")
                if cited_step and cited_step in owned_pairs:
                    comparison = [owned_pairs[cited_step]]
                elif cited_step:
                    findings.append(
                        _cited_step_lacks_metric_finding(
                            metric="auroc_ci",
                            label="AUROC confidence interval",
                            claimed=low,
                            cited_step=cited_step,
                        )
                    )
                    continue
                else:
                    comparison = ci_pairs
                if not any(
                    abs(low - bound_low) <= _METRIC_TOLERANCE
                    and abs(high - bound_high) <= _METRIC_TOLERANCE
                    for bound_low, bound_high in comparison
                ):
                    nearest_low, nearest_high = min(
                        comparison,
                        key=lambda pair: abs(low - pair[0]) + abs(high - pair[1]),
                    )
                    findings.append(
                        ValidationFinding(
                            validator="manuscript_numeric_auditor",
                            severity="error",
                            message=(
                                f"Manuscript AUROC 95% CI {low:.3g}-{high:.3g} does "
                                "not match "
                                + (
                                    f"the interval registered by step {cited_step!r} "
                                    if cited_step
                                    else "any registered AUROC CI "
                                )
                                + f"(nearest {nearest_low:.3g}-{nearest_high:.3g})."
                            ),
                            detail={
                                "metric": "auroc_ci",
                                "claimed_lower": low,
                                "claimed_upper": high,
                                "registered_lower": nearest_low,
                                "registered_upper": nearest_high,
                                "scoped_to_step": cited_step,
                                "reason": "ci_bounds_do_not_match_registered",
                            },
                        )
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
                        "Manuscript reports an AUROC confidence interval, but "
                        + (
                            "the step(s) it cites register no AUROC CI bounds: "
                            + ", ".join(sorted(ci_steps))
                            + "."
                            if ci_steps
                            else "no AUROC CI bounds are registered in "
                            "step_summary evidence."
                        )
                    ),
                    detail={
                        "metric": "auroc_ci",
                        "cited_steps": sorted(ci_steps),
                        "reason": (
                            "cited_step_does_not_register_metric"
                            if ci_steps
                            else "metric_not_registered"
                        ),
                    },
                )
            )

    registered_briers = _all_summary_scalars(summaries, _BRIER_SUMMARY_KEYS)
    if registered_briers:
        footnote_steps = footnote_step_ids(bound_manuscript)
        for claimed, footnote_id in _extract_metric_claims_with_footnote(
            bound_manuscript, r"\bBrier(?: score)?\b"
        ):
            scope = _scoped_registered_values(
                summaries=summaries,
                summary_owners=summary_owners,
                keys=_BRIER_SUMMARY_KEYS,
                footnote_steps=footnote_steps,
                footnote_id=footnote_id,
            )
            if scope.cited_step_lacks_metric:
                findings.append(
                    _cited_step_lacks_metric_finding(
                        metric="brier_score",
                        label="Brier score",
                        claimed=claimed,
                        cited_step=scope.cited_step,
                    )
                )
                continue
            scoped, step_id = scope.values, scope.step_id
            comparison = scoped or registered_briers
            nearest = min(comparison, key=lambda r: abs(claimed - r))
            if abs(claimed - nearest) > _METRIC_TOLERANCE:
                findings.append(
                    ValidationFinding(
                        validator="manuscript_numeric_auditor",
                        severity="error",
                        message=(
                            f"Manuscript Brier claim {claimed:.3g} does not match "
                            + (
                                f"the Brier score registered by step {step_id!r} "
                                f"(nearest {nearest:.3g})."
                                if scoped
                                else f"any registered Brier score (nearest {nearest:.3g})."
                            )
                        ),
                        detail={
                            "metric": "brier_score",
                            "claimed": claimed,
                            "registered": nearest,
                            "registered_all": sorted(set(registered_briers)),
                            "scoped_to_step": step_id,
                        },
                    )
                )

    registered_baselines = _all_summary_scalars(summaries, _PREVALENCE_SUMMARY_KEYS)
    if registered_baselines:
        footnote_steps = footnote_step_ids(bound_manuscript)
        for claimed, footnote_id in _extract_percent_claims_near_with_footnote(
            bound_manuscript,
            r"\b(?:baseline prevalence|mortality|death|outcome incidence)\b",
            skip_stratified_context=True,
        ):
            scope = _scoped_registered_values(
                summaries=summaries,
                summary_owners=summary_owners,
                keys=_PREVALENCE_SUMMARY_KEYS,
                footnote_steps=footnote_steps,
                footnote_id=footnote_id,
            )
            if scope.cited_step_lacks_metric:
                findings.append(
                    _cited_step_lacks_metric_finding(
                        metric="baseline_prevalence",
                        label="baseline prevalence",
                        claimed=claimed,
                        cited_step=scope.cited_step,
                    )
                )
                continue
            scoped, step_id = scope.values, scope.step_id
            comparison = scoped or registered_baselines
            nearest = min(comparison, key=lambda r: abs(claimed - r))
            if abs(claimed - nearest) > _METRIC_TOLERANCE:
                findings.append(
                    ValidationFinding(
                        validator="manuscript_numeric_auditor",
                        severity="error",
                        message=(
                            f"Manuscript prevalence claim {claimed:.3g} does not match "
                            + (
                                f"the baseline prevalence registered by step "
                                f"{step_id!r} (nearest {nearest:.3g})."
                                if scoped
                                else "any registered baseline prevalence "
                                f"(nearest {nearest:.3g})."
                            )
                        ),
                        detail={
                            "metric": "baseline_prevalence",
                            "claimed": claimed,
                            "registered": nearest,
                            "registered_all": sorted(set(registered_baselines)),
                            "scoped_to_step": step_id,
                        },
                    )
                )

    return findings


#: Ordinary two-decimal rounding (0.7769 -> 0.78) is fine; manuscript-friendly
#: drift is not. Shared so a point estimate and its interval are held to one
#: standard.
_METRIC_TOLERANCE = 0.015


#: ``AUROC ... 95% CI 0.71-0.84`` in the forms a writer actually produces.
_CI_SEPARATOR = r"(?:\s*(?:–|—|-|to|,)\s*)"
_CI_BOUNDS = r"[\(\[]?\s*([01]\.\d+)" + _CI_SEPARATOR + r"([01]\.\d+)\s*[\)\]]?"


def _extract_ci_claims(
    text: str, metric_pattern: str
) -> List[Tuple[float, float, Optional[str]]]:
    """The interval the manuscript printed, with its footnote id."""

    clean_text = _strip_manuscript_noise(text)
    pattern = re.compile(
        metric_pattern
        # ``.`` stops at a newline but not at a decimal point: the point
        # estimate between the metric name and its interval ("AUROC of 0.868
        # (95% CI ...") is full of decimal points, and excluding them was
        # enough to make this never match a real sentence.
        + r".{0,120}?9[05]\s*%\s*(?:CI|confidence\s+interval|credible\s+interval)"
        + r"[^0-9\(\[]{0,15}"
        + _CI_BOUNDS
        + r"(?:\[\^(?P<fid>[^\]]+)\])?",
        flags=re.IGNORECASE,
    )
    claims: List[Tuple[float, float, Optional[str]]] = []
    for match in pattern.finditer(clean_text):
        try:
            low = float(match.group(1))
            high = float(match.group(2))
        except (TypeError, ValueError):
            continue
        claims.append((low, high, match.group("fid")))
    return claims


def _summary_ci_pairs(
    summaries: Sequence[Dict[str, Any]],
    lower_keys: Sequence[str],
    upper_keys: Sequence[str],
) -> List[Tuple[float, float]]:
    """Intervals, each with both bounds taken from the same summary."""

    from ..scalar_utils import _first_present_scalar

    pairs: List[Tuple[float, float]] = []
    for summary in summaries:
        low = _first_present_scalar(summary, lower_keys)
        high = _first_present_scalar(summary, upper_keys)
        if low is None or high is None:
            continue
        try:
            pairs.append((float(low), float(high)))
        except (TypeError, ValueError):
            continue
    return pairs


def _summary_ci_pairs_by_owner(
    summaries: Sequence[Dict[str, Any]],
    owners: Sequence[str],
    lower_keys: Sequence[str],
    upper_keys: Sequence[str],
) -> Dict[str, Tuple[float, float]]:
    """The interval each step registered, keyed by that step."""

    from ..scalar_utils import _first_present_scalar

    owned: Dict[str, Tuple[float, float]] = {}
    for summary, owner in zip(summaries, owners):
        if not owner or owner in owned:
            continue
        low = _first_present_scalar(summary, lower_keys)
        high = _first_present_scalar(summary, upper_keys)
        if low is None or high is None:
            continue
        try:
            owned[owner] = (float(low), float(high))
        except (TypeError, ValueError):
            continue
    return owned


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


_FOOTNOTE_DEFINITION_RE = re.compile(
    r"^\[\^(?P<fid>[^\]]+)\]:\s*(?P<fields>.+)$", re.MULTILINE
)


def footnote_step_ids(bound_manuscript: str) -> Dict[str, str]:
    """Map each numeric footnote id to the step that produced its value.

    ``bind_numeric_values`` has already resolved every bound number to exactly
    one registered claim and printed the owning step in the footnote
    definition. Reading it back is what lets this auditor compare a claim
    against *its own* step instead of against every step in the run.
    """

    mapping: Dict[str, str] = {}
    for match in _FOOTNOTE_DEFINITION_RE.finditer(bound_manuscript or ""):
        for entry in match.group("fields").split(";"):
            name, _, value = entry.strip().partition("=")
            if name.strip() == "step" and value.strip():
                mapping[match.group("fid")] = value.strip()
                break
    return mapping


def _extract_metric_claims(text: str, metric_pattern: str) -> List[float]:
    return [
        value for value, _ in _extract_metric_claims_with_footnote(text, metric_pattern)
    ]


class _ScopedMetric(NamedTuple):
    """What the step a claim cites actually registered for one metric family."""

    #: Values that step registered. Empty when it registered none.
    values: List[float]
    #: The step whose values ``values`` holds — None when it holds none.
    step_id: Optional[str]
    #: The step the footnote resolved to, whether or not it owns the metric.
    #: This is what separates "unbound number" from "bound to the wrong step".
    cited_step: Optional[str]

    @property
    def cited_step_lacks_metric(self) -> bool:
        return bool(self.cited_step) and not self.values


def _scoped_registered_values(
    *,
    summaries: Sequence[Dict[str, Any]],
    summary_owners: Sequence[Optional[str]],
    keys: Sequence[str],
    footnote_steps: Mapping[str, str],
    footnote_id: Optional[str],
) -> _ScopedMetric:
    """Values registered by the step this claim's footnote names, if any.

    Match-any answers "is this number registered *somewhere* in the run?",
    which passes a sentence attributing the sensitivity model's value to the
    primary model. Every metric family gets the same treatment: the AUROC path
    was scoped first and leaving Brier, prevalence and the CI check on
    match-any made the auditor's own contract inconsistent.

    Three outcomes, and they are not the same failure:

    * a resolvable step that owns the metric → scope to it;
    * a resolvable step that owns *no* such metric → the citation is wrong, and
      falling back to match-any is what lets the sensitivity step's Brier score
      vouch for a sentence about the primary model. Reported by the caller as
      its own error, never fallen back from;
    * no resolvable step → ``cited_step`` is None and the caller falls back to
      match-any. That is an unbound number, which the untraced-numeric finding
      already reports.
    """

    cited_step = footnote_steps.get(footnote_id or "") if footnote_id else None
    if not cited_step:
        return _ScopedMetric([], None, None)
    scoped = _all_summary_scalars(
        [
            summary
            for summary, owner in zip(summaries, summary_owners)
            if owner == cited_step
        ],
        keys,
    )
    return _ScopedMetric(scoped, (cited_step if scoped else None), cited_step)


def _cited_step_lacks_metric_finding(
    *,
    metric: str,
    label: str,
    claimed: float,
    cited_step: str,
) -> ValidationFinding:
    """The sentence names a step; that step never registered this metric."""

    return ValidationFinding(
        validator="manuscript_numeric_auditor",
        severity="error",
        message=(
            f"Manuscript {label} claim {claimed:.3g} is footnoted to step "
            f"{cited_step!r}, which registers no {label}. The number may be "
            "correct for a different step, but the sentence attributes it to "
            "this one."
        ),
        detail={
            "metric": metric,
            "claimed": claimed,
            "cited_step": cited_step,
            "reason": "cited_step_does_not_register_metric",
        },
    )


def _extract_metric_claims_with_footnote(
    text: str, metric_pattern: str
) -> List[Tuple[float, Optional[str]]]:
    """Metric claims plus the footnote id the binder attached, when any."""

    claims: List[Tuple[float, Optional[str]]] = []
    clean_text = _strip_manuscript_noise(text)
    # Require a DECIMAL point: a real AUROC/Brier point estimate is always
    # reported with decimals (0.83, not 1). A bare integer near the metric word
    # is a figure/table number, an enumeration, or a step-id digit, never a
    # reported metric — excluding it removes that whole false-positive class
    # while still catching an implausible "1.0"/"0.0" claim written with decimals.
    pattern = re.compile(
        metric_pattern + r"([^0-9]{0,40})([01]\.\d+)(?:\[\^(?P<fid>[^\]]+)\])?",
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
            claims.append((value, match.group("fid")))
    return claims


def _extract_percent_claims_near(
    text: str,
    phrase_pattern: str,
    *,
    skip_stratified_context: bool = False,
) -> List[float]:
    return [
        value
        for value, _ in _extract_percent_claims_near_with_footnote(
            text, phrase_pattern, skip_stratified_context=skip_stratified_context
        )
    ]


def _extract_percent_claims_near_with_footnote(
    text: str,
    phrase_pattern: str,
    *,
    skip_stratified_context: bool = False,
) -> List[Tuple[float, Optional[str]]]:
    claims: List[Tuple[float, Optional[str]]] = []
    clean_text = _strip_manuscript_noise(text)
    pattern = re.compile(
        phrase_pattern
        + r".{0,80}?([0-9]+(?:\.[0-9]+)?)\s*%(?:\[\^(?P<fid>[^\]]+)\])?",
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
            claims.append((value, match.group("fid")))
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
