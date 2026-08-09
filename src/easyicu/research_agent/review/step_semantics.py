"""Compact, deterministic scientific summary consumed by ``CriticAgent``.

This reviewer owns interpretation of already-produced semantic receipts.  It
does not inspect raw patient rows, select a method, or duplicate the execution
gates that created those receipts.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal, Mapping

ScientificIssueSeverity = Literal["warning", "error"]
CriticDecisionStatus = Literal["pass", "needs_revision", "blocked"]


@dataclass(frozen=True)
class ScientificReviewMetric:
    """One bounded numeric fact copied from a host-owned semantic receipt."""

    variable: str
    semantics: str
    n_total: int
    eligible_n: int | None = None
    not_applicable_n: int | None = None
    observed_n: int | None = None
    missing_n: int | None = None
    event_present_n: int | None = None
    event_absent_n: int | None = None
    before_origin_n: int | None = None
    contradictory_n: int | None = None


@dataclass(frozen=True)
class ScientificReviewIssue:
    """Attributable issue emitted from a compact semantic receipt."""

    code: str
    severity: ScientificIssueSeverity
    message: str
    repair: str


@dataclass(frozen=True)
class StepScientificReviewSummary:
    """Bounded Critic input independent of files and broad mutable context."""

    metrics: tuple[ScientificReviewMetric, ...]
    issues: tuple[ScientificReviewIssue, ...]


@dataclass(frozen=True)
class StepCriticDecision:
    """Deterministic review decision returned to the agent adapter."""

    status: CriticDecisionStatus
    concerns: tuple[str, ...]
    semantic_repairs: tuple[str, ...]


def _integer(value: Any) -> int | None:
    if isinstance(value, bool):
        return None
    try:
        numeric = int(value)
    except (TypeError, ValueError):
        return None
    return numeric if numeric >= 0 else None


def _issue(
    code: str,
    severity: ScientificIssueSeverity,
    message: str,
    repair: str,
) -> ScientificReviewIssue:
    return ScientificReviewIssue(
        code=code,
        severity=severity,
        message=f"[scientific_semantics:{code}] {message}",
        repair=repair,
    )


def _positive_only_metric(
    variable: str,
    audit: Mapping[str, Any],
) -> tuple[ScientificReviewMetric, list[ScientificReviewIssue]]:
    total = _integer(audit.get("n_total"))
    present = _integer(audit.get("event_present_n"))
    absent = _integer(audit.get("event_absent_n"))
    contradiction_keys = (
        "invalid_pair_n",
        "discordant_n",
        "representative_invalid_n",
        "positive_representative_missing_n",
        "negative_representative_positive_n",
    )
    contradictions = [_integer(audit.get(key)) for key in contradiction_keys]
    issues: list[ScientificReviewIssue] = []
    if total is None or present is None or absent is None:
        issues.append(
            _issue(
                "event_partition_incomplete",
                "error",
                f"{variable} lacks a complete present/absent event partition.",
                "Regenerate the host event-presence receipt before accepting missingness.",
            )
        )
        total = total or 0
    elif present + absent != total:
        issues.append(
            _issue(
                "event_partition_not_closed",
                "error",
                f"{variable} event counts do not sum to n_total.",
                "Reconcile every row into exactly one present/absent event state.",
            )
        )
    known_contradictions = [value for value in contradictions if value is not None]
    contradictory_n = sum(known_contradictions) if known_contradictions else None
    if contradictory_n:
        issues.append(
            _issue(
                "event_receipt_contradictory",
                "error",
                f"{variable} has {contradictory_n} contradictory event rows.",
                "Resolve the source triad contradiction; do not relabel it as missingness.",
            )
        )
    return (
        ScientificReviewMetric(
            variable=variable,
            semantics="positive_only_event",
            n_total=total,
            event_present_n=present,
            event_absent_n=absent,
            contradictory_n=contradictory_n,
        ),
        issues,
    )


def _conditional_time_metric(
    variable: str,
    audit: Mapping[str, Any],
) -> tuple[ScientificReviewMetric, list[ScientificReviewIssue]]:
    total = _integer(audit.get("n_total"))
    eligible = _integer(audit.get("eligible_event_n"))
    not_applicable = _integer(audit.get("not_applicable_event_absent_n"))
    observed = _integer(audit.get("observed_event_time_n"))
    missing = _integer(audit.get("missing_event_time_n"))
    before_origin = _integer(audit.get("before_origin_n"))
    contradictory = _integer(audit.get("contradictory_event_absent_with_time_n"))
    issues: list[ScientificReviewIssue] = []
    if any(
        value is None
        for value in (total, eligible, not_applicable, observed, missing)
    ):
        issues.append(
            _issue(
                "conditional_time_partition_incomplete",
                "error",
                f"{variable} lacks a complete eligible/not-applicable time partition.",
                "Regenerate the host conditional-event-time receipt.",
            )
        )
        total = total or 0
    else:
        if eligible + not_applicable != total:
            issues.append(
                _issue(
                    "conditional_time_denominator_not_closed",
                    "error",
                    f"{variable} eligibility counts do not sum to n_total.",
                    "Partition every row by event eligibility before reporting missingness.",
                )
            )
        if observed + missing != eligible:
            issues.append(
                _issue(
                    "conditional_time_eligible_count_not_closed",
                    "error",
                    f"{variable} observed and missing event times do not sum to eligible events.",
                    "Compute event-time missingness only inside the event-positive denominator.",
                )
            )
    if contradictory:
        issues.append(
            _issue(
                "event_absent_with_time",
                "error",
                f"{variable} has {contradictory} event-absent rows with an event time.",
                "Resolve the contradictory event status and time before analysis.",
            )
        )
    return (
        ScientificReviewMetric(
            variable=variable,
            semantics="conditional_event_time",
            n_total=total,
            eligible_n=eligible,
            not_applicable_n=not_applicable,
            observed_n=observed,
            missing_n=missing,
            before_origin_n=before_origin,
            contradictory_n=contradictory,
        ),
        issues,
    )


def _interval_method_issues(
    step_summary: Mapping[str, Any],
) -> list[ScientificReviewIssue]:
    raw_contracts = step_summary.get("model_contracts")
    contracts = (
        [raw_contracts]
        if isinstance(raw_contracts, Mapping)
        else list(raw_contracts)
        if isinstance(raw_contracts, list)
        else []
    )
    diagnostics = step_summary.get("model_diagnostics")
    if isinstance(diagnostics, Mapping):
        contracts.append(diagnostics)
    issues: list[ScientificReviewIssue] = []
    for contract in contracts[:16]:
        if not isinstance(contract, Mapping):
            continue
        fit_method = str(contract.get("fit_method") or "").lower()
        interval_method = str(contract.get("interval_method") or "").lower()
        if "statsmodels" in fit_method and "profile_normal" in interval_method:
            issues.append(
                _issue(
                    "confidence_interval_method_mislabeled",
                    "warning",
                    "A default statsmodels interval is labeled profile likelihood.",
                    "Relabel the unchanged interval as Wald/asymptotic normal.",
                )
            )
            break
    return issues


def summarize_step_scientific_semantics(
    step_summary: Mapping[str, Any],
) -> StepScientificReviewSummary:
    """Build a bounded semantic summary and its deterministic review issues."""

    metrics: list[ScientificReviewMetric] = []
    issues: list[ScientificReviewIssue] = []
    interpretation = str(step_summary.get("interpretation_class") or "").strip()
    raw_audits = step_summary.get("observation_semantics_audit")
    if interpretation == "missingness_measurement_audit" and not isinstance(
        raw_audits, Mapping
    ):
        issues.append(
            _issue(
                "observation_semantics_receipt_missing",
                "error",
                "A missingness audit has no typed observation-semantics receipt.",
                "Regenerate missingness with event, measurement, and conditional-time semantics.",
            )
        )
    if isinstance(raw_audits, Mapping):
        for raw_variable, raw_audit in list(raw_audits.items())[:32]:
            if not isinstance(raw_audit, Mapping):
                continue
            variable = str(raw_variable)
            semantics = str(
                raw_audit.get("observation_semantics")
                or raw_audit.get("indicator_semantics")
                or ""
            )
            if semantics == "binary_event_presence":
                metric, found = _positive_only_metric(variable, raw_audit)
            elif semantics == "conditional_event_time":
                metric, found = _conditional_time_metric(variable, raw_audit)
            else:
                continue
            metrics.append(metric)
            issues.extend(found)

    temporal = step_summary.get("temporal_validity_audit")
    if isinstance(temporal, Mapping) and str(temporal.get("status") or "") == "blocked":
        if not any(issue.code == "event_time_before_origin" for issue in issues):
            issues.append(
                _issue(
                    "temporal_validity_blocked",
                    "error",
                    "The host temporal-validity receipt is blocked.",
                    "Resolve every temporal reason code before accepting the step.",
                )
            )
    issues.extend(_interval_method_issues(step_summary))
    deduplicated = {
        (issue.code, issue.message): issue
        for issue in issues
    }
    return StepScientificReviewSummary(
        metrics=tuple(metrics),
        issues=tuple(deduplicated.values()),
    )


def decide_step_scientific_review(
    *,
    step_summary: Mapping[str, Any],
    deterministic_findings: tuple[str, ...],
    evidence_present: bool,
) -> StepCriticDecision:
    """Combine typed semantic receipts with deterministic gate findings."""

    summary = summarize_step_scientific_semantics(step_summary)
    concerns = [message for message in deterministic_findings if message]
    concerns.extend(issue.message for issue in summary.issues)
    concerns.extend(
        (
            "[scientific_semantics:event_time_before_origin_reported] "
            f"{metric.variable} reports {metric.before_origin_n} event times "
            "before the declared origin; timing analyses must exclude or "
            "resolve them under the reviewed protocol."
        )
        for metric in summary.metrics
        if metric.before_origin_n
    )
    if not evidence_present:
        concerns.append("No evidence refs were registered for this step.")
        status: CriticDecisionStatus = "blocked"
    elif any(issue.severity == "error" for issue in summary.issues):
        status = "blocked"
    elif deterministic_findings or summary.issues:
        status = "needs_revision"
    else:
        status = "pass"
    repairs = tuple(dict.fromkeys(issue.repair for issue in summary.issues))
    return StepCriticDecision(
        status=status,
        concerns=tuple(concerns),
        semantic_repairs=repairs,
    )


__all__ = [
    "ScientificReviewIssue",
    "ScientificReviewMetric",
    "StepCriticDecision",
    "StepScientificReviewSummary",
    "decide_step_scientific_review",
    "summarize_step_scientific_semantics",
]
