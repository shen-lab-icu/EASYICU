"""Strict compiler for host-owned descriptive scientific claim payloads.

The executor owns computation.  This module consumes only its versioned,
digest-bound summary envelope and emits the minimal generic payload accepted by
``ScientificClaimDraft``.  Keeping that translation separate prevents the
association compiler from becoming a second descriptive-analysis executor.
"""

from __future__ import annotations

import json
import math
import statistics
from typing import Any, Mapping

from ..contracts.claim_ceiling import (
    BOUND_TYPED_COHORT_ANALYSIS_SET,
    EXPOSURE_OBSERVED_ANALYSIS_SET,
)
from ..contracts.dependence import PlannedDependenceRequirement

DESCRIPTIVE_CEILING = "descriptive_unadjusted_not_causal"
_ANALYSIS_ROLES = {
    "primary",
    "secondary",
    "sensitivity",
    "exploratory",
    "auxiliary",
}


def _text(payload: Mapping[str, Any], field: str, *, owner: str) -> str:
    value = " ".join(str(payload.get(field) or "").split())
    if not value:
        raise ValueError(f"{owner} cannot be derived without {field!r}")
    return value


def _number(payload: Mapping[str, Any], field: str, *, owner: str) -> float:
    value = payload.get(field)
    if isinstance(value, bool):
        raise ValueError(f"{owner}.{field} must be a finite number")
    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{owner}.{field} must be a finite number") from exc
    if not math.isfinite(number):
        raise ValueError(f"{owner}.{field} must be a finite number")
    return number


def _count(payload: Mapping[str, Any], field: str, *, owner: str) -> int:
    value = payload.get(field)
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"{owner}.{field} must be a non-negative integer")
    return value


def _format(value: float) -> str:
    rendered = f"{value:.6f}".rstrip("0").rstrip(".")
    return "0" if rendered in {"-0", ""} else rendered


def _level_key(value: Any) -> str:
    if value is None or isinstance(value, (str, bool, int, float)):
        if isinstance(value, float) and not math.isfinite(value):
            raise ValueError("descriptive estimate levels must be finite")
        return json.dumps(
            {"type": type(value).__name__, "value": value},
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        )
    raise ValueError("descriptive estimate levels must be JSON scalar values")


def _display_level(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, allow_nan=False)


def _interval(
    payload: Mapping[str, Any], *, owner: str, bounded: bool
) -> tuple[float, float, float, float, str, str, int | None]:
    estimate = _number(payload, "estimate_pct", owner=owner)
    standard_error = _number(payload, "standard_error_pct", owner=owner)
    low = _number(payload, "ci_low_pct", owner=owner)
    high = _number(payload, "ci_high_pct", owner=owner)
    confidence = _number(payload, "confidence_level", owner=owner)
    if low > estimate or estimate > high:
        raise ValueError(f"{owner} estimate must lie inside its confidence interval")
    if standard_error < 0.0:
        raise ValueError(f"{owner}.standard_error_pct cannot be negative")
    if not 0.0 < confidence < 1.0:
        raise ValueError(f"{owner}.confidence_level must lie strictly between 0 and 1")
    if bounded and not (0.0 <= low <= estimate <= high <= 100.0):
        raise ValueError(f"{owner} absolute risk must remain within 0 to 100 percent")
    method = _text(payload, "interval_method", owner=owner)
    covariance = _text(payload, "covariance", owner=owner)
    if covariance in {"cluster_robust", "hc1"} and standard_error <= 0.0:
        raise ValueError(f"{owner} robust standard error must be positive")
    clusters = None
    if payload.get("cluster_count") is not None:
        clusters = _count(payload, "cluster_count", owner=owner)
        if clusters < 2:
            raise ValueError(f"{owner}.cluster_count must be at least two")
    if (covariance == "cluster_robust") != (clusters is not None):
        raise ValueError(f"{owner} covariance and cluster_count are inconsistent")
    return estimate, low, high, confidence, method, covariance, clusters


def _estimand(
    *,
    name: str,
    interval: tuple[float, float, float, float, str, str, int | None],
    unit: str,
) -> str:
    estimate, low, high, confidence, method, covariance, clusters = interval
    cluster_text = f"; patient clusters: {clusters}" if clusters is not None else ""
    return (
        f"{name} was {_format(estimate)} {unit} "
        f"({_format(100.0 * confidence)}% CI, {_format(low)} to {_format(high)}; "
        f"interval: {method}; covariance: {covariance}{cluster_text})"
    )


def _require_interval_arithmetic(
    payload: Mapping[str, Any],
    *,
    owner: str,
    interval: tuple[float, float, float, float, str, str, int | None],
    events: int | None = None,
    denominator: int | None = None,
) -> None:
    """Recompute the declared interval rather than trusting its endpoints."""

    estimate, low, high, confidence, method, _, _ = interval
    standard_error = _number(payload, "standard_error_pct", owner=owner)
    z = statistics.NormalDist().inv_cdf(1.0 - (1.0 - confidence) / 2.0)
    if method == "wilson":
        if events is None or denominator is None or denominator <= 0:
            raise ValueError(f"{owner} Wilson interval lacks events/denominator")
        proportion = events / denominator
        centre = proportion + z * z / (2.0 * denominator)
        spread = z * math.sqrt(
            (
                proportion * (1.0 - proportion)
                + z * z / (4.0 * denominator)
            )
            / denominator
        )
        factor = 1.0 + z * z / denominator
        expected_low = 100.0 * max(0.0, (centre - spread) / factor)
        expected_high = 100.0 * min(1.0, (centre + spread) / factor)
    elif method == "patient_cluster_robust_wald":
        expected_low = max(0.0, estimate - z * standard_error)
        expected_high = min(100.0, estimate + z * standard_error)
    elif method == "linear_probability_wald":
        expected_low = estimate - z * standard_error
        expected_high = estimate + z * standard_error
    else:
        raise ValueError(f"{owner}.interval_method is unsupported")
    if abs(low - expected_low) > 2e-6 or abs(high - expected_high) > 2e-6:
        raise ValueError(f"{owner} confidence interval arithmetic is inconsistent")


def derive_descriptive_claim_payloads(
    summary: Mapping[str, Any],
) -> list[dict[str, Any]]:
    """Validate one versioned descriptive envelope and compile claim payloads."""

    owner = "scientific_claims exposure-outcome distribution"
    if _text(summary, "interpretation_ceiling", owner=owner) != DESCRIPTIVE_CEILING:
        raise ValueError(f"{owner} requires the exact descriptive interpretation ceiling")
    exposure = _text(summary, "exposure", owner=owner)
    outcome = _text(summary, "outcome", owner=owner)
    role = _text(summary, "analysis_role", owner=owner).lower()
    analysis_set = _text(summary, "analysis_set", owner=owner)
    if role not in _ANALYSIS_ROLES:
        raise ValueError(f"{owner} analysis_role is unsupported")
    if analysis_set not in {
        BOUND_TYPED_COHORT_ANALYSIS_SET,
        EXPOSURE_OBSERVED_ANALYSIS_SET,
    }:
        raise ValueError(f"{owner} requires a supported typed analysis_set")
    population = (
        "the bound typed cohort"
        if analysis_set == BOUND_TYPED_COHORT_ANALYSIS_SET
        else "rows with observed exposure in the bound typed cohort"
    )

    estimates = summary.get("descriptive_estimates")
    if not isinstance(estimates, dict):
        raise ValueError(f"{owner} requires a descriptive_estimates object")
    if (
        estimates.get("schema_version")
        != "easyicu.exposure_outcome_descriptive_estimates/1"
    ):
        raise ValueError(f"{owner} descriptive_estimates schema_version is unsupported")
    for field, expected in (
        ("analysis_role", role),
        ("analysis_set", analysis_set),
        ("interpretation_ceiling", DESCRIPTIVE_CEILING),
    ):
        if estimates.get(field) != expected:
            raise ValueError(f"{owner} descriptive_estimates.{field} drifted")

    if "dependence" not in estimates:
        raise ValueError(f"{owner} descriptive_estimates lacks dependence authority")
    raw_dependence = estimates.get("dependence")
    dependence = None
    if raw_dependence is not None:
        if not isinstance(raw_dependence, Mapping):
            raise ValueError(f"{owner} descriptive_estimates.dependence must be an object")
        dependence = PlannedDependenceRequirement.model_validate(raw_dependence)

    absolute_risks = estimates.get("outcome_absolute_risks")
    if not isinstance(absolute_risks, list) or not absolute_risks:
        raise ValueError(f"{owner} requires at least one outcome absolute risk")
    claims: list[dict[str, Any]] = []
    levels: dict[int, tuple[str, Any, int, float, tuple[float, float, float, float, str, str, int | None]]] = {}
    seen_level_keys: set[str] = set()
    confidence_levels: set[float] = set()
    for position, raw in enumerate(absolute_risks):
        item_owner = f"{owner}.outcome_absolute_risks[{position}]"
        if not isinstance(raw, dict):
            raise ValueError(f"{item_owner} must be an object")
        index = _count(raw, "level_index", owner=item_owner)
        if index in levels:
            raise ValueError(f"{owner} contains duplicate absolute-risk level_index")
        level = raw.get("level")
        level_key = _level_key(level)
        if level_key in seen_level_keys:
            raise ValueError(f"{owner} contains duplicate typed absolute-risk levels")
        seen_level_keys.add(level_key)
        events = _count(raw, "events", owner=item_owner)
        denominator = _count(raw, "denominator", owner=item_owner)
        if denominator <= 0 or events > denominator:
            raise ValueError(f"{item_owner} events/denominator are inconsistent")
        interval = _interval(raw, owner=item_owner, bounded=True)
        if dependence is None:
            if interval[4] != "wilson" or interval[5] != "binomial_independent":
                raise ValueError(
                    f"{item_owner} covariance contradicts the independent-row authority"
                )
        elif interval[4] != "patient_cluster_robust_wald" or interval[5] != "cluster_robust":
            raise ValueError(
                f"{item_owner} covariance contradicts the typed patient dependence authority"
            )
        if interval[6] is not None and interval[6] > denominator:
            raise ValueError(f"{item_owner}.cluster_count cannot exceed its denominator")
        _require_interval_arithmetic(
            raw,
            owner=item_owner,
            interval=interval,
            events=events,
            denominator=denominator,
        )
        if abs(interval[0] - 100.0 * events / denominator) > 1e-5:
            raise ValueError(f"{item_owner} estimate does not equal events/denominator")
        confidence_levels.add(interval[3])
        levels[index] = (level_key, level, denominator, interval[0], interval)
        claims.append(
            {
                "schema_version": "easyicu.scientific_claim/2",
                "claim_id": f"observed_absolute_risk_level_{index}",
                "claim_type": "descriptive_absolute_risk",
                "exposure": f"{exposure}={_display_level(level)}",
                "outcome": outcome,
                "direction": "descriptive_only",
                "estimand": _estimand(
                    name="observed absolute risk", interval=interval, unit="percent"
                ),
                "population": population,
                "analysis_role": role,
                "status": "supported",
            }
        )
    if set(levels) != set(range(len(absolute_risks))):
        raise ValueError(
            f"{owner} absolute-risk level_index values must be contiguous from zero"
        )

    contrast = estimates.get("risk_difference")
    if contrast is None:
        return claims
    if not isinstance(contrast, dict):
        raise ValueError(f"{owner}.risk_difference must be an object or null")
    contrast_owner = f"{owner}.risk_difference"
    if contrast.get("direction") != "comparison_minus_reference":
        raise ValueError(
            f"{contrast_owner} direction must be comparison_minus_reference"
        )
    if contrast.get("interpretation_ceiling") != DESCRIPTIVE_CEILING:
        raise ValueError(f"{contrast_owner} interpretation ceiling drifted")
    reference_index = _count(contrast, "reference_level_index", owner=contrast_owner)
    comparison_index = _count(contrast, "comparison_level_index", owner=contrast_owner)
    if reference_index == comparison_index:
        raise ValueError(f"{contrast_owner} must compare two different levels")
    try:
        (
            reference_key,
            reference_level,
            reference_n,
            reference_risk,
            reference_interval,
        ) = levels[reference_index]
        (
            comparison_key,
            comparison_level,
            comparison_n,
            comparison_risk,
            comparison_interval,
        ) = levels[comparison_index]
    except KeyError as exc:
        raise ValueError(
            f"{contrast_owner} refers to an absent absolute-risk level"
        ) from exc
    if _level_key(contrast.get("reference_level")) != reference_key or _level_key(
        contrast.get("comparison_level")
    ) != comparison_key:
        raise ValueError(f"{contrast_owner} typed level values drifted")
    interval = _interval(contrast, owner=contrast_owner, bounded=False)
    if confidence_levels != {interval[3]}:
        raise ValueError(f"{contrast_owner} confidence level differs from absolute risks")
    n = _count(contrast, "n", owner=contrast_owner)
    if n != reference_n + comparison_n:
        raise ValueError(
            f"{contrast_owner}.n must equal the two absolute-risk denominators"
        )
    expected_difference = comparison_risk - reference_risk
    if abs(interval[0] - expected_difference) > 1e-5:
        raise ValueError(
            f"{contrast_owner} estimate must equal comparison minus reference risk"
        )
    compatible_covariance = False
    if dependence is None:
        compatible_covariance = interval[5] == "hc1" and interval[6] is None
    elif interval[5] == "cluster_robust" and interval[6] is not None:
        absolute_cluster_counts = [
            count
            for count in (reference_interval[6], comparison_interval[6])
            if count is not None
        ]
        # The contrast is fitted on the union of the two level-specific rows.
        # Its patient count can exceed either stratum's count, but cannot be
        # smaller than either one or larger than its own analysis n.
        compatible_covariance = bool(
            len(absolute_cluster_counts) == 2
            and interval[6] >= max(absolute_cluster_counts)
            and interval[6] <= n
        )
    if interval[4] != "linear_probability_wald" or not compatible_covariance:
        raise ValueError(
            f"{contrast_owner} covariance authority differs from the absolute risks"
        )
    _require_interval_arithmetic(
        contrast,
        owner=contrast_owner,
        interval=interval,
    )
    claims.append(
        {
            "schema_version": "easyicu.scientific_claim/2",
            "claim_id": "prespecified_unadjusted_risk_difference",
            "claim_type": "descriptive_risk_difference",
            "exposure": (
                f"{exposure}={_display_level(comparison_level)} versus "
                f"{exposure}={_display_level(reference_level)}"
            ),
            "outcome": outcome,
            "direction": "descriptive_only",
            "estimand": _estimand(
                name=(
                    "prespecified unadjusted risk difference "
                    "(comparison minus reference)"
                ),
                interval=interval,
                unit="percentage points",
            ),
            "population": population,
            "analysis_role": role,
            "status": "supported",
        }
    )
    return claims


__all__ = ["DESCRIPTIVE_CEILING", "derive_descriptive_claim_payloads"]
