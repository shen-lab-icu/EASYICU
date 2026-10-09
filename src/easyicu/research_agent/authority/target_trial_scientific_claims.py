"""Host claims for the estimates of the signed target trial suite.

The suite (``execution.runners.target_trial_executor``) estimates the risk of
the outcome by the horizon under each strategy of an emulated target trial,
their difference and their ratio.  The stabilized weights as estimated give
the primary result; the weights truncated at the host's prespecified
percentiles give a sensitivity analysis; each comes with a bootstrap
percentile interval.  The suite opts in with a versioned
``easyicu.target_trial_reporting/1`` envelope.  Every claim is a
``target_trial_estimate``, whose one sentence is the fixed template of
:mod:`.target_trial_claim_terms`: it names the emulation's assumptions before
the estimate.  The unweighted estimates have no interval and are no claim;
the projection states them, the weight diagnostics, positivity, balance and
the ICU-exit weighting in neutral numeric sentences under the primary
results, and places the primary contrast in the abstract.
"""

from __future__ import annotations

import math
from typing import Any, Literal, Mapping, Optional

from pydantic import BaseModel, ConfigDict, Field, model_validator

from ..contracts.manuscript_result_structure import DEFAULT_PRIMARY_RESULT_HEADING
from .target_trial_claim_terms import (
    TARGET_TRIAL_CLAIM_TERMS_SCHEMA_VERSION,
    TargetTrialClaimTerms,
)

TARGET_TRIAL_REPORTING_KEY = "reportable_target_trial_results"
TARGET_TRIAL_REPORTING_SCHEMA_VERSION = "easyicu.target_trial_reporting/1"
TARGET_TRIAL_CLAIM_SCHEMA_VERSION = "easyicu.scientific_claim/5"
#: The primary claims, in reporting order; the contrast leads the abstract.
PRIMARY_CLAIM_IDS = (
    "strategy_risk_difference",
    "strategy_risk_ratio",
    "initiate_strategy_risk",
    "defer_strategy_risk",
)
SENSITIVITY_CLAIM_IDS = (
    "truncated_weight_risk_difference",
    "truncated_weight_risk_ratio",
)
_ABSTRACT_CLAIM_IDS = PRIMARY_CLAIM_IDS[:2]


class _Closed(BaseModel):
    model_config = ConfigDict(
        extra="forbid", frozen=True, strict=True, allow_inf_nan=False
    )


class _Estimate(_Closed):
    """One estimate with its bootstrap percentile interval."""

    estimate: float
    ci_low: float
    ci_high: float

    @model_validator(mode="after")
    def _interval_contains_estimate(self) -> "_Estimate":
        if not self.ci_low <= self.estimate <= self.ci_high:
            raise ValueError("a target trial interval must contain its estimate")
        return self


class _Weights(_Closed):
    """The weights the clones carry past the grace period, before truncation."""

    mean: float = Field(gt=0.0)
    maximum: float = Field(gt=0.0)
    p99: float = Field(gt=0.0)
    effective_sample_size: float = Field(gt=0.0)


class _Arm(_Closed):
    label: str = Field(min_length=1)
    n_clones: int = Field(ge=1)
    n_followed_past_grace: int = Field(ge=1)
    n_events: int = Field(ge=0)
    risk_percent: _Estimate
    risk_truncated_percent: _Estimate
    risk_unweighted_percent: float = Field(ge=0.0, le=100.0)
    weights: _Weights


class _EValue(_Closed):
    point: float = Field(ge=1.0)
    interval_bound: Optional[float] = Field(default=None, ge=1.0)


class _Positivity(_Closed):
    window_percent: tuple[float, float]
    n_outside: int = Field(ge=0)
    percent_outside: float = Field(ge=0.0, le=100.0)


class _Balance(_Closed):
    max_abs_smd_unweighted: Optional[float]
    max_abs_smd_weighted: Optional[float]
    flag_threshold: float = Field(gt=0.0)
    n_flagged_weighted: int = Field(ge=0)


class _IcuExit(_Closed):
    percent_in_grace: float = Field(ge=0.0, le=100.0)
    model_form: Literal["none", "hour_terms_and_covariates", "hour_terms_only"]


class _Bootstrap(_Closed):
    resamples: int = Field(ge=1)
    n_failed: int = Field(ge=0)
    confidence_level: float = Field(gt=0.5, lt=1.0)
    interval_method: Literal["bootstrap_percentile"]


class TargetTrialReporting(_Closed):
    """The suite's reporting envelope."""

    schema_version: Literal["easyicu.target_trial_reporting/1"]
    execution_owner: Literal["target_trial_executor_v1"]
    interpretation_ceiling: Literal["per_protocol_effect_under_emulation_assumptions"]
    evidence_ceiling: Literal["analysis_only"]
    treatment: str = Field(min_length=1)
    initiate_label: str = Field(min_length=1)
    defer_label: str = Field(min_length=1)
    outcome: str = Field(min_length=1)
    outcome_label: str = Field(min_length=1)
    analysis_unit: str = Field(min_length=1)
    population: str = Field(min_length=1)
    time_zero_hours: int = Field(ge=1)
    grace_period_hours: int = Field(ge=1)
    horizon_days: int = Field(gt=0)
    adjustment_columns: list[str] = Field(min_length=1)
    adjustment_labels: list[str] = Field(min_length=1)
    n_eligible: int = Field(ge=1)
    arms: dict[Literal["initiate", "defer"], _Arm]
    risk_difference_percentage_points: _Estimate
    risk_ratio: _Estimate
    truncated_risk_difference_percentage_points: _Estimate
    truncated_risk_ratio: _Estimate
    unweighted_risk_difference_percentage_points: float
    unweighted_risk_ratio: Optional[float]
    weight_truncation_percentiles: tuple[float, float]
    e_value: Optional[_EValue]
    positivity: _Positivity
    balance: _Balance
    icu_exit: _IcuExit
    late_starts: int = Field(ge=0)
    deaths_timed_by_calendar_day: int = Field(ge=0)
    other_deaths_timed_by_calendar_day: int = Field(ge=0)
    deaths_placed_at_icu_exit: int = Field(ge=0)
    bootstrap: _Bootstrap
    manuscript_projection: dict[str, Any]

    @model_validator(mode="after")
    def _coherent(self) -> "TargetTrialReporting":
        if set(self.arms) != {"initiate", "defer"}:
            raise ValueError("a target trial reports both strategies")
        if len(self.adjustment_columns) != len(self.adjustment_labels):
            raise ValueError("every adjustment column has its label")
        initiate, defer = self.arms["initiate"], self.arms["defer"]
        for difference, first, second in (
            (
                self.risk_difference_percentage_points,
                initiate.risk_percent,
                defer.risk_percent,
            ),
            (
                self.truncated_risk_difference_percentage_points,
                initiate.risk_truncated_percent,
                defer.risk_truncated_percent,
            ),
        ):
            if not math.isclose(
                difference.estimate,
                first.estimate - second.estimate,
                rel_tol=0.0,
                abs_tol=1e-9,
            ):
                raise ValueError("a risk difference is the strategies' difference")
        for ratio, first, second in (
            (self.risk_ratio, initiate.risk_percent, defer.risk_percent),
            (
                self.truncated_risk_ratio,
                initiate.risk_truncated_percent,
                defer.risk_truncated_percent,
            ),
        ):
            if not (
                second.estimate > 0.0
                and math.isclose(
                    ratio.estimate,
                    first.estimate / second.estimate,
                    rel_tol=1e-9,
                    abs_tol=0.0,
                )
            ):
                raise ValueError("a risk ratio is the strategies' ratio")
        if initiate.label != self.initiate_label or defer.label != self.defer_label:
            raise ValueError("each strategy keeps its label")
        if self.deaths_placed_at_icu_exit > (
            self.deaths_timed_by_calendar_day + self.other_deaths_timed_by_calendar_day
        ):
            raise ValueError("a death placed at the ICU exit was timed by its day")
        return self

    def terms(
        self, *, measure: str, strategy: Optional[str] = None, truncated: bool
    ) -> TargetTrialClaimTerms:
        return TargetTrialClaimTerms(
            schema_version=TARGET_TRIAL_CLAIM_TERMS_SCHEMA_VERSION,
            measure=measure,
            strategy=strategy,
            weighting="stabilized_truncated" if truncated else "stabilized",
            initiate_label=self.initiate_label,
            defer_label=self.defer_label,
            outcome_label=self.outcome_label,
            horizon_days=self.horizon_days,
            analysis_unit_label=self.analysis_unit,
            truncation_percentiles=(
                tuple(self.weight_truncation_percentiles) if truncated else None
            ),
        )


def target_trial_reporting_requests_claims(summary: Mapping[str, Any]) -> bool:
    reporting = summary.get(TARGET_TRIAL_REPORTING_KEY)
    return (
        isinstance(reporting, Mapping)
        and reporting.get("schema_version") == TARGET_TRIAL_REPORTING_SCHEMA_VERSION
    )


def _direction(estimate: _Estimate, null: float) -> str:
    if estimate.ci_low > null:
        return "positive"
    if estimate.ci_high < null:
        return "negative"
    return "no_clear_association"


def derive_target_trial_claim_payloads(
    summary: Mapping[str, Any],
) -> list[dict[str, Any]]:
    """Compile the suite's estimates into claims: the primary ones, then the sensitivity ones."""

    if summary.get("status") != "ok":
        raise ValueError("target trial claims require a completed owner summary")
    reporting = TargetTrialReporting.model_validate(
        summary[TARGET_TRIAL_REPORTING_KEY], strict=False
    )
    level = reporting.bootstrap.confidence_level

    def claim(
        claim_id: str,
        terms: TargetTrialClaimTerms,
        value: _Estimate,
        *,
        role: str,
        direction: str,
    ) -> dict[str, Any]:
        return {
            "schema_version": TARGET_TRIAL_CLAIM_SCHEMA_VERSION,
            "claim_id": claim_id,
            "claim_type": "target_trial_estimate",
            "exposure": terms.subject,
            "outcome": reporting.outcome,
            "direction": direction,
            "estimand": terms.estimand,
            "population": f"the eligible {reporting.analysis_unit}",
            "analysis_role": role,
            "status": "supported",
            "adjusted_for": list(reporting.adjustment_columns),
            "point_estimate": value.estimate,
            "interval_lower": value.ci_low,
            "interval_upper": value.ci_high,
            "confidence_level": level,
            "interval_method": "bootstrap_percentile",
            "effect_scale": terms.effect_scale,
            "target_trial_terms": terms.model_dump(mode="json"),
        }

    payloads = [
        claim(
            "strategy_risk_difference",
            reporting.terms(measure="risk_difference", truncated=False),
            reporting.risk_difference_percentage_points,
            role="primary",
            direction=_direction(reporting.risk_difference_percentage_points, 0.0),
        ),
        claim(
            "strategy_risk_ratio",
            reporting.terms(measure="risk_ratio", truncated=False),
            reporting.risk_ratio,
            role="primary",
            direction=_direction(reporting.risk_ratio, 1.0),
        ),
    ]
    for strategy in ("initiate", "defer"):
        payloads.append(
            claim(
                f"{strategy}_strategy_risk",
                reporting.terms(
                    measure="strategy_risk", strategy=strategy, truncated=False
                ),
                reporting.arms[strategy].risk_percent,
                role="primary",
                direction="descriptive_only",
            )
        )
    payloads.extend(
        [
            claim(
                "truncated_weight_risk_difference",
                reporting.terms(measure="risk_difference", truncated=True),
                reporting.truncated_risk_difference_percentage_points,
                role="sensitivity",
                direction=_direction(
                    reporting.truncated_risk_difference_percentage_points, 0.0
                ),
            ),
            claim(
                "truncated_weight_risk_ratio",
                reporting.terms(measure="risk_ratio", truncated=True),
                reporting.truncated_risk_ratio,
                role="sensitivity",
                direction=_direction(reporting.truncated_risk_ratio, 1.0),
            ),
        ]
    )
    return payloads


def _fragment_claim(claim_id: str, fragments: list[dict[str, str]]) -> dict[str, Any]:
    return {
        "claim_id": claim_id,
        "targets": [
            {"kind": "markdown_heading", "label": DEFAULT_PRIMARY_RESULT_HEADING}
        ],
        "fragments": fragments,
    }


def _text(value: str) -> dict[str, str]:
    return {"text": value}


def _number(path: str, spec: str) -> dict[str, str]:
    return {"numeric_path": path, "format_spec": spec}


_EXIT_FORM_WORDS = {
    "hour_terms_and_covariates": "a model with the hour terms and the confounders",
    "hour_terms_only": "a model with the hour terms only, because these exits were "
    "too few for the confounders",
}


def build_target_trial_manuscript_projection(
    reportable: Mapping[str, Any],
) -> dict[str, object]:
    """The reporting projection the signed target trial suite owns.

    The primary contrast's two claims lead the abstract Results.  Under the
    primary results, neutral numeric sentences state what every estimate
    rests on: the weights each strategy carried past the grace period (mean,
    maximum, 99th percentile and the effective sample size before
    truncation), positivity, balance, the share of stays censored at ICU exit
    and the form of the model that weighted it, and the unweighted contrast,
    which has no interval.  No sentence here names a cause: the estimates'
    only causal words are their claims' template.
    """

    initiate = str(reportable["initiate_label"])
    defer = str(reportable["defer_label"])
    weights = [
        _text("Past the grace period, the stabilized weights had a mean of "),
        _number("arms.initiate.weights.mean", ".3f"),
        _text(" (maximum "),
        _number("arms.initiate.weights.maximum", ".3f"),
        _text("; 99th percentile "),
        _number("arms.initiate.weights.p99", ".3f"),
        _text(f") in the {initiate} strategy and "),
        _number("arms.defer.weights.mean", ".3f"),
        _text(" (maximum "),
        _number("arms.defer.weights.maximum", ".3f"),
        _text("; 99th percentile "),
        _number("arms.defer.weights.p99", ".3f"),
        _text(f") in the {defer} strategy, with effective sample sizes of "),
        _number("arms.initiate.weights.effective_sample_size", ".1f"),
        _text(" and "),
        _number("arms.defer.weights.effective_sample_size", ".1f"),
        _text(" before truncation."),
    ]
    positivity = [
        _text("Of the eligible stays, "),
        _number("positivity.percent_outside", ".2f"),
        _text(
            "% had a modelled probability of starting within the grace period outside "
        ),
        _number("positivity.window_percent[0]", ".1f"),
        _text("% to "),
        _number("positivity.window_percent[1]", ".1f"),
        _text("%, and "),
        _number("icu_exit.percent_in_grace", ".2f"),
        _text("% left the ICU within the grace period before starting."),
    ]
    claims: list[dict[str, Any]] = [
        _fragment_claim("target_trial_weight_diagnostics", weights),
        _fragment_claim("target_trial_positivity", positivity),
    ]
    balance = reportable.get("balance") or {}
    if (
        balance.get("max_abs_smd_unweighted") is not None
        and balance.get("max_abs_smd_weighted") is not None
    ):
        claims.append(
            _fragment_claim(
                "target_trial_balance",
                [
                    _text(
                        "Against the eligible population, the largest absolute "
                        "standardized mean difference of a confounder among the "
                        "clones followed through the grace period was "
                    ),
                    _number("balance.max_abs_smd_unweighted", ".3f"),
                    _text(" without weights and "),
                    _number("balance.max_abs_smd_weighted", ".3f"),
                    _text(" with them."),
                ],
            )
        )
    form = str((reportable.get("icu_exit") or {}).get("model_form") or "")
    if form in _EXIT_FORM_WORDS:
        claims.append(
            _fragment_claim(
                "target_trial_icu_exit_weighting",
                [
                    _text(
                        "The censoring of clones at ICU exit within the grace "
                        f"period was weighted by {_EXIT_FORM_WORDS[form]}."
                    )
                ],
            )
        )
    claims.append(
        _fragment_claim(
            "target_trial_unweighted_contrast",
            [
                _text("Without weights, the difference in risk was "),
                _number("unweighted_risk_difference_percentage_points", ".2f"),
                _text(f" percentage points, {initiate} minus {defer}."),
            ],
        )
    )
    e_value = reportable.get("e_value")
    if isinstance(e_value, Mapping) and e_value.get("point") is not None:
        fragments = [
            _text("The E-value of the risk ratio was "),
            _number("e_value.point", ".2f"),
        ]
        if e_value.get("interval_bound") is not None:
            fragments += [
                _text(", and "),
                _number("e_value.interval_bound", ".2f"),
                _text(" for its confidence limit nearer the null"),
            ]
        fragments.append(_text("."))
        claims.append(_fragment_claim("target_trial_e_value", fragments))
    abstract = {"kind": "abstract_label", "label": "Results"}
    claims.extend(
        {
            "claim_id": f"abstract_{claim_id}",
            "targets": [abstract],
            "scientific_claim_id": claim_id,
        }
        for claim_id in _ABSTRACT_CLAIM_IDS
    )
    return {"schema_version": "easyicu.manuscript_projection/2", "claims": claims}


__all__ = [
    "PRIMARY_CLAIM_IDS",
    "SENSITIVITY_CLAIM_IDS",
    "TARGET_TRIAL_CLAIM_SCHEMA_VERSION",
    "TARGET_TRIAL_REPORTING_KEY",
    "TARGET_TRIAL_REPORTING_SCHEMA_VERSION",
    "TargetTrialReporting",
    "build_target_trial_manuscript_projection",
    "derive_target_trial_claim_payloads",
    "target_trial_reporting_requests_claims",
]
