"""Host claim for the adjusted ordered trend of a binary outcome.

The ordered-trend owner refits the primary association model with the ordered
exposure as one per-level term and publishes a versioned reporting envelope.
This compiler turns that envelope into one secondary association claim -- the
adjusted odds or risk ratio per level increment -- so the Writer reports the
trend through a host-rendered sentence.  It reads only the envelope; a
summary without one exposes no claim.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

ORDERED_TREND_REPORTING_KEY = "reportable_trend_results"

_ESTIMAND_BY_SCALE = {
    "odds_ratio": "adjusted odds ratio per one-level increase across its ordered levels",
    "risk_ratio": "adjusted risk ratio per one-level increase across its ordered levels",
}


class OrderedTrendReporting(BaseModel):
    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)

    schema_version: Literal["easyicu.ordered_trend_reporting/1"]
    exposure: str = Field(min_length=1)
    ordered_levels: list[str] = Field(min_length=3)
    outcome: str = Field(min_length=1)
    analysis_set: Literal["source_aware", "complete_case"]
    adjustment_covariates: list[str]
    effect_scale: Literal["odds_ratio", "risk_ratio"]
    estimate: float = Field(gt=0)
    lower: float = Field(gt=0)
    upper: float = Field(gt=0)
    p_value: float = Field(gt=0, le=1)
    n: int = Field(gt=0, strict=True)
    events: int = Field(ge=0, strict=True)

    @model_validator(mode="after")
    def _coherent(self) -> "OrderedTrendReporting":
        if not self.lower <= self.estimate <= self.upper:
            raise ValueError("trend interval must contain its estimate")
        if self.events > self.n:
            raise ValueError("trend events exceed the analysed records")
        if len(set(self.ordered_levels)) != len(self.ordered_levels):
            raise ValueError("ordered levels must be distinct")
        return self


def ordered_trend_reporting_requests_claims(summary: dict) -> bool:
    return ORDERED_TREND_REPORTING_KEY in summary


def derive_ordered_trend_claim_payloads(summary: dict) -> list[dict]:
    """Return the one secondary claim a validated trend envelope supports."""

    report = OrderedTrendReporting.model_validate(summary[ORDERED_TREND_REPORTING_KEY])
    if report.lower > 1.0:
        direction = "positive"
    elif report.upper < 1.0:
        direction = "negative"
    else:
        direction = "no_clear_association"
    return [
        {
            "claim_id": "adjusted_ordered_trend",
            "claim_type": "association",
            "exposure": report.exposure,
            "outcome": report.outcome,
            "direction": direction,
            "estimand": _ESTIMAND_BY_SCALE[report.effect_scale],
            "population": f"the {report.analysis_set.replace('_', ' ')} analysis set",
            "analysis_role": "secondary",
            "status": "supported",
            "adjusted_for": list(report.adjustment_covariates),
            "point_estimate": report.estimate,
            "interval_lower": report.lower,
            "interval_upper": report.upper,
        }
    ]


__all__ = [
    "ORDERED_TREND_REPORTING_KEY",
    "OrderedTrendReporting",
    "derive_ordered_trend_claim_payloads",
    "ordered_trend_reporting_requests_claims",
]
