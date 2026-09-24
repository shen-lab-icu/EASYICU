"""Host claims for a prespecified sensitivity refit of the primary association.

The binary sensitivity executor refits the primary adjusted model under one
reviewed variant and publishes a versioned reporting envelope.  This compiler
turns that envelope into one association claim with the sensitivity role, so
the Writer reports the refit through a host-rendered sentence and Results
placement puts it under the sensitivity subsection.  It reads only the
envelope; a summary without one exposes no claim.
"""

from __future__ import annotations

import re
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

_POPULATION_BY_STRATEGY = {
    "functional_form": (
        "the {analysis_set} analysis set with {covariate} modelled by a "
        "restricted cubic spline"
    ),
    "first_stay": (
        "the {analysis_set} analysis set restricted to the first ICU stay of "
        "each patient"
    ),
}


class BinarySensitivityReporting(BaseModel):
    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)

    schema_version: Literal["easyicu.binary_sensitivity_reporting/1"]
    analysis_id: str = Field(min_length=1)
    strategy: Literal["functional_form", "first_stay"]
    exposure: str = Field(min_length=1)
    outcome: str = Field(min_length=1)
    analysis_set: Literal["source_aware", "complete_case"]
    adjustment_covariates: list[str]
    covariate: str | None = Field(default=None, min_length=1)
    effect_scale: Literal["odds_ratio"]
    estimate: float = Field(gt=0)
    lower: float = Field(gt=0)
    upper: float = Field(gt=0)
    n: int = Field(gt=0, strict=True)
    events: int = Field(ge=0, strict=True)

    @model_validator(mode="after")
    def _coherent(self) -> "BinarySensitivityReporting":
        if not self.lower <= self.estimate <= self.upper:
            raise ValueError("sensitivity interval must contain its estimate")
        if self.events > self.n:
            raise ValueError("sensitivity events exceed the analysed records")
        if (self.strategy == "functional_form") != (self.covariate is not None):
            raise ValueError(
                "only a functional-form refit names the covariate it reshaped"
            )
        return self


def derive_sensitivity_claim_payloads(summary: dict) -> list[dict]:
    """Return the one claim payload a validated sensitivity envelope supports."""

    report = BinarySensitivityReporting.model_validate(
        summary["reportable_sensitivity_results"]
    )
    if report.lower > 1.0:
        direction = "positive"
    elif report.upper < 1.0:
        direction = "negative"
    else:
        direction = "no_clear_association"
    return [
        {
            "claim_id": "sensitivity_"
            + re.sub(r"[^a-z0-9]+", "_", report.analysis_id.lower()).strip("_"),
            "claim_type": "association",
            "exposure": report.exposure,
            "outcome": report.outcome,
            "direction": direction,
            "estimand": "adjusted odds ratio",
            "population": _POPULATION_BY_STRATEGY[report.strategy].format(
                analysis_set=report.analysis_set.replace("_", " "),
                covariate=report.covariate,
            ),
            "analysis_role": "sensitivity",
            "status": "supported",
            "adjusted_for": list(report.adjustment_covariates),
            "point_estimate": report.estimate,
            "interval_lower": report.lower,
            "interval_upper": report.upper,
        }
    ]


__all__ = ["BinarySensitivityReporting", "derive_sensitivity_claim_payloads"]
