"""The words of a target trial estimate, as its host claim states them.

A claim of the signed target trial suite
(``authority.target_trial_scientific_claims``) states one estimate of an
emulated target trial: the risk of the outcome by the horizon under one
strategy, or the difference or the ratio of the two strategies' risks, with
the weights as estimated or truncated.  An emulation answers its causal
question only under assumptions the data cannot test, so its sentence is a
fixed template that names them before the estimate.  This module owns those
words: a claim of this kind has no free text to fill, and no other sentence
of a manuscript states a causal estimate.

The numbers arrive formatted by the claim's own reader precision; the
templates avoid the words the causal-language audit reads as an asserted
mechanism (``review.causal_audit``).
"""

from __future__ import annotations

from typing import Literal, Optional

from pydantic import BaseModel, ConfigDict, Field, model_validator

TARGET_TRIAL_CLAIM_TERMS_SCHEMA_VERSION = "easyicu.target_trial_claim_terms/1"
#: The assumptions every estimate of an emulation rests on, in reading order.
TARGET_TRIAL_ASSUMPTIONS = (
    "no unmeasured confounding",
    "positivity",
    "correctly specified weight models",
    "censoring at ICU exit that the baseline covariates explain",
)
#: The assumptions as one phrase: the claim, the limitation and the researcher's
#: confirmation all print these words.
TARGET_TRIAL_ASSUMPTION_PHRASE = (
    ", ".join(TARGET_TRIAL_ASSUMPTIONS[:-1]) + " and " + TARGET_TRIAL_ASSUMPTIONS[-1]
)
_ASSUMPTIONS = "Under the emulation's assumptions of " + TARGET_TRIAL_ASSUMPTION_PHRASE
#: Labels carry no digit: every number the sentence prints binds to a result.
_READER = r"^[^{}\[\]<>`\\|*_#\n0-9]{1,80}$"
_EFFECT_SCALES = {
    "strategy_risk": "percent",
    "risk_difference": "percentage_points",
    "risk_ratio": "ratio",
}


def ordinal(value: float) -> str:
    """A whole percentile as an English ordinal: 1st, 2nd, 99th."""

    if not float(value).is_integer():
        raise ValueError(f"{value!r} is not a whole percentile")
    number = int(value)
    suffix = (
        "th"
        if 10 <= number % 100 <= 20
        else {1: "st", 2: "nd", 3: "rd"}.get(number % 10, "th")
    )
    return f"{number}{suffix}"


def _mid_sentence(text: str) -> str:
    """``Death`` reads ``death`` inside a sentence; ``ICU death`` keeps its case."""

    first = text.split(" ", 1)[0]
    if first[:1].isupper() and first[1:] == first[1:].lower():
        return text[:1].lower() + text[1:]
    return text


class TargetTrialClaimTerms(BaseModel):
    """What one target trial estimate is an estimate of, in typed fields.

    Not strict: a claim reloads it from its JSON record, where the
    percentiles are a list.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: Literal["easyicu.target_trial_claim_terms/1"]
    measure: Literal["strategy_risk", "risk_difference", "risk_ratio"]
    #: The strategy a risk is estimated under; a contrast names none.
    strategy: Optional[Literal["initiate", "defer"]]
    weighting: Literal["stabilized", "stabilized_truncated"]
    initiate_label: str = Field(pattern=_READER)
    defer_label: str = Field(pattern=_READER)
    outcome_label: str = Field(pattern=_READER)
    horizon_days: int = Field(gt=0)
    analysis_unit_label: str = Field(pattern=_READER)
    #: The percentiles each strategy's weights were truncated at, when they were.
    truncation_percentiles: Optional[tuple[float, float]]

    @model_validator(mode="after")
    def _one_estimate(self) -> "TargetTrialClaimTerms":
        if (self.measure == "strategy_risk") != (self.strategy is not None):
            raise ValueError("a strategy's risk names its strategy, and only it does")
        if (self.weighting == "stabilized_truncated") != (
            self.truncation_percentiles is not None
        ):
            raise ValueError(
                "truncated weights name their percentiles, and only they do"
            )
        if self.truncation_percentiles is not None:
            low, high = self.truncation_percentiles
            if not 0.0 < low < high < 100.0:
                raise ValueError("truncation percentiles increase inside 0 to 100")
            if not (float(low).is_integer() and float(high).is_integer()):
                raise ValueError(
                    "truncation percentiles are whole, as the sentence names them"
                )
        if self.initiate_label.casefold() == self.defer_label.casefold():
            raise ValueError("the two strategies need different labels")
        return self

    @property
    def effect_scale(self) -> str:
        return _EFFECT_SCALES[self.measure]

    @property
    def _risk(self) -> str:
        return f"risk of {_mid_sentence(self.outcome_label)} by day {self.horizon_days}"

    @property
    def _population(self) -> str:
        return f"in the eligible {self.analysis_unit_label}"

    @property
    def _trial(self) -> str:
        if self.truncation_percentiles is None:
            return "the emulated target trial"
        low, high = self.truncation_percentiles
        return (
            "the emulated target trial, with each strategy's weights truncated at "
            f"the {ordinal(low)} and {ordinal(high)} percentiles,"
        )

    @property
    def strategy_label(self) -> str:
        if self.strategy is None:
            raise ValueError("a contrast names no single strategy")
        return self.initiate_label if self.strategy == "initiate" else self.defer_label

    @property
    def subject(self) -> str:
        """The strategy, or the contrast, the estimate is for."""

        if self.strategy is not None:
            return self.strategy_label
        return f"{self.initiate_label} versus {self.defer_label}"

    @property
    def estimand(self) -> str:
        """The estimand in machine words: what is estimated, with which weights."""

        weights = (
            "stabilized weights"
            if self.truncation_percentiles is None
            else "truncated stabilized weights"
        )
        if self.measure == "strategy_risk":
            return f"{self._risk} under the {self.strategy_label} strategy, {weights}"
        if self.measure == "risk_difference":
            return (
                f"difference in the {self._risk}, {self.initiate_label} minus "
                f"{self.defer_label}, {weights}"
            )
        return (
            f"risk ratio of {_mid_sentence(self.outcome_label)} by day "
            f"{self.horizon_days}, {self.initiate_label} versus {self.defer_label}, "
            f"{weights}"
        )

    def result_sentence(
        self, *, point: str, low: str, high: str, confidence: str
    ) -> str:
        """The one sentence that states this estimate with its interval."""

        lead = f"{_ASSUMPTIONS}, {self._trial} estimated"
        interval = f"{confidence}% CI, {low}"
        if self.measure == "strategy_risk":
            return (
                f"{lead} a {self._risk} of {point}% ({interval}% to {high}%) under "
                f"the {self.strategy_label} strategy {self._population}."
            )
        if self.measure == "risk_difference":
            return (
                f"{lead} a difference in the {self._risk} of {point} percentage "
                f"points ({interval} to {high}) between the {self.initiate_label} and "
                f"{self.defer_label} strategies ({self.initiate_label} minus "
                f"{self.defer_label}) {self._population}."
            )
        return (
            f"{lead} a risk ratio of {point} ({interval} to {high}) for "
            f"{_mid_sentence(self.outcome_label)} by day {self.horizon_days}, "
            f"{self.initiate_label} versus {self.defer_label}, {self._population}."
        )

    def conclusion_sentence(self, direction: str) -> str:
        """The same estimate as a conclusion: its direction, without its numbers."""

        lead = f"{_ASSUMPTIONS}, {self._trial} estimated"
        if self.measure == "strategy_risk":
            return (
                f"{lead} the {self._risk} under the {self.strategy_label} strategy "
                f"{self._population}."
            )
        if direction == "no_clear_association":
            return (
                f"{lead} no clear difference in the {self._risk} between the "
                f"{self.initiate_label} and {self.defer_label} strategies "
                f"{self._population}."
            )
        level = {"negative": "lower", "positive": "higher"}[direction]
        return (
            f"{lead} a {level} {self._risk} under the {self.initiate_label} strategy "
            f"than under the {self.defer_label} strategy {self._population}."
        )


__all__ = [
    "TARGET_TRIAL_ASSUMPTIONS",
    "TARGET_TRIAL_ASSUMPTION_PHRASE",
    "TARGET_TRIAL_CLAIM_TERMS_SCHEMA_VERSION",
    "TargetTrialClaimTerms",
    "ordinal",
]
