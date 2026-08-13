"""Typed claim ceilings for plans that intentionally remain descriptive."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator


DescriptiveLimitation = Literal[
    "post_baseline_exposure_opportunity_unresolved",
]

BOUND_TYPED_COHORT_ANALYSIS_SET = "bound_typed_cohort"
EXPOSURE_OBSERVED_ANALYSIS_SET = (
    "exposure_observed_rows_within_bound_typed_cohort"
)


class DescriptiveClaimContract(BaseModel):
    """Forbid inferential interpretation while recording why it is withheld.

    This is not a text label.  Scientific review accepts the ceiling only for a
    closed descriptive method/product shape and only when the unresolved
    post-baseline exposure-opportunity limitation is explicitly retained.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: Literal["easyicu.descriptive_claim/1"] = (
        "easyicu.descriptive_claim/1"
    )
    claim_ceiling: Literal["descriptive_only"] = "descriptive_only"
    unresolved_limitations: tuple[DescriptiveLimitation, ...] = Field(
        min_length=1,
        max_length=8,
    )

    @model_validator(mode="after")
    def _unique_limitations(self) -> "DescriptiveClaimContract":
        if len(self.unresolved_limitations) != len(set(self.unresolved_limitations)):
            raise ValueError("unresolved_limitations must be unique")
        return self


__all__ = [
    "BOUND_TYPED_COHORT_ANALYSIS_SET",
    "DescriptiveClaimContract",
    "DescriptiveLimitation",
    "EXPOSURE_OBSERVED_ANALYSIS_SET",
]
