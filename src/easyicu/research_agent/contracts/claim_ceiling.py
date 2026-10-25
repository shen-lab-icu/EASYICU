"""Typed scientific claim ceilings and run-owned delivery scope."""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator
from pydantic.json_schema import SkipJsonSchema


DescriptiveLimitation = Literal[
    "post_baseline_exposure_opportunity_unresolved",
    "counts_only_design",
    "descriptive_study_design",
]

BOUND_TYPED_COHORT_ANALYSIS_SET = "bound_typed_cohort"
EXPOSURE_OBSERVED_ANALYSIS_SET = (
    "exposure_observed_rows_within_bound_typed_cohort"
)


class DescriptiveClaimContract(BaseModel):
    """Forbid inferential interpretation while recording why it is withheld.

    This is not a text label.  Scientific review accepts the ceiling only for a
    closed descriptive method/product shape with an explicit limitation:
    unresolved post-baseline exposure opportunity, a counts-only design
    that permits no uncertainty interval or inferential test, or a
    descriptive study design whose unadjusted group risks and their intervals
    describe the cohort without an adjusted or causal comparison.
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


ExecutionTier = Literal["first_result", "article"]
FIRST_RESULT_MARKER = "[easyicu.run-tier:first_result]"


class RunTierPlan(BaseModel):
    """Host-owned delivery scope carried by the run's frozen design contract."""

    execution_tier: SkipJsonSchema[ExecutionTier] = Field(
        default="article", frozen=True, exclude_if=lambda value: value == "article",
        description="Host-bound run delivery scope; immutable after approval.",
    )


def first_result(value: Any) -> bool:
    return getattr(value, "execution_tier", getattr(getattr(value, "design_selection", None), "execution_tier", None)) == "first_result"


def bind_plan_tier(plan: Any, tier: ExecutionTier) -> Any:
    selection = plan.design_selection
    if selection is None:
        if tier == "first_result":
            raise ValueError("first_result requires the run's reviewed design selection before approval")
        return plan
    return plan.model_copy(update={"design_selection": selection.model_copy(update={"execution_tier": tier})})


def tier_from_planning_contract(text: str) -> ExecutionTier:
    return "first_result" if FIRST_RESULT_MARKER in text.splitlines() else "article"


def first_result_instructions() -> str:
    return (
        FIRST_RESULT_MARKER + "\n"
        "This run delivers analysis-only first evidence, never a paper. Use only typed "
        "host-owned executors. No coder, free-form figure/report code, manuscript, "
        "Writer or LaTeX is permitted. Keep measurement_audit, cohort counts, primary "
        "estimates with confidence intervals, denominators and missingness. Preserve "
        "every question-stated outcome, adjustment, subgroup and sensitivity; if no "
        "typed owner can answer one, state the capability gap before approval. "
        "Optional article figures, robustness axes and covariate RCS checks are not "
        "required. Literature search and literature-based hypothesis generation are "
        "deferred to an article run. Do not claim novelty, literature comparisons or "
        "citation support in this result. An article needs a separate run and approval."
    )


def manuscript_disabled(pipeline: Any, *results: Any) -> bool:
    return (
        first_result(getattr(pipeline, "_config", None))
        or getattr(getattr(pipeline, "_config", None), "generate_manuscript", None) is False
        or any(first_result(getattr(result, "plan", None)) for result in results)
    )


def require_article_tier(plan: Any) -> None:
    if first_result(plan):
        raise ValueError("first_result cannot be revised or published as a manuscript; start an article run")
