"""Typed, pre-result authority for selecting among research designs.

Owner: research-agent planning authority.
Public contract: compare two to four scientifically distinct designs, select
exactly one before execution, and retain why every alternative was rejected.
The contract is case-neutral and carries an ``analysis_only`` claim ceiling.
"""

from __future__ import annotations

import re
from typing import Literal, Sequence

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from .literature_design_authority import CandidateLiteratureDesignDecision


_POST_RESULT_SELECTION = re.compile(
    r"\b(?:p[ -]?value|statistically significant|significance|lowest aic|"
    r"lowest bic|best auc|highest auc|results? (?:showed|indicated)|"
    r"observed effect)\b",
    re.IGNORECASE,
)


REVIEWABLE_PLAN_ITEM_ORDER = (
    "population_and_unit",
    "exposure_and_timing",
    "outcome_and_followup",
    "adjustment_and_model",
    "missing_data",
    "sensitivity_and_feasibility",
)


class ResearchDesignCandidate(BaseModel):
    """One pre-result design alternative for the same research question."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    design_id: str = Field(pattern=r"^[a-z][a-z0-9_]{2,79}$")
    analysis_type: str = Field(min_length=1, max_length=128)
    estimand: str = Field(min_length=8, max_length=600)
    time_zero: str = Field(min_length=8, max_length=400)
    observation_window: str = Field(min_length=8, max_length=400)
    primary_method: str = Field(min_length=3, max_length=300)
    required_variables: list[str] = Field(min_length=2, max_length=24)
    assumptions: list[str] = Field(min_length=1, max_length=8)
    literature_citation_keys: list[str] = Field(default_factory=list, max_length=8)
    literature_design_decisions: list[CandidateLiteratureDesignDecision] = Field(
        default_factory=list,
        max_length=7,
    )
    novelty_positioning: str = Field(min_length=8, max_length=600)
    figure_role: str = Field(min_length=8, max_length=400)
    supports: str = Field(min_length=8, max_length=500)
    cannot_prove: str = Field(min_length=8, max_length=500)
    reviewable_plan: list[str] | None = None
    disposition: Literal["selected", "rejected"]
    decision_reason: str = Field(min_length=12, max_length=600)

    @field_validator(
        "required_variables",
        "assumptions",
        "literature_citation_keys",
    )
    @classmethod
    def _unique_nonblank_roster(cls, values: list[str]) -> list[str]:
        cleaned = [" ".join(str(value or "").split()) for value in values]
        if any(not value for value in cleaned) or len(cleaned) != len(set(cleaned)):
            raise ValueError("design candidate rosters must be unique and non-empty")
        return cleaned

    @field_validator("literature_design_decisions")
    @classmethod
    def _unique_literature_dimensions(
        cls, values: list[CandidateLiteratureDesignDecision]
    ) -> list[CandidateLiteratureDesignDecision]:
        dimensions = [value.dimension for value in values]
        if len(dimensions) != len(set(dimensions)):
            raise ValueError("candidate literature design dimensions must be unique")
        return values

    @field_validator("decision_reason", "novelty_positioning")
    @classmethod
    def _pre_result_reasoning_only(cls, value: str) -> str:
        cleaned = " ".join(str(value or "").split())
        if _POST_RESULT_SELECTION.search(cleaned):
            raise ValueError(
                "design selection must be justified before results, not by "
                "significance or observed performance"
            )
        return cleaned


class ResearchDesignSelection(BaseModel):
    """Auditable comparison and choice among two to four candidate designs."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: Literal["easyicu.research_design_selection/1"] = (
        "easyicu.research_design_selection/1"
    )
    claim_ceiling: Literal["analysis_only"] = "analysis_only"
    candidates: list[ResearchDesignCandidate] = Field(min_length=2, max_length=4)

    @model_validator(mode="after")
    def _one_selected_distinct_roster(self) -> "ResearchDesignSelection":
        ids = [candidate.design_id for candidate in self.candidates]
        if len(ids) != len(set(ids)):
            raise ValueError("research design candidate ids must be unique")
        selected = [
            candidate
            for candidate in self.candidates
            if candidate.disposition == "selected"
        ]
        if len(selected) != 1:
            raise ValueError("research design selection requires exactly one selected")
        signatures = [
            (
                candidate.analysis_type.casefold(),
                candidate.estimand.casefold(),
                candidate.time_zero.casefold(),
                candidate.observation_window.casefold(),
                candidate.primary_method.casefold(),
            )
            for candidate in self.candidates
        ]
        if len(signatures) != len(set(signatures)):
            raise ValueError(
                "research design candidates must be scientifically distinct"
            )
        return self

    @property
    def selected(self) -> ResearchDesignCandidate:
        return next(
            candidate
            for candidate in self.candidates
            if candidate.disposition == "selected"
        )


class ResearchDesignSelectionError(ValueError):
    """Stable owner-attributable design-selection boundary failure."""

    def __init__(self, reason_code: str, message: str, *, path: str) -> None:
        self.reason_code = reason_code
        self.path = path
        super().__init__(message)


def validate_research_design_selection(
    selection: ResearchDesignSelection | None,
    *,
    selected_analysis_type: str,
    allowed_analysis_types: Sequence[str],
    allowed_variables: Sequence[str],
    allowed_literature_citation_keys: Sequence[str] = (),
    question_anchors: Sequence[str] = (),
    required: bool,
) -> None:
    """Validate one selection against the exact run authority."""

    if selection is None:
        if required:
            raise ResearchDesignSelectionError(
                "design_selection_missing",
                "fresh planning requires two to four candidate research designs",
                path="design_selection",
            )
        return
    allowed_types = set(allowed_analysis_types)
    unavailable_types = sorted(
        {
            candidate.analysis_type
            for candidate in selection.candidates
            if candidate.analysis_type not in allowed_types
        }
    )
    if unavailable_types:
        raise ResearchDesignSelectionError(
            "design_selection_analysis_type_unavailable",
            f"candidate analysis types are outside run authority: {unavailable_types!r}",
            path="design_selection.candidates.analysis_type",
        )
    if selection.selected.analysis_type != selected_analysis_type:
        raise ResearchDesignSelectionError(
            "design_selection_selected_type_mismatch",
            "the selected candidate analysis type must equal the outline analysis type",
            path="design_selection.candidates.disposition",
        )
    if selection.selected.reviewable_plan is None:
        raise ResearchDesignSelectionError(
            "design_selection_reviewable_plan_missing",
            "the selected design must include a complete Planner recommendation "
            "for researcher review before approval",
            path="design_selection.candidates.reviewable_plan",
        )
    if (
        len(selection.selected.reviewable_plan) != len(REVIEWABLE_PLAN_ITEM_ORDER)
        or any(
            len(" ".join(str(value or "").split())) < 8
            for value in selection.selected.reviewable_plan
        )
    ):
        raise ResearchDesignSelectionError(
            "design_selection_reviewable_plan_incomplete",
            "reviewable_plan must contain six substantive recommendations in "
            "the owner-declared order",
            path="design_selection.candidates.reviewable_plan",
        )
    allowed_variable_set = set(allowed_variables)
    unavailable_variables = sorted(
        {
            variable
            for candidate in selection.candidates
            for variable in candidate.required_variables
            if variable not in allowed_variable_set
        }
    )
    if unavailable_variables:
        raise ResearchDesignSelectionError(
            "design_selection_variable_unavailable",
            f"candidate designs use unavailable variables: {unavailable_variables!r}",
            path="design_selection.candidates.required_variables",
        )
    allowed_keys = set(allowed_literature_citation_keys)
    unknown_keys = sorted(
        {
            key
            for candidate in selection.candidates
            for key in (
                list(candidate.literature_citation_keys)
                + [
                    decision_key
                    for decision in candidate.literature_design_decisions
                    for decision_key in decision.citation_keys
                ]
            )
            if key not in allowed_keys
        }
    )
    if unknown_keys:
        raise ResearchDesignSelectionError(
            "design_selection_literature_key_unavailable",
            f"candidate designs cite keys outside the sealed bundle: {unknown_keys!r}",
            path="design_selection.candidates.literature_citation_keys",
        )
    anchors = {
        str(anchor or "").strip()
        for anchor in question_anchors
        if str(anchor or "").strip()
    }
    selected_variables = set(selection.selected.required_variables)
    if anchors and selected_variables.isdisjoint(anchors):
        raise ResearchDesignSelectionError(
            "design_selection_question_anchor_missing",
            "the selected design must bind at least one run-specific question anchor",
            path="design_selection.candidates.required_variables",
        )


__all__ = [
    "REVIEWABLE_PLAN_ITEM_ORDER",
    "ResearchDesignCandidate",
    "ResearchDesignSelection",
    "ResearchDesignSelectionError",
    "validate_research_design_selection",
]
