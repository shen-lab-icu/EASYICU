"""Typed design contract for the deterministic ordered-stratified adapter."""

from __future__ import annotations

import math
from typing import Any, List, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from .closed_levels import validate_closed_scalar_levels


class OrderedStratifiedSpec(BaseModel):
    """Planner-selected coordinates executed by the ordered-trend owner.

    The adapter does not choose variables or an event definition.  The
    progressive compiler binds them from the preceding typed primary model and
    the current step's explicit raw-input roster.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: Literal["easyicu.ordered_stratified/1"] = (
        "easyicu.ordered_stratified/1"
    )
    ordered_exposure: str
    ordered_levels: List[Any] = Field(min_length=3)
    cochran_armitage_scores: List[float] = Field(min_length=3)
    score_scheme: Literal["consecutive_ordinal_ranks"] = (
        "consecutive_ordinal_ranks"
    )
    binary_outcome: str
    binary_event_value: Literal[1] = 1
    continuous_outcome: str
    confidence_level: Literal[0.95] = 0.95
    stratified_product: Literal["table:ordered_stratified_outcomes"] = (
        "table:ordered_stratified_outcomes"
    )
    trend_product: str
    test_product: Literal["test:ordinal_trend"] = "test:ordinal_trend"

    @model_validator(mode="after")
    def _closed_design(self) -> "OrderedStratifiedSpec":
        names = [
            str(self.ordered_exposure or "").strip(),
            str(self.binary_outcome or "").strip(),
            str(self.continuous_outcome or "").strip(),
        ]
        if any(not name for name in names) or len(set(names)) != 3:
            raise ValueError(
                "ordered exposure, binary outcome, and continuous outcome must "
                "be three distinct non-empty columns"
            )
        levels = validate_closed_scalar_levels(
            self.ordered_levels, label="ordered-stratified exposure levels"
        )
        if len(levels) < 3:
            raise ValueError("ordered-stratified analysis requires at least three levels")
        scores = [float(value) for value in self.cochran_armitage_scores]
        if len(scores) != len(levels) or any(not math.isfinite(value) for value in scores):
            raise ValueError("one finite trend score is required per ordered level")
        if scores != [float(index) for index in range(len(levels))]:
            raise ValueError(
                "consecutive_ordinal_ranks requires scores 0..K-1 in level order"
            )
        if not str(self.trend_product or "").startswith("table:"):
            raise ValueError("trend_product must be a typed table product")
        return self
