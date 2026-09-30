"""An owner's receipt of the design it executed.

A signed owner applies a design fixed before the run: a time grid read from
its anchor, an eligibility minimum, a model and its selection rule.  The run
context's time window describes what the host materialized for the cohort,
not what a longitudinal owner read, and the strict Methods grammar admits a
numeric design detail only as an exact host fact.  An owner therefore states
its executed design as a closed, versioned block (``executed_method_design``
in its step summary); the reporting owner renders it and never infers it.
"""

from __future__ import annotations

from typing import Annotated, Any, Literal, Union

from pydantic import BaseModel, ConfigDict, Field, TypeAdapter, model_validator

EXECUTED_METHOD_DESIGN_KEY = "executed_method_design"
EXECUTED_METHOD_DESIGN_SCHEMA_VERSION = "easyicu.executed_method_design/1"


class _ExecutedDesign(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)

    schema_version: Literal["easyicu.executed_method_design/1"]


class FixedWindowRepresentationDesign(_ExecutedDesign):
    """Per-window summaries on a fixed grid, and the eligibility minimum."""

    design_kind: Literal["fixed_window_representation"]
    anchor: str = Field(pattern=r"^[a-z][a-z0-9]*(?:_[a-z0-9]+)*$")
    window_start_hours: int
    window_end_hours: int
    window_width_hours: int = Field(gt=0)
    n_windows: int = Field(ge=1)
    window_aggregation: Literal["max"]
    minimum_observed_windows: int = Field(ge=1)

    @model_validator(mode="after")
    def _grid_tiles_the_window(self) -> "FixedWindowRepresentationDesign":
        if (
            self.window_end_hours <= self.window_start_hours
            or self.window_end_hours - self.window_start_hours
            != self.n_windows * self.window_width_hours
        ):
            raise ValueError("executed time grid does not tile its window")
        if self.minimum_observed_windows > self.n_windows:
            raise ValueError("the window minimum exceeds the grid")
        return self


class LatentClassModelDesign(_ExecutedDesign):
    """The class model, its coordinate scale, and its selection rule."""

    design_kind: Literal["latent_class_model"]
    model_family: Literal[
        "latent_class_diagonal_gaussian_mixture", "latent_class_mixed_mode"
    ]
    coordinate_scaling: Literal[
        "pooled_coordinate_wise_z_score", "continuous_coordinate_wise_z_score"
    ]
    candidate_class_counts: list[int]
    selection_criterion: Literal["bic"]
    minimum_class_fraction: float = Field(gt=0.0, lt=1.0)

    @model_validator(mode="after")
    def _grid_is_increasing(self) -> "LatentClassModelDesign":
        counts = self.candidate_class_counts
        if (
            len(counts) < 2
            or counts[0] < 2
            or any(later <= earlier for earlier, later in zip(counts, counts[1:]))
        ):
            raise ValueError("candidate class counts must be an increasing grid from 2")
        # The mixed-mode model keeps ordinal levels; only a Gaussian-only model
        # z-scores every coordinate.
        if (self.model_family == "latent_class_mixed_mode") != (
            self.coordinate_scaling == "continuous_coordinate_wise_z_score"
        ):
            raise ValueError("the class model and its coordinate scaling disagree")
        return self


ExecutedMethodDesign = Annotated[
    Union[FixedWindowRepresentationDesign, LatentClassModelDesign],
    Field(discriminator="design_kind"),
]
_DESIGN_ADAPTER: TypeAdapter[Any] = TypeAdapter(ExecutedMethodDesign)


def validate_executed_method_design(payload: object) -> Any:
    """Parse one owner's design block into its closed type (fail closed)."""

    return _DESIGN_ADAPTER.validate_python(payload)


def executed_method_design_payload(design: Any) -> dict[str, Any]:
    return design.model_dump(mode="json")


__all__ = [
    "EXECUTED_METHOD_DESIGN_KEY",
    "EXECUTED_METHOD_DESIGN_SCHEMA_VERSION",
    "ExecutedMethodDesign",
    "FixedWindowRepresentationDesign",
    "LatentClassModelDesign",
    "executed_method_design_payload",
    "validate_executed_method_design",
]
