"""Typed article-to-design bindings emitted by the scientific Planner."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator


LiteratureDesignElement = Literal[
    "population",
    "time_zero",
    "exposure",
    "outcome",
    "estimand",
    "adjustment",
    "dependence",
    "missing_data",
    "robustness",
    "reporting",
]


class LiteratureDesignBinding(BaseModel):
    """Explain which exact design decisions one sealed source supports."""

    model_config = ConfigDict(extra="forbid")

    citation_key: str = Field(
        ...,
        pattern=r"^[A-Za-z0-9][A-Za-z0-9_.:-]{0,119}$",
    )
    design_elements: list[LiteratureDesignElement] = Field(min_length=1)
    application: str = Field(min_length=8, max_length=1200)
    divergence: str | None = Field(default=None, max_length=1200)

    @field_validator("design_elements")
    @classmethod
    def _unique_design_elements(
        cls, values: list[LiteratureDesignElement]
    ) -> list[LiteratureDesignElement]:
        if len(values) != len(set(values)):
            raise ValueError("literature design_elements must be unique")
        return values


__all__ = ["LiteratureDesignBinding", "LiteratureDesignElement"]
