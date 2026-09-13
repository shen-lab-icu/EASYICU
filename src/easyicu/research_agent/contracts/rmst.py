"""Exact time/event/group coordinates for a restricted-mean-survival step.

This plan contract is not an execution capability by itself.  A host-owned
executor must still claim the exact action, validate its cohort, run the
reviewed kernel, and fail closed when the declared columns or groups drift.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


class RMSTSpec(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, allow_inf_nan=False)

    time_column: str = Field(pattern=r"^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$")
    event_column: str = Field(pattern=r"^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$")
    group_column: str = Field(pattern=r"^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$")
    group_levels: tuple[str, str]
    tau: float = Field(gt=0)
    time_unit: Literal["days", "hours"] = "days"
    event_code: float = 1.0

    @field_validator("group_levels")
    @classmethod
    def _clean_levels(cls, values: tuple[str, str]) -> tuple[str, str]:
        cleaned = [str(value or "").strip() for value in values]
        if any(not value or len(value) > 120 for value in cleaned):
            raise ValueError("rmst_spec group levels must be short non-empty labels")
        return cleaned[0], cleaned[1]

    @model_validator(mode="after")
    def _distinct_coordinates(self) -> "RMSTSpec":
        columns = (self.time_column, self.event_column, self.group_column)
        if len(set(columns)) != len(columns):
            raise ValueError("rmst_spec time/event/group columns must be distinct")
        if self.group_levels[0] == self.group_levels[1]:
            raise ValueError("rmst_spec group levels must be two distinct labels")
        return self
