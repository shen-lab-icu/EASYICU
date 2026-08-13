"""Dependency-neutral Planner contract for manuscript figure panels.

The declared input products establish lineage, but they do not establish what
reader-facing article role a panel serves or which chart grammar it will use.
This contract keeps those semantics explicit without importing plan, figure,
or reporting owners.
"""

from __future__ import annotations

import re
from typing import List, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator


class PlannedFigurePanelSpec(BaseModel):
    """Planner-owned article role and chart grammar for one figure panel."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: Literal["easyicu.planned_figure_panel/1"] = (
        "easyicu.planned_figure_panel/1"
    )
    panel_id: str = Field(pattern=r"^[a-z][a-z0-9_]{0,79}$")
    figure_output: str = Field(pattern=r"^figure:[a-z][a-z0-9_]{0,79}$")
    article_role: str = Field(pattern=r"^[a-z][a-z0-9_]{0,79}$")
    chart_type: str = Field(pattern=r"^[a-z][a-z0-9_]{0,79}$")
    source_products: List[str] = Field(min_length=1, max_length=16)

    @field_validator("source_products")
    @classmethod
    def _source_products_are_unique_typed_inputs(
        cls, values: List[str]
    ) -> List[str]:
        cleaned = [str(value or "").strip() for value in values]
        if any(
            not re.fullmatch(r"[a-z][a-z0-9_]*:[a-z][a-z0-9_]*", value)
            for value in cleaned
        ):
            raise ValueError(
                "source_products must contain canonical typed kind:product inputs"
            )
        if len(cleaned) != len(set(cleaned)):
            raise ValueError("source_products must be unique")
        return cleaned


__all__ = ["PlannedFigurePanelSpec"]
