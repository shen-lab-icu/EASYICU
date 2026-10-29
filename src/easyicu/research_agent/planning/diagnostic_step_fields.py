"""Compact Planner transport for the two closed descriptive diagnostics."""

from __future__ import annotations

import json
from typing import Annotated, Any

from pydantic import BaseModel, Field, WithJsonSchema, field_validator

from ..contracts.covariate_balance import CovariateBalanceSpec
from ..contracts.cluster_sizes import ClusterSizeSpec

_WIRE = {"anyOf": [{"type": "string"}, {"type": "null"}]}


class DiagnosticPlanFields(BaseModel):
    covariate_balance_spec: Annotated[CovariateBalanceSpec | None, WithJsonSchema(_WIRE)] = Field(
        default=None, exclude_if=lambda value: value is None,
        description="JSON CovariateBalanceSpec; units, treatment, covariates, optional fixed weights and explicit missing/denominator policy.",
    )
    cluster_size_spec: Annotated[ClusterSizeSpec | None, WithJsonSchema(_WIRE)] = Field(
        default=None, exclude_if=lambda value: value is None,
        description="JSON ClusterSizeSpec; cohort identity, frozen assignment source/identity/label, closed labels and missing-label policy.",
    )

    @field_validator("covariate_balance_spec", "cluster_size_spec", mode="before")
    @classmethod
    def _decode(cls, value: Any) -> Any:
        return json.loads(value) if isinstance(value, str) else value
