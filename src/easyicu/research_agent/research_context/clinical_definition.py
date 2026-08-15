"""Typed owner-issued identity for derived ICU clinical concepts."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field


class ClinicalDefinitionReference(BaseModel):
    """Separate clinical phenotype time zero from observation windows.

    A phenotype can be defined relative to a clinical event while exported
    rows are selected in a different physical window.  This immutable owner
    receipt prevents consumers from treating those coordinates as equivalent.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    contract_id: str
    definition: str
    version: str
    source_id: str
    definition_time_anchor: str | None = None
    status: str
    validation_status: str
    canonical_definition: bool
    ascertainment_limitations: list[str] = Field(default_factory=list)
    database_conformance: dict[
        str,
        Literal["not_assessed", "mapping_only", "algorithm_golden"],
    ] = Field(
        default_factory=dict,
        description=(
            "Owner-issued per-database validation depth. mapping_only proves "
            "physical concept mapping, not clinical algorithm equivalence."
        ),
    )


__all__ = ["ClinicalDefinitionReference"]
