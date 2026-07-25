"""Strict schemas for resources selected into bounded agent context."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

RESOURCE_DESCRIPTOR_SCHEMA = "easyicu.resource_descriptor/1"
RESOURCE_SELECTION_RECEIPT_SCHEMA = "easyicu.resource_selection_receipt/1"

ResourceKind = Literal["protocol", "action", "software", "data"]
ResourceReviewStatus = Literal[
    "curated_mvp",
    "clinical_reviewed",
    "validated",
    "approved",
    "unreviewed",
]
ResourcePermission = Literal[
    "planner_context",
    "coder_context",
    "sandbox_import",
    "data_read",
]


class ResourceDescriptor(BaseModel):
    """One immutable resource that may be exposed to Planner or Coder."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: Literal["easyicu.resource_descriptor/1"] = (
        RESOURCE_DESCRIPTOR_SCHEMA
    )
    resource_id: str = Field(pattern=r"^[a-z][a-z0-9_.:-]{2,127}$")
    version: str = Field(pattern=r"^(?:0|[1-9][0-9]*)\.[0-9]+\.[0-9]+$")
    sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    kind: ResourceKind
    analysis_families: tuple[str, ...] = ()
    required_input_roles: tuple[str, ...] = ()
    produced_output_roles: tuple[str, ...] = ()
    permissions: tuple[ResourcePermission, ...]
    review_status: ResourceReviewStatus
    search_terms: tuple[str, ...] = ()
    prompt_projection: str = ""

    @field_validator(
        "analysis_families",
        "required_input_roles",
        "produced_output_roles",
        "permissions",
        "search_terms",
    )
    @classmethod
    def _unique_coordinates(cls, values: tuple[str, ...]) -> tuple[str, ...]:
        cleaned = tuple(str(value).strip() for value in values if str(value).strip())
        if len(cleaned) != len(set(cleaned)):
            raise ValueError("resource coordinates must be unique")
        return cleaned


class ResourceSelectionPolicy(BaseModel):
    """Host policy applied before deterministic ranking."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    allowed_kinds: tuple[ResourceKind, ...]
    allowed_review_statuses: tuple[ResourceReviewStatus, ...]
    allowed_permissions: tuple[ResourcePermission, ...]
    max_protocols: int = Field(default=3, ge=0, le=5)
    max_actions: int = Field(default=8, ge=0, le=16)
    max_software: int = Field(default=3, ge=0, le=8)
    max_data: int = Field(default=4, ge=0, le=16)

    @model_validator(mode="after")
    def _nonempty_policy(self) -> "ResourceSelectionPolicy":
        if not self.allowed_kinds:
            raise ValueError("resource policy must allow at least one kind")
        if not self.allowed_review_statuses:
            raise ValueError("resource policy must allow at least one review status")
        if not self.allowed_permissions:
            raise ValueError("resource policy must allow at least one permission")
        return self

    def limit_for(self, kind: ResourceKind) -> int:
        return {
            "protocol": self.max_protocols,
            "action": self.max_actions,
            "software": self.max_software,
            "data": self.max_data,
        }[kind]


class ResourceSelectionQuery(BaseModel):
    """Scientific coordinates supplied by the host, never inferred by a tool."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    purpose: Literal["planner", "coder"]
    query: str = Field(min_length=1, max_length=8_000)
    analysis_family: str = Field(min_length=1, max_length=120)
    step_role: str | None = Field(default=None, max_length=120)
    database: str | None = Field(default=None, max_length=80)
    available_input_roles: tuple[str, ...] = ()


class SelectedResource(BaseModel):
    """One selected resource and the deterministic reasons for selection."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    resource_id: str
    version: str
    sha256: str
    kind: ResourceKind
    score: float = Field(ge=0.0, le=1.0)
    reasons: tuple[str, ...]


class ResourceSelectionReceipt(BaseModel):
    """Digest-bound proof of a zero-provider-call resource selection."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: Literal["easyicu.resource_selection_receipt/1"] = (
        RESOURCE_SELECTION_RECEIPT_SCHEMA
    )
    query: ResourceSelectionQuery
    policy: ResourceSelectionPolicy
    catalog_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    allowlist_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    candidate_count: int = Field(ge=0)
    selected: tuple[SelectedResource, ...]
    projection_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    projection_bytes: int = Field(ge=0)
    provider_calls: Literal[0] = 0
    authority: Literal["host_allowlist_then_deterministic_rank"] = (
        "host_allowlist_then_deterministic_rank"
    )
