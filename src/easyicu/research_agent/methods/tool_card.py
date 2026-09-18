"""Tool Card schema for reusable capability sedimentation (R4).

A sandbox run that succeeds once must not masquerade as a ``verified_tool``.
A Tool Card binds the five facts R4 requires before any reuse identity can be
considered: pinned version, declared inputs, population/timing assumptions,
declared outputs, and validation evidence.  A package being installed never
counts as validation evidence.

This module owns only the schema and its digest.  Whether a card may carry
the ``verified_tool`` identity is decided by the host in
:mod:`easyicu.research_agent.authority.tool_promotion`, never here.
"""

from __future__ import annotations

import hashlib
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from ..canonical_json import canonical_json as _canonical_json

TOOL_CARD_SCHEMA = "easyicu.tool_card/1"

ToolCapabilityIdentity = Literal[
    "sandbox_code", "candidate", "composed_workflow", "verified_tool"
]

#: Identities a single sandbox success may claim.  ``verified_tool`` and
#: ``composed_workflow`` are deliberately absent: one passing run can only
#: ever be ``sandbox_code`` (run recorded) or ``candidate`` (run recorded and
#: filed for promotion).
SINGLE_RUN_IDENTITIES: tuple[str, str] = ("sandbox_code", "candidate")

_UNPINNED_VERSIONS = frozenset({"", "latest", "main", "master", "head", "unpinned"})


def _nonblank(value: str, *, field: str) -> str:
    if not str(value).strip():
        raise ValueError(f"Tool Card {field} must be non-blank")
    return str(value)


class ToolInputSpec(BaseModel):
    """One declared tool input; the card cannot bind an unnamed input."""

    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)

    name: str = Field(min_length=1)
    value_kind: str = Field(min_length=1)
    required: bool = True
    description: str = ""

    @model_validator(mode="after")
    def _verify_nonblank(self) -> "ToolInputSpec":
        _nonblank(self.name, field="input name")
        _nonblank(self.value_kind, field="input value_kind")
        return self


class ToolOutputSpec(BaseModel):
    """One declared tool output; consumers bind outputs by these names."""

    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)

    name: str = Field(min_length=1)
    value_kind: str = Field(min_length=1)
    description: str = ""

    @model_validator(mode="after")
    def _verify_nonblank(self) -> "ToolOutputSpec":
        _nonblank(self.name, field="output name")
        _nonblank(self.value_kind, field="output value_kind")
        return self


class ValidationEvidence(BaseModel):
    """One validation receipt filed against the card's origin output."""

    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)

    evidence_id: str = Field(min_length=1)
    kind: Literal["sandbox_run", "independent_reproduction", "applicability_check"]
    passed: bool
    output_sha256: str | None = Field(
        default=None,
        pattern=r"^[0-9a-f]{64}$",
    )
    note: str = ""


class ToolCard(BaseModel):
    """The five R4 bindings for one reusable capability candidate."""

    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)

    schema_version: Literal["easyicu.tool_card/1"]
    tool_name: str = Field(min_length=1, max_length=128)
    tool_version: str = Field(min_length=1, max_length=64)
    method_meaning: str = Field(min_length=1)
    inputs: list[ToolInputSpec] = Field(min_length=1)
    outputs: list[ToolOutputSpec] = Field(min_length=1)
    population_assumption: str = Field(min_length=1)
    timing_assumption: str = Field(min_length=1)
    origin_output_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    validation_evidence: list[ValidationEvidence] = Field(default_factory=list)

    @model_validator(mode="after")
    def _verify_bindings(self) -> "ToolCard":
        _nonblank(self.tool_name, field="tool_name")
        version = _nonblank(self.tool_version, field="tool_version")
        if version.strip().lower() in _UNPINNED_VERSIONS:
            raise ValueError(
                "Tool Card tool_version must be pinned; "
                f"{version!r} is not a version"
            )
        _nonblank(self.method_meaning, field="method_meaning")
        _nonblank(self.population_assumption, field="population_assumption")
        _nonblank(self.timing_assumption, field="timing_assumption")
        input_names = [item.name for item in self.inputs]
        if len(set(input_names)) != len(input_names):
            raise ValueError("Tool Card input names must be unique")
        output_names = [item.name for item in self.outputs]
        if len(set(output_names)) != len(output_names):
            raise ValueError("Tool Card output names must be unique")
        return self


def tool_card_sha256(card: ToolCard) -> str:
    """Return the canonical digest of one Tool Card."""

    if not isinstance(card, ToolCard):
        raise TypeError("tool_card_sha256 requires a ToolCard")
    payload = card.model_dump(mode="json")
    return hashlib.sha256(_canonical_json(payload).encode("utf-8")).hexdigest()


def tool_card_completeness_issues(card: ToolCard) -> list[str]:
    """Return human-readable gaps blocking any promotion of ``card``.

    A constructed card already passed structural validation, so this is a
    defense-in-depth re-check plus the one semantic rule the schema cannot
    see at construction time: the origin sandbox run must be filed as passed
    ``sandbox_run`` evidence bound to the card's origin digest.
    """

    issues: list[str] = []
    if not isinstance(card, ToolCard):
        return ["tool_card_missing"]
    if not card.tool_version.strip() or card.tool_version.strip().lower() in (
        _UNPINNED_VERSIONS
    ):
        issues.append("incomplete_card: tool_version is not pinned")
    if not card.method_meaning.strip():
        issues.append("incomplete_card: method_meaning is blank")
    if not card.population_assumption.strip():
        issues.append("incomplete_card: population_assumption is blank")
    if not card.timing_assumption.strip():
        issues.append("incomplete_card: timing_assumption is blank")
    if not card.inputs:
        issues.append("incomplete_card: inputs are empty")
    if not card.outputs:
        issues.append("incomplete_card: outputs are empty")
    if not any(
        item.kind == "sandbox_run"
        and item.passed
        and item.output_sha256 == card.origin_output_sha256
        for item in card.validation_evidence
    ):
        issues.append("incomplete_card: no passed sandbox_run bound to origin digest")
    return issues


def has_recorded_origin_run(card: ToolCard) -> bool:
    """True once the single sandbox success is filed on the card."""

    return any(
        item.kind == "sandbox_run"
        and item.passed
        and item.output_sha256 == card.origin_output_sha256
        for item in card.validation_evidence
    )


__all__ = [
    "SINGLE_RUN_IDENTITIES",
    "TOOL_CARD_SCHEMA",
    "ToolCapabilityIdentity",
    "ToolCard",
    "ToolInputSpec",
    "ToolOutputSpec",
    "ValidationEvidence",
    "has_recorded_origin_run",
    "tool_card_completeness_issues",
    "tool_card_sha256",
]
