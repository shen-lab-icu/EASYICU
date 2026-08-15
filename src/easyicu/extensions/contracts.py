"""Dependency-neutral public contracts for EasyICU user extensions.

The extension owner deliberately exposes immutable, path-free activation
descriptors.  Skill source text remains in a content-addressed object store and
MCP credentials are outside the v1 contract.
"""

from __future__ import annotations

import hashlib
import json
from typing import Any, Dict, Literal, Mapping, Sequence, Tuple

from pydantic import BaseModel, ConfigDict, Field, model_validator

NAME_PATTERN = r"^[a-z0-9][a-z0-9-]{0,62}[a-z0-9]$|^[a-z0-9]$"
TOOL_PATTERN = r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,127}$"
SHA256_PATTERN = r"^[a-f0-9]{64}$"
SkillStage = Literal["conversation", "writing"]


class ExtensionRegistryError(ValueError):
    """Stable, owner-attributable extension validation failure."""

    def __init__(
        self,
        code: str,
        message: str,
        *,
        details: Mapping[str, Any] | None = None,
    ) -> None:
        self.code = str(code or "extension_invalid")
        self.message = str(message or "The extension is invalid.")
        self.details = dict(details or {})
        super().__init__(self.message)

    @property
    def detail(self) -> Dict[str, Any]:
        return {
            "error": self.code,
            "message": self.message,
            "reason": self.message,
            "owner": "easyicu.extensions",
            **({"details": self.details} if self.details else {}),
        }


class SkillActivation(BaseModel):
    """One exact Skill revision frozen into a session or pipeline run."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    name: str = Field(pattern=NAME_PATTERN, max_length=64)
    description: str = Field(min_length=1, max_length=1024)
    digest: str = Field(pattern=SHA256_PATTERN)
    stages: Tuple[SkillStage, ...] = Field(min_length=1, max_length=2)
    disable_model_invocation: bool = False

    @model_validator(mode="after")
    def _unique_stages(self) -> "SkillActivation":
        if len(set(self.stages)) != len(self.stages):
            raise ValueError("skill stages must be unique")
        return self


class McpServerActivation(BaseModel):
    """One read-only Streamable HTTP MCP server frozen into activation."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    name: str = Field(pattern=NAME_PATTERN, max_length=64)
    url: str = Field(min_length=1, max_length=2048)
    transport: Literal["streamable-http"] = "streamable-http"
    allowed_tools: Tuple[str, ...] = Field(min_length=1, max_length=32)

    @model_validator(mode="after")
    def _valid_tools(self) -> "McpServerActivation":
        import re

        if len(set(self.allowed_tools)) != len(self.allowed_tools):
            raise ValueError("MCP allowed_tools must be unique")
        invalid = [name for name in self.allowed_tools if not re.fullmatch(TOOL_PATTERN, name)]
        if invalid:
            raise ValueError(f"invalid MCP tool name: {invalid[0]}")
        return self


def _canonical_json(value: Mapping[str, Any]) -> bytes:
    return json.dumps(
        dict(value),
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    ).encode("utf-8")


class ExtensionActivationSnapshot(BaseModel):
    """Path-free immutable extension set captured at a host boundary."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: Literal["easyicu.extension-activation/1"] = (
        "easyicu.extension-activation/1"
    )
    revision: int = Field(default=0, ge=0)
    skills: Tuple[SkillActivation, ...] = ()
    mcp_servers: Tuple[McpServerActivation, ...] = ()
    activation_sha256: str = Field(pattern=SHA256_PATTERN)

    @staticmethod
    def _unsigned_payload(
        *,
        revision: int,
        skills: Sequence[SkillActivation],
        mcp_servers: Sequence[McpServerActivation],
    ) -> Dict[str, Any]:
        return {
            "schema_version": "easyicu.extension-activation/1",
            "revision": int(revision),
            "skills": [item.model_dump(mode="json") for item in skills],
            "mcp_servers": [item.model_dump(mode="json") for item in mcp_servers],
        }

    @classmethod
    def build(
        cls,
        *,
        revision: int = 0,
        skills: Sequence[SkillActivation] = (),
        mcp_servers: Sequence[McpServerActivation] = (),
    ) -> "ExtensionActivationSnapshot":
        ordered_skills = tuple(sorted(skills, key=lambda item: item.name))
        ordered_mcp = tuple(sorted(mcp_servers, key=lambda item: item.name))
        payload = cls._unsigned_payload(
            revision=revision,
            skills=ordered_skills,
            mcp_servers=ordered_mcp,
        )
        return cls(
            **payload,
            activation_sha256=hashlib.sha256(_canonical_json(payload)).hexdigest(),
        )

    @model_validator(mode="after")
    def _verify_digest_and_names(self) -> "ExtensionActivationSnapshot":
        if len({item.name for item in self.skills}) != len(self.skills):
            raise ValueError("skill activation names must be unique")
        if len({item.name for item in self.mcp_servers}) != len(self.mcp_servers):
            raise ValueError("MCP activation names must be unique")
        payload = self._unsigned_payload(
            revision=self.revision,
            skills=self.skills,
            mcp_servers=self.mcp_servers,
        )
        expected = hashlib.sha256(_canonical_json(payload)).hexdigest()
        if expected != self.activation_sha256:
            raise ValueError("extension activation digest mismatch")
        return self


EMPTY_EXTENSION_ACTIVATION = ExtensionActivationSnapshot.build()


__all__ = [
    "EMPTY_EXTENSION_ACTIVATION",
    "ExtensionActivationSnapshot",
    "ExtensionRegistryError",
    "McpServerActivation",
    "NAME_PATTERN",
    "SHA256_PATTERN",
    "SkillActivation",
    "SkillStage",
    "TOOL_PATTERN",
]
