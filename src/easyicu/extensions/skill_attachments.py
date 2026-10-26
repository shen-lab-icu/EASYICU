"""Bounded, path-free attachment identities, independent of Planner schemas."""

from __future__ import annotations

from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field, TypeAdapter, ValidationError

from .contracts import ExtensionRegistryError

MAX_ATTACHED_SKILLS = 8


class BuiltinSkillAttachment(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    source: Literal["builtin"]
    id: str = Field(pattern=r"^[a-z0-9][a-z0-9-]{0,95}$")
    version: str = Field(min_length=1, max_length=80)
    digest: str | None = Field(default=None, pattern=r"^[a-f0-9]{64}$")


class UserSkillAttachment(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    source: Literal["user"]
    id: str = Field(pattern=r"^[a-z0-9][a-z0-9-]{0,63}$")
    digest: str = Field(pattern=r"^[a-f0-9]{64}$")


SkillAttachment = Annotated[
    BuiltinSkillAttachment | UserSkillAttachment, Field(discriminator="source")
]
AttachedSkills = Annotated[list[SkillAttachment], Field(max_length=MAX_ATTACHED_SKILLS)]
_ADAPTER = TypeAdapter(AttachedSkills)


def parse_attachments(value: object) -> list[SkillAttachment]:
    try:
        items = _ADAPTER.validate_python(value)
    except ValidationError as exc:
        raise ExtensionRegistryError("skill_attachment_invalid", "skill_attachment_invalid") from exc
    if len({(item.source, item.id) for item in items}) != len(items):
        raise ExtensionRegistryError("skill_attachment_duplicate", "skill_attachment_duplicate")
    return items


def attachment_payload(value: object) -> list[dict]:
    return [item.model_dump(mode="json", exclude_none=True) for item in parse_attachments(value)]
