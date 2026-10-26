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

# Mentions extend the sk2 turn attachment boundary. A skill mention contains
# the original skill reference; it never creates a second skill identity.
MentionKind = Literal["database", "concept", "file", "skill"]
ConceptRole = Literal["candidate", "exposure", "outcome", "covariate"]
Digest = Annotated[str, Field(pattern=r"^[a-f0-9]{64}$")]
ObjectId = Annotated[str, Field(pattern=r"^[a-zA-Z0-9][a-zA-Z0-9_.:-]{0,159}$")]


class ObjectMention(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    id: ObjectId
    digest: Digest


class DatabaseMention(ObjectMention):
    kind: Literal["database"] = "database"
    source: Literal["database", "registered_source"] = "database"


class ConceptMention(ObjectMention):
    kind: Literal["concept"] = "concept"
    role: ConceptRole = "candidate"


class FileMention(ObjectMention):
    kind: Literal["file"] = "file"


class SkillMention(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    kind: Literal["skill"] = "skill"
    attachment: SkillAttachment


Mention = Annotated[DatabaseMention | ConceptMention | FileMention | SkillMention, Field(discriminator="kind")]
Mentions = Annotated[list[Mention], Field(max_length=16)]
_MENTIONS = TypeAdapter(Mentions)


def mention_identity(ref: Mention) -> tuple[str, str, str]:
    if ref.kind == "skill":
        return (ref.kind, ref.attachment.source, ref.attachment.id)
    return (ref.kind, getattr(ref, "source", ""), ref.id)


def parse_mentions(value: object) -> list[Mention]:
    try:
        items = _MENTIONS.validate_python(value)
    except ValidationError as exc:
        raise ExtensionRegistryError("mention_invalid", "mention_invalid") from exc
    if len({mention_identity(item) for item in items}) != len(items):
        raise ExtensionRegistryError("mention_duplicate", "mention_duplicate")
    if sum(item.kind == "database" for item in items) > 1:
        raise ExtensionRegistryError("mention_database_conflict", "mention_database_conflict")
    for role in ("exposure", "outcome"):
        if sum(item.kind == "concept" and item.role == role for item in items) > 1:
            raise ExtensionRegistryError("mention_role_conflict", "mention_role_conflict")
    return items


def mention_payload(value: object) -> list[dict]:
    return [item.model_dump(mode="json", exclude_none=True) for item in parse_mentions(value)]


class TurnAttachments(BaseModel):
    """One admission envelope for the original sk2 references and @ mentions."""

    model_config = ConfigDict(extra="forbid")
    attached_skills: AttachedSkills = Field(default_factory=list)
    mentions: Mentions = Field(default_factory=list)


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
