"""Typed, digest-bound projections of catalogue and project material owners."""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict
from typing import Annotated, Any, Literal

from pydantic import BaseModel, ConfigDict, Field, TypeAdapter, ValidationError

from easyicu.extensions.contracts import ExtensionRegistryError
from easyicu.extensions.skill_attachments import Digest, Mention, SkillAttachment, parse_mentions
from easyicu.research_agent.concept_availability import explain_concept_availability


class MentionObject(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class DatabaseObject(MentionObject):
    kind: Literal["database"] = "database"
    database: str
    label: str
    source_id: str | None = None
    identity_sha256: Digest | None = None
    path_sha256: Digest | None = None


class MentionAvailability(MentionObject):
    concept: str
    status: str
    available: bool
    reason: str | None = None


class ConceptObject(MentionObject):
    kind: Literal["concept"] = "concept"
    concept_id: str
    definition: dict[str, Any]
    definition_sha256: Digest
    availability: dict[str, MentionAvailability]
    availability_basis: Literal["packaged_mappings_not_patient_coverage"] = "packaged_mappings_not_patient_coverage"


class FileObject(MentionObject):
    kind: Literal["file"] = "file"
    project_id: str
    file: str
    sha256: Digest
    size: int = Field(ge=0)
    media_type: str
    authority_class: Literal["workspace_artifact"] = "workspace_artifact"
    scientific_evidence: Literal[False] = False


class SkillObject(MentionObject):
    kind: Literal["skill"] = "skill"
    attachment: SkillAttachment


class ResolvedMention(MentionObject):
    reference: Mention
    object: Annotated[DatabaseObject | ConceptObject | FileObject | SkillObject, Field(discriminator="kind")]


_RESOLVED = TypeAdapter(Annotated[list[ResolvedMention], Field(max_length=16)])


def object_digest(value: MentionObject) -> str:
    return hashlib.sha256(json.dumps(value.model_dump(mode="json"), sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def resolved_payload(value: object) -> list[dict]:
    try:
        rows = _RESOLVED.validate_python(value)
        parse_mentions([row.reference for row in rows])
        for row in rows:
            ref, obj = row.reference, row.object
            valid = ref.kind == obj.kind
            if ref.kind == "skill":
                valid = valid and ref.attachment == obj.attachment
            else:
                valid = valid and ref.digest == object_digest(obj)
                if ref.kind == "concept":
                    valid = valid and ref.id == obj.concept_id
                if ref.kind == "database":
                    valid = valid and ref.id == (obj.source_id if ref.source == "registered_source" else obj.database)
                if ref.kind == "file":
                    valid = valid and ref.id == file_id(obj.project_id, obj.file)
            if not valid:
                raise ExtensionRegistryError("mention_stale", "mention_stale")
        payload = [row.model_dump(mode="json", exclude_none=True) for row in rows]
        if len(json.dumps(payload).encode()) > 64_000:
            raise ExtensionRegistryError("mention_context_too_large", "mention_context_too_large")
        return payload
    except (ValidationError, AttributeError) as exc:
        raise ExtensionRegistryError("mention_invalid", "mention_invalid") from exc


def concept_object(concept_id: str) -> ConceptObject:
    from easyicu.concept.catalog import CONCEPT_DICTIONARY
    from easyicu.concept.export_metadata import metadata_definition_for_concept
    from easyicu.databases.profiles import iter_database_profiles
    from easyicu.resources import load_dictionary

    dictionary = load_dictionary(include_sofa2=True)
    if concept_id not in dictionary and concept_id not in CONCEPT_DICTIONARY:
        raise ExtensionRegistryError("mention_unknown", "mention_unknown")
    definition = metadata_definition_for_concept(concept_id, "", dictionary)
    # Full definition identity, including callback and source bindings, but no
    # source data. JSON scalars normalize Timedelta and other owner values.
    full = json.loads(json.dumps(asdict(definition), default=str))
    snapshot = {key: full[key] for key in (
        "description", "units", "category", "class_name", "minimum", "maximum", "callback",
        "aggregate", "sub_concepts", "depends_on", "definition_source", "definition_version",
        "canonical_definition", "clinical_status", "clinical_contract_id",
    ) if full.get(key) is not None and full[key] != []}
    availability = {}
    for profile in iter_database_profiles(public_only=True):
        item = explain_concept_availability(concept=concept_id, database=profile.key)
        availability[profile.key] = MentionAvailability(concept=item.concept, status=item.status, available=item.available, reason=item.reason)
    return ConceptObject(concept_id=concept_id, definition=snapshot,
                         definition_sha256=hashlib.sha256(json.dumps(full, sort_keys=True, separators=(",", ":")).encode()).hexdigest(),
                         availability=availability)


def file_id(project_id: str, relative_file: str) -> str:
    return hashlib.sha256(json.dumps([project_id, relative_file], separators=(",", ":")).encode()).hexdigest()


def file_object(project_id: str, material: dict) -> FileObject:
    return FileObject(project_id=project_id, **{key: material[key] for key in ("file", "sha256", "size", "media_type")})


def require_mentioned_file(project_id: str, refs: object, material: dict) -> None:
    """A later workspace read cannot replace the bytes pinned for this turn."""
    obj = file_object(project_id, material)
    for ref in parse_mentions(refs):
        if ref.kind == "file" and ref.id == file_id(project_id, obj.file) and ref.digest != object_digest(obj):
            raise ExtensionRegistryError("mention_stale", "mention_stale")
