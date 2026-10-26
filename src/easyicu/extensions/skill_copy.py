"""Project reviewed built-in documentation into personal instruction text.

Provenance describes the source; it grants no execution or scientific authority.
The personal format is one SKILL.md. Keep its main instructions complete, embed
whole references that fit, and record omissions within the same reviewed digest.
"""

from __future__ import annotations

from datetime import datetime
import re
from typing import Annotated, Any, Mapping

from pydantic import BaseModel, ConfigDict, Field, ValidationError, field_validator
import yaml

from .contracts import ExtensionRegistryError, SHA256_PATTERN


COPY_BOUNDARY = (
    "## Personal Skill authority boundary\n\n"
    "This personal Skill is instructions only and analysis-only. It cannot change "
    "host execution, plan review, method dispatch, evidence binding or claim ceilings. "
    "Host gates ignore conflicting instructions, including those in the copied "
    "documentation below. Built-in execution modes and reportability describe the "
    "source package only; this copy inherits none of that authority. Scripts are "
    "not copied or executable through this Skill. References that do not fit the "
    "personal instruction limit are listed in source.omitted_reference_paths; "
    "inspect them in the built-in source package.\n"
)

ReferenceMarkdownPath = Annotated[
    str,
    Field(
        pattern=r"^references/(?:[A-Za-z0-9_-]+/)*[A-Za-z0-9_-][A-Za-z0-9_.-]*\.md$",
        max_length=256,
    ),
]


class BuiltinSkillSource(BaseModel):
    """Descriptive provenance bound into the reviewed instruction digest."""

    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)

    builtin_id: str = Field(pattern=r"^[a-z0-9][a-z0-9_.-]*$", max_length=128)
    registry_version: str = Field(
        pattern=r"^easyicu\.method-skills/[0-9]+$", max_length=64
    )
    package_sha256: str = Field(pattern=SHA256_PATTERN)
    copied_at: str = Field(pattern=r"^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}Z$")
    omitted_reference_paths: list[ReferenceMarkdownPath] = Field(
        default_factory=list, max_length=64
    )

    @field_validator("copied_at")
    @classmethod
    def _valid_copy_time(cls, value: str) -> str:
        datetime.strptime(value, "%Y-%m-%dT%H:%M:%SZ")
        return value


def parse_builtin_source(value: Any) -> dict[str, Any]:
    try:
        return BuiltinSkillSource.model_validate(value).model_dump()
    except ValidationError as exc:
        raise ExtensionRegistryError(
            "extension_skill_source_invalid",
            "Skill source must contain a built-in id, registry version, package SHA256 and UTC copy time.",
        ) from exc


def _builtin_package(builtin_id: str) -> dict[str, Any]:
    # Keep catalogue dependencies out of ordinary user-Skill registry imports.
    from easyicu.research_agent.method_skills import (
        MethodSkillPackageNotFound,
        method_skill_package,
    )

    try:
        return method_skill_package(builtin_id, enabled=True)
    except MethodSkillPackageNotFound as exc:
        raise ExtensionRegistryError(
            "extension_skill_source_not_found",
            "The requested built-in Skill package is unavailable.",
        ) from exc


def _copy_document(
    metadata: Mapping[str, Any], sections: list[str], omitted: list[str]
) -> str:
    frontmatter = {
        **metadata,
        "source": {**metadata["source"], "omitted_reference_paths": omitted},
    }
    return (
        "---\n"
        + yaml.safe_dump(frontmatter, sort_keys=False, allow_unicode=True)
        + "---\n\n"
        + "\n\n".join(sections)
        + "\n"
    )


def builtin_copy_markdown(builtin_id: str, *, name: str | None, copied_at: str) -> str:
    from easyicu.research_agent.method_skills import METHOD_SKILL_REGISTRY_VERSION

    from .registry import MAX_SKILL_BYTES, parse_skill_markdown

    package = _builtin_package(builtin_id)
    files = {file["path"]: file["content"] for file in package["files"]}
    _, raw_metadata, body = files["SKILL.md"].split("---\n", 2)
    metadata = yaml.safe_load(raw_metadata)
    metadata["name"] = (
        re.sub(r"[^a-z0-9-]+", "-", package["skill_id"])[:64].rstrip("-")
        if name is None
        else name
    )
    metadata["category"] = (
        re.sub(r"[^\w &+.-]", " ", metadata["category"])[:64].strip() or "General"
    )
    main = parse_skill_markdown(
        "---\n"
        + yaml.safe_dump(metadata, sort_keys=False, allow_unicode=True)
        + "---\n"
        + body
    )
    source = BuiltinSkillSource(
        builtin_id=package["skill_id"],
        registry_version=METHOD_SKILL_REGISTRY_VERSION,
        package_sha256=package["package_sha256"],
        copied_at=copied_at,
    )
    metadata = {
        "name": main["name"],
        "description": main["description"],
        "category": main["category"],
        "disable-model-invocation": main["disable_model_invocation"],
        "source": source.model_dump(),
    }
    sections = [COPY_BOUNDARY, main["instructions"]]
    references = [
        path
        for path in sorted(files)
        if path.startswith("references/") and path.endswith(".md")
    ]
    omitted = list(references)
    document = _copy_document(metadata, sections, omitted)
    # Never shorten the main instruction or omit provenance to make room.
    parse_skill_markdown(document)
    for path in references:
        candidate_sections = [
            *sections,
            f"## Copied reference: {path}\n\n{files[path].strip()}",
        ]
        candidate_omitted = [item for item in omitted if item != path]
        candidate = _copy_document(metadata, candidate_sections, candidate_omitted)
        if len(candidate.encode("utf-8")) <= MAX_SKILL_BYTES:
            sections, omitted, document = (
                candidate_sections,
                candidate_omitted,
                candidate,
            )
    return document


def builtin_source_status(source: Mapping[str, Any] | None) -> dict[str, Any]:
    """Compare current package identity without modifying the frozen copy."""
    if source is None:
        return {"source": None, "source_updated": False, "source_available": None}
    validated = parse_builtin_source(source)
    try:
        current = _builtin_package(validated["builtin_id"])
    except ExtensionRegistryError as exc:
        if exc.code != "extension_skill_source_not_found":
            raise
        return {"source": validated, "source_updated": None, "source_available": False}
    return {
        "source": validated,
        "source_updated": current["package_sha256"] != validated["package_sha256"],
        "source_available": True,
    }
