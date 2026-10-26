"""Host read models for the data map, outside Planner structured output."""

from typing import Literal

from pydantic import Field, field_serializer

from easyicu.databases.profiles import DatabaseProfile
from easyicu.databases.research_capabilities import (
    ConceptAvailability, Fact, MortalityCapability, NumericRepresentation, ResearchCapabilityProfile,
)
from easyicu.databases.research_requirements import CapabilityDecision
from easyicu.research_agent.research_context.export_selection import SourceIdentity


class UndeclaredMetadata(Fact):
    value: None = None
    reason_code: Literal["not_declared_by_metadata_owners"] = "not_declared_by_metadata_owners"


class DatabaseCapabilities(ResearchCapabilityProfile):
    # The selected concept page is returned separately, never the whole matrix.
    concepts: tuple[ConceptAvailability, ...] = Field(default=(), exclude=True)


class ConceptCell(Fact):
    concept: str
    unit: str
    audience: Literal["clinical", "internal"]
    status: Literal["available", "substituted", "missing"]
    capability: ConceptAvailability
    substitute: NumericRepresentation | MortalityCapability | None = None
    reason_code: str | None = None


class StudyCell(Fact):
    family: str
    status: Literal["exact", "substitution", "refused"]
    scope: Literal["family_minimum_requirements"] = "family_minimum_requirements"
    decision: CapabilityDecision


class RegisteredSource(Fact):
    identity: SourceIdentity
    kind: Literal["synthetic", "public_demo", "unrecorded"]
    is_demo: bool
    basis: Literal["stored_registration"] = "stored_registration"

    @field_serializer("identity")
    def serialize_identity(self, value: SourceIdentity) -> dict[str, str]:
        return value.to_dict()


class DatabaseMapRow(Fact):
    database: str
    profile: DatabaseProfile
    capabilities: DatabaseCapabilities
    years: UndeclaredMetadata = Field(default_factory=UndeclaredMetadata)
    population_size: UndeclaredMetadata = Field(default_factory=UndeclaredMetadata)
    sampling_resolution: UndeclaredMetadata = Field(default_factory=UndeclaredMetadata)
    real_data_acceptance: Literal["not_accepted"] = "not_accepted"
    concepts: tuple[ConceptCell, ...]
    studies: tuple[StudyCell, ...]
    sources: tuple[RegisteredSource, ...]
    source_total: int
    source_next_offset: int | None


class DataMap(Fact):
    schema_version: Literal["easyicu.data-map/1"] = "easyicu.data-map/1"
    basis: Literal["packaged_metadata"] = "packaged_metadata"
    databases: tuple[DatabaseMapRow, ...]
    concept_total: int
    concept_offset: int
    concept_limit: int
    concept_next_offset: int | None
    source_offset: int
    source_limit: int
