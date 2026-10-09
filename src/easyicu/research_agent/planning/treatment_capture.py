"""What an absent record of a treatment means, per database and concept.

A target trial's comparison strategy -- do not start the treatment within the
grace period -- is read from records: a stay with no record of the treatment
in that window is taken as untreated.  Whether that reading holds is a fact
about the source, not about a question, so it is stated once per database and
treatment concept, in a registry beside the concept dictionaries
(``data/treatment-capture-registry.json``):

* where the source records the treatment (``capture_setting``);
* what an absent record there means (``absent_in_capture``): the endpoint
  vocabulary's ``absent_row_is_no_event`` or ``absent_row_is_unmeasured``
  (``contracts.endpoint``), or ``unknown``;
* whether a use before ICU admission is visible;
* which drugs the concept records (``agents``), each tied to the dictionary
  component or source item it is read from, so the list can be checked
  against the dictionaries rather than trusted as written.

Each entry carries its ``basis``.  A ``development_assumption`` is a reading
the development adopted: each study's researcher confirms it at plan
approval, and only an explicit act of a data owner or a user upgrades it to
``data_owner_attested`` or ``user_attested``.  The registry also names the
drug classes a study may mean (``treatment_classes``), so the host can tell
whether a treatment's concepts record every drug of its class; which drugs a
class counts is a development reading as well, and is confirmed with it.  The
registry lives outside the locked concept dictionaries; a compiled trial
records its sha256.
"""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Literal, Mapping, Optional

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

TREATMENT_CAPTURE_REGISTRY_SCHEMA_VERSION = "easyicu.treatment_capture_registry/1"
TREATMENT_CAPTURE_REGISTRY_FILENAME = "treatment-capture-registry.json"

AbsentInCapture = Literal[
    "absent_row_is_no_event", "absent_row_is_unmeasured", "unknown"
]
CaptureBasis = Literal["development_assumption", "data_owner_attested", "user_attested"]
CaptureSetting = Literal["icu"]
DefinitionDictionary = Literal["concept-dict.json", "sofa2-dict.json"]

_NAME = r"^[a-z][a-z0-9_]{0,63}$"


def _names_once(value: tuple[str, ...], what: str) -> tuple[str, ...]:
    if len(set(value)) != len(value):
        raise ValueError(f"name each {what} once")
    return tuple(sorted(value))


class TreatmentClass(BaseModel):
    """The drugs a class of treatment consists of."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    agents: tuple[str, ...] = Field(min_length=1)
    note: str = Field(min_length=2, max_length=400)

    @field_validator("agents")
    @classmethod
    def _agents(cls, value: tuple[str, ...]) -> tuple[str, ...]:
        return _names_once(value, "agent of a class")


class ComponentAgent(BaseModel):
    """A concept the treatment concept is derived from, and the drug it records."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    component: str = Field(pattern=_NAME)
    agent: str = Field(pattern=_NAME)


class SourceItemAgent(BaseModel):
    """A source item the treatment concept reads, its label, and the drug it records."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    item_id: int = Field(gt=0)
    #: The item's label as the dictionary's source note gives it.
    label: str = Field(min_length=2, max_length=80)
    agent: str = Field(pattern=_NAME)


class CaptureDefinition(BaseModel):
    """Where the dictionaries define the drugs a concept records."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    dictionary: DefinitionDictionary
    #: The concepts the concept is derived from, each with its drug.
    components: tuple[ComponentAgent, ...] = ()
    #: The source items a concept that reads items directly maps, each with its drug.
    source_items: tuple[SourceItemAgent, ...] = ()

    @model_validator(mode="after")
    def _stated(self) -> "CaptureDefinition":
        if not self.components and not self.source_items:
            raise ValueError("a definition names its components or its source items")
        names = [item.component for item in self.components]
        ids = [item.item_id for item in self.source_items]
        if len(set(names)) != len(names) or len(set(ids)) != len(ids):
            raise ValueError("a definition names each component and source item once")
        return self

    @property
    def agents(self) -> frozenset[str]:
        """The drugs the definition's components and source items record."""

        return frozenset(item.agent for item in (*self.components, *self.source_items))


class CaptureEntry(BaseModel):
    """How one database records one treatment concept."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    database: str = Field(pattern=_NAME)
    concept: str = Field(pattern=_NAME)
    agents: tuple[str, ...] = Field(min_length=1)
    definition: CaptureDefinition
    capture_setting: CaptureSetting
    absent_in_capture: AbsentInCapture
    pre_admission_visible: bool
    basis: CaptureBasis
    declared_by: str = Field(min_length=2, max_length=200)
    declared_at: str = Field(pattern=r"^\d{4}-\d{2}-\d{2}$")
    note: str = Field(min_length=2, max_length=400)

    @field_validator("agents")
    @classmethod
    def _agents(cls, value: tuple[str, ...]) -> tuple[str, ...]:
        return _names_once(value, "agent of an entry")

    @model_validator(mode="after")
    def _agents_read_from_the_definition(self) -> "CaptureEntry":
        if frozenset(self.agents) != self.definition.agents:
            raise ValueError(
                f"{self.database}/{self.concept} lists agents its definition does "
                "not read, or reads agents it does not list"
            )
        return self


class TreatmentCaptureRegistry(BaseModel):
    """Every database's statement of how it records each treatment concept."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: Literal["easyicu.treatment_capture_registry/1"]
    description: str = Field(min_length=2)
    treatment_classes: Mapping[str, TreatmentClass]
    entries: tuple[CaptureEntry, ...]

    @model_validator(mode="after")
    def _closed(self) -> "TreatmentCaptureRegistry":
        for name in self.treatment_classes:
            if not re.fullmatch(_NAME, name):
                raise ValueError(f"treatment class name {name!r} is not a token")
        keys = [(entry.database, entry.concept) for entry in self.entries]
        if len(set(keys)) != len(keys):
            raise ValueError("state each database's concept once")
        known = {
            agent for item in self.treatment_classes.values() for agent in item.agents
        }
        for entry in self.entries:
            unknown = sorted(set(entry.agents) - known)
            if unknown:
                raise ValueError(
                    f"{entry.database}/{entry.concept} records agents no class "
                    f"names: {unknown}"
                )
        return self

    def entry(self, database: str, concept: str) -> Optional[CaptureEntry]:
        """The entry for ``concept`` in ``database``; ``None`` when none is stated."""

        key = (str(database or "").strip().lower(), str(concept or "").strip())
        return next(
            (item for item in self.entries if (item.database, item.concept) == key),
            None,
        )

    def class_agents(self, name: str) -> Optional[frozenset[str]]:
        """The drugs of the class ``name``; ``None`` for a class it does not name."""

        item = self.treatment_classes.get(str(name or "").strip())
        return frozenset(item.agents) if item is not None else None


@dataclass(frozen=True)
class LoadedCaptureRegistry:
    """The registry as read, with the sha256 of the bytes it was read from."""

    registry: TreatmentCaptureRegistry
    sha256: str


def treatment_capture_registry_path() -> Path:
    """The packaged registry file."""

    return (
        Path(__file__).resolve().parents[2]
        / "data"
        / TREATMENT_CAPTURE_REGISTRY_FILENAME
    )


def load_treatment_capture_registry(
    path: Optional[Path] = None,
) -> LoadedCaptureRegistry:
    """Read and validate a registry file; the packaged one by default."""

    raw = Path(path or treatment_capture_registry_path()).read_bytes()
    return LoadedCaptureRegistry(
        registry=TreatmentCaptureRegistry.model_validate_json(raw),
        sha256=hashlib.sha256(raw).hexdigest(),
    )


@lru_cache(maxsize=1)
def packaged_treatment_capture_registry() -> LoadedCaptureRegistry:
    """The packaged registry, read once per process."""

    return load_treatment_capture_registry()


__all__ = [
    "TREATMENT_CAPTURE_REGISTRY_FILENAME",
    "TREATMENT_CAPTURE_REGISTRY_SCHEMA_VERSION",
    "AbsentInCapture",
    "CaptureBasis",
    "CaptureDefinition",
    "CaptureEntry",
    "CaptureSetting",
    "ComponentAgent",
    "LoadedCaptureRegistry",
    "SourceItemAgent",
    "TreatmentCaptureRegistry",
    "TreatmentClass",
    "load_treatment_capture_registry",
    "packaged_treatment_capture_registry",
    "treatment_capture_registry_path",
]
