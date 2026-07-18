"""Typed, case-neutral projection of concept metadata onto physical columns.

This leaf module separates three authorities that older call sites conflated:

* the packaged dictionary describes extraction units, bounds and availability;
* the run/export declares the database that actually produced this column;
* the analysis layer may supply a distinct plausibility range.

Callers must provide an explicit :class:`ColumnProjectionSpec`.  In particular,
this module never guesses that an arbitrary ``*_max`` column is a numeric
aggregate: event-presence and categorical summaries can use the same suffix.
"""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass
from enum import Enum
from typing import Any, Mapping, Optional

from .schema import ConceptDefinition, ConceptSource

METADATA_SCHEMA_VERSION = "easyicu.concept_column_metadata/1"
_RANGE_PRESERVING_AGGREGATIONS = {
    "first",
    "last",
    "max",
    "mean",
    "median",
    "min",
}
_SUPPORTED_NUMERIC_AGGREGATIONS = _RANGE_PRESERVING_AGGREGATIONS | {"sum"}


class MetadataProjectionError(ValueError):
    """Raised when a typed column projection is internally inconsistent."""


class ConceptColumnRole(str, Enum):
    """Physical role of one analysis column relative to its source concept."""

    VALUE = "value"
    NUMERIC_AGGREGATE = "numeric_aggregate"
    COUNT = "count"
    MEASUREMENT_STATUS = "measurement_status"
    FIRST_OBSERVATION_TIME = "first_observation_time"
    LAST_OBSERVATION_TIME = "last_observation_time"
    EVENT_TIME = "event_time"


@dataclass(frozen=True, slots=True)
class NumericBounds:
    """Finite, ordered numeric bounds; either endpoint may be unspecified."""

    minimum: Optional[float] = None
    maximum: Optional[float] = None

    def __post_init__(self) -> None:
        for label, value in (("minimum", self.minimum), ("maximum", self.maximum)):
            if value is None:
                continue
            if isinstance(value, bool):
                raise MetadataProjectionError(f"{label} must be finite when present")
            try:
                numeric = float(value)
            except (TypeError, ValueError, OverflowError) as exc:
                raise MetadataProjectionError(
                    f"{label} must be numeric when present"
                ) from exc
            if not math.isfinite(numeric):
                raise MetadataProjectionError(f"{label} must be finite when present")
            object.__setattr__(self, label, numeric)
        if (
            self.minimum is not None
            and self.maximum is not None
            and float(self.minimum) > float(self.maximum)
        ):
            raise MetadataProjectionError("minimum must not exceed maximum")

    def to_dict(self) -> dict[str, Optional[float]]:
        return {
            "minimum": None if self.minimum is None else float(self.minimum),
            "maximum": None if self.maximum is None else float(self.maximum),
        }


@dataclass(frozen=True, slots=True)
class ColumnProjectionSpec:
    """Explicit binding from one physical column to one source concept role."""

    column_name: str
    source_concept: str
    role: ConceptColumnRole
    aggregation: Optional[str] = None
    time_origin: Optional[str] = None
    time_unit: Optional[str] = None

    def __post_init__(self) -> None:
        if not isinstance(self.role, ConceptColumnRole):
            raise MetadataProjectionError("role must be a ConceptColumnRole")
        if not isinstance(self.column_name, str) or not isinstance(
            self.source_concept, str
        ):
            raise MetadataProjectionError(
                "column_name and source_concept must be strings"
            )
        if not self.column_name.strip() or not self.source_concept.strip():
            raise MetadataProjectionError(
                "column_name and source_concept must be non-empty"
            )
        normalized = str(self.aggregation or "").strip().lower() or None
        normalized_origin = _clean_optional(self.time_origin)
        normalized_time_unit = _clean_optional(self.time_unit)
        if self.role is ConceptColumnRole.NUMERIC_AGGREGATE and normalized is None:
            raise MetadataProjectionError(
                "numeric aggregate projections require an explicit aggregation"
            )
        if (
            self.role is ConceptColumnRole.NUMERIC_AGGREGATE
            and normalized not in _SUPPORTED_NUMERIC_AGGREGATIONS
        ):
            raise MetadataProjectionError(
                f"unsupported numeric aggregation: {normalized!r}"
            )
        if self.role is not ConceptColumnRole.NUMERIC_AGGREGATE and normalized:
            raise MetadataProjectionError(
                "aggregation is only valid for numeric aggregate projections"
            )
        time_like = self.role in {
            ConceptColumnRole.FIRST_OBSERVATION_TIME,
            ConceptColumnRole.LAST_OBSERVATION_TIME,
            ConceptColumnRole.EVENT_TIME,
        }
        if time_like and (normalized_origin is None or normalized_time_unit is None):
            raise MetadataProjectionError(
                "time projections require explicit time_origin and time_unit"
            )
        if not time_like and (
            normalized_origin is not None or normalized_time_unit is not None
        ):
            raise MetadataProjectionError(
                "time_origin and time_unit are only valid for time projections"
            )
        object.__setattr__(self, "aggregation", normalized)
        object.__setattr__(self, "time_origin", normalized_origin)
        object.__setattr__(self, "time_unit", normalized_time_unit)


@dataclass(frozen=True, slots=True)
class SourceLineage:
    """One dictionary source entry for the database that produced the column."""

    database: str
    table: Optional[str]
    selector_variable: Optional[str]
    selector_regex: Optional[str]
    item_ids_json: tuple[str, ...]
    value_variable: Optional[str]
    unit_variable: Optional[str]
    time_variable: Optional[str]
    duration_variable: Optional[str]
    source_class_name: Optional[str]
    target: Optional[str]
    callback: Optional[str]
    interval_iso8601: Optional[str]
    semantic_parameters: tuple[tuple[str, str], ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "database": self.database,
            "table": self.table,
            "selector_variable": self.selector_variable,
            "selector_regex": self.selector_regex,
            "item_ids": [json.loads(value_json) for value_json in self.item_ids_json],
            "value_variable": self.value_variable,
            "unit_variable": self.unit_variable,
            "time_variable": self.time_variable,
            "duration_variable": self.duration_variable,
            "source_class_name": self.source_class_name,
            "target": self.target,
            "callback": self.callback,
            "interval_iso8601": self.interval_iso8601,
            "semantic_parameters": {
                name: json.loads(value_json)
                for name, value_json in self.semantic_parameters
            },
        }


@dataclass(frozen=True, slots=True)
class ConceptColumnMetadata:
    """Canonical typed metadata for one physical analysis column.

    ``available_databases`` contains dictionary-declared database keys.  It is
    not, by itself, proof that the current export contains usable rows; callers
    must interpret it together with ``source_declared_for_database`` and
    ``availability_basis`` (and later runtime-manifest evidence).
    """

    column_name: str
    source_concept: str
    role: ConceptColumnRole
    aggregation: Optional[str]
    canonical_unit: Optional[str]
    accepted_units: tuple[str, ...]
    extraction_bounds: Optional[NumericBounds]
    analysis_plausibility_range: Optional[NumericBounds]
    allowed_values: Optional[tuple[int, ...]]
    time_origin: Optional[str]
    time_unit: Optional[str]
    source_database: Optional[str]
    available_databases: tuple[str, ...]
    source_declared_for_database: Optional[bool]
    availability_basis: str
    source_lineage: tuple[SourceLineage, ...]
    description: Optional[str]
    category: Optional[str]
    class_name: Optional[str]
    derived_from_concepts: tuple[str, ...]
    schema_version: str = METADATA_SCHEMA_VERSION

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "column_name": self.column_name,
            "source_concept": self.source_concept,
            "role": self.role.value,
            "aggregation": self.aggregation,
            "canonical_unit": self.canonical_unit,
            "accepted_units": list(self.accepted_units),
            "extraction_bounds": (
                self.extraction_bounds.to_dict() if self.extraction_bounds else None
            ),
            "analysis_plausibility_range": (
                self.analysis_plausibility_range.to_dict()
                if self.analysis_plausibility_range
                else None
            ),
            "allowed_values": (
                list(self.allowed_values) if self.allowed_values is not None else None
            ),
            "time_origin": self.time_origin,
            "time_unit": self.time_unit,
            "source_database": self.source_database,
            "available_databases": list(self.available_databases),
            "source_declared_for_database": self.source_declared_for_database,
            "availability_basis": self.availability_basis,
            "source_lineage": [entry.to_dict() for entry in self.source_lineage],
            "description": self.description,
            "category": self.category,
            "class_name": self.class_name,
            "derived_from_concepts": list(self.derived_from_concepts),
        }


def _clean_optional(value: object) -> Optional[str]:
    text = str(value or "").strip()
    return text or None


def _canonical_json_value(value: object, *, path: str) -> object:
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise MetadataProjectionError(f"{path} must not contain non-finite values")
        return value
    if isinstance(value, Mapping):
        canonical: dict[str, object] = {}
        for key, item in value.items():
            if not isinstance(key, str):
                raise MetadataProjectionError(f"{path} mapping keys must be strings")
            canonical[key] = _canonical_json_value(item, path=f"{path}.{key}")
        return {key: canonical[key] for key in sorted(canonical)}
    if isinstance(value, (list, tuple)):
        return [
            _canonical_json_value(item, path=f"{path}[{index}]")
            for index, item in enumerate(value)
        ]
    raise MetadataProjectionError(
        f"{path} contains unsupported value type {type(value).__name__}"
    )


def _semantic_parameters(source: ConceptSource) -> tuple[tuple[str, str], ...]:
    pairs: list[tuple[str, str]] = []
    raw_parameters = source.params or {}
    if any(not isinstance(name, str) for name in raw_parameters):
        raise MetadataProjectionError("source parameter names must be strings")
    for name, value in sorted(raw_parameters.items()):
        if name == "_comment":
            continue
        canonical = _canonical_json_value(value, path=f"source.params.{name}")
        pairs.append(
            (
                name,
                json.dumps(
                    canonical,
                    ensure_ascii=False,
                    sort_keys=True,
                    separators=(",", ":"),
                    allow_nan=False,
                ),
            )
        )
    return tuple(pairs)


def _source_lineage(database: str, source: ConceptSource) -> SourceLineage:
    ids = tuple(
        sorted(
            {
                json.dumps(
                    _canonical_json_value(value, path="source.ids"),
                    ensure_ascii=False,
                    sort_keys=True,
                    separators=(",", ":"),
                    allow_nan=False,
                )
                for value in (source.ids or [])
            }
        )
    )
    return SourceLineage(
        database=database,
        table=_clean_optional(source.table),
        selector_variable=_clean_optional(source.sub_var),
        selector_regex=_clean_optional(source.regex),
        item_ids_json=ids,
        value_variable=_clean_optional(source.value_var),
        unit_variable=_clean_optional(source.unit_var),
        time_variable=_clean_optional(source.index_var),
        duration_variable=_clean_optional(source.dur_var),
        source_class_name=_clean_optional(source.class_name),
        target=_clean_optional(source.target),
        callback=_clean_optional(source.callback),
        interval_iso8601=(
            _clean_optional(source.interval.isoformat())
            if source.interval is not None
            else None
        ),
        semantic_parameters=_semantic_parameters(source),
    )


def _lineage_sort_key(entry: SourceLineage) -> tuple[object, ...]:
    return (
        entry.database,
        entry.table or "",
        entry.selector_variable or "",
        entry.selector_regex or "",
        entry.item_ids_json,
        entry.value_variable or "",
        entry.unit_variable or "",
        entry.time_variable or "",
        entry.duration_variable or "",
        entry.source_class_name or "",
        entry.target or "",
        entry.callback or "",
        entry.interval_iso8601 or "",
        entry.semantic_parameters,
    )


def _dictionary_bounds(definition: ConceptDefinition) -> Optional[NumericBounds]:
    if definition.minimum is None and definition.maximum is None:
        return None
    return NumericBounds(definition.minimum, definition.maximum)


def project_concept_column_metadata(
    definition: ConceptDefinition,
    *,
    spec: ColumnProjectionSpec,
    source_database: Optional[str],
    analysis_plausibility_range: Optional[NumericBounds] = None,
) -> ConceptColumnMetadata:
    """Project dictionary/run/analysis authorities without conflating them."""

    if definition.name != spec.source_concept:
        raise MetadataProjectionError(
            "projection source_concept does not match ConceptDefinition.name"
        )
    if analysis_plausibility_range is not None and spec.role not in {
        ConceptColumnRole.VALUE,
        ConceptColumnRole.NUMERIC_AGGREGATE,
    }:
        raise MetadataProjectionError(
            "analysis plausibility ranges are only valid for value-like columns"
        )
    if (
        analysis_plausibility_range is not None
        and spec.role is ConceptColumnRole.NUMERIC_AGGREGATE
        and spec.aggregation not in _RANGE_PRESERVING_AGGREGATIONS
    ):
        raise MetadataProjectionError(
            "analysis plausibility ranges require a range-preserving aggregation"
        )

    units = tuple(
        dict.fromkeys(
            value
            for value in (str(item).strip() for item in (definition.units or []))
            if value
        )
    )
    value_like = spec.role in {
        ConceptColumnRole.VALUE,
        ConceptColumnRole.NUMERIC_AGGREGATE,
    }
    range_preserving = spec.role is ConceptColumnRole.VALUE or (
        spec.role is ConceptColumnRole.NUMERIC_AGGREGATE
        and spec.aggregation in _RANGE_PRESERVING_AGGREGATIONS
    )
    time_like = spec.role in {
        ConceptColumnRole.FIRST_OBSERVATION_TIME,
        ConceptColumnRole.LAST_OBSERVATION_TIME,
        ConceptColumnRole.EVENT_TIME,
    }

    normalized_database = _clean_optional(source_database)
    if normalized_database is not None:
        normalized_database = normalized_database.lower()
    available_databases = tuple(sorted(str(name) for name in definition.sources))
    actual_sources = (
        tuple(definition.sources.get(normalized_database, ()))
        if normalized_database is not None
        else ()
    )
    lineage = tuple(
        sorted(
            (
                _source_lineage(normalized_database or "", item)
                for item in actual_sources
            ),
            key=_lineage_sort_key,
        )
    )
    derived = tuple(
        sorted(
            dict.fromkeys(
                [
                    *(str(item) for item in definition.sub_concepts),
                    *(str(item) for item in definition.depends_on),
                ]
            )
        )
    )
    database_declared = (
        normalized_database in definition.sources
        if normalized_database is not None
        else False
    )
    has_derived_definition = bool(derived or definition.callback)
    if normalized_database is None:
        source_declared_for_database: Optional[bool] = None
        availability_basis = "source_database_not_supplied"
    elif actual_sources:
        source_declared_for_database = True
        availability_basis = "direct_source"
    elif database_declared and has_derived_definition:
        source_declared_for_database = True
        availability_basis = "declared_derived_or_unresolved"
    elif database_declared:
        source_declared_for_database = True
        availability_basis = "declared_without_direct_source"
    elif has_derived_definition:
        source_declared_for_database = False
        availability_basis = "derived_dependencies_not_resolved"
    elif definition.sources:
        source_declared_for_database = False
        availability_basis = "source_not_declared"
    else:
        source_declared_for_database = None
        availability_basis = "no_source_metadata"

    return ConceptColumnMetadata(
        column_name=spec.column_name,
        source_concept=spec.source_concept,
        role=spec.role,
        aggregation=spec.aggregation,
        canonical_unit=(units[0] if units and value_like else None),
        accepted_units=(units if value_like else ()),
        extraction_bounds=(
            _dictionary_bounds(definition) if range_preserving else None
        ),
        analysis_plausibility_range=(
            analysis_plausibility_range if range_preserving else None
        ),
        allowed_values=(
            (0, 1) if spec.role is ConceptColumnRole.MEASUREMENT_STATUS else None
        ),
        time_origin=(spec.time_origin if time_like else None),
        time_unit=(spec.time_unit if time_like else None),
        source_database=normalized_database,
        available_databases=available_databases,
        source_declared_for_database=source_declared_for_database,
        availability_basis=availability_basis,
        source_lineage=lineage,
        description=_clean_optional(definition.description),
        category=_clean_optional(definition.category),
        class_name=_clean_optional(definition.class_name),
        derived_from_concepts=derived,
    )


def canonical_metadata_bytes(metadata: ConceptColumnMetadata) -> bytes:
    """Return deterministic JSON bytes suitable for sidecar/replay binding."""

    return json.dumps(
        metadata.to_dict(),
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")


def metadata_sha256(metadata: ConceptColumnMetadata) -> str:
    """Digest the complete typed metadata projection."""

    return hashlib.sha256(canonical_metadata_bytes(metadata)).hexdigest()


def metadata_payload_sha256(payload: Mapping[str, ConceptColumnMetadata]) -> str:
    """Digest a column mapping independently of insertion order."""

    for column, metadata in payload.items():
        if not isinstance(column, str):
            raise MetadataProjectionError("metadata payload keys must be strings")
        if column != metadata.column_name:
            raise MetadataProjectionError(
                "metadata payload key must match metadata.column_name"
            )
    canonical = {
        str(column): metadata.to_dict()
        for column, metadata in sorted(payload.items(), key=lambda item: str(item[0]))
    }
    return hashlib.sha256(
        json.dumps(
            canonical,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    ).hexdigest()


__all__ = [
    "METADATA_SCHEMA_VERSION",
    "ColumnProjectionSpec",
    "ConceptColumnMetadata",
    "ConceptColumnRole",
    "MetadataProjectionError",
    "NumericBounds",
    "SourceLineage",
    "canonical_metadata_bytes",
    "metadata_payload_sha256",
    "metadata_sha256",
    "project_concept_column_metadata",
]
