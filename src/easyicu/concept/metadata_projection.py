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
from typing import Any, Mapping, Optional, Sequence

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
    EVENT_STATUS = "event_status"
    EVENT_FRACTION = "event_fraction"
    FIRST_OBSERVATION_TIME = "first_observation_time"
    LAST_OBSERVATION_TIME = "last_observation_time"
    EVENT_TIME = "event_time"


def is_range_preserving_projection(
    role: ConceptColumnRole,
    aggregation: Optional[str],
) -> bool:
    """Whether a physical projection preserves single-measurement ranges.

    This is representation semantics only; it does not assign an analysis
    role.  Callers must pass the already canonical aggregation recorded by the
    typed projection authority.
    """

    if not isinstance(role, ConceptColumnRole):
        raise MetadataProjectionError("role must be a ConceptColumnRole")
    return role is ConceptColumnRole.VALUE or (
        role is ConceptColumnRole.NUMERIC_AGGREGATE
        and aggregation in _RANGE_PRESERVING_AGGREGATIONS
    )


_DERIVED_ROLE_TRANSITIONS = {
    ConceptColumnRole.VALUE: frozenset(
        {
            ConceptColumnRole.VALUE,
            ConceptColumnRole.NUMERIC_AGGREGATE,
            ConceptColumnRole.COUNT,
            ConceptColumnRole.MEASUREMENT_STATUS,
            ConceptColumnRole.FIRST_OBSERVATION_TIME,
            ConceptColumnRole.LAST_OBSERVATION_TIME,
        }
    ),
    ConceptColumnRole.EVENT_STATUS: frozenset(
        {
            ConceptColumnRole.EVENT_STATUS,
            ConceptColumnRole.EVENT_FRACTION,
            ConceptColumnRole.COUNT,
            ConceptColumnRole.MEASUREMENT_STATUS,
            ConceptColumnRole.FIRST_OBSERVATION_TIME,
            ConceptColumnRole.LAST_OBSERVATION_TIME,
            ConceptColumnRole.EVENT_TIME,
        }
    ),
}


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
            if numeric == 0.0:
                numeric = 0.0
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

    @classmethod
    def from_dict(cls, payload: Mapping[str, object]) -> "NumericBounds":
        _require_exact_keys(payload, {"minimum", "maximum"}, label="numeric bounds")
        return cls(payload["minimum"], payload["maximum"])  # type: ignore[arg-type]


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
        if self.aggregation is not None and not isinstance(self.aggregation, str):
            raise MetadataProjectionError("aggregation must be a string when present")
        for label, value in (
            ("time_origin", self.time_origin),
            ("time_unit", self.time_unit),
        ):
            if value is not None and not isinstance(value, str):
                raise MetadataProjectionError(f"{label} must be a string when present")
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
        if self.role is ConceptColumnRole.EVENT_FRACTION and normalized != "mean":
            raise MetadataProjectionError(
                "event-fraction projections require aggregation='mean'"
            )
        if self.role is ConceptColumnRole.EVENT_STATUS and normalized not in {
            None,
            "all",
            "any",
            "first",
            "last",
            "max",
            "min",
        }:
            raise MetadataProjectionError(
                f"unsupported event-status aggregation: {normalized!r}"
            )
        if (
            self.role
            not in {
                ConceptColumnRole.NUMERIC_AGGREGATE,
                ConceptColumnRole.EVENT_STATUS,
                ConceptColumnRole.EVENT_FRACTION,
            }
            and normalized
        ):
            raise MetadataProjectionError(
                "aggregation is only valid for aggregate or event projections"
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
    """One dictionary source entry selected for the actual source database."""

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

    @classmethod
    def from_dict(cls, payload: Mapping[str, object]) -> "SourceLineage":
        expected = {
            "database",
            "table",
            "selector_variable",
            "selector_regex",
            "item_ids",
            "value_variable",
            "unit_variable",
            "time_variable",
            "duration_variable",
            "source_class_name",
            "target",
            "callback",
            "interval_iso8601",
            "semantic_parameters",
        }
        _require_exact_keys(payload, expected, label="source lineage")
        database = _required_string(payload["database"], label="lineage database")
        item_ids = payload["item_ids"]
        parameters = payload["semantic_parameters"]
        if not isinstance(item_ids, list):
            raise MetadataProjectionError("lineage item_ids must be a list")
        if not isinstance(parameters, Mapping):
            raise MetadataProjectionError(
                "lineage semantic_parameters must be an object"
            )
        if any(not isinstance(key, str) for key in parameters):
            raise MetadataProjectionError(
                "lineage semantic parameter names must be strings"
            )
        canonical_item_ids = tuple(
            sorted(
                json.dumps(
                    _canonical_json_value(value, path="lineage.item_ids"),
                    ensure_ascii=False,
                    sort_keys=True,
                    separators=(",", ":"),
                    allow_nan=False,
                )
                for value in item_ids
            )
        )
        if len(set(canonical_item_ids)) != len(canonical_item_ids):
            raise MetadataProjectionError(
                "lineage item_ids must not contain duplicates"
            )
        parsed = cls(
            database=database,
            table=_optional_string(payload["table"], label="lineage table"),
            selector_variable=_optional_string(
                payload["selector_variable"], label="lineage selector_variable"
            ),
            selector_regex=_optional_string(
                payload["selector_regex"], label="lineage selector_regex"
            ),
            item_ids_json=canonical_item_ids,
            value_variable=_optional_string(
                payload["value_variable"], label="lineage value_variable"
            ),
            unit_variable=_optional_string(
                payload["unit_variable"], label="lineage unit_variable"
            ),
            time_variable=_optional_string(
                payload["time_variable"], label="lineage time_variable"
            ),
            duration_variable=_optional_string(
                payload["duration_variable"], label="lineage duration_variable"
            ),
            source_class_name=_optional_string(
                payload["source_class_name"], label="lineage source_class_name"
            ),
            target=_optional_string(payload["target"], label="lineage target"),
            callback=_optional_string(payload["callback"], label="lineage callback"),
            interval_iso8601=_optional_string(
                payload["interval_iso8601"], label="lineage interval_iso8601"
            ),
            semantic_parameters=tuple(
                (
                    key,
                    json.dumps(
                        _canonical_json_value(
                            parameters[key], path=f"lineage.semantic_parameters.{key}"
                        ),
                        ensure_ascii=False,
                        sort_keys=True,
                        separators=(",", ":"),
                        allow_nan=False,
                    ),
                )
                for key in sorted(parameters)
            ),
        )
        if parsed.to_dict() != dict(payload):
            raise MetadataProjectionError("source lineage is not canonical")
        return parsed


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
    dictionary_source_database: Optional[str]
    source_resolution_chain: tuple[str, ...]
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
            "dictionary_source_database": self.dictionary_source_database,
            "source_resolution_chain": list(self.source_resolution_chain),
            "available_databases": list(self.available_databases),
            "source_declared_for_database": self.source_declared_for_database,
            "availability_basis": self.availability_basis,
            "source_lineage": [entry.to_dict() for entry in self.source_lineage],
            "description": self.description,
            "category": self.category,
            "class_name": self.class_name,
            "derived_from_concepts": list(self.derived_from_concepts),
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, object]) -> "ConceptColumnMetadata":
        """Parse one strict canonical metadata record from an authority payload."""

        expected = {
            "schema_version",
            "column_name",
            "source_concept",
            "role",
            "aggregation",
            "canonical_unit",
            "accepted_units",
            "extraction_bounds",
            "analysis_plausibility_range",
            "allowed_values",
            "time_origin",
            "time_unit",
            "source_database",
            "dictionary_source_database",
            "source_resolution_chain",
            "available_databases",
            "source_declared_for_database",
            "availability_basis",
            "source_lineage",
            "description",
            "category",
            "class_name",
            "derived_from_concepts",
        }
        _require_exact_keys(payload, expected, label="concept column metadata")
        if payload["schema_version"] != METADATA_SCHEMA_VERSION:
            raise MetadataProjectionError("unsupported concept metadata schema")
        try:
            role = ConceptColumnRole(payload["role"])
        except (TypeError, ValueError) as exc:
            raise MetadataProjectionError("invalid concept column role") from exc
        spec = ColumnProjectionSpec(
            column_name=_required_string(
                payload["column_name"], label="metadata column_name"
            ),
            source_concept=_required_string(
                payload["source_concept"], label="metadata source_concept"
            ),
            role=role,
            aggregation=_optional_string(
                payload["aggregation"], label="metadata aggregation"
            ),
            time_origin=_optional_string(
                payload["time_origin"], label="metadata time_origin"
            ),
            time_unit=_optional_string(
                payload["time_unit"], label="metadata time_unit"
            ),
        )
        accepted_units = _canonical_string_list(
            payload["accepted_units"], label="metadata accepted_units", sorted_=False
        )
        source_chain = _canonical_string_list(
            payload["source_resolution_chain"],
            label="metadata source_resolution_chain",
            sorted_=False,
        )
        available_databases = _canonical_string_list(
            payload["available_databases"],
            label="metadata available_databases",
            sorted_=True,
        )
        derived = _canonical_string_list(
            payload["derived_from_concepts"],
            label="metadata derived_from_concepts",
            sorted_=True,
        )
        raw_allowed = payload["allowed_values"]
        if raw_allowed is None:
            allowed_values = None
        else:
            if not isinstance(raw_allowed, list) or any(
                not isinstance(value, int) or isinstance(value, bool)
                for value in raw_allowed
            ):
                raise MetadataProjectionError(
                    "metadata allowed_values must be an integer list or null"
                )
            if len(set(raw_allowed)) != len(raw_allowed):
                raise MetadataProjectionError(
                    "metadata allowed_values must not contain duplicates"
                )
            allowed_values = tuple(raw_allowed)
        raw_lineage = payload["source_lineage"]
        if not isinstance(raw_lineage, list) or not all(
            isinstance(value, Mapping) for value in raw_lineage
        ):
            raise MetadataProjectionError("metadata source_lineage must be a list")
        source_declared = payload["source_declared_for_database"]
        if source_declared is not None and not isinstance(source_declared, bool):
            raise MetadataProjectionError(
                "metadata source_declared_for_database must be boolean or null"
            )
        parsed = cls(
            column_name=spec.column_name,
            source_concept=spec.source_concept,
            role=role,
            aggregation=spec.aggregation,
            canonical_unit=_optional_string(
                payload["canonical_unit"], label="metadata canonical_unit"
            ),
            accepted_units=accepted_units,
            extraction_bounds=_optional_bounds(payload["extraction_bounds"]),
            analysis_plausibility_range=_optional_bounds(
                payload["analysis_plausibility_range"]
            ),
            allowed_values=allowed_values,
            time_origin=spec.time_origin,
            time_unit=spec.time_unit,
            source_database=_optional_string(
                payload["source_database"], label="metadata source_database"
            ),
            dictionary_source_database=_optional_string(
                payload["dictionary_source_database"],
                label="metadata dictionary_source_database",
            ),
            source_resolution_chain=source_chain,
            available_databases=available_databases,
            source_declared_for_database=source_declared,
            availability_basis=_required_string(
                payload["availability_basis"], label="metadata availability_basis"
            ),
            source_lineage=tuple(
                SourceLineage.from_dict(value) for value in raw_lineage
            ),
            description=_optional_string(
                payload["description"], label="metadata description"
            ),
            category=_optional_string(payload["category"], label="metadata category"),
            class_name=_optional_string(
                payload["class_name"], label="metadata class_name"
            ),
            derived_from_concepts=derived,
        )
        _validate_metadata_role(parsed)
        if parsed.to_dict() != dict(payload):
            raise MetadataProjectionError("concept column metadata is not canonical")
        return parsed


def _clean_optional(value: object) -> Optional[str]:
    text = str(value or "").strip()
    return text or None


def _require_exact_keys(
    payload: Mapping[str, object], expected: set[str], *, label: str
) -> None:
    if not isinstance(payload, Mapping) or any(
        not isinstance(key, str) for key in payload
    ):
        raise MetadataProjectionError(f"{label} must be an object with string keys")
    actual = set(payload)
    if actual != expected:
        missing = sorted(expected - actual)
        extra = sorted(actual - expected)
        raise MetadataProjectionError(
            f"{label} keys do not match schema (missing={missing}, extra={extra})"
        )


def _required_string(value: object, *, label: str) -> str:
    if not isinstance(value, str) or not value.strip() or value != value.strip():
        raise MetadataProjectionError(f"{label} must be a canonical non-empty string")
    return value


def _optional_string(value: object, *, label: str) -> Optional[str]:
    if value is None:
        return None
    return _required_string(value, label=label)


def _canonical_string_list(
    value: object, *, label: str, sorted_: bool
) -> tuple[str, ...]:
    if not isinstance(value, list):
        raise MetadataProjectionError(f"{label} must be a list")
    parsed = tuple(_required_string(item, label=label) for item in value)
    if len(set(parsed)) != len(parsed):
        raise MetadataProjectionError(f"{label} must not contain duplicates")
    if sorted_ and list(parsed) != sorted(parsed):
        raise MetadataProjectionError(f"{label} must be canonically sorted")
    return parsed


def _optional_bounds(value: object) -> Optional[NumericBounds]:
    if value is None:
        return None
    if not isinstance(value, Mapping):
        raise MetadataProjectionError("metadata bounds must be an object or null")
    return NumericBounds.from_dict(value)


def _validate_metadata_role(metadata: ConceptColumnMetadata) -> None:
    role = metadata.role
    time_roles = {
        ConceptColumnRole.FIRST_OBSERVATION_TIME,
        ConceptColumnRole.LAST_OBSERVATION_TIME,
        ConceptColumnRole.EVENT_TIME,
    }
    physiological = (
        metadata.canonical_unit is not None
        or bool(metadata.accepted_units)
        or metadata.extraction_bounds is not None
        or metadata.analysis_plausibility_range is not None
    )
    if (
        role
        in {
            ConceptColumnRole.COUNT,
            ConceptColumnRole.MEASUREMENT_STATUS,
            ConceptColumnRole.EVENT_STATUS,
            ConceptColumnRole.EVENT_FRACTION,
            *time_roles,
        }
        and physiological
    ):
        raise MetadataProjectionError(
            f"{role.value} metadata must not inherit physiological unit or ranges"
        )
    if role in time_roles:
        if metadata.time_origin is None or metadata.time_unit is None:
            raise MetadataProjectionError("time metadata requires origin and unit")
        if metadata.allowed_values is not None:
            raise MetadataProjectionError("time metadata cannot declare allowed_values")
    elif metadata.time_origin is not None or metadata.time_unit is not None:
        raise MetadataProjectionError(
            "non-time metadata cannot declare time coordinates"
        )
    if role in {
        ConceptColumnRole.MEASUREMENT_STATUS,
        ConceptColumnRole.EVENT_STATUS,
    }:
        if metadata.allowed_values != (0, 1):
            raise MetadataProjectionError(
                f"{role.value} metadata requires allowed_values [0, 1]"
            )
    elif metadata.allowed_values is not None:
        raise MetadataProjectionError(
            f"{role.value} metadata must not declare allowed_values"
        )
    if metadata.canonical_unit is None and metadata.accepted_units:
        raise MetadataProjectionError("accepted_units require a canonical_unit")
    if metadata.canonical_unit is not None and (
        not metadata.accepted_units
        or metadata.accepted_units[0] != metadata.canonical_unit
    ):
        raise MetadataProjectionError("canonical_unit must be the first accepted unit")
    if metadata.source_database is None:
        if metadata.source_resolution_chain:
            raise MetadataProjectionError(
                "source resolution chain requires source_database"
            )
    elif (
        not metadata.source_resolution_chain
        or metadata.source_resolution_chain[0] != metadata.source_database
    ):
        raise MetadataProjectionError(
            "source resolution chain must start with source_database"
        )
    if metadata.dictionary_source_database is not None and (
        metadata.dictionary_source_database not in metadata.source_resolution_chain
        or metadata.dictionary_source_database not in metadata.available_databases
    ):
        raise MetadataProjectionError(
            "dictionary source database is not authorized by metadata"
        )
    if any(
        entry.database != metadata.dictionary_source_database
        for entry in metadata.source_lineage
    ):
        raise MetadataProjectionError(
            "source lineage database does not match dictionary source"
        )


def _canonical_json_value(value: object, *, path: str) -> object:
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise MetadataProjectionError(f"{path} must not contain non-finite values")
        return 0.0 if value == 0.0 else value
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


def _source_has_executable_anchor(source: ConceptSource) -> bool:
    """Return whether a source entry can identify a loader or physical relation.

    Selector IDs, units, intervals and extra parameters only qualify an already
    anchored source.  Treating those attachments as a direct source by
    themselves would overstate dictionary availability.
    """

    return bool(source.table or source.callback or source.class_name or source.target)


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
    source_database_class_prefixes: Sequence[str] = (),
    analysis_plausibility_range: Optional[NumericBounds] = None,
) -> ConceptColumnMetadata:
    """Project dictionary/run/analysis authorities without conflating them."""

    if source_database is not None and not isinstance(source_database, str):
        raise MetadataProjectionError("source_database must be a string when present")
    if isinstance(source_database, str) and not source_database.strip():
        raise MetadataProjectionError("source_database must be non-empty when present")
    if analysis_plausibility_range is not None and not isinstance(
        analysis_plausibility_range, NumericBounds
    ):
        raise MetadataProjectionError(
            "analysis_plausibility_range must be NumericBounds when present"
        )
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
        and not is_range_preserving_projection(spec.role, spec.aggregation)
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
    range_preserving = is_range_preserving_projection(spec.role, spec.aggregation)
    time_like = spec.role in {
        ConceptColumnRole.FIRST_OBSERVATION_TIME,
        ConceptColumnRole.LAST_OBSERVATION_TIME,
        ConceptColumnRole.EVENT_TIME,
    }

    normalized_database = _clean_optional(source_database)
    if normalized_database is not None:
        normalized_database = normalized_database.lower()
    if isinstance(source_database_class_prefixes, (str, bytes)):
        raise MetadataProjectionError(
            "source_database_class_prefixes must be a sequence of database names"
        )
    normalized_prefixes: list[str] = []
    for value in source_database_class_prefixes:
        if not isinstance(value, str) or not value.strip():
            raise MetadataProjectionError(
                "source database class prefixes must be non-empty strings"
            )
        normalized_prefixes.append(value.strip().lower())
    if normalized_database is None and normalized_prefixes:
        raise MetadataProjectionError(
            "source database class prefixes require source_database"
        )
    resolution_chain = tuple(
        dict.fromkeys(
            [
                *([normalized_database] if normalized_database is not None else []),
                *normalized_prefixes,
            ]
        )
    )
    available_databases = tuple(sorted(str(name) for name in definition.sources))
    dictionary_source_database = next(
        (name for name in resolution_chain if name in definition.sources),
        None,
    )
    actual_sources = (
        tuple(
            source
            for source in definition.sources.get(dictionary_source_database, ())
            if _source_has_executable_anchor(source)
        )
        if dictionary_source_database is not None
        else ()
    )
    lineage = tuple(
        sorted(
            (
                _source_lineage(dictionary_source_database or "", item)
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
    database_declared = dictionary_source_database is not None
    inherited_source = bool(
        dictionary_source_database is not None
        and dictionary_source_database != normalized_database
    )
    has_derived_definition = bool(derived or definition.callback)
    if normalized_database is None:
        source_declared_for_database: Optional[bool] = None
        availability_basis = "source_database_not_supplied"
    elif actual_sources:
        source_declared_for_database = True
        availability_basis = (
            "inherited_direct_source" if inherited_source else "direct_source"
        )
    elif database_declared and has_derived_definition:
        source_declared_for_database = True
        availability_basis = (
            "inherited_declared_derived_or_unresolved"
            if inherited_source
            else "declared_derived_or_unresolved"
        )
    elif database_declared:
        source_declared_for_database = True
        availability_basis = (
            "inherited_declared_without_direct_source"
            if inherited_source
            else "declared_without_direct_source"
        )
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
            (0, 1)
            if spec.role
            in {
                ConceptColumnRole.MEASUREMENT_STATUS,
                ConceptColumnRole.EVENT_STATUS,
            }
            else None
        ),
        time_origin=(spec.time_origin if time_like else None),
        time_unit=(spec.time_unit if time_like else None),
        source_database=normalized_database,
        dictionary_source_database=dictionary_source_database,
        source_resolution_chain=resolution_chain,
        available_databases=available_databases,
        source_declared_for_database=source_declared_for_database,
        availability_basis=availability_basis,
        source_lineage=lineage,
        description=_clean_optional(definition.description),
        category=_clean_optional(definition.category),
        class_name=_clean_optional(definition.class_name),
        derived_from_concepts=derived,
    )


def derive_concept_column_metadata(
    source: ConceptColumnMetadata,
    *,
    spec: ColumnProjectionSpec,
) -> ConceptColumnMetadata:
    """Derive one materialized-column contract from sealed source metadata.

    This is deliberately a *projection of an existing authority*, not a second
    dictionary lookup.  A cohort materializer may change the physical column
    name, aggregation, representation role, or time coordinate, while every
    source/database/lineage coordinate remains byte-for-byte inherited from
    the verified export metadata.
    """

    if not isinstance(source, ConceptColumnMetadata):
        raise MetadataProjectionError("source metadata must be ConceptColumnMetadata")
    if not isinstance(spec, ColumnProjectionSpec):
        raise MetadataProjectionError("spec must be ColumnProjectionSpec")
    if spec.source_concept != source.source_concept:
        raise MetadataProjectionError(
            "derived projection source_concept does not match source authority"
        )
    permitted_roles = _DERIVED_ROLE_TRANSITIONS.get(source.role)
    if permitted_roles is None or spec.role not in permitted_roles:
        raise MetadataProjectionError(
            "derived projection role is not authorized by the sealed source role "
            f"({source.role.value!r} -> {spec.role.value!r})"
        )

    value_like = spec.role in {
        ConceptColumnRole.VALUE,
        ConceptColumnRole.NUMERIC_AGGREGATE,
    }
    range_preserving = is_range_preserving_projection(spec.role, spec.aggregation)
    time_like = spec.role in {
        ConceptColumnRole.FIRST_OBSERVATION_TIME,
        ConceptColumnRole.LAST_OBSERVATION_TIME,
        ConceptColumnRole.EVENT_TIME,
    }
    derived = ConceptColumnMetadata(
        column_name=spec.column_name,
        source_concept=source.source_concept,
        role=spec.role,
        aggregation=spec.aggregation,
        canonical_unit=(source.canonical_unit if value_like else None),
        accepted_units=(source.accepted_units if value_like else ()),
        extraction_bounds=(source.extraction_bounds if range_preserving else None),
        analysis_plausibility_range=(
            source.analysis_plausibility_range if range_preserving else None
        ),
        allowed_values=(
            (0, 1)
            if spec.role
            in {
                ConceptColumnRole.MEASUREMENT_STATUS,
                ConceptColumnRole.EVENT_STATUS,
            }
            else None
        ),
        time_origin=(spec.time_origin if time_like else None),
        time_unit=(spec.time_unit if time_like else None),
        source_database=source.source_database,
        dictionary_source_database=source.dictionary_source_database,
        source_resolution_chain=source.source_resolution_chain,
        available_databases=source.available_databases,
        source_declared_for_database=source.source_declared_for_database,
        availability_basis=source.availability_basis,
        source_lineage=source.source_lineage,
        description=source.description,
        category=source.category,
        class_name=source.class_name,
        derived_from_concepts=source.derived_from_concepts,
    )
    # Exercise the strict parser here as a defense against future additions to
    # ConceptColumnMetadata that this explicit authority-preserving copy must
    # not silently omit.
    return ConceptColumnMetadata.from_dict(derived.to_dict())


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
    "derive_concept_column_metadata",
    "is_range_preserving_projection",
    "metadata_payload_sha256",
    "metadata_sha256",
    "project_concept_column_metadata",
]
