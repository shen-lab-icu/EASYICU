"""Typed authority for host-materialized long-format ICU trajectories.

The trajectory is a scientific input, not a convenience side file.  Typed
cohorts therefore expose it only through an explicit, content-addressed
authority selected by the canonical ``*_provenance.json`` sibling.  Publication
and exact run staging are descriptor-anchored so a directory swap cannot
redirect writes through a symbolic link.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import math
import os
from pathlib import Path
from types import MappingProxyType
from typing import Mapping, Optional, Sequence

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from easyicu.concept.metadata_projection import ConceptColumnRole
from easyicu.concept.metadata_sidecar import (
    ColumnMetadataBinding,
    binding_payload_sha256,
    parse_column_metadata_sidecar,
)

from ..authority.filesystem import AnchoredDirectory, AuthorityFilesystemError
from .materialized_metadata import (
    MaterializedCohortAuthorityRef,
    MaterializedMetadataError,
    SourceColumnRef,
    VerifiedMaterializedCohortAuthority,
    canonical_parameters_sha256,
    load_verified_materialized_cohort_authority,
    read_verified_materialized_cohort_table,
)

MATERIALIZED_TRAJECTORY_AUTHORITY_SCHEMA = "easyicu.materialized_trajectory_authority/1"
MATERIALIZED_TRAJECTORY_AUTHORITY_REF_SCHEMA = (
    "easyicu.materialized_trajectory_authority_ref/1"
)
MATERIALIZED_TRAJECTORY_DESCRIPTOR_SCHEMA = (
    "easyicu.materialized_trajectory_descriptor/1"
)
MATERIALIZED_TRAJECTORY_TRANSACTION_SCHEMA = (
    "easyicu.materialized_trajectory_transaction/1"
)

TRAJECTORY_COLUMNS = (
    "stay_id",
    "charttime",
    "concept",
    "value_num",
    "value_str",
)
TRAJECTORY_SCHEMA = pa.schema(
    (
        pa.field("stay_id", pa.int64(), nullable=False),
        pa.field("charttime", pa.float64(), nullable=False),
        pa.field("concept", pa.string(), nullable=False),
        pa.field("value_num", pa.float64(), nullable=True),
        pa.field("value_str", pa.string(), nullable=False),
    )
)

_MAX_AUTHORITY_BYTES = 64 * 1024 * 1024
_MAX_SELECTOR_BYTES = 4 * 1024 * 1024
_HEX = frozenset("0123456789abcdef")


class MaterializedTrajectoryError(MaterializedMetadataError):
    """A trajectory artifact cannot be proven to match typed authority."""


def _digest(value: object, *, label: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in _HEX for character in value)
    ):
        raise MaterializedTrajectoryError(f"{label} must be 64 lowercase hex digits")
    return value


def _string(value: object, *, label: str) -> str:
    if not isinstance(value, str) or not value or value != value.strip():
        raise MaterializedTrajectoryError(f"{label} must be a canonical string")
    return value


def _component(value: object, *, label: str) -> str:
    name = _string(value, label=label)
    if Path(name).name != name or name in {".", ".."}:
        raise MaterializedTrajectoryError(f"{label} must be one path component")
    return name


def _nonnegative_int(value: object, *, label: str) -> int:
    if not isinstance(value, int) or isinstance(value, bool) or value < 0:
        raise MaterializedTrajectoryError(f"{label} must be a non-negative integer")
    return value


def _finite(value: object, *, label: str) -> float:
    if not isinstance(value, (int, float)) or isinstance(value, bool):
        raise MaterializedTrajectoryError(f"{label} must be a finite number")
    number = float(value)
    if not math.isfinite(number):
        raise MaterializedTrajectoryError(f"{label} must be a finite number")
    return 0.0 if number == 0.0 else number


def _exact_keys(
    payload: Mapping[str, object], expected: set[str], *, label: str
) -> None:
    if set(payload) != expected:
        raise MaterializedTrajectoryError(f"{label} keys do not match schema")


def _canonical_strings(values: Sequence[object], *, label: str) -> tuple[str, ...]:
    parsed = tuple(_string(value, label=label) for value in values)
    if len(parsed) != len(set(parsed)):
        raise MaterializedTrajectoryError(f"{label} values must be unique")
    return parsed


def _reject_duplicate_pairs(
    pairs: Sequence[tuple[str, object]],
) -> dict[str, object]:
    result: dict[str, object] = {}
    for key, value in pairs:
        if key in result:
            raise MaterializedTrajectoryError(f"duplicate JSON key: {key!r}")
        result[key] = value
    return result


def _reject_constant(value: str) -> object:
    raise MaterializedTrajectoryError(f"non-finite JSON constant is forbidden: {value}")


def _freeze_json(value: object) -> object:
    if isinstance(value, Mapping):
        return MappingProxyType(
            {str(key): _freeze_json(nested) for key, nested in value.items()}
        )
    if isinstance(value, list):
        return tuple(_freeze_json(item) for item in value)
    return value


def _thaw_json(value: object) -> object:
    if isinstance(value, Mapping):
        return {str(key): _thaw_json(nested) for key, nested in value.items()}
    if isinstance(value, tuple):
        return [_thaw_json(item) for item in value]
    return value


def _canonical_mapping(
    value: Mapping[str, object], *, label: str
) -> Mapping[str, object]:
    try:
        raw = json.dumps(
            _thaw_json(value),
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
        parsed = json.loads(
            raw.decode("utf-8"),
            object_pairs_hook=_reject_duplicate_pairs,
            parse_constant=_reject_constant,
        )
    except MaterializedTrajectoryError:
        raise
    except (TypeError, ValueError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise MaterializedTrajectoryError(f"{label} is not canonical JSON") from exc
    if not isinstance(parsed, dict):
        raise MaterializedTrajectoryError(f"{label} must be an object")
    frozen = _freeze_json(parsed)
    assert isinstance(frozen, Mapping)
    return frozen


@dataclass(frozen=True, slots=True)
class MaterializedTrajectoryAuthorityRef:
    file: str
    sha256: str
    size: int
    schema_version: str = MATERIALIZED_TRAJECTORY_AUTHORITY_REF_SCHEMA

    def __post_init__(self) -> None:
        object.__setattr__(self, "file", _component(self.file, label="authority file"))
        object.__setattr__(
            self, "sha256", _digest(self.sha256, label="authority sha256")
        )
        object.__setattr__(
            self, "size", _nonnegative_int(self.size, label="authority size")
        )
        if self.schema_version != MATERIALIZED_TRAJECTORY_AUTHORITY_REF_SCHEMA:
            raise MaterializedTrajectoryError(
                "unsupported trajectory authority reference schema"
            )

    def to_dict(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "file": self.file,
            "sha256": self.sha256,
            "size": self.size,
        }

    @classmethod
    def from_dict(
        cls, payload: Mapping[str, object]
    ) -> "MaterializedTrajectoryAuthorityRef":
        _exact_keys(
            payload,
            {"schema_version", "file", "sha256", "size"},
            label="trajectory authority reference",
        )
        parsed = cls(
            file=payload["file"],  # type: ignore[arg-type]
            sha256=payload["sha256"],  # type: ignore[arg-type]
            size=payload["size"],  # type: ignore[arg-type]
            schema_version=payload["schema_version"],  # type: ignore[arg-type]
        )
        if parsed.to_dict() != dict(payload):
            raise MaterializedTrajectoryError(
                "trajectory authority reference is not canonical"
            )
        return parsed


@dataclass(frozen=True, slots=True)
class TrajectoryWindow:
    start_hours: float
    end_hours: float
    origin: str = "icu_admission"
    unit: str = "h"
    inclusive: bool = True

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "start_hours", _finite(self.start_hours, label="window start")
        )
        object.__setattr__(
            self, "end_hours", _finite(self.end_hours, label="window end")
        )
        if self.start_hours > self.end_hours:
            raise MaterializedTrajectoryError("window start must not exceed end")
        if (
            self.origin != "icu_admission"
            or self.unit != "h"
            or self.inclusive is not True
        ):
            raise MaterializedTrajectoryError("unsupported trajectory time window")

    def to_dict(self) -> dict[str, object]:
        return {
            "origin": self.origin,
            "unit": self.unit,
            "start_hours": self.start_hours,
            "end_hours": self.end_hours,
            "inclusive": self.inclusive,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, object]) -> "TrajectoryWindow":
        _exact_keys(
            payload,
            {"origin", "unit", "start_hours", "end_hours", "inclusive"},
            label="trajectory window",
        )
        parsed = cls(
            origin=payload["origin"],  # type: ignore[arg-type]
            unit=payload["unit"],  # type: ignore[arg-type]
            start_hours=payload["start_hours"],  # type: ignore[arg-type]
            end_hours=payload["end_hours"],  # type: ignore[arg-type]
            inclusive=payload["inclusive"],  # type: ignore[arg-type]
        )
        if parsed.to_dict() != dict(payload):
            raise MaterializedTrajectoryError("trajectory window is not canonical")
        return parsed


@dataclass(frozen=True, slots=True)
class TrajectoryConceptBinding:
    concept_id: str
    source: SourceColumnRef
    binding: ColumnMetadataBinding

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "concept_id", _string(self.concept_id, label="trajectory concept")
        )
        if not isinstance(self.source, SourceColumnRef) or not isinstance(
            self.binding, ColumnMetadataBinding
        ):
            raise MaterializedTrajectoryError(
                "trajectory concept binding has invalid coordinates"
            )
        if self.binding.metadata.source_concept != self.concept_id:
            raise MaterializedTrajectoryError(
                "trajectory concept does not match source metadata"
            )
        if self.binding.metadata.column_name != self.source.column:
            raise MaterializedTrajectoryError(
                "trajectory source column does not match source metadata"
            )

    def to_dict(self) -> dict[str, object]:
        return {
            "concept_id": self.concept_id,
            "source": self.source.to_dict(),
            "binding": self.binding.to_dict(),
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, object]) -> "TrajectoryConceptBinding":
        _exact_keys(
            payload,
            {"concept_id", "source", "binding"},
            label="trajectory concept binding",
        )
        source = payload["source"]
        binding = payload["binding"]
        if not isinstance(source, Mapping) or not isinstance(binding, Mapping):
            raise MaterializedTrajectoryError(
                "trajectory concept binding coordinates must be objects"
            )
        parsed = cls(
            concept_id=payload["concept_id"],  # type: ignore[arg-type]
            source=SourceColumnRef.from_dict(source),
            binding=ColumnMetadataBinding.from_dict(binding),
        )
        if parsed.to_dict() != dict(payload):
            raise MaterializedTrajectoryError(
                "trajectory concept binding is not canonical"
            )
        return parsed


@dataclass(frozen=True, slots=True)
class MaterializedTrajectoryAuthority:
    trajectory_file: str
    trajectory_sha256: str
    trajectory_size: int
    trajectory_rows: int
    trajectory_columns: tuple[str, ...]
    trajectory_schema_sha256: str
    identity_column: str
    time_column: str
    time_origin: str
    time_unit: str
    concept_column: str
    numeric_value_column: str
    text_value_column: str
    ordered_row_key_sha256: str
    stay_identity_set_sha256: str
    trajectory_stays: int
    bound_universe_file: str
    bound_universe_authority: MaterializedCohortAuthorityRef
    bound_universe_row_identity_sha256: str
    source_export_authority_sha256: str
    concept_bindings: tuple[TrajectoryConceptBinding, ...]
    requested_concepts: tuple[str, ...]
    materialized_concepts: tuple[str, ...]
    available_unobserved_concepts: tuple[str, ...]
    unavailable_concepts: tuple[str, ...]
    window: Optional[TrajectoryWindow]
    producer: str
    producer_implementation_sha256: str
    producer_parameters: Mapping[str, object]
    producer_parameters_sha256: str
    semantic_provenance: Mapping[str, object]
    parent_trajectory_authority: Optional[MaterializedTrajectoryAuthorityRef] = None
    schema_version: str = MATERIALIZED_TRAJECTORY_AUTHORITY_SCHEMA

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "trajectory_file",
            _component(self.trajectory_file, label="trajectory file"),
        )
        for name in (
            "trajectory_sha256",
            "trajectory_schema_sha256",
            "ordered_row_key_sha256",
            "stay_identity_set_sha256",
            "bound_universe_row_identity_sha256",
            "source_export_authority_sha256",
            "producer_implementation_sha256",
            "producer_parameters_sha256",
        ):
            object.__setattr__(self, name, _digest(getattr(self, name), label=name))
        for name in ("trajectory_size", "trajectory_rows", "trajectory_stays"):
            object.__setattr__(
                self, name, _nonnegative_int(getattr(self, name), label=name)
            )
        columns = tuple(self.trajectory_columns)
        if columns != TRAJECTORY_COLUMNS:
            raise MaterializedTrajectoryError(
                "trajectory columns do not match the canonical long schema"
            )
        object.__setattr__(self, "trajectory_columns", columns)
        expected_roles = {
            "identity_column": "stay_id",
            "time_column": "charttime",
            "time_origin": "icu_admission",
            "time_unit": "h",
            "concept_column": "concept",
            "numeric_value_column": "value_num",
            "text_value_column": "value_str",
        }
        for field_name, expected in expected_roles.items():
            if getattr(self, field_name) != expected:
                raise MaterializedTrajectoryError(
                    f"unsupported trajectory coordinate: {field_name}"
                )
        object.__setattr__(
            self,
            "bound_universe_file",
            _component(self.bound_universe_file, label="bound universe file"),
        )
        if not isinstance(
            self.bound_universe_authority, MaterializedCohortAuthorityRef
        ):
            raise MaterializedTrajectoryError(
                "bound universe authority reference is invalid"
            )
        bindings = tuple(self.concept_bindings)
        if any(not isinstance(item, TrajectoryConceptBinding) for item in bindings):
            raise MaterializedTrajectoryError("trajectory concept bindings are invalid")
        binding_ids = tuple(item.concept_id for item in bindings)
        if len(binding_ids) != len(set(binding_ids)):
            raise MaterializedTrajectoryError(
                "trajectory concepts must have one source binding"
            )
        object.__setattr__(self, "concept_bindings", bindings)
        requested = _canonical_strings(
            self.requested_concepts, label="requested trajectory concept"
        )
        materialized = _canonical_strings(
            self.materialized_concepts, label="materialized trajectory concept"
        )
        available_unobserved = _canonical_strings(
            self.available_unobserved_concepts,
            label="available unobserved trajectory concept",
        )
        unavailable = _canonical_strings(
            self.unavailable_concepts, label="unavailable trajectory concept"
        )
        materialized_set = set(materialized)
        available_unobserved_set = set(available_unobserved)
        unavailable_set = set(unavailable)
        if not materialized_set.issubset(requested):
            raise MaterializedTrajectoryError(
                "materialized trajectory concepts must be requested"
            )
        if (
            not available_unobserved_set.issubset(requested)
            or not unavailable_set.issubset(requested)
            or materialized_set & available_unobserved_set
            or materialized_set & unavailable_set
            or available_unobserved_set & unavailable_set
        ):
            raise MaterializedTrajectoryError(
                "trajectory availability sets do not close"
            )
        if materialized_set | available_unobserved_set | unavailable_set != set(
            requested
        ):
            raise MaterializedTrajectoryError(
                "trajectory availability sets do not cover every requested concept"
            )
        available_binding_order = tuple(
            concept
            for concept in requested
            if concept in materialized_set or concept in available_unobserved_set
        )
        if binding_ids != available_binding_order:
            raise MaterializedTrajectoryError(
                "trajectory concept bindings must match source-available concept order"
            )
        object.__setattr__(self, "requested_concepts", requested)
        object.__setattr__(self, "materialized_concepts", materialized)
        object.__setattr__(self, "available_unobserved_concepts", available_unobserved)
        object.__setattr__(self, "unavailable_concepts", unavailable)
        if self.window is not None and not isinstance(self.window, TrajectoryWindow):
            raise MaterializedTrajectoryError("trajectory window is invalid")
        object.__setattr__(
            self,
            "producer",
            _string(self.producer, label="trajectory producer"),
        )
        parameters = _canonical_mapping(
            self.producer_parameters, label="trajectory producer parameters"
        )
        provenance = _canonical_mapping(
            self.semantic_provenance, label="trajectory semantic provenance"
        )
        if canonical_parameters_sha256(parameters) != self.producer_parameters_sha256:
            raise MaterializedTrajectoryError(
                "trajectory producer parameters digest mismatch"
            )
        object.__setattr__(self, "producer_parameters", parameters)
        object.__setattr__(self, "semantic_provenance", provenance)
        parent = self.parent_trajectory_authority
        if parent is not None and not isinstance(
            parent, MaterializedTrajectoryAuthorityRef
        ):
            raise MaterializedTrajectoryError("parent trajectory authority is invalid")
        if self.schema_version != MATERIALIZED_TRAJECTORY_AUTHORITY_SCHEMA:
            raise MaterializedTrajectoryError("unsupported trajectory authority schema")

    def to_dict(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "trajectory_file": self.trajectory_file,
            "trajectory_sha256": self.trajectory_sha256,
            "trajectory_size": self.trajectory_size,
            "trajectory_rows": self.trajectory_rows,
            "trajectory_columns": list(self.trajectory_columns),
            "trajectory_schema_sha256": self.trajectory_schema_sha256,
            "identity_column": self.identity_column,
            "time_column": self.time_column,
            "time_origin": self.time_origin,
            "time_unit": self.time_unit,
            "concept_column": self.concept_column,
            "numeric_value_column": self.numeric_value_column,
            "text_value_column": self.text_value_column,
            "ordered_row_key_sha256": self.ordered_row_key_sha256,
            "stay_identity_set_sha256": self.stay_identity_set_sha256,
            "trajectory_stays": self.trajectory_stays,
            "bound_universe_file": self.bound_universe_file,
            "bound_universe_authority": self.bound_universe_authority.to_dict(),
            "bound_universe_row_identity_sha256": (
                self.bound_universe_row_identity_sha256
            ),
            "source_export_authority_sha256": self.source_export_authority_sha256,
            "concept_bindings": [item.to_dict() for item in self.concept_bindings],
            "requested_concepts": list(self.requested_concepts),
            "materialized_concepts": list(self.materialized_concepts),
            "available_unobserved_concepts": list(self.available_unobserved_concepts),
            "unavailable_concepts": list(self.unavailable_concepts),
            "window": self.window.to_dict() if self.window is not None else None,
            "producer": self.producer,
            "producer_implementation_sha256": self.producer_implementation_sha256,
            "producer_parameters": _thaw_json(self.producer_parameters),
            "producer_parameters_sha256": self.producer_parameters_sha256,
            "semantic_provenance": _thaw_json(self.semantic_provenance),
            "parent_trajectory_authority": (
                self.parent_trajectory_authority.to_dict()
                if self.parent_trajectory_authority is not None
                else None
            ),
        }

    @classmethod
    def from_dict(
        cls, payload: Mapping[str, object]
    ) -> "MaterializedTrajectoryAuthority":
        expected = set(cls.__dataclass_fields__)  # type: ignore[attr-defined]
        _exact_keys(payload, expected, label="materialized trajectory authority")
        raw_universe = payload["bound_universe_authority"]
        raw_bindings = payload["concept_bindings"]
        raw_window = payload["window"]
        raw_parent = payload["parent_trajectory_authority"]
        if not isinstance(raw_universe, Mapping):
            raise MaterializedTrajectoryError(
                "bound universe authority must be an object"
            )
        if not isinstance(raw_bindings, list) or not all(
            isinstance(item, Mapping) for item in raw_bindings
        ):
            raise MaterializedTrajectoryError("concept_bindings must be a list")
        if raw_window is not None and not isinstance(raw_window, Mapping):
            raise MaterializedTrajectoryError("trajectory window must be an object")
        if raw_parent is not None and not isinstance(raw_parent, Mapping):
            raise MaterializedTrajectoryError("parent trajectory must be an object")
        for name in (
            "trajectory_columns",
            "requested_concepts",
            "materialized_concepts",
            "available_unobserved_concepts",
            "unavailable_concepts",
        ):
            if not isinstance(payload[name], list):
                raise MaterializedTrajectoryError(f"{name} must be a list")
        for name in ("producer_parameters", "semantic_provenance"):
            if not isinstance(payload[name], Mapping):
                raise MaterializedTrajectoryError(f"{name} must be an object")
        parsed = cls(
            trajectory_file=payload["trajectory_file"],  # type: ignore[arg-type]
            trajectory_sha256=payload["trajectory_sha256"],  # type: ignore[arg-type]
            trajectory_size=payload["trajectory_size"],  # type: ignore[arg-type]
            trajectory_rows=payload["trajectory_rows"],  # type: ignore[arg-type]
            trajectory_columns=tuple(payload["trajectory_columns"]),  # type: ignore[arg-type]
            trajectory_schema_sha256=payload["trajectory_schema_sha256"],  # type: ignore[arg-type]
            identity_column=payload["identity_column"],  # type: ignore[arg-type]
            time_column=payload["time_column"],  # type: ignore[arg-type]
            time_origin=payload["time_origin"],  # type: ignore[arg-type]
            time_unit=payload["time_unit"],  # type: ignore[arg-type]
            concept_column=payload["concept_column"],  # type: ignore[arg-type]
            numeric_value_column=payload["numeric_value_column"],  # type: ignore[arg-type]
            text_value_column=payload["text_value_column"],  # type: ignore[arg-type]
            ordered_row_key_sha256=payload["ordered_row_key_sha256"],  # type: ignore[arg-type]
            stay_identity_set_sha256=payload["stay_identity_set_sha256"],  # type: ignore[arg-type]
            trajectory_stays=payload["trajectory_stays"],  # type: ignore[arg-type]
            bound_universe_file=payload["bound_universe_file"],  # type: ignore[arg-type]
            bound_universe_authority=MaterializedCohortAuthorityRef.from_dict(
                raw_universe
            ),
            bound_universe_row_identity_sha256=payload[
                "bound_universe_row_identity_sha256"
            ],  # type: ignore[arg-type]
            source_export_authority_sha256=payload[
                "source_export_authority_sha256"
            ],  # type: ignore[arg-type]
            concept_bindings=tuple(
                TrajectoryConceptBinding.from_dict(item) for item in raw_bindings
            ),
            requested_concepts=tuple(payload["requested_concepts"]),  # type: ignore[arg-type]
            materialized_concepts=tuple(payload["materialized_concepts"]),  # type: ignore[arg-type]
            available_unobserved_concepts=tuple(
                payload["available_unobserved_concepts"]  # type: ignore[arg-type]
            ),
            unavailable_concepts=tuple(payload["unavailable_concepts"]),  # type: ignore[arg-type]
            window=(TrajectoryWindow.from_dict(raw_window) if raw_window else None),
            producer=payload["producer"],  # type: ignore[arg-type]
            producer_implementation_sha256=payload[
                "producer_implementation_sha256"
            ],  # type: ignore[arg-type]
            producer_parameters=payload["producer_parameters"],  # type: ignore[arg-type]
            producer_parameters_sha256=payload[
                "producer_parameters_sha256"
            ],  # type: ignore[arg-type]
            semantic_provenance=payload["semantic_provenance"],  # type: ignore[arg-type]
            parent_trajectory_authority=(
                MaterializedTrajectoryAuthorityRef.from_dict(raw_parent)
                if raw_parent is not None
                else None
            ),
            schema_version=payload["schema_version"],  # type: ignore[arg-type]
        )
        if parsed.to_dict() != dict(payload):
            raise MaterializedTrajectoryError(
                "materialized trajectory authority is not canonical"
            )
        return parsed


@dataclass(frozen=True, slots=True)
class VerifiedMaterializedTrajectoryAuthority:
    reference: MaterializedTrajectoryAuthorityRef
    authority: MaterializedTrajectoryAuthority
    provenance: Mapping[str, object]


@dataclass(frozen=True, slots=True)
class VerifiedLegacyTrajectoryCapsuleReceipt:
    """Digest-bound permission to replay one archived v2 raw trajectory.

    Fresh typed runs must carry a materialized trajectory authority.  The only
    compatibility exception is a trajectory that was already sealed by an
    archived v2 run-input capsule before typed trajectory authority existed.
    The receipt is created only after that capsule and all selected bytes have
    been verified; the runner rechecks these coordinates before exposing the
    raw trajectory to sandboxed code.
    """

    capsule_sha256: str
    trajectory_relative_path: str
    trajectory_sha256: str
    trajectory_size: int
    universe_authority_sha256: str
    schema_version: str = "easyicu.run_input_capsule/2"

    def __post_init__(self) -> None:
        for name in (
            "capsule_sha256",
            "trajectory_sha256",
            "universe_authority_sha256",
        ):
            object.__setattr__(self, name, _digest(getattr(self, name), label=name))
        object.__setattr__(
            self,
            "trajectory_relative_path",
            _component(
                self.trajectory_relative_path,
                label="legacy capsule trajectory path",
            ),
        )
        object.__setattr__(
            self,
            "trajectory_size",
            _nonnegative_int(self.trajectory_size, label="legacy trajectory size"),
        )
        if self.schema_version != "easyicu.run_input_capsule/2":
            raise MaterializedTrajectoryError(
                "legacy trajectory receipt must come from a v2 run-input capsule"
            )


@dataclass(frozen=True, slots=True)
class StagedTrajectoryBinding:
    """Exact run-internal trajectory coordinates passed across control layers."""

    path: Path
    sha256: str
    size: int
    authority_ref: Optional[MaterializedTrajectoryAuthorityRef] = None
    legacy_capsule_receipt: Optional[VerifiedLegacyTrajectoryCapsuleReceipt] = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "path", Path(self.path))
        object.__setattr__(
            self, "sha256", _digest(self.sha256, label="trajectory sha256")
        )
        object.__setattr__(
            self,
            "size",
            _nonnegative_int(self.size, label="trajectory size"),
        )
        if self.authority_ref is not None and not isinstance(
            self.authority_ref, MaterializedTrajectoryAuthorityRef
        ):
            raise MaterializedTrajectoryError(
                "staged trajectory authority reference is invalid"
            )
        if self.legacy_capsule_receipt is not None and not isinstance(
            self.legacy_capsule_receipt, VerifiedLegacyTrajectoryCapsuleReceipt
        ):
            raise MaterializedTrajectoryError(
                "legacy trajectory capsule receipt is invalid"
            )
        if self.authority_ref is not None and self.legacy_capsule_receipt is not None:
            raise MaterializedTrajectoryError(
                "trajectory cannot use typed authority and a legacy capsule receipt"
            )
        receipt = self.legacy_capsule_receipt
        if receipt is not None and (
            receipt.trajectory_sha256 != self.sha256
            or receipt.trajectory_size != self.size
            or receipt.trajectory_relative_path != self.path.name
        ):
            raise MaterializedTrajectoryError(
                "legacy trajectory receipt does not bind the staged artifact"
            )


def materialized_trajectory_provenance_path(trajectory_path: Path) -> Path:
    path = Path(trajectory_path)
    return path.with_name(f"{path.stem}_provenance.json")


def _canonical_authority_bytes(authority: MaterializedTrajectoryAuthority) -> bytes:
    return json.dumps(
        authority.to_dict(),
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")


def _write_authority_at(
    root: AnchoredDirectory, authority: MaterializedTrajectoryAuthority
) -> MaterializedTrajectoryAuthorityRef:
    raw = _canonical_authority_bytes(authority)
    digest = hashlib.sha256(raw).hexdigest()
    name = f"trajectory_authority.sha256-{digest}.json"
    root.publish_immutable_bytes(name, raw)
    return MaterializedTrajectoryAuthorityRef(file=name, sha256=digest, size=len(raw))


def _atomic_json_at(
    root: AnchoredDirectory,
    *,
    name: str,
    payload: Mapping[str, object],
    require_absent: bool = False,
) -> None:
    raw = json.dumps(
        _thaw_json(payload),
        ensure_ascii=False,
        sort_keys=True,
        indent=2,
        allow_nan=False,
    ).encode("utf-8")
    root.replace_bytes(name, raw, require_absent=require_absent)


def _read_json_bytes(raw: bytes, *, label: str) -> Mapping[str, object]:
    try:
        parsed = json.loads(
            raw.decode("utf-8"),
            object_pairs_hook=_reject_duplicate_pairs,
            parse_constant=_reject_constant,
        )
    except MaterializedTrajectoryError:
        raise
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise MaterializedTrajectoryError(f"cannot parse {label}") from exc
    if not isinstance(parsed, Mapping):
        raise MaterializedTrajectoryError(f"{label} must be an object")
    return parsed


def _canonical_frame(frame: pd.DataFrame) -> pa.Table:
    if tuple(frame.columns) != TRAJECTORY_COLUMNS:
        raise MaterializedTrajectoryError(
            "trajectory frame must have the exact canonical column order"
        )
    data = frame.copy()
    if data["stay_id"].isna().any():
        raise MaterializedTrajectoryError("trajectory stay_id contains null values")
    try:
        stay_numeric = pd.to_numeric(data["stay_id"], errors="raise")
    except (TypeError, ValueError) as exc:
        raise MaterializedTrajectoryError("trajectory stay_id must be integer") from exc
    if any(float(value) != int(value) for value in stay_numeric.tolist()):
        raise MaterializedTrajectoryError("trajectory stay_id must be integer")
    data["stay_id"] = stay_numeric.astype("int64")
    try:
        data["charttime"] = pd.to_numeric(data["charttime"], errors="raise").astype(
            "float64"
        )
    except (TypeError, ValueError) as exc:
        raise MaterializedTrajectoryError(
            "trajectory charttime must be numeric"
        ) from exc
    if data["charttime"].isna().any() or not all(
        math.isfinite(float(value)) for value in data["charttime"].tolist()
    ):
        raise MaterializedTrajectoryError("trajectory charttime must be finite")
    if data["concept"].isna().any():
        raise MaterializedTrajectoryError("trajectory concept contains null values")
    data["concept"] = data["concept"].astype("string")
    if any(
        not str(value).strip() or str(value) != str(value).strip()
        for value in data["concept"]
    ):
        raise MaterializedTrajectoryError("trajectory concept must be canonical")
    try:
        data["value_num"] = pd.to_numeric(data["value_num"], errors="raise").astype(
            "float64"
        )
    except (TypeError, ValueError) as exc:
        raise MaterializedTrajectoryError(
            "trajectory value_num must be numeric"
        ) from exc
    if any(
        not math.isfinite(float(value)) for value in data["value_num"].dropna().tolist()
    ):
        raise MaterializedTrajectoryError("trajectory value_num must be finite")
    if data["value_str"].isna().any():
        raise MaterializedTrajectoryError("trajectory value_str contains null values")
    data["value_str"] = data["value_str"].astype("string")
    if data[["value_num", "value_str"]].isna().all(axis=1).any():
        raise MaterializedTrajectoryError("trajectory row has no recorded value")
    data = data.sort_values(
        list(TRAJECTORY_COLUMNS), kind="mergesort", na_position="last"
    ).reset_index(drop=True)
    try:
        return pa.Table.from_pandas(
            data, schema=TRAJECTORY_SCHEMA, preserve_index=False, safe=True
        )
    except (pa.ArrowException, ValueError, TypeError) as exc:
        raise MaterializedTrajectoryError(
            "trajectory frame cannot be encoded in the canonical schema"
        ) from exc


def _canonical_scalar(value: object) -> object:
    if isinstance(value, float):
        if not math.isfinite(value):
            raise MaterializedTrajectoryError("trajectory coordinate is not finite")
        return 0.0 if value == 0.0 else value
    return value


def _coordinate_digests(table: pa.Table) -> tuple[str, str, int, set[object]]:
    stay_values = table.column("stay_id").combine_chunks().to_pylist()
    time_values = table.column("charttime").combine_chunks().to_pylist()
    concepts = table.column("concept").combine_chunks().to_pylist()
    keys = [
        [_canonical_scalar(stay), _canonical_scalar(time), concept]
        for stay, time, concept in zip(stay_values, time_values, concepts)
    ]
    key_digest = hashlib.sha256(
        json.dumps(
            keys,
            ensure_ascii=False,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
    ).hexdigest()
    stay_set = set(stay_values)
    stay_digest = hashlib.sha256(
        json.dumps(
            sorted(_canonical_scalar(value) for value in stay_set),
            ensure_ascii=False,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
    ).hexdigest()
    return key_digest, stay_digest, len(stay_set), stay_set


def _parquet_envelope_at(
    root: AnchoredDirectory, name: str
) -> tuple[str, int, pa.Table]:
    digest = hashlib.sha256()
    try:
        with root.open_regular(name) as handle:
            info = os.fstat(handle.fileno())
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
            handle.seek(0)
            table = pq.read_table(handle)
    except (AuthorityFilesystemError, OSError, ValueError, pa.ArrowException) as exc:
        raise MaterializedTrajectoryError("cannot read trajectory parquet") from exc
    return digest.hexdigest(), int(info.st_size), table


def _source_sidecar_at(
    root: AnchoredDirectory,
    bound_universe: VerifiedMaterializedCohortAuthority,
):
    reference = bound_universe.authority.source_column_metadata
    raw = root.read_bytes(
        reference.file,
        max_bytes=_MAX_AUTHORITY_BYTES,
        expected_size=reference.size,
        expected_sha256=reference.sha256,
    )
    return parse_column_metadata_sidecar(raw)


def _concept_bindings_at(
    root: AnchoredDirectory,
    bound_universe: VerifiedMaterializedCohortAuthority,
    concepts: tuple[str, ...],
) -> tuple[TrajectoryConceptBinding, ...]:
    source_sidecar = _source_sidecar_at(root, bound_universe)
    result: list[TrajectoryConceptBinding] = []
    for concept in concepts:
        candidates: list[tuple[str, str, ColumnMetadataBinding]] = []
        for file_binding in source_sidecar.files:
            for column, binding in file_binding.columns.items():
                if (
                    binding.metadata.source_concept == concept
                    and binding.metadata.role
                    in {ConceptColumnRole.VALUE, ConceptColumnRole.EVENT_STATUS}
                ):
                    candidates.append((file_binding.relative_path, column, binding))
        exact = [item for item in candidates if item[1] == concept]
        selected = exact if len(exact) == 1 else candidates
        if len(selected) != 1:
            raise MaterializedTrajectoryError(
                f"trajectory concept {concept!r} lacks one typed source owner"
            )
        relative_path, column, binding = selected[0]
        source = SourceColumnRef(
            authority_sha256=(bound_universe.authority.source_export_authority_sha256),
            file=relative_path,
            column=column,
            binding_sha256=binding_payload_sha256({column: binding}),
        )
        result.append(
            TrajectoryConceptBinding(
                concept_id=concept,
                source=source,
                binding=binding,
            )
        )
    return tuple(result)


def _descriptor(
    *,
    authority: MaterializedTrajectoryAuthorityRef,
    relative_path: str,
) -> dict[str, object]:
    return {
        "schema_version": MATERIALIZED_TRAJECTORY_DESCRIPTOR_SCHEMA,
        "authority": authority.to_dict(),
        "relative_path": relative_path,
    }


def _parse_descriptor(
    payload: Mapping[str, object],
) -> tuple[MaterializedTrajectoryAuthorityRef, str]:
    _exact_keys(
        payload,
        {"schema_version", "authority", "relative_path"},
        label="trajectory descriptor",
    )
    if payload["schema_version"] != MATERIALIZED_TRAJECTORY_DESCRIPTOR_SCHEMA:
        raise MaterializedTrajectoryError("unsupported trajectory descriptor schema")
    raw_ref = payload["authority"]
    if not isinstance(raw_ref, Mapping):
        raise MaterializedTrajectoryError("trajectory authority ref must be an object")
    relative_path = _component(payload["relative_path"], label="trajectory path")
    return MaterializedTrajectoryAuthorityRef.from_dict(raw_ref), relative_path


def _validate_table_contract(
    table: pa.Table,
    authority: MaterializedTrajectoryAuthority,
    *,
    bound_universe: VerifiedMaterializedCohortAuthority,
    bound_universe_path: Path,
) -> None:
    if table.schema != TRAJECTORY_SCHEMA:
        raise MaterializedTrajectoryError("trajectory parquet schema mismatch")
    if table.num_rows != authority.trajectory_rows:
        raise MaterializedTrajectoryError("trajectory row count mismatch")
    schema_sha = hashlib.sha256(table.schema.serialize().to_pybytes()).hexdigest()
    if schema_sha != authority.trajectory_schema_sha256:
        raise MaterializedTrajectoryError("trajectory schema digest mismatch")
    key_sha, stay_sha, stays, stay_set = _coordinate_digests(table)
    if (
        key_sha != authority.ordered_row_key_sha256
        or stay_sha != authority.stay_identity_set_sha256
        or stays != authority.trajectory_stays
    ):
        raise MaterializedTrajectoryError("trajectory coordinate digest mismatch")
    cohort = read_verified_materialized_cohort_table(
        Path(bound_universe_path), verified=bound_universe
    )
    cohort_ids = set(
        cohort.column(bound_universe.authority.identity_column)
        .combine_chunks()
        .to_pylist()
    )
    if not stay_set.issubset(cohort_ids):
        raise MaterializedTrajectoryError(
            "trajectory contains an identity outside its bound universe"
        )
    actual_concepts = tuple(
        dict.fromkeys(table.column("concept").combine_chunks().to_pylist())
    )
    if set(actual_concepts) != set(authority.materialized_concepts):
        raise MaterializedTrajectoryError("trajectory concept coverage mismatch")
    times = table.column("charttime").combine_chunks().to_pylist()
    if authority.window is not None and any(
        float(value) < authority.window.start_hours
        or float(value) > authority.window.end_hours
        for value in times
    ):
        raise MaterializedTrajectoryError(
            "trajectory contains a time outside its window"
        )


def _validate_stage_parent_projection(
    authority: MaterializedTrajectoryAuthority,
    *,
    parent: MaterializedTrajectoryAuthority,
    parent_ref: MaterializedTrajectoryAuthorityRef,
) -> None:
    """Prove a staged trajectory is the exact deterministic parent projection."""

    expected_parameters = _canonical_mapping(
        {
            "source_trajectory_authority_sha256": parent_ref.sha256,
            "source_trajectory_sha256": parent.trajectory_sha256,
            "source_universe_authority_sha256": (
                parent.bound_universe_authority.sha256
            ),
            "target_universe_authority_sha256": (
                authority.bound_universe_authority.sha256
            ),
            "target_file": authority.trajectory_file,
            "transform": "identity_stage_copy",
        },
        label="expected trajectory stage parameters",
    )
    expected_provenance = _canonical_mapping(
        {
            **dict(_thaw_json(parent.semantic_provenance)),
            "staged_from_trajectory_authority_sha256": parent_ref.sha256,
        },
        label="expected trajectory stage provenance",
    )
    if (
        authority.parent_trajectory_authority != parent_ref
        or authority.trajectory_sha256 != parent.trajectory_sha256
        or authority.trajectory_size != parent.trajectory_size
        or authority.trajectory_rows != parent.trajectory_rows
        or authority.trajectory_columns != parent.trajectory_columns
        or authority.trajectory_schema_sha256 != parent.trajectory_schema_sha256
        or authority.identity_column != parent.identity_column
        or authority.time_column != parent.time_column
        or authority.time_origin != parent.time_origin
        or authority.time_unit != parent.time_unit
        or authority.concept_column != parent.concept_column
        or authority.numeric_value_column != parent.numeric_value_column
        or authority.text_value_column != parent.text_value_column
        or authority.ordered_row_key_sha256 != parent.ordered_row_key_sha256
        or authority.stay_identity_set_sha256 != parent.stay_identity_set_sha256
        or authority.trajectory_stays != parent.trajectory_stays
        or authority.bound_universe_row_identity_sha256
        != parent.bound_universe_row_identity_sha256
        or authority.source_export_authority_sha256
        != parent.source_export_authority_sha256
        or authority.concept_bindings != parent.concept_bindings
        or authority.requested_concepts != parent.requested_concepts
        or authority.materialized_concepts != parent.materialized_concepts
        or authority.available_unobserved_concepts
        != parent.available_unobserved_concepts
        or authority.unavailable_concepts != parent.unavailable_concepts
        or authority.window != parent.window
        or authority.producer != "research_agent_run_stage"
        or authority.producer_parameters != expected_parameters
        or authority.semantic_provenance != expected_provenance
    ):
        raise MaterializedTrajectoryError(
            "staged trajectory is not the exact deterministic parent projection"
        )


def _validate_initial_producer_receipts(
    root: AnchoredDirectory,
    authority: MaterializedTrajectoryAuthority,
    *,
    universe: VerifiedMaterializedCohortAuthority,
) -> None:
    source_available_concepts = tuple(
        concept
        for concept in authority.requested_concepts
        if concept in set(authority.materialized_concepts)
        or concept in set(authority.available_unobserved_concepts)
    )
    expected_bindings = _concept_bindings_at(
        root,
        universe,
        source_available_concepts,
    )
    expected_parameters = _canonical_mapping(
        {
            "database": universe.sidecar.source_database,
            "requested_concepts": list(authority.requested_concepts),
            "materialized_concepts": list(authority.materialized_concepts),
            "available_unobserved_concepts": list(
                authority.available_unobserved_concepts
            ),
            "unavailable_concepts": list(authority.unavailable_concepts),
            "window": (
                [authority.window.start_hours, authority.window.end_hours]
                if authority.window is not None
                else None
            ),
            "bound_universe_authority_sha256": universe.reference.sha256,
        },
        label="expected trajectory producer parameters",
    )
    provenance = authority.semantic_provenance
    if (
        authority.producer != "cohort_materializer"
        or authority.concept_bindings != expected_bindings
        or authority.producer_parameters != expected_parameters
        or provenance.get("n_rows") != authority.trajectory_rows
        or provenance.get("n_stays") != authority.trajectory_stays
        or tuple(provenance.get("trajectory_concepts_materialized") or ())
        != authority.materialized_concepts
        or tuple(provenance.get("available_unobserved_concepts") or ())
        != authority.available_unobserved_concepts
        or tuple(provenance.get("unavailable_concepts") or ())
        != authority.unavailable_concepts
    ):
        raise MaterializedTrajectoryError(
            "initial trajectory producer receipts do not match typed inputs"
        )


def publish_materialized_trajectory_authority(
    frame: pd.DataFrame,
    target_path: Path,
    *,
    bound_universe_path: Path,
    bound_universe: VerifiedMaterializedCohortAuthority,
    requested_concepts: Sequence[str],
    materialized_concepts: Sequence[str],
    available_unobserved_concepts: Sequence[str],
    unavailable_concepts: Sequence[str],
    window: Optional[tuple[float, float]],
    semantic_provenance: Mapping[str, object],
    producer_implementation_sha256: str,
    producer_parameters: Mapping[str, object],
) -> VerifiedMaterializedTrajectoryAuthority:
    """Publish one canonical trajectory and bind it to an exact wide universe."""

    target_path = Path(target_path)
    bound_universe_path = Path(bound_universe_path)
    if target_path.parent != bound_universe_path.parent:
        raise MaterializedTrajectoryError(
            "trajectory and bound universe must share one authority directory"
        )
    selected_universe = load_verified_materialized_cohort_authority(
        bound_universe_path, expected_authority=bound_universe.reference
    )
    if selected_universe is None:
        raise MaterializedTrajectoryError("bound trajectory universe lost authority")
    table = _canonical_frame(frame)
    requested = _canonical_strings(
        tuple(requested_concepts), label="requested trajectory concept"
    )
    materialized = _canonical_strings(
        tuple(materialized_concepts), label="materialized trajectory concept"
    )
    available_unobserved = _canonical_strings(
        tuple(available_unobserved_concepts),
        label="available unobserved trajectory concept",
    )
    unavailable = _canonical_strings(
        tuple(unavailable_concepts), label="unavailable trajectory concept"
    )
    trajectory_window = (
        TrajectoryWindow(start_hours=window[0], end_hours=window[1])
        if window is not None
        else None
    )
    root_path = target_path.parent
    selector_name = materialized_trajectory_provenance_path(target_path).name
    universe_table = read_verified_materialized_cohort_table(
        bound_universe_path, verified=selected_universe
    )
    universe_ids = set(
        universe_table.column(selected_universe.authority.identity_column)
        .combine_chunks()
        .to_pylist()
    )
    key_sha, stay_sha, stays, stay_set = _coordinate_digests(table)
    if not stay_set.issubset(universe_ids):
        raise MaterializedTrajectoryError(
            "trajectory contains an identity outside its bound universe"
        )
    if trajectory_window is not None and any(
        float(value) < trajectory_window.start_hours
        or float(value) > trajectory_window.end_hours
        for value in table.column("charttime").combine_chunks().to_pylist()
    ):
        raise MaterializedTrajectoryError(
            "trajectory contains a time outside its declared window"
        )
    actual_concepts = set(table.column("concept").combine_chunks().to_pylist())
    if actual_concepts != set(materialized):
        raise MaterializedTrajectoryError(
            "materialized concept receipt does not match trajectory rows"
        )
    try:
        with AnchoredDirectory.open(root_path) as root:
            # Resolve all typed source bindings before creating the prepared
            # selector.  Invalid science metadata leaves no transaction debris.
            source_available = tuple(
                concept
                for concept in requested
                if concept in set(materialized) or concept in set(available_unobserved)
            )
            concept_bindings = _concept_bindings_at(
                root, selected_universe, source_available
            )
            root.require_absent(target_path.name, selector_name)
            _atomic_json_at(
                root,
                name=selector_name,
                payload={
                    "schema_version": MATERIALIZED_TRAJECTORY_TRANSACTION_SCHEMA,
                    "trajectory_authority_required": True,
                    "trajectory_authority": None,
                    "authority_transaction_state": "prepared",
                },
                require_absent=True,
            )
            temporary_name, descriptor = root.create_temporary(stem=target_path.name)
            try:
                with os.fdopen(descriptor, "wb") as handle:
                    descriptor = -1
                    pq.write_table(table, handle)
                    handle.flush()
                    os.fsync(handle.fileno())
                trajectory_sha, trajectory_size, stored = _parquet_envelope_at(
                    root, temporary_name
                )
                if not stored.equals(table):
                    raise MaterializedTrajectoryError(
                        "stored trajectory does not match canonical input table"
                    )
                parameters = _canonical_mapping(
                    producer_parameters, label="trajectory producer parameters"
                )
                authority = MaterializedTrajectoryAuthority(
                    trajectory_file=target_path.name,
                    trajectory_sha256=trajectory_sha,
                    trajectory_size=trajectory_size,
                    trajectory_rows=stored.num_rows,
                    trajectory_columns=TRAJECTORY_COLUMNS,
                    trajectory_schema_sha256=hashlib.sha256(
                        stored.schema.serialize().to_pybytes()
                    ).hexdigest(),
                    identity_column="stay_id",
                    time_column="charttime",
                    time_origin="icu_admission",
                    time_unit="h",
                    concept_column="concept",
                    numeric_value_column="value_num",
                    text_value_column="value_str",
                    ordered_row_key_sha256=key_sha,
                    stay_identity_set_sha256=stay_sha,
                    trajectory_stays=stays,
                    bound_universe_file=bound_universe_path.name,
                    bound_universe_authority=selected_universe.reference,
                    bound_universe_row_identity_sha256=(
                        selected_universe.authority.row_identity_sha256
                    ),
                    source_export_authority_sha256=(
                        selected_universe.authority.source_export_authority_sha256
                    ),
                    concept_bindings=concept_bindings,
                    requested_concepts=requested,
                    materialized_concepts=materialized,
                    available_unobserved_concepts=available_unobserved,
                    unavailable_concepts=unavailable,
                    window=trajectory_window,
                    producer="cohort_materializer",
                    producer_implementation_sha256=producer_implementation_sha256,
                    producer_parameters=parameters,
                    producer_parameters_sha256=canonical_parameters_sha256(parameters),
                    semantic_provenance=semantic_provenance,
                )
                authority_ref = _write_authority_at(root, authority)
                root.replace_temporary(
                    temporary_name, target_path.name, require_absent=True
                )
                temporary_name = ""
                _atomic_json_at(
                    root,
                    name=selector_name,
                    payload={
                        **dict(_thaw_json(semantic_provenance)),
                        "schema_version": MATERIALIZED_TRAJECTORY_TRANSACTION_SCHEMA,
                        "trajectory_authority_required": True,
                        "trajectory_authority": _descriptor(
                            authority=authority_ref,
                            relative_path=target_path.name,
                        ),
                        "authority_transaction_state": "committed",
                    },
                )
                root.assert_still_selected()
            finally:
                if descriptor >= 0:
                    os.close(descriptor)
                if temporary_name:
                    root.unlink(temporary_name, missing_ok=True)
    except AuthorityFilesystemError as exc:
        raise MaterializedTrajectoryError(
            "cannot publish materialized trajectory authority"
        ) from exc
    result = load_verified_materialized_trajectory_authority(
        target_path,
        expected_authority=authority_ref,
        expected_universe_authority=selected_universe.reference,
    )
    if result is None:
        raise MaterializedTrajectoryError("published trajectory lost its authority")
    return result


def load_verified_materialized_trajectory_authority(
    trajectory_path: Path,
    *,
    expected_authority: Optional[MaterializedTrajectoryAuthorityRef] = None,
    expected_universe_authority: Optional[MaterializedCohortAuthorityRef] = None,
    expected_parent_universe_authority: Optional[MaterializedCohortAuthorityRef] = None,
) -> Optional[VerifiedMaterializedTrajectoryAuthority]:
    """Load only the selector-chosen trajectory authority; never scan blobs."""

    trajectory_path = Path(trajectory_path)
    selector_path = materialized_trajectory_provenance_path(trajectory_path)
    if not selector_path.exists():
        if expected_authority is not None:
            raise MaterializedTrajectoryError(
                "trajectory authority selector is missing"
            )
        return None
    try:
        with AnchoredDirectory.open(trajectory_path.parent) as root:
            selector_raw = root.read_bytes(
                selector_path.name, max_bytes=_MAX_SELECTOR_BYTES
            )
            selector = _read_json_bytes(selector_raw, label="trajectory provenance")
            if (
                selector.get("schema_version")
                != MATERIALIZED_TRAJECTORY_TRANSACTION_SCHEMA
            ):
                raise MaterializedTrajectoryError(
                    "unsupported trajectory transaction schema"
                )
            if selector.get("trajectory_authority_required") is not True:
                raise MaterializedTrajectoryError(
                    "trajectory authority selector lacks required marker"
                )
            if selector.get("authority_transaction_state") != "committed":
                raise MaterializedTrajectoryError(
                    "trajectory authority publication is incomplete"
                )
            raw_descriptor = selector.get("trajectory_authority")
            if not isinstance(raw_descriptor, Mapping):
                raise MaterializedTrajectoryError(
                    "trajectory authority descriptor is missing"
                )
            reference, relative_path = _parse_descriptor(raw_descriptor)
            if relative_path != trajectory_path.name:
                raise MaterializedTrajectoryError(
                    "trajectory descriptor selects a different artifact"
                )
            expected_name = f"trajectory_authority.sha256-{reference.sha256}.json"
            if reference.file != expected_name:
                raise MaterializedTrajectoryError(
                    "trajectory authority is not content addressed"
                )
            if expected_authority is not None and reference != expected_authority:
                raise MaterializedTrajectoryError(
                    "trajectory authority does not match selected reference"
                )
            authority_raw = root.read_bytes(
                reference.file,
                max_bytes=_MAX_AUTHORITY_BYTES,
                expected_size=reference.size,
                expected_sha256=reference.sha256,
            )
            authority_payload = _read_json_bytes(
                authority_raw, label="trajectory authority"
            )
            authority = MaterializedTrajectoryAuthority.from_dict(authority_payload)
            if authority.trajectory_file != trajectory_path.name:
                raise MaterializedTrajectoryError(
                    "trajectory authority selects a different artifact"
                )
            if (
                expected_universe_authority is not None
                and authority.bound_universe_authority != expected_universe_authority
            ):
                raise MaterializedTrajectoryError(
                    "trajectory is bound to a different universe authority"
                )
            universe_path = trajectory_path.parent / authority.bound_universe_file
            universe = load_verified_materialized_cohort_authority(
                universe_path,
                expected_authority=authority.bound_universe_authority,
            )
            if universe is None:
                raise MaterializedTrajectoryError(
                    "trajectory bound universe lost its authority"
                )
            if (
                universe.authority.row_identity_sha256
                != authority.bound_universe_row_identity_sha256
                or universe.authority.source_export_authority_sha256
                != authority.source_export_authority_sha256
            ):
                raise MaterializedTrajectoryError(
                    "trajectory universe lineage does not match authority"
                )
            trajectory_sha, trajectory_size, table = _parquet_envelope_at(
                root, trajectory_path.name
            )
            if (
                trajectory_sha != authority.trajectory_sha256
                or trajectory_size != authority.trajectory_size
            ):
                raise MaterializedTrajectoryError("trajectory artifact digest mismatch")
            _validate_table_contract(
                table,
                authority,
                bound_universe=universe,
                bound_universe_path=universe_path,
            )
            if authority.parent_trajectory_authority is not None:
                parent_ref = authority.parent_trajectory_authority
                parent_raw = root.read_bytes(
                    parent_ref.file,
                    max_bytes=_MAX_AUTHORITY_BYTES,
                    expected_size=parent_ref.size,
                    expected_sha256=parent_ref.sha256,
                )
                parent_payload = _read_json_bytes(
                    parent_raw, label="parent trajectory authority"
                )
                parent = MaterializedTrajectoryAuthority.from_dict(parent_payload)
                if (
                    expected_parent_universe_authority is not None
                    and parent.bound_universe_authority
                    != expected_parent_universe_authority
                ):
                    raise MaterializedTrajectoryError(
                        "trajectory parent is bound to a different source universe"
                    )
                _validate_stage_parent_projection(
                    authority,
                    parent=parent,
                    parent_ref=parent_ref,
                )
            else:
                if expected_parent_universe_authority is not None:
                    raise MaterializedTrajectoryError(
                        "trajectory parent universe was required but no parent exists"
                    )
                _validate_initial_producer_receipts(
                    root,
                    authority,
                    universe=universe,
                )
            root.assert_still_selected()
    except AuthorityFilesystemError as exc:
        raise MaterializedTrajectoryError(
            "cannot verify materialized trajectory authority"
        ) from exc
    return VerifiedMaterializedTrajectoryAuthority(
        reference=reference,
        authority=authority,
        provenance=_canonical_mapping(selector, label="trajectory provenance"),
    )


def verify_materialized_trajectory_envelope(
    trajectory_path: Path,
    *,
    expected_authority: MaterializedTrajectoryAuthorityRef,
    expected_universe_authority: MaterializedCohortAuthorityRef,
) -> MaterializedTrajectoryAuthority:
    """Reverify selector, authority and artifact bytes without loading the table.

    Full schema/row/concept validation belongs at intake, staging and resume.
    Per-step pre/post checks only need to prove that the already-validated
    content-addressed envelope did not change while generated code executed.
    """

    trajectory_path = Path(trajectory_path)
    selector_path = materialized_trajectory_provenance_path(trajectory_path)
    try:
        with AnchoredDirectory.open(trajectory_path.parent) as root:
            selector = _read_json_bytes(
                root.read_bytes(
                    selector_path.name,
                    max_bytes=_MAX_SELECTOR_BYTES,
                ),
                label="trajectory provenance",
            )
            if (
                selector.get("schema_version")
                != MATERIALIZED_TRAJECTORY_TRANSACTION_SCHEMA
                or selector.get("trajectory_authority_required") is not True
                or selector.get("authority_transaction_state") != "committed"
            ):
                raise MaterializedTrajectoryError(
                    "trajectory authority publication is not committed"
                )
            raw_descriptor = selector.get("trajectory_authority")
            if not isinstance(raw_descriptor, Mapping):
                raise MaterializedTrajectoryError(
                    "trajectory authority descriptor is missing"
                )
            reference, relative_path = _parse_descriptor(raw_descriptor)
            if reference != expected_authority or relative_path != trajectory_path.name:
                raise MaterializedTrajectoryError(
                    "trajectory selector changed its authority"
                )
            authority = MaterializedTrajectoryAuthority.from_dict(
                _read_json_bytes(
                    root.read_bytes(
                        reference.file,
                        max_bytes=_MAX_AUTHORITY_BYTES,
                        expected_size=reference.size,
                        expected_sha256=reference.sha256,
                    ),
                    label="trajectory authority",
                )
            )
            if (
                authority.trajectory_file != trajectory_path.name
                or authority.bound_universe_authority != expected_universe_authority
            ):
                raise MaterializedTrajectoryError(
                    "trajectory envelope changed its universe binding"
                )
            digest = hashlib.sha256()
            size = 0
            with root.open_regular(trajectory_path.name) as handle:
                for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                    digest.update(chunk)
                    size += len(chunk)
            if (
                digest.hexdigest() != authority.trajectory_sha256
                or size != authority.trajectory_size
            ):
                raise MaterializedTrajectoryError(
                    "trajectory artifact changed after validation"
                )
            root.assert_still_selected()
    except AuthorityFilesystemError as exc:
        raise MaterializedTrajectoryError(
            "cannot verify materialized trajectory envelope"
        ) from exc
    return authority


def stage_materialized_trajectory_authority(
    source_path: Path,
    target_path: Path,
    *,
    source_universe_path: Path,
    target_universe_path: Path,
    expected_source_authority: MaterializedTrajectoryAuthorityRef,
    expected_target_universe_authority: MaterializedCohortAuthorityRef,
    producer_implementation_sha256: str,
) -> VerifiedMaterializedTrajectoryAuthority:
    """Exact-copy a typed trajectory and bind it to the staged run universe."""

    source_path = Path(source_path)
    target_path = Path(target_path)
    source_universe = load_verified_materialized_cohort_authority(source_universe_path)
    if source_universe is None:
        raise MaterializedTrajectoryError("source trajectory universe is untyped")
    source = load_verified_materialized_trajectory_authority(
        source_path,
        expected_authority=expected_source_authority,
        expected_universe_authority=source_universe.reference,
    )
    if source is None:
        raise MaterializedTrajectoryError("source trajectory authority is missing")
    target_universe = load_verified_materialized_cohort_authority(
        target_universe_path,
        expected_authority=expected_target_universe_authority,
    )
    if target_universe is None:
        raise MaterializedTrajectoryError("staged trajectory universe is untyped")
    if (
        target_universe.authority.parent_authority_sha256
        != source_universe.reference.sha256
    ):
        raise MaterializedTrajectoryError(
            "staged trajectory universe does not descend from source universe"
        )
    selector_name = materialized_trajectory_provenance_path(target_path).name
    try:
        with AnchoredDirectory.open(source_path.parent) as source_root:
            with AnchoredDirectory.open(target_path.parent) as target_root:
                if (
                    source_root.identity == target_root.identity
                    and source_path.name == target_path.name
                ):
                    raise MaterializedTrajectoryError(
                        "trajectory source and target must differ"
                    )
                target_root.require_absent(target_path.name, selector_name)
                _atomic_json_at(
                    target_root,
                    name=selector_name,
                    payload={
                        "schema_version": MATERIALIZED_TRAJECTORY_TRANSACTION_SCHEMA,
                        "trajectory_authority_required": True,
                        "trajectory_authority": None,
                        "authority_transaction_state": "prepared",
                    },
                    require_absent=True,
                )
                temporary_name, descriptor = target_root.create_temporary(
                    stem=target_path.name
                )
                digest = hashlib.sha256()
                copied = 0
                try:
                    with source_root.open_regular(source_path.name) as input_handle:
                        with os.fdopen(descriptor, "wb") as output_handle:
                            descriptor = -1
                            for chunk in iter(
                                lambda: input_handle.read(1024 * 1024), b""
                            ):
                                digest.update(chunk)
                                copied += len(chunk)
                                output_handle.write(chunk)
                            output_handle.flush()
                            os.fsync(output_handle.fileno())
                    if (
                        digest.hexdigest() != source.authority.trajectory_sha256
                        or copied != source.authority.trajectory_size
                    ):
                        raise MaterializedTrajectoryError(
                            "staged trajectory exact-copy digest mismatch"
                        )
                    parent_raw = source_root.read_bytes(
                        source.reference.file,
                        max_bytes=_MAX_AUTHORITY_BYTES,
                        expected_size=source.reference.size,
                        expected_sha256=source.reference.sha256,
                    )
                    target_root.publish_immutable_bytes(
                        source.reference.file, parent_raw
                    )
                    parameters = {
                        "source_trajectory_authority_sha256": source.reference.sha256,
                        "source_trajectory_sha256": source.authority.trajectory_sha256,
                        "source_universe_authority_sha256": (
                            source_universe.reference.sha256
                        ),
                        "target_universe_authority_sha256": (
                            target_universe.reference.sha256
                        ),
                        "target_file": target_path.name,
                        "transform": "identity_stage_copy",
                    }
                    authority = MaterializedTrajectoryAuthority(
                        trajectory_file=target_path.name,
                        trajectory_sha256=source.authority.trajectory_sha256,
                        trajectory_size=source.authority.trajectory_size,
                        trajectory_rows=source.authority.trajectory_rows,
                        trajectory_columns=source.authority.trajectory_columns,
                        trajectory_schema_sha256=(
                            source.authority.trajectory_schema_sha256
                        ),
                        identity_column=source.authority.identity_column,
                        time_column=source.authority.time_column,
                        time_origin=source.authority.time_origin,
                        time_unit=source.authority.time_unit,
                        concept_column=source.authority.concept_column,
                        numeric_value_column=source.authority.numeric_value_column,
                        text_value_column=source.authority.text_value_column,
                        ordered_row_key_sha256=(
                            source.authority.ordered_row_key_sha256
                        ),
                        stay_identity_set_sha256=(
                            source.authority.stay_identity_set_sha256
                        ),
                        trajectory_stays=source.authority.trajectory_stays,
                        bound_universe_file=target_universe_path.name,
                        bound_universe_authority=target_universe.reference,
                        bound_universe_row_identity_sha256=(
                            target_universe.authority.row_identity_sha256
                        ),
                        source_export_authority_sha256=(
                            target_universe.authority.source_export_authority_sha256
                        ),
                        concept_bindings=source.authority.concept_bindings,
                        requested_concepts=source.authority.requested_concepts,
                        materialized_concepts=source.authority.materialized_concepts,
                        available_unobserved_concepts=(
                            source.authority.available_unobserved_concepts
                        ),
                        unavailable_concepts=source.authority.unavailable_concepts,
                        window=source.authority.window,
                        producer="research_agent_run_stage",
                        producer_implementation_sha256=producer_implementation_sha256,
                        producer_parameters=parameters,
                        producer_parameters_sha256=canonical_parameters_sha256(
                            parameters
                        ),
                        semantic_provenance={
                            **dict(_thaw_json(source.authority.semantic_provenance)),
                            "staged_from_trajectory_authority_sha256": (
                                source.reference.sha256
                            ),
                        },
                        parent_trajectory_authority=source.reference,
                    )
                    authority_ref = _write_authority_at(target_root, authority)
                    source_root.assert_still_selected()
                    target_root.replace_temporary(
                        temporary_name, target_path.name, require_absent=True
                    )
                    temporary_name = ""
                    _atomic_json_at(
                        target_root,
                        name=selector_name,
                        payload={
                            **dict(_thaw_json(authority.semantic_provenance)),
                            "schema_version": (
                                MATERIALIZED_TRAJECTORY_TRANSACTION_SCHEMA
                            ),
                            "trajectory_authority_required": True,
                            "trajectory_authority": _descriptor(
                                authority=authority_ref,
                                relative_path=target_path.name,
                            ),
                            "authority_transaction_state": "committed",
                        },
                    )
                    target_root.assert_still_selected()
                finally:
                    if descriptor >= 0:
                        os.close(descriptor)
                    if temporary_name:
                        target_root.unlink(temporary_name, missing_ok=True)
    except AuthorityFilesystemError as exc:
        raise MaterializedTrajectoryError(
            "cannot stage materialized trajectory authority"
        ) from exc
    result = load_verified_materialized_trajectory_authority(
        target_path,
        expected_authority=authority_ref,
        expected_universe_authority=target_universe.reference,
    )
    if result is None:
        raise MaterializedTrajectoryError("staged trajectory lost its authority")
    return result


def stage_legacy_trajectory_exact(
    source_path: Path,
    target_path: Path,
    *,
    expected_sha256: str,
    expected_size: int,
) -> Path:
    """Stage one legacy trajectory exactly, without inventing typed lineage."""

    source_path = Path(source_path)
    target_path = Path(target_path)
    expected_sha256 = _digest(expected_sha256, label="legacy trajectory sha256")
    expected_size = _nonnegative_int(expected_size, label="legacy trajectory size")
    try:
        with AnchoredDirectory.open(source_path.parent) as source_root:
            with AnchoredDirectory.open(target_path.parent) as target_root:
                if (
                    source_root.identity == target_root.identity
                    and source_path.name == target_path.name
                ):
                    raise MaterializedTrajectoryError(
                        "legacy trajectory source and target must differ"
                    )
                target_root.require_absent(target_path.name)
                temporary_name, descriptor = target_root.create_temporary(
                    stem=target_path.name
                )
                digest = hashlib.sha256()
                copied = 0
                try:
                    with source_root.open_regular(source_path.name) as input_handle:
                        with os.fdopen(descriptor, "wb") as output_handle:
                            descriptor = -1
                            for chunk in iter(
                                lambda: input_handle.read(1024 * 1024), b""
                            ):
                                digest.update(chunk)
                                copied += len(chunk)
                                output_handle.write(chunk)
                            output_handle.flush()
                            os.fsync(output_handle.fileno())
                    if copied != expected_size or digest.hexdigest() != expected_sha256:
                        raise MaterializedTrajectoryError(
                            "legacy trajectory changed while being staged"
                        )
                    source_root.assert_still_selected()
                    target_root.replace_temporary(
                        temporary_name, target_path.name, require_absent=True
                    )
                    temporary_name = ""
                    target_root.assert_still_selected()
                finally:
                    if descriptor >= 0:
                        os.close(descriptor)
                    if temporary_name:
                        target_root.unlink(temporary_name, missing_ok=True)
    except AuthorityFilesystemError as exc:
        raise MaterializedTrajectoryError(
            "cannot stage legacy trajectory exactly"
        ) from exc
    return target_path


__all__ = [
    "MATERIALIZED_TRAJECTORY_AUTHORITY_REF_SCHEMA",
    "MATERIALIZED_TRAJECTORY_AUTHORITY_SCHEMA",
    "MATERIALIZED_TRAJECTORY_DESCRIPTOR_SCHEMA",
    "MaterializedTrajectoryAuthority",
    "MaterializedTrajectoryAuthorityRef",
    "MaterializedTrajectoryError",
    "StagedTrajectoryBinding",
    "TrajectoryConceptBinding",
    "TrajectoryWindow",
    "VerifiedMaterializedTrajectoryAuthority",
    "load_verified_materialized_trajectory_authority",
    "materialized_trajectory_provenance_path",
    "publish_materialized_trajectory_authority",
    "stage_materialized_trajectory_authority",
    "stage_legacy_trajectory_exact",
    "verify_materialized_trajectory_envelope",
]
