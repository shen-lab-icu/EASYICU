"""Strict content-addressed sidecars for typed physical-column metadata.

The sidecar is a concept-layer authority shared by the Web export producer and
research-agent intake.  It contains no study-design decisions and does not
discover files: callers must bind the returned reference from their manifest.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import stat
import uuid
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from types import MappingProxyType
from typing import Mapping, Optional, Sequence

from .metadata_projection import (
    METADATA_SCHEMA_VERSION,
    ConceptColumnMetadata,
    MetadataProjectionError,
)

COLUMN_METADATA_SIDECAR_SCHEMA = "easyicu.column_metadata_sidecar/1"
EXPORT_PHYSICAL_SCOPE = "export_physical_columns"
MATERIALIZED_COHORT_SCOPE = "materialized_cohort_columns"
_SCOPES = {EXPORT_PHYSICAL_SCOPE, MATERIALIZED_COHORT_SCOPE}
_MAX_SIDECAR_BYTES = 64 * 1024 * 1024


class MetadataSidecarError(ValueError):
    """Raised when sidecar bytes cannot establish typed metadata authority."""


def _required_string(value: object, *, label: str) -> str:
    if not isinstance(value, str) or not value.strip() or value != value.strip():
        raise MetadataSidecarError(f"{label} must be a canonical non-empty string")
    return value


def _optional_string(value: object, *, label: str) -> Optional[str]:
    if value is None:
        return None
    return _required_string(value, label=label)


def _exact_keys(
    payload: Mapping[str, object], expected: set[str], *, label: str
) -> None:
    if any(not isinstance(key, str) for key in payload):
        raise MetadataSidecarError(f"{label} keys must be strings")
    actual = set(payload)
    if actual != expected:
        raise MetadataSidecarError(
            f"{label} keys do not match schema "
            f"(missing={sorted(expected - actual)}, extra={sorted(actual - expected)})"
        )


def _relative_path(value: object) -> str:
    raw = _required_string(value, label="relative_path")
    path = PurePosixPath(raw)
    if path.is_absolute() or ".." in path.parts or path.as_posix() != raw:
        raise MetadataSidecarError("relative_path must be normalized and contained")
    return raw


def _finite_number(value: object, *, label: str) -> float:
    if isinstance(value, bool):
        raise MetadataSidecarError(f"{label} must be finite")
    try:
        number = float(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise MetadataSidecarError(f"{label} must be finite") from exc
    if not math.isfinite(number):
        raise MetadataSidecarError(f"{label} must be finite")
    return 0.0 if number == 0.0 else number


@dataclass(frozen=True, slots=True)
class DerivationWindow:
    """Explicit fixed window used to derive one physical output column."""

    origin: str
    start_hours: float
    end_hours: float

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "origin", _required_string(self.origin, label="window origin")
        )
        object.__setattr__(
            self,
            "start_hours",
            _finite_number(self.start_hours, label="window start_hours"),
        )
        object.__setattr__(
            self,
            "end_hours",
            _finite_number(self.end_hours, label="window end_hours"),
        )
        if self.start_hours > self.end_hours:
            raise MetadataSidecarError("window start_hours must not exceed end_hours")

    def to_dict(self) -> dict[str, object]:
        return {
            "origin": self.origin,
            "start_hours": self.start_hours,
            "end_hours": self.end_hours,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, object]) -> "DerivationWindow":
        _exact_keys(
            payload,
            {"origin", "start_hours", "end_hours"},
            label="derivation_window",
        )
        parsed = cls(
            origin=payload["origin"],  # type: ignore[arg-type]
            start_hours=payload["start_hours"],  # type: ignore[arg-type]
            end_hours=payload["end_hours"],  # type: ignore[arg-type]
        )
        if parsed.to_dict() != dict(payload):
            raise MetadataSidecarError("derivation_window is not canonical")
        return parsed


@dataclass(frozen=True, slots=True)
class ColumnMetadataBinding:
    """Typed metadata plus the producer-owned derivation coordinates."""

    metadata: ConceptColumnMetadata
    derivation_window: Optional[DerivationWindow] = None
    representation_transform: Optional[str] = None

    def __post_init__(self) -> None:
        if not isinstance(self.metadata, ConceptColumnMetadata):
            raise MetadataSidecarError("binding metadata must be ConceptColumnMetadata")
        if self.derivation_window is not None and not isinstance(
            self.derivation_window, DerivationWindow
        ):
            raise MetadataSidecarError("derivation_window has an invalid type")
        object.__setattr__(
            self,
            "representation_transform",
            _optional_string(
                self.representation_transform, label="representation_transform"
            ),
        )

    def to_dict(self) -> dict[str, object]:
        return {
            "metadata": self.metadata.to_dict(),
            "derivation_window": (
                self.derivation_window.to_dict() if self.derivation_window else None
            ),
            "representation_transform": self.representation_transform,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, object]) -> "ColumnMetadataBinding":
        _exact_keys(
            payload,
            {"metadata", "derivation_window", "representation_transform"},
            label="column binding",
        )
        raw_metadata = payload["metadata"]
        raw_window = payload["derivation_window"]
        if not isinstance(raw_metadata, Mapping):
            raise MetadataSidecarError("column binding metadata must be an object")
        if raw_window is not None and not isinstance(raw_window, Mapping):
            raise MetadataSidecarError("derivation_window must be an object or null")
        try:
            metadata = ConceptColumnMetadata.from_dict(raw_metadata)
        except MetadataProjectionError as exc:
            raise MetadataSidecarError(str(exc)) from exc
        parsed = cls(
            metadata=metadata,
            derivation_window=(
                DerivationWindow.from_dict(raw_window)
                if isinstance(raw_window, Mapping)
                else None
            ),
            representation_transform=_optional_string(
                payload["representation_transform"],
                label="representation_transform",
            ),
        )
        if parsed.to_dict() != dict(payload):
            raise MetadataSidecarError("column binding is not canonical")
        return parsed


@dataclass(frozen=True, slots=True)
class TimeCoordinate:
    """File-level shared time coordinate, not owned by one source concept."""

    column: str
    origin: str
    unit: str

    def __post_init__(self) -> None:
        for field_name in ("column", "origin", "unit"):
            object.__setattr__(
                self,
                field_name,
                _required_string(getattr(self, field_name), label=f"time {field_name}"),
            )

    def to_dict(self) -> dict[str, str]:
        return {"column": self.column, "origin": self.origin, "unit": self.unit}

    @classmethod
    def from_dict(cls, payload: Mapping[str, object]) -> "TimeCoordinate":
        _exact_keys(payload, {"column", "origin", "unit"}, label="time coordinate")
        parsed = cls(
            column=payload["column"],  # type: ignore[arg-type]
            origin=payload["origin"],  # type: ignore[arg-type]
            unit=payload["unit"],  # type: ignore[arg-type]
        )
        if parsed.to_dict() != dict(payload):
            raise MetadataSidecarError("time coordinate is not canonical")
        return parsed


def binding_payload_sha256(columns: Mapping[str, ColumnMetadataBinding]) -> str:
    canonical: dict[str, object] = {}
    for raw_name, binding in columns.items():
        name = _required_string(raw_name, label="column metadata key")
        if not isinstance(binding, ColumnMetadataBinding):
            raise MetadataSidecarError("column metadata values must be bindings")
        if binding.metadata.column_name != name:
            raise MetadataSidecarError(
                "column metadata key must match metadata.column_name"
            )
        canonical[name] = binding.to_dict()
    return hashlib.sha256(
        json.dumps(
            {name: canonical[name] for name in sorted(canonical)},
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
    ).hexdigest()


@dataclass(frozen=True, slots=True)
class ColumnMetadataFileBinding:
    """Exact manifest-file binding for typed value columns and shared time."""

    relative_path: str
    module: str
    identity_column: str
    time_coordinates: tuple[TimeCoordinate, ...]
    columns: Mapping[str, ColumnMetadataBinding]

    def __post_init__(self) -> None:
        object.__setattr__(self, "relative_path", _relative_path(self.relative_path))
        object.__setattr__(
            self, "module", _required_string(self.module, label="module")
        )
        object.__setattr__(
            self,
            "identity_column",
            _required_string(self.identity_column, label="identity_column"),
        )
        coordinates = tuple(self.time_coordinates)
        if any(not isinstance(item, TimeCoordinate) for item in coordinates):
            raise MetadataSidecarError("time_coordinates contains an invalid item")
        coordinate_names = [item.column for item in coordinates]
        if len(set(coordinate_names)) != len(coordinate_names):
            raise MetadataSidecarError("time coordinate columns must be unique")
        if self.identity_column in coordinate_names:
            raise MetadataSidecarError(
                "identity and time coordinate columns must be disjoint"
            )
        canonical_columns = dict(self.columns)
        binding_payload_sha256(canonical_columns)
        structural_names = {self.identity_column, *coordinate_names}
        overlap = structural_names.intersection(canonical_columns)
        if overlap:
            raise MetadataSidecarError(
                "identity/time coordinates must not also be value bindings: "
                f"{sorted(overlap)}"
            )
        object.__setattr__(self, "time_coordinates", coordinates)
        object.__setattr__(
            self,
            "columns",
            MappingProxyType(
                {name: canonical_columns[name] for name in sorted(canonical_columns)}
            ),
        )

    @property
    def metadata_payload_sha256(self) -> str:
        return binding_payload_sha256(self.columns)

    def to_dict(self) -> dict[str, object]:
        return {
            "relative_path": self.relative_path,
            "module": self.module,
            "identity_column": self.identity_column,
            "time_coordinates": [item.to_dict() for item in self.time_coordinates],
            "metadata_payload_sha256": self.metadata_payload_sha256,
            "columns": {
                name: binding.to_dict() for name, binding in self.columns.items()
            },
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, object]) -> "ColumnMetadataFileBinding":
        _exact_keys(
            payload,
            {
                "relative_path",
                "module",
                "identity_column",
                "time_coordinates",
                "metadata_payload_sha256",
                "columns",
            },
            label="file metadata binding",
        )
        raw_times = payload["time_coordinates"]
        raw_columns = payload["columns"]
        if not isinstance(raw_times, list) or not all(
            isinstance(item, Mapping) for item in raw_times
        ):
            raise MetadataSidecarError("time_coordinates must be a list of objects")
        if not isinstance(raw_columns, Mapping) or any(
            not isinstance(name, str) or not isinstance(value, Mapping)
            for name, value in raw_columns.items()
        ):
            raise MetadataSidecarError("columns must be an object of bindings")
        parsed = cls(
            relative_path=payload["relative_path"],  # type: ignore[arg-type]
            module=payload["module"],  # type: ignore[arg-type]
            identity_column=payload["identity_column"],  # type: ignore[arg-type]
            time_coordinates=tuple(
                TimeCoordinate.from_dict(item) for item in raw_times
            ),
            columns={
                name: ColumnMetadataBinding.from_dict(value)
                for name, value in raw_columns.items()
            },
        )
        digest = _required_string(
            payload["metadata_payload_sha256"], label="metadata_payload_sha256"
        )
        if digest != parsed.metadata_payload_sha256:
            raise MetadataSidecarError("metadata payload digest mismatch")
        if parsed.to_dict() != dict(payload):
            raise MetadataSidecarError("file metadata binding is not canonical")
        return parsed


@dataclass(frozen=True, slots=True)
class ColumnMetadataSidecar:
    """One immutable column-metadata authority for an export or cohort file set."""

    source_database: str
    source_database_class_prefixes: tuple[str, ...]
    scope: str
    files: tuple[ColumnMetadataFileBinding, ...]
    schema_version: str = COLUMN_METADATA_SIDECAR_SCHEMA
    metadata_schema_version: str = METADATA_SCHEMA_VERSION

    def __post_init__(self) -> None:
        if self.schema_version != COLUMN_METADATA_SIDECAR_SCHEMA:
            raise MetadataSidecarError("unsupported column metadata sidecar schema")
        if self.metadata_schema_version != METADATA_SCHEMA_VERSION:
            raise MetadataSidecarError("unsupported concept metadata schema")
        database = _required_string(
            self.source_database, label="source_database"
        ).lower()
        if database != self.source_database:
            raise MetadataSidecarError("source_database must be lowercase")
        prefixes = tuple(self.source_database_class_prefixes)
        if any(
            not isinstance(value, str) or not value or value != value.strip().lower()
            for value in prefixes
        ) or len(set(prefixes)) != len(prefixes):
            raise MetadataSidecarError(
                "source database class prefixes must be unique lowercase strings"
            )
        if self.scope not in _SCOPES:
            raise MetadataSidecarError("unsupported column metadata scope")
        files = tuple(self.files)
        if not files or any(
            not isinstance(item, ColumnMetadataFileBinding) for item in files
        ):
            raise MetadataSidecarError("sidecar must contain file metadata bindings")
        paths = [item.relative_path for item in files]
        if len(set(paths)) != len(paths):
            raise MetadataSidecarError("sidecar file paths must be unique")
        for file_binding in files:
            for binding in file_binding.columns.values():
                metadata = binding.metadata
                if metadata.source_database != database:
                    raise MetadataSidecarError(
                        "column source_database does not match sidecar authority"
                    )
                expected_chain = tuple(dict.fromkeys((database, *prefixes)))
                if metadata.source_resolution_chain != expected_chain:
                    raise MetadataSidecarError(
                        "column source resolution chain does not match sidecar authority"
                    )
        object.__setattr__(self, "source_database", database)
        object.__setattr__(self, "source_database_class_prefixes", prefixes)
        object.__setattr__(
            self, "files", tuple(sorted(files, key=lambda item: item.relative_path))
        )

    @property
    def record_count(self) -> int:
        return sum(len(item.columns) for item in self.files)

    def to_dict(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "metadata_schema_version": self.metadata_schema_version,
            "source_database": self.source_database,
            "source_database_class_prefixes": list(self.source_database_class_prefixes),
            "scope": self.scope,
            "record_count": self.record_count,
            "files": [item.to_dict() for item in self.files],
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, object]) -> "ColumnMetadataSidecar":
        _exact_keys(
            payload,
            {
                "schema_version",
                "metadata_schema_version",
                "source_database",
                "source_database_class_prefixes",
                "scope",
                "record_count",
                "files",
            },
            label="column metadata sidecar",
        )
        raw_prefixes = payload["source_database_class_prefixes"]
        raw_files = payload["files"]
        raw_count = payload["record_count"]
        if not isinstance(raw_prefixes, list) or not all(
            isinstance(value, str) for value in raw_prefixes
        ):
            raise MetadataSidecarError(
                "source_database_class_prefixes must be a string list"
            )
        if not isinstance(raw_files, list) or not all(
            isinstance(value, Mapping) for value in raw_files
        ):
            raise MetadataSidecarError("sidecar files must be a list of objects")
        if (
            not isinstance(raw_count, int)
            or isinstance(raw_count, bool)
            or raw_count < 0
        ):
            raise MetadataSidecarError("record_count must be a non-negative integer")
        parsed = cls(
            source_database=payload["source_database"],  # type: ignore[arg-type]
            source_database_class_prefixes=tuple(raw_prefixes),
            scope=payload["scope"],  # type: ignore[arg-type]
            files=tuple(
                ColumnMetadataFileBinding.from_dict(value) for value in raw_files
            ),
            schema_version=payload["schema_version"],  # type: ignore[arg-type]
            metadata_schema_version=payload["metadata_schema_version"],  # type: ignore[arg-type]
        )
        if raw_count != parsed.record_count:
            raise MetadataSidecarError("sidecar record_count mismatch")
        if parsed.to_dict() != dict(payload):
            raise MetadataSidecarError("column metadata sidecar is not canonical")
        return parsed


@dataclass(frozen=True, slots=True)
class SidecarRef:
    file: str
    sha256: str
    size: int
    record_count: int
    schema_version: str = COLUMN_METADATA_SIDECAR_SCHEMA
    metadata_schema_version: str = METADATA_SCHEMA_VERSION

    def __post_init__(self) -> None:
        object.__setattr__(self, "file", _relative_path(self.file))
        digest = _required_string(self.sha256, label="sidecar sha256")
        if len(digest) != 64 or any(ch not in "0123456789abcdef" for ch in digest):
            raise MetadataSidecarError("sidecar sha256 must be 64 lowercase hex digits")
        for label, value in (
            ("sidecar size", self.size),
            ("sidecar record_count", self.record_count),
        ):
            if not isinstance(value, int) or isinstance(value, bool) or value < 0:
                raise MetadataSidecarError(f"{label} must be a non-negative integer")
        if self.schema_version != COLUMN_METADATA_SIDECAR_SCHEMA:
            raise MetadataSidecarError("unsupported column metadata sidecar schema")
        if self.metadata_schema_version != METADATA_SCHEMA_VERSION:
            raise MetadataSidecarError("unsupported concept metadata schema")

    def to_dict(self) -> dict[str, object]:
        return {
            "file": self.file,
            "sha256": self.sha256,
            "size": self.size,
            "record_count": self.record_count,
            "schema_version": self.schema_version,
            "metadata_schema_version": self.metadata_schema_version,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, object]) -> "SidecarRef":
        _exact_keys(
            payload,
            {
                "file",
                "sha256",
                "size",
                "record_count",
                "schema_version",
                "metadata_schema_version",
            },
            label="column metadata descriptor",
        )
        parsed = cls(
            file=payload["file"],  # type: ignore[arg-type]
            sha256=payload["sha256"],  # type: ignore[arg-type]
            size=payload["size"],  # type: ignore[arg-type]
            record_count=payload["record_count"],  # type: ignore[arg-type]
            schema_version=payload["schema_version"],  # type: ignore[arg-type]
            metadata_schema_version=payload["metadata_schema_version"],  # type: ignore[arg-type]
        )
        if parsed.to_dict() != dict(payload):
            raise MetadataSidecarError("column metadata descriptor is not canonical")
        return parsed


def canonical_sidecar_bytes(sidecar: ColumnMetadataSidecar) -> bytes:
    if not isinstance(sidecar, ColumnMetadataSidecar):
        raise MetadataSidecarError("sidecar must be ColumnMetadataSidecar")
    return json.dumps(
        sidecar.to_dict(),
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")


def sidecar_sha256(sidecar: ColumnMetadataSidecar) -> str:
    return hashlib.sha256(canonical_sidecar_bytes(sidecar)).hexdigest()


def _reject_duplicate_pairs(pairs: Sequence[tuple[str, object]]) -> dict[str, object]:
    result: dict[str, object] = {}
    for key, value in pairs:
        if key in result:
            raise MetadataSidecarError(f"duplicate JSON key: {key!r}")
        result[key] = value
    return result


def _reject_constant(value: str) -> object:
    raise MetadataSidecarError(f"non-finite JSON constant is forbidden: {value}")


def parse_column_metadata_sidecar(raw: bytes) -> ColumnMetadataSidecar:
    if not isinstance(raw, bytes) or len(raw) > _MAX_SIDECAR_BYTES:
        raise MetadataSidecarError("column metadata sidecar bytes are invalid")
    try:
        payload = json.loads(
            raw.decode("utf-8"),
            object_pairs_hook=_reject_duplicate_pairs,
            parse_constant=_reject_constant,
        )
    except MetadataSidecarError:
        raise
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise MetadataSidecarError("invalid column metadata JSON") from exc
    if not isinstance(payload, Mapping):
        raise MetadataSidecarError("column metadata sidecar must be an object")
    parsed = ColumnMetadataSidecar.from_dict(payload)
    if canonical_sidecar_bytes(parsed) != raw:
        raise MetadataSidecarError("column metadata sidecar bytes are not canonical")
    return parsed


def read_content_addressed_sidecar(
    path: Path,
    *,
    expected_sha256: str,
    expected_size: int,
) -> ColumnMetadataSidecar:
    if (
        not isinstance(expected_sha256, str)
        or len(expected_sha256) != 64
        or any(ch not in "0123456789abcdef" for ch in expected_sha256)
    ):
        raise MetadataSidecarError("expected sidecar sha256 is invalid")
    if (
        not isinstance(expected_size, int)
        or isinstance(expected_size, bool)
        or expected_size < 0
    ):
        raise MetadataSidecarError("expected sidecar size is invalid")
    if path.is_symlink():
        raise MetadataSidecarError("column metadata sidecar must not be a symlink")
    try:
        info = path.stat()
        if not stat.S_ISREG(info.st_mode) or info.st_size != expected_size:
            raise MetadataSidecarError("column metadata sidecar size/type mismatch")
        if info.st_size > _MAX_SIDECAR_BYTES:
            raise MetadataSidecarError("column metadata sidecar exceeds size limit")
        raw = path.read_bytes()
    except OSError as exc:
        raise MetadataSidecarError("cannot read column metadata sidecar") from exc
    if hashlib.sha256(raw).hexdigest() != expected_sha256:
        raise MetadataSidecarError("column metadata sidecar digest mismatch")
    return parse_column_metadata_sidecar(raw)


def write_content_addressed_sidecar(
    root: Path,
    sidecar: ColumnMetadataSidecar,
    *,
    stem: str = "column_metadata",
) -> SidecarRef:
    root = Path(root)
    if root.is_symlink() or not root.is_dir():
        raise MetadataSidecarError("sidecar root must be an existing real directory")
    stem = _required_string(stem, label="sidecar stem")
    if "/" in stem or "\\" in stem or stem in {".", ".."}:
        raise MetadataSidecarError("sidecar stem must be one path component")
    raw = canonical_sidecar_bytes(sidecar)
    digest = hashlib.sha256(raw).hexdigest()
    file_name = f"{stem}.sha256-{digest}.json"
    target = root / file_name
    temporary = root / f".{file_name}.{uuid.uuid4().hex}.tmp"
    fd: Optional[int] = None
    try:
        fd = os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
        view = memoryview(raw)
        while view:
            written = os.write(fd, view)
            view = view[written:]
        os.fsync(fd)
        os.close(fd)
        fd = None
        try:
            os.link(temporary, target, follow_symlinks=False)
        except FileExistsError:
            if (
                target.is_symlink()
                or not target.is_file()
                or target.read_bytes() != raw
            ):
                raise MetadataSidecarError(
                    "existing content-addressed sidecar has different bytes"
                )
        directory_fd = os.open(root, os.O_RDONLY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
    except OSError as exc:
        raise MetadataSidecarError("cannot publish column metadata sidecar") from exc
    finally:
        if fd is not None:
            os.close(fd)
        try:
            temporary.unlink()
        except FileNotFoundError:
            pass
    return SidecarRef(
        file=file_name,
        sha256=digest,
        size=len(raw),
        record_count=sidecar.record_count,
    )


__all__ = [
    "COLUMN_METADATA_SIDECAR_SCHEMA",
    "EXPORT_PHYSICAL_SCOPE",
    "MATERIALIZED_COHORT_SCOPE",
    "ColumnMetadataBinding",
    "ColumnMetadataFileBinding",
    "ColumnMetadataSidecar",
    "DerivationWindow",
    "MetadataSidecarError",
    "SidecarRef",
    "TimeCoordinate",
    "binding_payload_sha256",
    "canonical_sidecar_bytes",
    "parse_column_metadata_sidecar",
    "read_content_addressed_sidecar",
    "sidecar_sha256",
    "write_content_addressed_sidecar",
]
