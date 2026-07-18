"""Typed metadata authority for host-materialized wide cohort columns.

The collector is deliberately downstream of a verified export-package
sidecar.  It records transformations the deterministic cohort materializer
actually executed; it never resolves concepts from column names or re-queries
the mutable concept dictionary.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path
import stat
from types import MappingProxyType
from typing import Mapping, Optional, Sequence

import pyarrow as pa
import pyarrow.parquet as pq

from easyicu.concept.metadata_projection import (
    ColumnProjectionSpec,
    ConceptColumnRole,
    derive_concept_column_metadata,
)
from easyicu.concept.metadata_sidecar import (
    EXPORT_PHYSICAL_SCOPE,
    MATERIALIZED_COHORT_SCOPE,
    ColumnMetadataBinding,
    ColumnMetadataFileBinding,
    ColumnMetadataSidecar,
    DerivationWindow,
    MetadataSidecarError,
    SidecarRef,
    binding_payload_sha256,
    canonical_sidecar_bytes,
    read_content_addressed_sidecar,
)

from ..authority_fs import AnchoredDirectory, AuthorityFilesystemError
from .export_package import ExportPackage, resolve_exported_concept

_PRIMARY_ROLES = {ConceptColumnRole.VALUE, ConceptColumnRole.EVENT_STATUS}
_VALUE_AGGREGATIONS = {"first", "last", "max", "mean", "median", "min", "sum"}
_EVENT_AGGREGATIONS = {"all", "any", "first", "last", "max", "min"}
MATERIALIZED_COHORT_AUTHORITY_SCHEMA = "easyicu.materialized_cohort_authority/1"
MATERIALIZED_COHORT_AUTHORITY_REF_SCHEMA = "easyicu.materialized_cohort_authority_ref/1"
MATERIALIZED_COHORT_DESCRIPTOR_SCHEMA = "easyicu.materialized_cohort_descriptor/1"
_MAX_AUTHORITY_BYTES = 64 * 1024 * 1024
_MAX_SELECTOR_BYTES = 4 * 1024 * 1024
_MAX_AUTHORITY_ANCESTRY = 256
_HEX = frozenset("0123456789abcdef")


class MaterializedMetadataError(MetadataSidecarError):
    """Raised when materializer output cannot be bound to source authority."""


def _require_real_directory(path: Path, *, label: str) -> Path:
    return _prepare_real_directory(path, label=label, create=False)


def prepare_real_directory(path: Path, *, label: str) -> Path:
    """Create a directory only through already verified non-symlink parents."""

    return _prepare_real_directory(path, label=label, create=True)


def _prepare_real_directory(path: Path, *, label: str, create: bool) -> Path:
    absolute = Path(path).expanduser()
    absolute = absolute if absolute.is_absolute() else absolute.absolute()
    current = Path(absolute.anchor)
    for part in absolute.parts[1:]:
        current = current / part
        try:
            info = current.lstat()
        except FileNotFoundError:
            if not create:
                raise MaterializedMetadataError(f"{label} is not readable")
            try:
                current.mkdir()
                info = current.lstat()
            except OSError as exc:
                raise MaterializedMetadataError(f"cannot create {label}") from exc
        except OSError as exc:
            raise MaterializedMetadataError(f"{label} is not readable") from exc
        if stat.S_ISLNK(info.st_mode):
            # macOS exposes /var and /tmp as fixed system aliases.  Canonicalize
            # only those platform roots; reject project/user-controlled links.
            if str(current) not in {"/var", "/tmp", "/etc"}:
                raise MaterializedMetadataError(
                    f"{label} must contain only real directory components"
                )
            current = current.resolve(strict=True)
            continue
        if not stat.S_ISDIR(info.st_mode):
            raise MaterializedMetadataError(
                f"{label} must contain only real directory components"
            )
    return current


def _file_stability_coordinate(path: Path) -> tuple[int, int, int, int]:
    try:
        info = path.stat()
    except OSError as exc:
        raise MaterializedMetadataError("cannot stat materialized cohort") from exc
    if not stat.S_ISREG(info.st_mode):
        raise MaterializedMetadataError("materialized cohort must be a regular file")
    return (
        int(info.st_dev),
        int(info.st_ino),
        int(info.st_size),
        int(info.st_mtime_ns),
    )


def _read_bounded_regular_file(
    path: Path,
    *,
    label: str,
    max_bytes: int,
    expected_size: Optional[int] = None,
) -> bytes:
    """Read one descriptor-anchored regular file without following symlinks."""

    flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0) | getattr(os, "O_NONBLOCK", 0)
    fd: Optional[int] = None
    try:
        if not getattr(os, "O_NOFOLLOW", 0) and path.is_symlink():
            raise MaterializedMetadataError(f"{label} must not be a symlink")
        fd = os.open(path, flags)
        info = os.fstat(fd)
        if not stat.S_ISREG(info.st_mode):
            raise MaterializedMetadataError(f"{label} must be a regular file")
        size = int(info.st_size)
        if size > max_bytes or (expected_size is not None and size != expected_size):
            raise MaterializedMetadataError(f"{label} size/type mismatch")
        chunks: list[bytes] = []
        remaining = size
        while remaining:
            chunk = os.read(fd, min(1024 * 1024, remaining))
            if not chunk:
                raise MaterializedMetadataError(f"{label} ended before its stat size")
            chunks.append(chunk)
            remaining -= len(chunk)
        if os.read(fd, 1):
            raise MaterializedMetadataError(f"{label} changed while being read")
        return b"".join(chunks)
    except MaterializedMetadataError:
        raise
    except OSError as exc:
        raise MaterializedMetadataError(f"cannot read {label}") from exc
    finally:
        if fd is not None:
            os.close(fd)


def _publish_content_addressed_snapshot(
    source: Path,
    target_root: Path,
    *,
    reference: SidecarRef | MaterializedCohortAuthorityRef,
    label: str = "source column metadata",
) -> None:
    """Publish a verified content-addressed blob without a mutable alias."""

    try:
        with AnchoredDirectory.open(source.parent) as source_directory:
            with AnchoredDirectory.open(target_root) as target_directory:
                _publish_content_addressed_snapshot_at(
                    source_directory,
                    target_directory,
                    source_name=source.name,
                    reference=reference,
                    label=label,
                )
    except AuthorityFilesystemError as exc:
        raise MaterializedMetadataError(f"cannot publish {label} snapshot") from exc


def _publish_content_addressed_snapshot_at(
    source_directory: AnchoredDirectory,
    target_directory: AnchoredDirectory,
    *,
    source_name: str,
    reference: SidecarRef | MaterializedCohortAuthorityRef,
    label: str,
) -> None:
    raw = source_directory.read_bytes(
        source_name,
        max_bytes=_MAX_AUTHORITY_BYTES,
        expected_size=reference.size,
        expected_sha256=reference.sha256,
    )
    try:
        target_directory.publish_immutable_bytes(reference.file, raw)
    except AuthorityFilesystemError as exc:
        raise MaterializedMetadataError(f"cannot publish {label} snapshot") from exc


def _write_content_addressed_sidecar_at(
    root: AnchoredDirectory,
    sidecar: ColumnMetadataSidecar,
    *,
    stem: str,
) -> SidecarRef:
    if not stem or stem in {".", ".."} or "/" in stem or "\\" in stem:
        raise MaterializedMetadataError("sidecar stem must be one path component")
    raw = canonical_sidecar_bytes(sidecar)
    digest = hashlib.sha256(raw).hexdigest()
    file_name = f"{stem}.sha256-{digest}.json"
    try:
        root.publish_immutable_bytes(file_name, raw)
    except AuthorityFilesystemError as exc:
        raise MaterializedMetadataError(
            "cannot publish materialized column metadata"
        ) from exc
    return SidecarRef(
        file=file_name,
        sha256=digest,
        size=len(raw),
        record_count=sidecar.record_count,
    )


def _copy_verified_regular_to_temporary(
    source_root: AnchoredDirectory,
    target_root: AnchoredDirectory,
    *,
    source_name: str,
    target_stem: str,
    expected_size: int,
    expected_sha256: str,
) -> str:
    """Copy one exact source fd into a temporary file under a target fd."""

    temporary_name, target_fd = target_root.create_temporary(stem=target_stem)
    digest = hashlib.sha256()
    copied = 0
    try:
        with source_root.open_regular(source_name) as source:
            source_before = os.fstat(source.fileno())
            with os.fdopen(target_fd, "wb") as target:
                target_fd = -1
                for chunk in iter(lambda: source.read(1024 * 1024), b""):
                    digest.update(chunk)
                    copied += len(chunk)
                    target.write(chunk)
                target.flush()
                os.fsync(target.fileno())
            source_after = os.fstat(source.fileno())
        if (
            source_before.st_dev,
            source_before.st_ino,
            source_before.st_size,
            source_before.st_mtime_ns,
        ) != (
            source_after.st_dev,
            source_after.st_ino,
            source_after.st_size,
            source_after.st_mtime_ns,
        ):
            raise MaterializedMetadataError(
                "source cohort changed while staging exact bytes"
            )
        if copied != expected_size or digest.hexdigest() != expected_sha256:
            raise MaterializedMetadataError("staged cohort copy digest mismatch")
        return temporary_name
    except BaseException:
        if target_fd >= 0:
            os.close(target_fd)
        target_root.unlink(temporary_name, missing_ok=True)
        raise


def _digest(value: object, *, label: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in _HEX for character in value)
    ):
        raise MaterializedMetadataError(f"{label} must be 64 lowercase hex digits")
    return value


def _nonempty(value: object, *, label: str) -> str:
    if not isinstance(value, str) or not value or value != value.strip():
        raise MaterializedMetadataError(f"{label} must be a canonical string")
    return value


def _nonnegative_int(value: object, *, label: str) -> int:
    if not isinstance(value, int) or isinstance(value, bool) or value < 0:
        raise MaterializedMetadataError(f"{label} must be a non-negative integer")
    return value


def _exact_keys(
    payload: Mapping[str, object], expected: set[str], *, label: str
) -> None:
    if set(payload) != expected:
        raise MaterializedMetadataError(f"{label} keys do not match schema")


@dataclass(frozen=True, slots=True)
class MaterializedCohortAuthorityRef:
    file: str
    sha256: str
    size: int
    schema_version: str = MATERIALIZED_COHORT_AUTHORITY_REF_SCHEMA

    def __post_init__(self) -> None:
        file_name = _nonempty(self.file, label="authority file")
        if Path(file_name).name != file_name or file_name in {".", ".."}:
            raise MaterializedMetadataError("authority file must be one path component")
        object.__setattr__(
            self, "sha256", _digest(self.sha256, label="authority sha256")
        )
        object.__setattr__(
            self, "size", _nonnegative_int(self.size, label="authority size")
        )
        if self.schema_version != MATERIALIZED_COHORT_AUTHORITY_REF_SCHEMA:
            raise MaterializedMetadataError("unsupported authority reference schema")

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
    ) -> "MaterializedCohortAuthorityRef":
        _exact_keys(
            payload,
            {"schema_version", "file", "sha256", "size"},
            label="materialized cohort authority reference",
        )
        parsed = cls(
            file=payload["file"],  # type: ignore[arg-type]
            sha256=payload["sha256"],  # type: ignore[arg-type]
            size=payload["size"],  # type: ignore[arg-type]
            schema_version=payload["schema_version"],  # type: ignore[arg-type]
        )
        if parsed.to_dict() != dict(payload):
            raise MaterializedMetadataError("authority reference is not canonical")
        return parsed


@dataclass(frozen=True, slots=True)
class SourceColumnRef:
    """Exact upstream authority coordinate consumed by one derived column."""

    authority_sha256: str
    file: str
    column: str
    binding_sha256: str

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "authority_sha256",
            _digest(self.authority_sha256, label="source authority sha256"),
        )
        file_name = _nonempty(self.file, label="source file")
        if Path(file_name).is_absolute() or ".." in Path(file_name).parts:
            raise MaterializedMetadataError("source file must be a contained path")
        object.__setattr__(
            self, "column", _nonempty(self.column, label="source column")
        )
        object.__setattr__(
            self,
            "binding_sha256",
            _digest(self.binding_sha256, label="source binding sha256"),
        )

    def to_dict(self) -> dict[str, str]:
        return {
            "authority_sha256": self.authority_sha256,
            "file": self.file,
            "column": self.column,
            "binding_sha256": self.binding_sha256,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, object]) -> "SourceColumnRef":
        _exact_keys(
            payload,
            {"authority_sha256", "file", "column", "binding_sha256"},
            label="source column reference",
        )
        parsed = cls(
            authority_sha256=payload["authority_sha256"],  # type: ignore[arg-type]
            file=payload["file"],  # type: ignore[arg-type]
            column=payload["column"],  # type: ignore[arg-type]
            binding_sha256=payload["binding_sha256"],  # type: ignore[arg-type]
        )
        if parsed.to_dict() != dict(payload):
            raise MaterializedMetadataError("source column reference is not canonical")
        return parsed


@dataclass(frozen=True, slots=True)
class OutputDerivation:
    """Exact source coordinates and transform receipt for one output column."""

    output_column: str
    sources: tuple[SourceColumnRef, ...]
    transform_id: str

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "output_column",
            _nonempty(self.output_column, label="output column"),
        )
        sources = tuple(self.sources)
        if not sources or any(
            not isinstance(item, SourceColumnRef) for item in sources
        ):
            raise MaterializedMetadataError("output derivation requires source columns")
        canonical = tuple(
            sorted(
                sources,
                key=lambda item: (
                    item.authority_sha256,
                    item.file,
                    item.column,
                    item.binding_sha256,
                ),
            )
        )
        if len(
            {json.dumps(item.to_dict(), sort_keys=True) for item in canonical}
        ) != len(canonical):
            raise MaterializedMetadataError("output derivation sources must be unique")
        object.__setattr__(self, "sources", canonical)
        object.__setattr__(
            self,
            "transform_id",
            _nonempty(self.transform_id, label="transform id"),
        )

    def to_dict(self) -> dict[str, object]:
        return {
            "output_column": self.output_column,
            "sources": [item.to_dict() for item in self.sources],
            "transform_id": self.transform_id,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, object]) -> "OutputDerivation":
        _exact_keys(
            payload,
            {"output_column", "sources", "transform_id"},
            label="output derivation",
        )
        raw_sources = payload["sources"]
        if not isinstance(raw_sources, list) or not all(
            isinstance(item, Mapping) for item in raw_sources
        ):
            raise MaterializedMetadataError("output derivation sources must be a list")
        parsed = cls(
            output_column=payload["output_column"],  # type: ignore[arg-type]
            sources=tuple(SourceColumnRef.from_dict(item) for item in raw_sources),
            transform_id=payload["transform_id"],  # type: ignore[arg-type]
        )
        if parsed.to_dict() != dict(payload):
            raise MaterializedMetadataError("output derivation is not canonical")
        return parsed


@dataclass(frozen=True, slots=True)
class MaterializedCohortAuthority:
    cohort_file: str
    cohort_sha256: str
    cohort_size: int
    cohort_rows: int
    cohort_columns: tuple[str, ...]
    cohort_schema_sha256: str
    identity_column: str
    row_identity_sha256: str
    column_metadata: SidecarRef
    column_metadata_scope: str
    file_metadata_payload_sha256: str
    source_export_authority_sha256: str
    source_column_metadata: SidecarRef
    source_column_metadata_sha256: str
    producer: str
    producer_implementation_sha256: str
    producer_parameters: Mapping[str, object]
    producer_parameters_sha256: str
    semantic_provenance: Mapping[str, object]
    output_derivations: tuple[OutputDerivation, ...]
    parent_authority_sha256: Optional[str] = None
    schema_version: str = MATERIALIZED_COHORT_AUTHORITY_SCHEMA

    def __post_init__(self) -> None:
        file_name = _nonempty(self.cohort_file, label="cohort file")
        if Path(file_name).name != file_name or file_name in {".", ".."}:
            raise MaterializedMetadataError("cohort file must be one path component")
        object.__setattr__(
            self, "cohort_sha256", _digest(self.cohort_sha256, label="cohort sha256")
        )
        object.__setattr__(
            self, "cohort_size", _nonnegative_int(self.cohort_size, label="cohort size")
        )
        object.__setattr__(
            self, "cohort_rows", _nonnegative_int(self.cohort_rows, label="cohort rows")
        )
        columns = tuple(self.cohort_columns)
        if (
            not columns
            or any(not isinstance(item, str) or not item for item in columns)
            or len(set(columns)) != len(columns)
        ):
            raise MaterializedMetadataError("cohort columns must be unique strings")
        object.__setattr__(self, "cohort_columns", columns)
        if _nonempty(self.identity_column, label="identity column") not in columns:
            raise MaterializedMetadataError(
                "identity column is absent from cohort columns"
            )
        if not isinstance(self.column_metadata, SidecarRef) or not isinstance(
            self.source_column_metadata, SidecarRef
        ):
            raise MaterializedMetadataError(
                "column metadata references must be SidecarRef"
            )
        if self.column_metadata_scope != MATERIALIZED_COHORT_SCOPE:
            raise MaterializedMetadataError("materialized metadata scope is invalid")
        for label in (
            "file_metadata_payload_sha256",
            "source_export_authority_sha256",
            "source_column_metadata_sha256",
            "producer_implementation_sha256",
            "producer_parameters_sha256",
            "cohort_schema_sha256",
            "row_identity_sha256",
        ):
            object.__setattr__(self, label, _digest(getattr(self, label), label=label))
        if self.source_column_metadata.sha256 != self.source_column_metadata_sha256:
            raise MaterializedMetadataError(
                "source column metadata reference digest mismatch"
            )
        parameters = _canonical_mapping(
            self.producer_parameters, label="producer parameters"
        )
        if canonical_parameters_sha256(parameters) != self.producer_parameters_sha256:
            raise MaterializedMetadataError("producer parameter digest mismatch")
        object.__setattr__(self, "producer_parameters", parameters)
        object.__setattr__(
            self,
            "semantic_provenance",
            _canonical_mapping(self.semantic_provenance, label="semantic provenance"),
        )
        if self.parent_authority_sha256 is not None:
            object.__setattr__(
                self,
                "parent_authority_sha256",
                _digest(self.parent_authority_sha256, label="parent authority sha256"),
            )
        object.__setattr__(self, "producer", _nonempty(self.producer, label="producer"))
        derivations = tuple(self.output_derivations)
        if any(not isinstance(item, OutputDerivation) for item in derivations):
            raise MaterializedMetadataError(
                "output derivations contain invalid entries"
            )
        derivations = tuple(sorted(derivations, key=lambda item: item.output_column))
        names = [item.output_column for item in derivations]
        if len(names) != len(set(names)):
            raise MaterializedMetadataError("output derivation columns must be unique")
        object.__setattr__(self, "output_derivations", derivations)
        if self.schema_version != MATERIALIZED_COHORT_AUTHORITY_SCHEMA:
            raise MaterializedMetadataError("unsupported materialized authority schema")

    def to_dict(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "cohort_file": self.cohort_file,
            "cohort_sha256": self.cohort_sha256,
            "cohort_size": self.cohort_size,
            "cohort_rows": self.cohort_rows,
            "cohort_columns": list(self.cohort_columns),
            "cohort_schema_sha256": self.cohort_schema_sha256,
            "identity_column": self.identity_column,
            "row_identity_sha256": self.row_identity_sha256,
            "column_metadata": self.column_metadata.to_dict(),
            "column_metadata_scope": self.column_metadata_scope,
            "file_metadata_payload_sha256": self.file_metadata_payload_sha256,
            "source_export_authority_sha256": self.source_export_authority_sha256,
            "source_column_metadata": self.source_column_metadata.to_dict(),
            "source_column_metadata_sha256": self.source_column_metadata_sha256,
            "producer": self.producer,
            "producer_implementation_sha256": self.producer_implementation_sha256,
            "producer_parameters": _thaw_json(self.producer_parameters),
            "producer_parameters_sha256": self.producer_parameters_sha256,
            "semantic_provenance": _thaw_json(self.semantic_provenance),
            "output_derivations": [item.to_dict() for item in self.output_derivations],
            "parent_authority_sha256": self.parent_authority_sha256,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, object]) -> "MaterializedCohortAuthority":
        expected = {
            "schema_version",
            "cohort_file",
            "cohort_sha256",
            "cohort_size",
            "cohort_rows",
            "cohort_columns",
            "cohort_schema_sha256",
            "identity_column",
            "row_identity_sha256",
            "column_metadata",
            "column_metadata_scope",
            "file_metadata_payload_sha256",
            "source_export_authority_sha256",
            "source_column_metadata",
            "source_column_metadata_sha256",
            "producer",
            "producer_implementation_sha256",
            "producer_parameters",
            "producer_parameters_sha256",
            "semantic_provenance",
            "output_derivations",
            "parent_authority_sha256",
        }
        _exact_keys(payload, expected, label="materialized cohort authority")
        raw_columns = payload["cohort_columns"]
        raw_sidecar = payload["column_metadata"]
        raw_source_sidecar = payload["source_column_metadata"]
        raw_derivations = payload["output_derivations"]
        if not isinstance(raw_columns, list) or not all(
            isinstance(item, str) for item in raw_columns
        ):
            raise MaterializedMetadataError("cohort_columns must be a string list")
        if not isinstance(raw_sidecar, Mapping) or not isinstance(
            raw_source_sidecar, Mapping
        ):
            raise MaterializedMetadataError(
                "column metadata references must be objects"
            )
        if not isinstance(raw_derivations, list) or not all(
            isinstance(item, Mapping) for item in raw_derivations
        ):
            raise MaterializedMetadataError("output_derivations must be a list")
        if not isinstance(payload["producer_parameters"], Mapping):
            raise MaterializedMetadataError("producer_parameters must be an object")
        if not isinstance(payload["semantic_provenance"], Mapping):
            raise MaterializedMetadataError("semantic_provenance must be an object")
        parsed = cls(
            cohort_file=payload["cohort_file"],  # type: ignore[arg-type]
            cohort_sha256=payload["cohort_sha256"],  # type: ignore[arg-type]
            cohort_size=payload["cohort_size"],  # type: ignore[arg-type]
            cohort_rows=payload["cohort_rows"],  # type: ignore[arg-type]
            cohort_columns=tuple(raw_columns),
            cohort_schema_sha256=payload["cohort_schema_sha256"],  # type: ignore[arg-type]
            identity_column=payload["identity_column"],  # type: ignore[arg-type]
            row_identity_sha256=payload["row_identity_sha256"],  # type: ignore[arg-type]
            column_metadata=SidecarRef.from_dict(raw_sidecar),
            column_metadata_scope=payload["column_metadata_scope"],  # type: ignore[arg-type]
            file_metadata_payload_sha256=payload["file_metadata_payload_sha256"],  # type: ignore[arg-type]
            source_export_authority_sha256=payload["source_export_authority_sha256"],  # type: ignore[arg-type]
            source_column_metadata=SidecarRef.from_dict(raw_source_sidecar),
            source_column_metadata_sha256=payload["source_column_metadata_sha256"],  # type: ignore[arg-type]
            producer=payload["producer"],  # type: ignore[arg-type]
            producer_implementation_sha256=payload["producer_implementation_sha256"],  # type: ignore[arg-type]
            producer_parameters=payload["producer_parameters"],  # type: ignore[arg-type]
            producer_parameters_sha256=payload["producer_parameters_sha256"],  # type: ignore[arg-type]
            semantic_provenance=payload["semantic_provenance"],  # type: ignore[arg-type]
            output_derivations=tuple(
                OutputDerivation.from_dict(item) for item in raw_derivations
            ),
            parent_authority_sha256=payload["parent_authority_sha256"],  # type: ignore[arg-type]
            schema_version=payload["schema_version"],  # type: ignore[arg-type]
        )
        if parsed.to_dict() != dict(payload):
            raise MaterializedMetadataError("materialized authority is not canonical")
        return parsed


@dataclass(frozen=True, slots=True)
class VerifiedMaterializedCohortAuthority:
    reference: MaterializedCohortAuthorityRef
    authority: MaterializedCohortAuthority
    sidecar: ColumnMetadataSidecar
    provenance: Mapping[str, object]


def _canonical_authority_bytes(authority: MaterializedCohortAuthority) -> bytes:
    return json.dumps(
        authority.to_dict(),
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def implementation_bundle_sha256(paths: Sequence[Path]) -> str:
    digest = hashlib.sha256()
    members = [Path(path).resolve() for path in paths]
    if not members:
        raise MaterializedMetadataError("implementation bundle must not be empty")
    member_digests: list[str] = []
    for path in members:
        if path.is_symlink() or not path.is_file():
            raise MaterializedMetadataError("implementation bundle member is invalid")
        member_digests.append(hashlib.sha256(path.read_bytes()).hexdigest())
    for member_digest in sorted(member_digests):
        digest.update(member_digest.encode("ascii"))
        digest.update(b"\0")
    return digest.hexdigest()


def canonical_parameters_sha256(parameters: Mapping[str, object]) -> str:
    try:
        raw = json.dumps(
            _thaw_json(parameters),
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
    except (TypeError, ValueError) as exc:
        raise MaterializedMetadataError(
            "producer parameters are not canonical JSON"
        ) from exc
    return hashlib.sha256(raw).hexdigest()


def _freeze_json(value: object) -> object:
    if isinstance(value, dict):
        return MappingProxyType(
            {str(key): _freeze_json(item) for key, item in value.items()}
        )
    if isinstance(value, list):
        return tuple(_freeze_json(item) for item in value)
    return value


def _thaw_json(value: object) -> object:
    if isinstance(value, Mapping):
        return {str(key): _thaw_json(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return [_thaw_json(item) for item in value]
    return value


def _canonical_mapping(
    payload: Mapping[str, object], *, label: str
) -> Mapping[str, object]:
    """Round-trip one immutable JSON object into its canonical value form."""

    try:
        raw = json.dumps(
            _thaw_json(payload),
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
    except MaterializedMetadataError:
        raise
    except (TypeError, ValueError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise MaterializedMetadataError(f"{label} is not canonical JSON") from exc
    if not isinstance(parsed, dict):
        raise MaterializedMetadataError(f"{label} must be an object")
    frozen = _freeze_json(parsed)
    if not isinstance(frozen, Mapping):  # pragma: no cover - parsed is a dict
        raise MaterializedMetadataError(f"{label} must be an object")
    return frozen


def _write_authority(
    root: Path, authority: MaterializedCohortAuthority
) -> MaterializedCohortAuthorityRef:
    try:
        with AnchoredDirectory.open(root) as anchored:
            return _write_authority_at(anchored, authority)
    except AuthorityFilesystemError as exc:
        raise MaterializedMetadataError(
            "authority root must be an existing real directory"
        ) from exc


def _write_authority_at(
    root: AnchoredDirectory,
    authority: MaterializedCohortAuthority,
) -> MaterializedCohortAuthorityRef:
    raw = _canonical_authority_bytes(authority)
    digest = hashlib.sha256(raw).hexdigest()
    name = f"cohort_authority.sha256-{digest}.json"
    try:
        root.publish_immutable_bytes(name, raw)
    except AuthorityFilesystemError as exc:
        raise MaterializedMetadataError(
            "cannot publish materialized authority"
        ) from exc
    return MaterializedCohortAuthorityRef(file=name, sha256=digest, size=len(raw))


def _reject_duplicate_pairs(
    pairs: Sequence[tuple[str, object]],
) -> dict[str, object]:
    result: dict[str, object] = {}
    for key, value in pairs:
        if key in result:
            raise MaterializedMetadataError(f"duplicate JSON key: {key!r}")
        result[key] = value
    return result


def _reject_constant(value: str) -> object:
    raise MaterializedMetadataError(f"non-finite JSON constant is forbidden: {value}")


def _read_authority(
    path: Path,
    *,
    reference: MaterializedCohortAuthorityRef,
) -> MaterializedCohortAuthority:
    raw = _read_bounded_regular_file(
        path,
        label="materialized authority",
        max_bytes=_MAX_AUTHORITY_BYTES,
        expected_size=reference.size,
    )
    if hashlib.sha256(raw).hexdigest() != reference.sha256:
        raise MaterializedMetadataError("materialized authority digest mismatch")
    try:
        payload = json.loads(
            raw.decode("utf-8"),
            object_pairs_hook=_reject_duplicate_pairs,
            parse_constant=_reject_constant,
        )
    except MaterializedMetadataError:
        raise
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise MaterializedMetadataError("invalid materialized authority JSON") from exc
    if not isinstance(payload, Mapping):
        raise MaterializedMetadataError("materialized authority must be an object")
    authority = MaterializedCohortAuthority.from_dict(payload)
    if _canonical_authority_bytes(authority) != raw:
        raise MaterializedMetadataError(
            "materialized authority bytes are not canonical"
        )
    return authority


def _parquet_envelope(
    path: Path,
) -> tuple[str, int, int, tuple[str, ...], str]:
    if path.is_symlink():
        raise MaterializedMetadataError("materialized cohort must not be a symlink")
    try:
        info = path.stat()
        if not stat.S_ISREG(info.st_mode):
            raise MaterializedMetadataError(
                "materialized cohort must be a regular file"
            )
        parquet = pq.ParquetFile(path)
        rows = int(parquet.metadata.num_rows)
        columns = tuple(parquet.schema_arrow.names)
        schema_sha256 = hashlib.sha256(
            parquet.schema_arrow.serialize().to_pybytes()
        ).hexdigest()
    except MaterializedMetadataError:
        raise
    except (OSError, ValueError, pa.ArrowException) as exc:
        raise MaterializedMetadataError(
            "cannot inspect materialized cohort parquet"
        ) from exc
    if len(columns) != len(set(columns)):
        raise MaterializedMetadataError("materialized cohort columns must be unique")
    return _sha256_file(path), int(info.st_size), rows, columns, schema_sha256


def _parquet_envelope_at(
    root: AnchoredDirectory,
    name: str,
) -> tuple[str, int, int, tuple[str, ...], str]:
    """Inspect one parquet object through the transaction's directory fd."""

    try:
        with root.open_regular(name) as handle:
            before = os.fstat(handle.fileno())
            digest = hashlib.sha256()
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
            handle.seek(0)
            parquet = pq.ParquetFile(handle)
            rows = int(parquet.metadata.num_rows)
            columns = tuple(parquet.schema_arrow.names)
            schema_sha256 = hashlib.sha256(
                parquet.schema_arrow.serialize().to_pybytes()
            ).hexdigest()
            after = os.fstat(handle.fileno())
    except AuthorityFilesystemError as exc:
        raise MaterializedMetadataError(
            "cannot inspect materialized cohort parquet"
        ) from exc
    except (OSError, ValueError, pa.ArrowException) as exc:
        raise MaterializedMetadataError(
            "cannot inspect materialized cohort parquet"
        ) from exc
    if (
        before.st_dev,
        before.st_ino,
        before.st_size,
        before.st_mtime_ns,
    ) != (
        after.st_dev,
        after.st_ino,
        after.st_size,
        after.st_mtime_ns,
    ):
        raise MaterializedMetadataError(
            "materialized cohort changed while being inspected"
        )
    if len(columns) != len(set(columns)):
        raise MaterializedMetadataError("materialized cohort columns must be unique")
    return (
        digest.hexdigest(),
        int(before.st_size),
        rows,
        columns,
        schema_sha256,
    )


def _canonical_identity_values(values: Sequence[object]) -> tuple[str, ...]:
    if any(value is None for value in values):
        raise MaterializedMetadataError("cohort identity column contains null values")
    try:
        canonical = [
            json.dumps(
                value,
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
                allow_nan=False,
            )
            for value in values
        ]
    except (TypeError, ValueError) as exc:
        raise MaterializedMetadataError(
            "cohort identity values are not canonical"
        ) from exc
    if len(canonical) != len(set(canonical)):
        raise MaterializedMetadataError("cohort identity values must be unique")
    return tuple(canonical)


def _row_identity_canonical_values(
    path: Path, *, identity_column: str
) -> tuple[str, ...]:
    try:
        table = pq.read_table(path, columns=[identity_column])
    except (OSError, ValueError, pa.ArrowException) as exc:
        raise MaterializedMetadataError("cannot read cohort identity column") from exc
    return _canonical_identity_values(
        table.column(identity_column).combine_chunks().to_pylist()
    )


def _row_identity_sha256(path: Path, *, identity_column: str) -> str:
    canonical = _row_identity_canonical_values(path, identity_column=identity_column)
    return hashlib.sha256(
        json.dumps(
            canonical,
            ensure_ascii=False,
            separators=(",", ":"),
        ).encode("utf-8")
    ).hexdigest()


def _row_identity_sha256_at(
    root: AnchoredDirectory,
    name: str,
    *,
    identity_column: str,
) -> str:
    try:
        with root.open_regular(name) as handle:
            table = pq.read_table(handle, columns=[identity_column])
    except AuthorityFilesystemError as exc:
        raise MaterializedMetadataError("cannot read cohort identity column") from exc
    except (OSError, ValueError, pa.ArrowException) as exc:
        raise MaterializedMetadataError("cannot read cohort identity column") from exc
    canonical = _canonical_identity_values(
        table.column(identity_column).combine_chunks().to_pylist()
    )
    return hashlib.sha256(
        json.dumps(
            canonical,
            ensure_ascii=False,
            separators=(",", ":"),
        ).encode("utf-8")
    ).hexdigest()


def _descriptor(
    *,
    authority: MaterializedCohortAuthorityRef,
    sidecar: SidecarRef,
    file_binding: ColumnMetadataFileBinding,
) -> dict[str, object]:
    return {
        "schema_version": MATERIALIZED_COHORT_DESCRIPTOR_SCHEMA,
        "authority": authority.to_dict(),
        "sidecar": sidecar.to_dict(),
        "scope": MATERIALIZED_COHORT_SCOPE,
        "file_relative_path": file_binding.relative_path,
        "file_metadata_payload_sha256": file_binding.metadata_payload_sha256,
    }


def _parse_descriptor(
    payload: Mapping[str, object],
) -> tuple[MaterializedCohortAuthorityRef, SidecarRef, str, str]:
    _exact_keys(
        payload,
        {
            "schema_version",
            "authority",
            "sidecar",
            "scope",
            "file_relative_path",
            "file_metadata_payload_sha256",
        },
        label="materialized cohort descriptor",
    )
    if payload["schema_version"] != MATERIALIZED_COHORT_DESCRIPTOR_SCHEMA:
        raise MaterializedMetadataError("unsupported materialized cohort descriptor")
    if payload["scope"] != MATERIALIZED_COHORT_SCOPE:
        raise MaterializedMetadataError("materialized cohort descriptor scope mismatch")
    raw_authority = payload["authority"]
    raw_sidecar = payload["sidecar"]
    if not isinstance(raw_authority, Mapping) or not isinstance(raw_sidecar, Mapping):
        raise MaterializedMetadataError(
            "materialized descriptor references are invalid"
        )
    authority = MaterializedCohortAuthorityRef.from_dict(raw_authority)
    sidecar = SidecarRef.from_dict(raw_sidecar)
    relative_path = _nonempty(
        payload["file_relative_path"], label="descriptor file_relative_path"
    )
    payload_sha = _digest(
        payload["file_metadata_payload_sha256"],
        label="descriptor metadata payload sha256",
    )
    return authority, sidecar, relative_path, payload_sha


@dataclass(frozen=True, slots=True)
class _SourceOwner:
    binding: ColumnMetadataBinding
    coordinate: SourceColumnRef


class MaterializedColumnMetadataCollector:
    """Collect exact output bindings for one deterministic cohort materialization."""

    def __init__(self, package: Optional[ExportPackage]) -> None:
        self._enabled = bool(package and package.column_metadata_sha256)
        self._concept_index = package.concept_index if self._enabled and package else {}
        self._source_database = package.database if self._enabled and package else None
        self._source_export_authority_sha256 = (
            package.authority_sha256 if self._enabled and package else None
        )
        self._source_column_metadata_sha256 = (
            package.column_metadata_sha256 if self._enabled and package else None
        )
        self._source_column_metadata_path: Optional[Path] = None
        self._source_column_metadata_ref: Optional[SidecarRef] = None
        if self._enabled and package is not None:
            if not package.column_metadata_file or not package.column_metadata_sha256:
                raise MaterializedMetadataError(
                    "typed export package lost its source metadata reference"
                )
            source_sidecar_path = package.root / package.column_metadata_file
            try:
                source_sidecar_size = int(source_sidecar_path.stat().st_size)
                source_sidecar = read_content_addressed_sidecar(
                    source_sidecar_path,
                    expected_sha256=package.column_metadata_sha256,
                    expected_size=source_sidecar_size,
                )
            except (OSError, MetadataSidecarError) as exc:
                raise MaterializedMetadataError(
                    "cannot bind source column metadata snapshot"
                ) from exc
            self._source_column_metadata_path = source_sidecar_path
            self._source_column_metadata_ref = SidecarRef(
                file=package.column_metadata_file,
                sha256=package.column_metadata_sha256,
                size=source_sidecar_size,
                record_count=source_sidecar.record_count,
            )
        self._columns: dict[str, ColumnMetadataBinding] = {}
        self._derivations: dict[str, OutputDerivation] = {}

    @property
    def enabled(self) -> bool:
        return self._enabled

    def _source_owner(self, concept: str) -> Optional[_SourceOwner]:
        if not self._enabled:
            return None
        resolved = resolve_exported_concept(self._concept_index, concept)
        if resolved is None:
            return None
        info = self._concept_index[resolved]
        raw = info.get("column_metadata_binding")
        if not isinstance(raw, ColumnMetadataBinding):
            raise MaterializedMetadataError(
                f"verified concept {concept!r} lacks a typed source binding"
            )
        if (
            raw.metadata.source_concept != concept
            or raw.metadata.role not in _PRIMARY_ROLES
        ):
            raise MaterializedMetadataError(
                f"verified concept {concept!r} does not resolve to one primary owner"
            )
        relative_path = info.get("relative_path")
        if not isinstance(relative_path, str) or not relative_path:
            raise MaterializedMetadataError(
                f"verified concept {concept!r} lacks a source file coordinate"
            )
        return _SourceOwner(
            binding=raw,
            coordinate=SourceColumnRef(
                authority_sha256=str(self._source_export_authority_sha256),
                file=relative_path,
                column=resolved,
                binding_sha256=binding_payload_sha256({raw.metadata.column_name: raw}),
            ),
        )

    def source_role(self, concept: str) -> Optional[ConceptColumnRole]:
        owner = self._source_owner(concept)
        return owner.binding.metadata.role if owner is not None else None

    def source_binding(self, concept: str) -> Optional[ColumnMetadataBinding]:
        """Return the exact verified physical binding consumed for a concept."""

        owner = self._source_owner(concept)
        return owner.binding if owner is not None else None

    def require_source_role(self, concept: str) -> Optional[ConceptColumnRole]:
        role = self.source_role(concept)
        if self._enabled and role is None:
            raise MaterializedMetadataError(
                f"typed materialization concept {concept!r} has no unique source owner"
            )
        return role

    def role_for_output(self, column_name: str) -> Optional[ConceptColumnRole]:
        binding = self._columns.get(column_name)
        return binding.metadata.role if binding is not None else None

    def binding_for_output(self, column_name: str) -> Optional[ColumnMetadataBinding]:
        """Return the exact pending binding for deterministic domain checks."""

        return self._columns.get(column_name)

    @property
    def owned_columns(self) -> frozenset[str]:
        return frozenset(self._columns)

    def _add(
        self,
        source: _SourceOwner,
        *,
        column_name: str,
        role: ConceptColumnRole,
        aggregation: Optional[str] = None,
        derivation_window: Optional[DerivationWindow] = None,
        representation_transform: str,
        time_origin: Optional[str] = None,
        time_unit: Optional[str] = None,
    ) -> None:
        binding = ColumnMetadataBinding(
            metadata=derive_concept_column_metadata(
                source.binding.metadata,
                spec=ColumnProjectionSpec(
                    column_name=column_name,
                    source_concept=source.binding.metadata.source_concept,
                    role=role,
                    aggregation=aggregation,
                    time_origin=time_origin,
                    time_unit=time_unit,
                ),
            ),
            derivation_window=derivation_window,
            representation_transform=representation_transform,
        )
        previous = self._columns.get(column_name)
        if previous is not None and previous != binding:
            raise MaterializedMetadataError(
                f"materialized column {column_name!r} has conflicting derivations"
            )
        self._columns[column_name] = binding
        derivation = OutputDerivation(
            output_column=column_name,
            sources=(source.coordinate,),
            transform_id=representation_transform,
        )
        previous_derivation = self._derivations.get(column_name)
        if previous_derivation is not None and previous_derivation != derivation:
            raise MaterializedMetadataError(
                f"materialized column {column_name!r} has conflicting source receipts"
            )
        self._derivations[column_name] = derivation

    def add_static(self, concept: str, *, output_columns: Sequence[str]) -> None:
        source = self._source_owner(concept)
        if source is None or concept not in set(output_columns):
            return
        self._add(
            source,
            column_name=concept,
            role=source.binding.metadata.role,
            aggregation=source.binding.metadata.aggregation,
            representation_transform="stay_level_unique_value",
        )

    def add_timeseries(
        self,
        concept: str,
        *,
        output_columns: Sequence[str],
        window: tuple[float, float],
    ) -> None:
        source = self._source_owner(concept)
        if source is None:
            return
        names = set(output_columns)
        event_like = source.binding.metadata.role is ConceptColumnRole.EVENT_STATUS
        derivation_window = DerivationWindow(
            origin="icu_admission",
            start_hours=window[0],
            end_hours=window[1],
        )
        for aggregation in ("max", "min", "mean", "first"):
            column = f"{concept}_{aggregation}"
            if column not in names:
                continue
            if event_like and aggregation == "mean":
                role = ConceptColumnRole.EVENT_FRACTION
            elif event_like:
                role = ConceptColumnRole.EVENT_STATUS
            else:
                role = ConceptColumnRole.NUMERIC_AGGREGATE
            self._add(
                source,
                column_name=column,
                role=role,
                aggregation=aggregation,
                derivation_window=derivation_window,
                representation_transform=(
                    f"window_presence_{aggregation}"
                    if event_like
                    else f"window_numeric_{aggregation}"
                ),
            )
        if f"{concept}_n" in names:
            self._add(
                source,
                column_name=f"{concept}_n",
                role=ConceptColumnRole.COUNT,
                derivation_window=derivation_window,
                representation_transform="window_nonnull_count",
            )
        if f"{concept}_measured" in names:
            self._add(
                source,
                column_name=f"{concept}_measured",
                role=ConceptColumnRole.MEASUREMENT_STATUS,
                derivation_window=derivation_window,
                representation_transform="window_measurement_status",
            )
        for suffix, role in (
            ("first_time", ConceptColumnRole.FIRST_OBSERVATION_TIME),
            ("last_time", ConceptColumnRole.LAST_OBSERVATION_TIME),
        ):
            column = f"{concept}_{suffix}"
            if column in names:
                self._add(
                    source,
                    column_name=column,
                    role=role,
                    derivation_window=derivation_window,
                    representation_transform=f"window_{suffix}",
                    time_origin="icu_admission",
                    time_unit="h",
                )

    def add_outcome(self, concept: str, *, output_columns: Sequence[str]) -> None:
        source = self._source_owner(concept)
        if source is None:
            return
        if source.binding.metadata.role is not ConceptColumnRole.EVENT_STATUS:
            raise MaterializedMetadataError(
                f"typed outcome {concept!r} is not authorized as an event status"
            )
        names = set(output_columns)
        if concept in names:
            self._add(
                source,
                column_name=concept,
                role=ConceptColumnRole.EVENT_STATUS,
                aggregation="any",
                representation_transform="whole_stay_any_truthy",
            )
        event_time = f"{concept}_time"
        if event_time in names:
            self._add(
                source,
                column_name=event_time,
                role=ConceptColumnRole.EVENT_TIME,
                representation_transform="first_truthy_event_time",
                time_origin="icu_admission",
                time_unit="h",
            )

    def add_predicate(
        self,
        concept: str,
        *,
        output_columns: Sequence[str],
        source_has_time: bool,
        aggregation: str,
        window: tuple[float, float],
        anchor: str,
    ) -> None:
        source = self._source_owner(concept)
        if source is None or concept not in set(output_columns):
            return
        normalized_anchor = str(anchor or "").strip().lower()
        if normalized_anchor not in {"icu_admit", "icu_admission"}:
            raise MaterializedMetadataError(
                f"typed predicate {concept!r} uses unsupported time anchor "
                f"{anchor!r}; source time is ICU-admission relative"
            )
        if not source_has_time:
            self.add_static(concept, output_columns=output_columns)
            return
        normalized = str(aggregation or "").strip().lower()
        event_source = source.binding.metadata.role is ConceptColumnRole.EVENT_STATUS
        if normalized == "count":
            role = ConceptColumnRole.COUNT
            spec_aggregation = None
        elif normalized in {"any", "all"}:
            if not event_source:
                raise MaterializedMetadataError(
                    f"predicate aggregation {normalized!r} requires an event-status source"
                )
            role = ConceptColumnRole.EVENT_STATUS
            spec_aggregation = normalized
        elif event_source and normalized == "mean":
            role = ConceptColumnRole.EVENT_FRACTION
            spec_aggregation = normalized
        elif event_source and normalized in _EVENT_AGGREGATIONS:
            role = ConceptColumnRole.EVENT_STATUS
            spec_aggregation = normalized
        elif not event_source and normalized in _VALUE_AGGREGATIONS:
            role = ConceptColumnRole.NUMERIC_AGGREGATE
            spec_aggregation = normalized
        else:
            raise MaterializedMetadataError(
                f"predicate aggregation {normalized!r} cannot be represented safely "
                f"for source role {source.binding.metadata.role.value!r}"
            )
        self._add(
            source,
            column_name=concept,
            role=role,
            aggregation=spec_aggregation,
            derivation_window=DerivationWindow(
                origin="icu_admission",
                start_hours=window[0],
                end_hours=window[1],
            ),
            representation_transform=f"cohort_predicate_{normalized}",
        )

    def build_sidecar(
        self,
        *,
        relative_path: str,
        identity_column: str,
        cohort_columns: Sequence[str],
        source_database: str,
    ) -> Optional[tuple[ColumnMetadataSidecar, ColumnMetadataFileBinding]]:
        if not self._enabled:
            return None
        expected = set(cohort_columns) - {identity_column}
        actual = set(self._columns)
        if actual != expected:
            raise MaterializedMetadataError(
                "materialized column metadata coverage mismatch "
                f"(missing={sorted(expected - actual)}, extra={sorted(actual - expected)})"
            )
        if not actual:
            raise MaterializedMetadataError(
                "typed materialized cohort must contain at least one value column"
            )
        if set(self._derivations) != actual:
            raise MaterializedMetadataError(
                "materialized source-receipt coverage does not match metadata coverage"
            )
        chains = {
            binding.metadata.source_resolution_chain
            for binding in self._columns.values()
        }
        if len(chains) != 1:
            raise MaterializedMetadataError(
                "materialized columns do not share one source resolution authority"
            )
        chain = next(iter(chains))
        if not chain or chain[0] != source_database:
            raise MaterializedMetadataError(
                "materialized source database does not match typed column authority"
            )
        file_binding = ColumnMetadataFileBinding(
            relative_path=relative_path,
            module="cohort_materializer",
            identity_column=identity_column,
            time_coordinates=(),
            columns=self._columns,
        )
        sidecar = ColumnMetadataSidecar(
            source_database=source_database,
            source_database_class_prefixes=chain[1:],
            scope=MATERIALIZED_COHORT_SCOPE,
            files=(file_binding,),
        )
        return sidecar, file_binding

    def seal_existing_cohort(
        self,
        *,
        cohort_path: Path,
        identity_column: str,
        source_database: str,
        producer: str,
        producer_implementation_sha256: str,
        producer_parameters: Mapping[str, object],
        semantic_provenance: Mapping[str, object],
    ) -> Optional[dict[str, object]]:
        if not self._enabled:
            return None
        cohort_path = Path(cohort_path)
        try:
            with AnchoredDirectory.open(cohort_path.parent) as root:
                (
                    cohort_sha,
                    cohort_size,
                    cohort_rows,
                    cohort_columns,
                    cohort_schema_sha256,
                ) = _parquet_envelope_at(root, cohort_path.name)
                built = self.build_sidecar(
                    relative_path=cohort_path.name,
                    identity_column=identity_column,
                    cohort_columns=cohort_columns,
                    source_database=source_database,
                )
                if built is None:  # pragma: no cover - guarded above
                    return None
                sidecar, file_binding = built
                sidecar_ref = _write_content_addressed_sidecar_at(
                    root, sidecar, stem="cohort_column_metadata"
                )
                assert self._source_export_authority_sha256 is not None
                assert self._source_column_metadata_sha256 is not None
                assert self._source_column_metadata_path is not None
                assert self._source_column_metadata_ref is not None
                with AnchoredDirectory.open(
                    self._source_column_metadata_path.parent
                ) as source_root:
                    _publish_content_addressed_snapshot_at(
                        source_root,
                        root,
                        source_name=self._source_column_metadata_path.name,
                        reference=self._source_column_metadata_ref,
                        label="source column metadata",
                    )
                authority = MaterializedCohortAuthority(
                    cohort_file=cohort_path.name,
                    cohort_sha256=cohort_sha,
                    cohort_size=cohort_size,
                    cohort_rows=cohort_rows,
                    cohort_columns=cohort_columns,
                    cohort_schema_sha256=cohort_schema_sha256,
                    identity_column=identity_column,
                    row_identity_sha256=_row_identity_sha256_at(
                        root,
                        cohort_path.name,
                        identity_column=identity_column,
                    ),
                    column_metadata=sidecar_ref,
                    column_metadata_scope=MATERIALIZED_COHORT_SCOPE,
                    file_metadata_payload_sha256=file_binding.metadata_payload_sha256,
                    source_export_authority_sha256=(
                        self._source_export_authority_sha256
                    ),
                    source_column_metadata=self._source_column_metadata_ref,
                    source_column_metadata_sha256=(self._source_column_metadata_sha256),
                    producer=producer,
                    producer_implementation_sha256=producer_implementation_sha256,
                    producer_parameters=producer_parameters,
                    producer_parameters_sha256=canonical_parameters_sha256(
                        producer_parameters
                    ),
                    semantic_provenance=semantic_provenance,
                    output_derivations=tuple(self._derivations.values()),
                )
                authority_ref = _write_authority_at(root, authority)
        except AuthorityFilesystemError as exc:
            raise MaterializedMetadataError(
                "cannot seal materialized cohort authority"
            ) from exc
        return _descriptor(
            authority=authority_ref,
            sidecar=sidecar_ref,
            file_binding=file_binding,
        )


def materialized_provenance_path(cohort_path: Path) -> Path:
    cohort_path = Path(cohort_path)
    return cohort_path.with_name(f"{cohort_path.stem}_provenance.json")


def _atomic_write_json(path: Path, payload: Mapping[str, object]) -> None:
    try:
        with AnchoredDirectory.open(path.parent) as root:
            _atomic_write_json_at(root, name=path.name, payload=payload)
    except AuthorityFilesystemError as exc:
        raise MaterializedMetadataError("cannot publish cohort provenance") from exc


def _atomic_write_json_at(
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
    try:
        root.replace_bytes(name, raw, require_absent=require_absent)
    except AuthorityFilesystemError as exc:
        raise MaterializedMetadataError("cannot publish cohort provenance") from exc


def _directory_declares_materialized_authority(cohort_path: Path) -> bool:
    """Detect an orphaned typed authority without selecting a replacement."""

    root = _require_real_directory(
        cohort_path.parent, label="materialized cohort directory"
    )
    for candidate in root.glob("cohort_authority.sha256-*.json"):
        name = candidate.name
        prefix = "cohort_authority.sha256-"
        if not name.startswith(prefix) or not name.endswith(".json"):
            continue
        digest = name[len(prefix) : -len(".json")]
        try:
            info = candidate.stat()
            reference = MaterializedCohortAuthorityRef(
                file=name,
                sha256=digest,
                size=int(info.st_size),
            )
            authority = _read_authority(candidate, reference=reference)
        except (OSError, MaterializedMetadataError) as exc:
            raise MaterializedMetadataError(
                "cannot classify an orphaned materialized authority"
            ) from exc
        if authority.cohort_file == cohort_path.name:
            return True
    return False


def _read_source_column_metadata(
    root: Path,
    *,
    authority: MaterializedCohortAuthority,
) -> ColumnMetadataSidecar:
    """Load the exact export-sidecar snapshot selected by one authority."""

    reference = authority.source_column_metadata
    try:
        sidecar = read_content_addressed_sidecar(
            root / reference.file,
            expected_sha256=reference.sha256,
            expected_size=reference.size,
        )
    except MetadataSidecarError as exc:
        raise MaterializedMetadataError(str(exc)) from exc
    if (
        sidecar.scope != EXPORT_PHYSICAL_SCOPE
        or sidecar.record_count != reference.record_count
        or reference.sha256 != authority.source_column_metadata_sha256
    ):
        raise MaterializedMetadataError(
            "source column metadata snapshot does not match authority"
        )
    return sidecar


def _source_binding_for_receipt(
    source_sidecar: ColumnMetadataSidecar,
    *,
    source: SourceColumnRef,
) -> ColumnMetadataBinding:
    """Resolve one exact export file/column/binding coordinate."""

    matching_files = [
        item for item in source_sidecar.files if item.relative_path == source.file
    ]
    if len(matching_files) != 1:
        raise MaterializedMetadataError(
            f"source metadata file binding mismatch for {source.file!r}"
        )
    binding = matching_files[0].columns.get(source.column)
    if binding is None:
        raise MaterializedMetadataError(
            f"source metadata column binding mismatch for {source.column!r}"
        )
    if binding_payload_sha256({source.column: binding}) != source.binding_sha256:
        raise MaterializedMetadataError(
            f"source metadata digest mismatch for {source.column!r}"
        )
    return binding


def _validate_derivation_contract(
    authority: MaterializedCohortAuthority,
    *,
    file_binding: ColumnMetadataFileBinding,
    source_sidecar: ColumnMetadataSidecar,
) -> None:
    derivations = {item.output_column: item for item in authority.output_derivations}
    expected_columns = set(authority.cohort_columns) - {authority.identity_column}
    if set(derivations) != expected_columns:
        raise MaterializedMetadataError("materialized derivation coverage mismatch")
    if authority.producer == "cohort_materializer":
        provenance = _thaw_json(authority.semantic_provenance)
        export_authority = provenance.get("export_authority")
        if (
            not isinstance(export_authority, Mapping)
            or export_authority.get("authority_sha256")
            != authority.source_export_authority_sha256
            or provenance.get("database") != source_sidecar.source_database
            or provenance.get("n_stays_after_inclusion_exclusion")
            != authority.cohort_rows
            or provenance.get("columns") != list(authority.cohort_columns)
        ):
            raise MaterializedMetadataError(
                "initial materialization provenance does not match authority"
            )
        for column, binding in file_binding.columns.items():
            derivation = derivations[column]
            if derivation.transform_id != binding.representation_transform:
                raise MaterializedMetadataError(
                    f"initial transform receipt mismatch for {column!r}"
                )
            if len(derivation.sources) != 1:
                raise MaterializedMetadataError(
                    f"initial source receipt cardinality mismatch for {column!r}"
                )
            for source in derivation.sources:
                if source.authority_sha256 != authority.source_export_authority_sha256:
                    raise MaterializedMetadataError(
                        f"initial source authority mismatch for {column!r}"
                    )
                source_binding = _source_binding_for_receipt(
                    source_sidecar,
                    source=source,
                )
                if (
                    source_binding.metadata.source_concept
                    != binding.metadata.source_concept
                ):
                    raise MaterializedMetadataError(
                        f"initial source concept mismatch for {column!r}"
                    )
        return
    if authority.producer == "research_agent_run_stage":
        parent = authority.parent_authority_sha256
        if parent is None:
            raise MaterializedMetadataError("staged authority lacks an upstream anchor")
        parameters = authority.producer_parameters
        if (
            parameters.get("source_authority_sha256") != parent
            or parameters.get("source_cohort_sha256") != authority.cohort_sha256
            or parameters.get("source_cohort_rows") != authority.cohort_rows
            or parameters.get("source_cohort_schema_sha256")
            != authority.cohort_schema_sha256
            or parameters.get("source_row_identity_sha256")
            != authority.row_identity_sha256
            or parameters.get("target_file") != authority.cohort_file
            or parameters.get("transform") != "identity_stage_copy"
        ):
            raise MaterializedMetadataError("staged cohort copy receipt mismatch")
        for column, derivation in derivations.items():
            if derivation.transform_id != "identity_stage_copy" or any(
                source.authority_sha256 != parent for source in derivation.sources
            ):
                raise MaterializedMetadataError(
                    f"staged transform receipt mismatch for {column!r}"
                )
        return
    if authority.producer == "analysis_cohort_ordered_subset":
        parent = authority.parent_authority_sha256
        if parent is None:
            raise MaterializedMetadataError("analysis cohort lacks a parent authority")
        for column, derivation in derivations.items():
            if derivation.transform_id != "ordered_row_subset" or any(
                source.authority_sha256 != parent for source in derivation.sources
            ):
                raise MaterializedMetadataError(
                    f"analysis subset receipt mismatch for {column!r}"
                )
        return
    raise MaterializedMetadataError(
        f"unsupported materialized cohort producer {authority.producer!r}"
    )


def _local_authority_reference(
    root: Path,
    *,
    authority_sha256: str,
) -> tuple[MaterializedCohortAuthorityRef, MaterializedCohortAuthority]:
    """Resolve one explicitly named local authority without latest-file scans."""

    name = f"cohort_authority.sha256-{authority_sha256}.json"
    path = root / name
    if not path.exists() or path.is_symlink():
        raise MaterializedMetadataError("analysis cohort parent authority is missing")
    try:
        info = path.stat()
    except OSError as exc:
        raise MaterializedMetadataError(
            "cannot stat analysis cohort parent authority"
        ) from exc
    reference = MaterializedCohortAuthorityRef(
        file=name,
        sha256=authority_sha256,
        size=int(info.st_size),
    )
    return reference, _read_authority(path, reference=reference)


def _validate_stage_parent_receipts(
    authority: MaterializedCohortAuthority,
    *,
    sidecar: ColumnMetadataSidecar,
    parent: VerifiedMaterializedCohortAuthority,
) -> None:
    """Prove that a staged authority is the full deterministic parent projection."""

    expected_sidecar, _ = _rebind_sidecar(
        parent,
        relative_path=authority.cohort_file,
        module="research_agent_run_stage",
    )
    expected_parameters = {
        "source_authority_sha256": parent.reference.sha256,
        "source_cohort_sha256": parent.authority.cohort_sha256,
        "source_cohort_rows": parent.authority.cohort_rows,
        "source_cohort_schema_sha256": parent.authority.cohort_schema_sha256,
        "source_row_identity_sha256": parent.authority.row_identity_sha256,
        "target_file": authority.cohort_file,
        "transform": "identity_stage_copy",
    }
    parent_binding = parent.sidecar.files[0]
    expected_derivations = tuple(
        sorted(
            (
                OutputDerivation(
                    output_column=column,
                    sources=(
                        SourceColumnRef(
                            authority_sha256=parent.reference.sha256,
                            file=parent.authority.cohort_file,
                            column=column,
                            binding_sha256=binding_payload_sha256({column: binding}),
                        ),
                    ),
                    transform_id="identity_stage_copy",
                )
                for column, binding in parent_binding.columns.items()
            ),
            key=lambda item: item.output_column,
        )
    )
    expected_semantic_provenance = {
        **dict(_thaw_json(parent.authority.semantic_provenance)),
        "staged_from_authority_sha256": parent.reference.sha256,
    }
    if (
        authority.parent_authority_sha256 != parent.reference.sha256
        or authority.cohort_sha256 != parent.authority.cohort_sha256
        or authority.cohort_size != parent.authority.cohort_size
        or authority.cohort_rows != parent.authority.cohort_rows
        or authority.cohort_columns != parent.authority.cohort_columns
        or authority.cohort_schema_sha256 != parent.authority.cohort_schema_sha256
        or authority.identity_column != parent.authority.identity_column
        or authority.row_identity_sha256 != parent.authority.row_identity_sha256
        or authority.source_export_authority_sha256
        != parent.authority.source_export_authority_sha256
        or authority.source_column_metadata != parent.authority.source_column_metadata
        or authority.source_column_metadata_sha256
        != parent.authority.source_column_metadata_sha256
        or authority.producer_parameters
        != _canonical_mapping(expected_parameters, label="expected stage parameters")
        or authority.semantic_provenance
        != _canonical_mapping(
            expected_semantic_provenance,
            label="expected stage semantic provenance",
        )
        or authority.output_derivations != expected_derivations
        or sidecar != expected_sidecar
    ):
        raise MaterializedMetadataError(
            "staged cohort is not the exact deterministic parent projection"
        )


def _validate_analysis_parent_receipts(
    authority: MaterializedCohortAuthority,
    *,
    child_reference: MaterializedCohortAuthorityRef,
    sidecar: ColumnMetadataSidecar,
    parent: VerifiedMaterializedCohortAuthority,
    parent_path: Path,
    child_path: Path,
) -> None:
    """Bind every analysis-child column to the verified local parent binding."""

    parent_binding = parent.sidecar.files[0]
    expected_sidecar, _ = _rebind_sidecar(
        parent,
        relative_path=authority.cohort_file,
        module="analysis_cohort_ordered_subset",
    )
    if (
        authority.parent_authority_sha256 != parent.reference.sha256
        or authority.cohort_columns != parent.authority.cohort_columns
        or authority.identity_column != parent.authority.identity_column
        or authority.source_export_authority_sha256
        != parent.authority.source_export_authority_sha256
        or authority.source_column_metadata != parent.authority.source_column_metadata
        or authority.source_column_metadata_sha256
        != parent.authority.source_column_metadata_sha256
        or sidecar != expected_sidecar
    ):
        raise MaterializedMetadataError("analysis cohort parent binding mismatch")
    derivations = {item.output_column: item for item in authority.output_derivations}
    for column, binding in parent_binding.columns.items():
        expected_source = SourceColumnRef(
            authority_sha256=parent.reference.sha256,
            file=parent.authority.cohort_file,
            column=column,
            binding_sha256=binding_payload_sha256({column: binding}),
        )
        if derivations[column].sources != (expected_source,):
            raise MaterializedMetadataError(
                f"analysis subset source binding mismatch for {column!r}"
            )
    parent_table = _read_verified_parent_table(parent_path, verified=parent)
    child_table = _read_verified_parent_table(
        child_path,
        verified=VerifiedMaterializedCohortAuthority(
            reference=child_reference,
            authority=authority,
            sidecar=sidecar,
            provenance=authority.semantic_provenance,
        ),
    )
    parent_identities = _canonical_identity_values(
        parent_table.column(parent.authority.identity_column)
        .combine_chunks()
        .to_pylist()
    )
    child_identities = _canonical_identity_values(
        child_table.column(authority.identity_column).combine_chunks().to_pylist()
    )
    parent_positions = {
        identity: position for position, identity in enumerate(parent_identities)
    }
    try:
        selected_positions = tuple(
            parent_positions[identity] for identity in child_identities
        )
    except KeyError as exc:
        raise MaterializedMetadataError(
            "analysis cohort contains an identity outside its parent"
        ) from exc
    if any(
        left >= right for left, right in zip(selected_positions, selected_positions[1:])
    ):
        raise MaterializedMetadataError(
            "analysis cohort is not an ordered parent-row subset"
        )
    parameters = _thaw_json(authority.producer_parameters)
    provenance = _thaw_json(authority.semantic_provenance)
    if (
        parameters.get("selected_row_count") != len(selected_positions)
        or parameters.get("selected_row_positions_sha256")
        != _ordered_positions_sha256(selected_positions)
        or parameters.get("parent_authority_sha256") != parent.reference.sha256
        or parameters.get("transform") != "ordered_row_subset"
        or provenance.get("n_analysis_cohort") != authority.cohort_rows
        or provenance.get("n_universe") != parent.authority.cohort_rows
        or provenance.get("cohort_sha256") != parameters.get("cohort_definition_sha256")
        or provenance.get("cohort_definition") != parameters.get("cohort_definition")
        or provenance.get("predicate_column_bindings")
        != parameters.get("predicate_column_bindings")
    ):
        raise MaterializedMetadataError("analysis subset position receipt mismatch")
    try:
        expected_child = parent_table.take(
            pa.array(selected_positions, type=pa.int64())
        )
    except (ValueError, pa.ArrowException) as exc:
        raise MaterializedMetadataError(
            "cannot verify analysis cohort parent-row subset"
        ) from exc
    if not child_table.equals(expected_child):
        raise MaterializedMetadataError(
            "analysis cohort values are not an exact parent-row subset"
        )


def load_verified_materialized_cohort_authority(
    cohort_path: Path,
    *,
    provenance_path: Optional[Path] = None,
    expected_authority: Optional[MaterializedCohortAuthorityRef] = None,
    _ancestor_chain: frozenset[str] = frozenset(),
) -> Optional[VerifiedMaterializedCohortAuthority]:
    """Load only the provenance-selected authority; never scan for a latest blob."""

    cohort_path = Path(cohort_path).expanduser()
    if cohort_path.name in {"", ".", ".."} or ".." in cohort_path.parts:
        raise MaterializedMetadataError("materialized cohort path is not canonical")
    cohort_root = _require_real_directory(
        cohort_path.parent, label="materialized cohort directory"
    )
    cohort_path = cohort_root / cohort_path.name
    provenance_path = Path(provenance_path or materialized_provenance_path(cohort_path))
    if (
        provenance_path.parent != cohort_path.parent
        or provenance_path.name != materialized_provenance_path(cohort_path).name
    ):
        raise MaterializedMetadataError(
            "cohort provenance must be the canonical sibling selector"
        )
    if not provenance_path.exists():
        if (
            expected_authority is not None
            or _directory_declares_materialized_authority(cohort_path)
        ):
            raise MaterializedMetadataError(
                "materialized cohort authority selector is missing"
            )
        return None
    try:
        raw_provenance = _read_bounded_regular_file(
            provenance_path,
            label="cohort provenance",
            max_bytes=_MAX_SELECTOR_BYTES,
        )
        provenance = json.loads(
            raw_provenance.decode("utf-8"),
            object_pairs_hook=_reject_duplicate_pairs,
            parse_constant=_reject_constant,
        )
    except MaterializedMetadataError:
        raise
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise MaterializedMetadataError("cannot parse cohort provenance") from exc
    if not isinstance(provenance, Mapping):
        raise MaterializedMetadataError("cohort provenance must be an object")
    transaction_state = provenance.get("authority_transaction_state")
    if transaction_state is not None:
        if not isinstance(transaction_state, str):
            raise MaterializedMetadataError(
                "materialized authority transaction state must be a string"
            )
        if transaction_state == "prepared":
            if provenance.get("column_metadata") is not None:
                raise MaterializedMetadataError(
                    "prepared materialized authority cannot expose a descriptor"
                )
            raise MaterializedMetadataError(
                "materialized authority publication is incomplete"
            )
        if transaction_state != "committed":
            raise MaterializedMetadataError(
                "unknown materialized authority transaction state"
            )
    authority_required = provenance.get("materialized_authority_required")
    if authority_required is not None and not isinstance(authority_required, bool):
        raise MaterializedMetadataError(
            "materialized_authority_required must be boolean when present"
        )
    raw_descriptor = provenance.get("column_metadata")
    if raw_descriptor is None:
        if (
            transaction_state == "committed"
            or authority_required is True
            or expected_authority is not None
            or _directory_declares_materialized_authority(cohort_path)
        ):
            raise MaterializedMetadataError(
                "required materialized cohort authority descriptor is missing"
            )
        return None
    if authority_required is not True:
        raise MaterializedMetadataError(
            "materialized authority descriptor lacks a required marker"
        )
    if not isinstance(raw_descriptor, Mapping):
        raise MaterializedMetadataError("column_metadata descriptor must be an object")
    authority_ref, sidecar_ref, relative_path, payload_sha = _parse_descriptor(
        raw_descriptor
    )
    if authority_ref.sha256 in _ancestor_chain:
        raise MaterializedMetadataError(
            "materialized authority ancestry contains a cycle"
        )
    if expected_authority is not None and authority_ref != expected_authority:
        raise MaterializedMetadataError(
            "materialized authority does not match the caller-selected reference"
        )
    if relative_path != cohort_path.name:
        raise MaterializedMetadataError(
            "cohort descriptor selects a different artifact"
        )
    expected_authority_name = f"cohort_authority.sha256-{authority_ref.sha256}.json"
    if authority_ref.file != expected_authority_name:
        raise MaterializedMetadataError("authority reference is not content addressed")
    expected_sidecar_name = f"cohort_column_metadata.sha256-{sidecar_ref.sha256}.json"
    if sidecar_ref.file != expected_sidecar_name:
        raise MaterializedMetadataError("sidecar reference is not content addressed")
    authority = _read_authority(
        cohort_path.parent / authority_ref.file,
        reference=authority_ref,
    )
    if authority.cohort_file != cohort_path.name:
        raise MaterializedMetadataError("authority selects a different cohort file")
    for key, expected_value in authority.semantic_provenance.items():
        if key not in provenance or provenance[key] != _thaw_json(expected_value):
            raise MaterializedMetadataError(
                f"cohort semantic provenance mismatch for {key!r}"
            )
    stability_before = _file_stability_coordinate(cohort_path)
    observed = _parquet_envelope(cohort_path)
    expected = (
        authority.cohort_sha256,
        authority.cohort_size,
        authority.cohort_rows,
        authority.cohort_columns,
        authority.cohort_schema_sha256,
    )
    if observed != expected:
        raise MaterializedMetadataError("cohort artifact no longer matches authority")
    if (
        _row_identity_sha256(cohort_path, identity_column=authority.identity_column)
        != authority.row_identity_sha256
    ):
        raise MaterializedMetadataError("cohort row identity digest mismatch")
    if authority.column_metadata != sidecar_ref:
        raise MaterializedMetadataError(
            "authority and provenance select different sidecars"
        )
    try:
        sidecar = read_content_addressed_sidecar(
            cohort_path.parent / sidecar_ref.file,
            expected_sha256=sidecar_ref.sha256,
            expected_size=sidecar_ref.size,
        )
    except MaterializedMetadataError:
        raise
    except MetadataSidecarError as exc:
        raise MaterializedMetadataError(str(exc)) from exc
    if sidecar.scope != MATERIALIZED_COHORT_SCOPE or len(sidecar.files) != 1:
        raise MaterializedMetadataError("materialized sidecar scope/files are invalid")
    if sidecar.record_count != sidecar_ref.record_count:
        raise MaterializedMetadataError("materialized sidecar record count mismatch")
    file_binding = sidecar.files[0]
    if (
        file_binding.relative_path != cohort_path.name
        or file_binding.identity_column != authority.identity_column
        or file_binding.metadata_payload_sha256 != payload_sha
        or payload_sha != authority.file_metadata_payload_sha256
    ):
        raise MaterializedMetadataError("materialized file metadata binding mismatch")
    expected_columns = set(authority.cohort_columns) - {authority.identity_column}
    if set(file_binding.columns) != expected_columns:
        raise MaterializedMetadataError("materialized metadata coverage mismatch")
    source_sidecar = _read_source_column_metadata(
        cohort_path.parent,
        authority=authority,
    )
    _validate_derivation_contract(
        authority,
        file_binding=file_binding,
        source_sidecar=source_sidecar,
    )
    if authority.producer == "research_agent_run_stage":
        parent_sha = authority.parent_authority_sha256
        if parent_sha is None:  # pragma: no cover - checked by the contract above
            raise MaterializedMetadataError("staged cohort lacks a parent authority")
        parent_ref, parent_authority = _local_authority_reference(
            cohort_path.parent,
            authority_sha256=parent_sha,
        )
        try:
            parent_sidecar = read_content_addressed_sidecar(
                cohort_path.parent / parent_authority.column_metadata.file,
                expected_sha256=parent_authority.column_metadata.sha256,
                expected_size=parent_authority.column_metadata.size,
            )
        except MetadataSidecarError as exc:
            raise MaterializedMetadataError(str(exc)) from exc
        if (
            parent_sidecar.scope != MATERIALIZED_COHORT_SCOPE
            or parent_sidecar.record_count
            != parent_authority.column_metadata.record_count
            or len(parent_sidecar.files) != 1
        ):
            raise MaterializedMetadataError(
                "staged parent metadata snapshot is invalid"
            )
        parent_binding = parent_sidecar.files[0]
        if (
            parent_authority.producer != "cohort_materializer"
            or parent_binding.relative_path != parent_authority.cohort_file
            or parent_binding.identity_column != parent_authority.identity_column
            or parent_binding.metadata_payload_sha256
            != parent_authority.file_metadata_payload_sha256
            or set(parent_binding.columns)
            != set(parent_authority.cohort_columns) - {parent_authority.identity_column}
        ):
            raise MaterializedMetadataError(
                "staged parent authority snapshot is not an initial materialization"
            )
        parent_source_sidecar = _read_source_column_metadata(
            cohort_path.parent,
            authority=parent_authority,
        )
        _validate_derivation_contract(
            parent_authority,
            file_binding=parent_binding,
            source_sidecar=parent_source_sidecar,
        )
        _validate_stage_parent_receipts(
            authority,
            sidecar=sidecar,
            parent=VerifiedMaterializedCohortAuthority(
                reference=parent_ref,
                authority=parent_authority,
                sidecar=parent_sidecar,
                provenance=parent_authority.semantic_provenance,
            ),
        )
    elif authority.producer == "analysis_cohort_ordered_subset":
        if len(_ancestor_chain) >= _MAX_AUTHORITY_ANCESTRY:
            raise MaterializedMetadataError(
                "materialized authority ancestry exceeds the verification limit"
            )
        parent_sha = authority.parent_authority_sha256
        if parent_sha is None:  # pragma: no cover - checked by the contract above
            raise MaterializedMetadataError(
                "materialized cohort lacks a parent authority"
            )
        parent_ref, parent_authority = _local_authority_reference(
            cohort_path.parent,
            authority_sha256=parent_sha,
        )
        parent_verified = load_verified_materialized_cohort_authority(
            cohort_path.parent / parent_authority.cohort_file,
            expected_authority=parent_ref,
            _ancestor_chain=_ancestor_chain | {authority_ref.sha256},
        )
        if parent_verified is None:  # pragma: no cover - expected ref forbids legacy
            raise MaterializedMetadataError("materialized cohort parent lost authority")
        _validate_analysis_parent_receipts(
            authority,
            child_reference=authority_ref,
            sidecar=sidecar,
            parent=parent_verified,
            parent_path=cohort_path.parent / parent_authority.cohort_file,
            child_path=cohort_path,
        )
    if (
        _file_stability_coordinate(cohort_path) != stability_before
        or _parquet_envelope(cohort_path) != expected
    ):
        raise MaterializedMetadataError(
            "materialized cohort changed during authority verification"
        )
    return VerifiedMaterializedCohortAuthority(
        reference=authority_ref,
        authority=authority,
        sidecar=sidecar,
        provenance=authority.semantic_provenance,
    )


def _rebind_sidecar(
    verified: VerifiedMaterializedCohortAuthority,
    *,
    relative_path: str,
    module: str,
) -> tuple[ColumnMetadataSidecar, ColumnMetadataFileBinding]:
    parent_binding = verified.sidecar.files[0]
    columns = {
        column: ColumnMetadataBinding(
            metadata=binding.metadata,
            derivation_window=binding.derivation_window,
            # An exact byte copy does not change the physical representation.
            # The stage operation is recorded separately in OutputDerivation.
            representation_transform=binding.representation_transform,
        )
        for column, binding in parent_binding.columns.items()
    }
    file_binding = ColumnMetadataFileBinding(
        relative_path=relative_path,
        module=module,
        identity_column=parent_binding.identity_column,
        time_coordinates=parent_binding.time_coordinates,
        columns=columns,
    )
    sidecar = ColumnMetadataSidecar(
        source_database=verified.sidecar.source_database,
        source_database_class_prefixes=(
            verified.sidecar.source_database_class_prefixes
        ),
        scope=MATERIALIZED_COHORT_SCOPE,
        files=(file_binding,),
    )
    return sidecar, file_binding


def stage_materialized_cohort_authority(
    source_path: Path,
    target_path: Path,
    *,
    producer_implementation_sha256: str,
    expected_source_authority: Optional[MaterializedCohortAuthorityRef] = None,
) -> Optional[VerifiedMaterializedCohortAuthority]:
    """Exact-copy one selected typed cohort and seal a parent-bound run copy."""

    source_path = Path(source_path)
    target_path = Path(target_path)
    if target_path.name in {"", ".", ".."} or ".." in target_path.parts:
        raise MaterializedMetadataError("staged cohort target path is not canonical")
    verified = load_verified_materialized_cohort_authority(
        source_path,
        expected_authority=expected_source_authority,
    )
    if verified is None:
        return None
    if verified.authority.producer != "cohort_materializer":
        raise MaterializedMetadataError(
            "run staging currently requires an initial typed materialization"
        )
    target_root = prepare_real_directory(
        target_path.parent, label="staged materialized cohort directory"
    )
    target_path = target_root / target_path.name
    try:
        with AnchoredDirectory.open(source_path.parent) as source_root:
            with AnchoredDirectory.open(target_root) as target_directory:
                authority_ref = _stage_materialized_cohort_authority_at(
                    source_root,
                    target_directory,
                    source_name=source_path.name,
                    target_name=target_path.name,
                    verified=verified,
                    producer_implementation_sha256=(producer_implementation_sha256),
                )
                source_root.assert_still_selected()
                target_directory.assert_still_selected()
    except AuthorityFilesystemError as exc:
        raise MaterializedMetadataError("cannot stage materialized cohort") from exc
    result = load_verified_materialized_cohort_authority(
        target_path,
        expected_authority=authority_ref,
    )
    if result is None:  # pragma: no cover - selector was just written
        raise MaterializedMetadataError("staged materialized cohort lost authority")
    return result


def _stage_materialized_cohort_authority_at(
    source_root: AnchoredDirectory,
    target_root: AnchoredDirectory,
    *,
    source_name: str,
    target_name: str,
    verified: VerifiedMaterializedCohortAuthority,
    producer_implementation_sha256: str,
) -> MaterializedCohortAuthorityRef:
    selector_name = materialized_provenance_path(Path(target_name)).name
    if source_root.identity == target_root.identity and source_name == target_name:
        raise MaterializedMetadataError(
            "staged cohort source and target must be different artifacts"
        )
    try:
        target_root.require_absent(target_name, selector_name)
    except AuthorityFilesystemError as exc:
        raise MaterializedMetadataError(
            "staged cohort target already exists and cannot be overwritten"
        ) from exc
    _atomic_write_json_at(
        target_root,
        name=selector_name,
        payload={
            "schema_version": "easyicu.materialized_cohort_transaction/1",
            "materialized_authority_required": True,
            "column_metadata": None,
            "authority_transaction_state": "prepared",
        },
        require_absent=True,
    )
    temporary_name = _copy_verified_regular_to_temporary(
        source_root,
        target_root,
        source_name=source_name,
        target_stem=target_name,
        expected_size=verified.authority.cohort_size,
        expected_sha256=verified.authority.cohort_sha256,
    )
    try:
        sidecar, file_binding = _rebind_sidecar(
            verified,
            relative_path=target_name,
            module="research_agent_run_stage",
        )
        sidecar_ref = _write_content_addressed_sidecar_at(
            target_root, sidecar, stem="cohort_column_metadata"
        )
        for reference, label in (
            (verified.authority.source_column_metadata, "source column metadata"),
            (verified.reference, "parent materialized authority"),
            (
                verified.authority.column_metadata,
                "parent materialized column metadata",
            ),
        ):
            _publish_content_addressed_snapshot_at(
                source_root,
                target_root,
                source_name=reference.file,
                reference=reference,
                label=label,
            )
        (
            cohort_sha,
            cohort_size,
            cohort_rows,
            cohort_columns,
            cohort_schema_sha256,
        ) = _parquet_envelope_at(target_root, temporary_name)
        parent_binding = verified.sidecar.files[0]
        derivations = tuple(
            OutputDerivation(
                output_column=column,
                sources=(
                    SourceColumnRef(
                        authority_sha256=verified.reference.sha256,
                        file=verified.authority.cohort_file,
                        column=column,
                        binding_sha256=binding_payload_sha256(
                            {column: parent_binding.columns[column]}
                        ),
                    ),
                ),
                transform_id="identity_stage_copy",
            )
            for column in parent_binding.columns
        )
        stage_parameters = {
            "source_authority_sha256": verified.reference.sha256,
            "source_cohort_sha256": verified.authority.cohort_sha256,
            "source_cohort_rows": verified.authority.cohort_rows,
            "source_cohort_schema_sha256": (verified.authority.cohort_schema_sha256),
            "source_row_identity_sha256": verified.authority.row_identity_sha256,
            "target_file": target_name,
            "transform": "identity_stage_copy",
        }
        authority = MaterializedCohortAuthority(
            cohort_file=target_name,
            cohort_sha256=cohort_sha,
            cohort_size=cohort_size,
            cohort_rows=cohort_rows,
            cohort_columns=cohort_columns,
            cohort_schema_sha256=cohort_schema_sha256,
            identity_column=verified.authority.identity_column,
            row_identity_sha256=_row_identity_sha256_at(
                target_root,
                temporary_name,
                identity_column=verified.authority.identity_column,
            ),
            column_metadata=sidecar_ref,
            column_metadata_scope=MATERIALIZED_COHORT_SCOPE,
            file_metadata_payload_sha256=file_binding.metadata_payload_sha256,
            source_export_authority_sha256=(
                verified.authority.source_export_authority_sha256
            ),
            source_column_metadata=verified.authority.source_column_metadata,
            source_column_metadata_sha256=(
                verified.authority.source_column_metadata_sha256
            ),
            producer="research_agent_run_stage",
            producer_implementation_sha256=producer_implementation_sha256,
            producer_parameters=stage_parameters,
            producer_parameters_sha256=canonical_parameters_sha256(stage_parameters),
            semantic_provenance={
                **dict(verified.provenance),
                "staged_from_authority_sha256": verified.reference.sha256,
            },
            output_derivations=derivations,
            parent_authority_sha256=verified.reference.sha256,
        )
        authority_ref = _write_authority_at(target_root, authority)
        source_root.assert_still_selected()
        target_root.replace_temporary(temporary_name, target_name, require_absent=True)
        temporary_name = ""
        provenance = dict(verified.provenance)
        provenance["column_metadata"] = _descriptor(
            authority=authority_ref,
            sidecar=sidecar_ref,
            file_binding=file_binding,
        )
        provenance["materialized_authority_required"] = True
        provenance["staged_from_authority_sha256"] = verified.reference.sha256
        _atomic_write_json_at(
            target_root,
            name=selector_name,
            payload=provenance,
        )
        return authority_ref
    finally:
        if temporary_name:
            target_root.unlink(temporary_name, missing_ok=True)


def _ordered_positions_sha256(positions: Sequence[int]) -> str:
    return hashlib.sha256(
        json.dumps(list(positions), separators=(",", ":")).encode("ascii")
    ).hexdigest()


def _read_verified_parent_table(
    parent_path: Path,
    *,
    verified: VerifiedMaterializedCohortAuthority,
) -> pa.Table:
    """Read one verified parent through a single descriptor-anchored fd."""

    try:
        with parent_path.open("rb") as handle:
            before = os.fstat(handle.fileno())
            digest = hashlib.sha256()
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
            if digest.hexdigest() != verified.authority.cohort_sha256:
                raise MaterializedMetadataError(
                    "parent cohort changed before ordered-subset materialization"
                )
            handle.seek(0)
            table = pq.read_table(handle)
            after = os.fstat(handle.fileno())
    except MaterializedMetadataError:
        raise
    except (OSError, ValueError, pa.ArrowException) as exc:
        raise MaterializedMetadataError(
            "cannot read verified parent cohort snapshot"
        ) from exc
    if (
        before.st_dev,
        before.st_ino,
        before.st_size,
        before.st_mtime_ns,
    ) != (
        after.st_dev,
        after.st_ino,
        after.st_size,
        after.st_mtime_ns,
    ):
        raise MaterializedMetadataError(
            "parent cohort changed during ordered-subset materialization"
        )
    if (
        tuple(table.schema.names) != verified.authority.cohort_columns
        or table.num_rows != verified.authority.cohort_rows
        or hashlib.sha256(table.schema.serialize().to_pybytes()).hexdigest()
        != verified.authority.cohort_schema_sha256
    ):
        raise MaterializedMetadataError("parent cohort snapshot schema/rows mismatch")
    return table


def _read_verified_parent_table_at(
    root: AnchoredDirectory,
    *,
    name: str,
    verified: VerifiedMaterializedCohortAuthority,
) -> pa.Table:
    try:
        with root.open_regular(name) as handle:
            before = os.fstat(handle.fileno())
            digest = hashlib.sha256()
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
            if digest.hexdigest() != verified.authority.cohort_sha256:
                raise MaterializedMetadataError(
                    "parent cohort changed before ordered-subset materialization"
                )
            handle.seek(0)
            table = pq.read_table(handle)
            after = os.fstat(handle.fileno())
    except MaterializedMetadataError:
        raise
    except AuthorityFilesystemError as exc:
        raise MaterializedMetadataError(
            "cannot read verified parent cohort snapshot"
        ) from exc
    except (OSError, ValueError, pa.ArrowException) as exc:
        raise MaterializedMetadataError(
            "cannot read verified parent cohort snapshot"
        ) from exc
    if (
        before.st_dev,
        before.st_ino,
        before.st_size,
        before.st_mtime_ns,
    ) != (
        after.st_dev,
        after.st_ino,
        after.st_size,
        after.st_mtime_ns,
    ):
        raise MaterializedMetadataError(
            "parent cohort changed during ordered-subset materialization"
        )
    if (
        tuple(table.schema.names) != verified.authority.cohort_columns
        or table.num_rows != verified.authority.cohort_rows
        or hashlib.sha256(table.schema.serialize().to_pybytes()).hexdigest()
        != verified.authority.cohort_schema_sha256
    ):
        raise MaterializedMetadataError("parent cohort snapshot schema/rows mismatch")
    return table


def read_verified_materialized_cohort_table(
    cohort_path: Path,
    *,
    verified: VerifiedMaterializedCohortAuthority,
) -> pa.Table:
    """Read exactly the artifact bytes selected by a verified authority."""

    return _read_verified_parent_table(Path(cohort_path), verified=verified)


def publish_ordered_subset_materialized_cohort(
    parent_path: Path,
    target_path: Path,
    *,
    selected_row_positions: Sequence[int],
    semantic_provenance: Mapping[str, object],
    producer_implementation_sha256: str,
    producer_parameters: Mapping[str, object],
    expected_parent_authority: Optional[MaterializedCohortAuthorityRef] = None,
) -> Optional[VerifiedMaterializedCohortAuthority]:
    """Publish an exact ordered-row child of one typed materialized cohort.

    This leaf commits caller-selected row positions only.  It never selects a
    cohort, exposure, outcome, method, or estimand.
    """

    parent_path = Path(parent_path)
    verified = load_verified_materialized_cohort_authority(
        parent_path,
        expected_authority=expected_parent_authority,
    )
    if verified is None:
        return None
    positions = tuple(selected_row_positions)
    if any(not isinstance(item, int) or isinstance(item, bool) for item in positions):
        raise MaterializedMetadataError("ordered subset positions must be integers")
    if any(item < 0 or item >= verified.authority.cohort_rows for item in positions):
        raise MaterializedMetadataError("ordered subset position is out of range")
    if any(left >= right for left, right in zip(positions, positions[1:])):
        raise MaterializedMetadataError(
            "ordered subset positions must be unique and strictly increasing"
        )
    target_path = Path(target_path)
    if target_path.name in {"", ".", ".."} or ".." in target_path.parts:
        raise MaterializedMetadataError("analysis cohort target path is not canonical")
    target_root = prepare_real_directory(
        target_path.parent, label="analysis cohort directory"
    )
    target_path = target_root / target_path.name
    try:
        with AnchoredDirectory.open(parent_path.parent) as parent_root:
            with AnchoredDirectory.open(target_root) as target_directory:
                if parent_root.identity != target_directory.identity:
                    raise MaterializedMetadataError(
                        "analysis cohort parent authority must be staged in "
                        "the target directory"
                    )
                authority_ref = _publish_ordered_subset_at(
                    target_directory,
                    parent_name=parent_path.name,
                    target_name=target_path.name,
                    positions=positions,
                    verified=verified,
                    semantic_provenance=semantic_provenance,
                    producer_implementation_sha256=(producer_implementation_sha256),
                    producer_parameters=producer_parameters,
                )
                target_directory.assert_still_selected()
    except AuthorityFilesystemError as exc:
        raise MaterializedMetadataError(
            "cannot publish ordered analysis cohort"
        ) from exc
    result = load_verified_materialized_cohort_authority(
        target_path, expected_authority=authority_ref
    )
    if result is None:  # pragma: no cover - required selector was just committed
        raise MaterializedMetadataError("analysis cohort lost its typed authority")
    return result


def _publish_ordered_subset_at(
    root: AnchoredDirectory,
    *,
    parent_name: str,
    target_name: str,
    positions: tuple[int, ...],
    verified: VerifiedMaterializedCohortAuthority,
    semantic_provenance: Mapping[str, object],
    producer_implementation_sha256: str,
    producer_parameters: Mapping[str, object],
) -> MaterializedCohortAuthorityRef:
    if parent_name == target_name:
        raise MaterializedMetadataError(
            "analysis cohort parent and target must be different artifacts"
        )
    selector_name = materialized_provenance_path(Path(target_name)).name
    try:
        root.require_absent(target_name, selector_name)
    except AuthorityFilesystemError as exc:
        raise MaterializedMetadataError(
            "analysis cohort target already exists and cannot be overwritten"
        ) from exc
    table = _read_verified_parent_table_at(root, name=parent_name, verified=verified)
    child = table.take(pa.array(positions, type=pa.int64()))
    parent_binding = verified.sidecar.files[0]
    child_binding = ColumnMetadataFileBinding(
        relative_path=target_name,
        module="analysis_cohort_ordered_subset",
        identity_column=parent_binding.identity_column,
        time_coordinates=parent_binding.time_coordinates,
        columns=parent_binding.columns,
    )
    child_sidecar = ColumnMetadataSidecar(
        source_database=verified.sidecar.source_database,
        source_database_class_prefixes=(
            verified.sidecar.source_database_class_prefixes
        ),
        scope=MATERIALIZED_COHORT_SCOPE,
        files=(child_binding,),
    )
    _atomic_write_json_at(
        root,
        name=selector_name,
        payload={
            "schema_version": "easyicu.materialized_cohort_transaction/1",
            "materialized_authority_required": True,
            "column_metadata": None,
            "authority_transaction_state": "prepared",
        },
        require_absent=True,
    )
    sidecar_ref = _write_content_addressed_sidecar_at(
        root, child_sidecar, stem="cohort_column_metadata"
    )
    _publish_content_addressed_snapshot_at(
        root,
        root,
        source_name=verified.authority.source_column_metadata.file,
        reference=verified.authority.source_column_metadata,
        label="source column metadata",
    )
    temporary_name, descriptor = root.create_temporary(stem=target_name)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            descriptor = -1
            pq.write_table(child, handle)
            handle.flush()
            os.fsync(handle.fileno())
        (
            cohort_sha,
            cohort_size,
            cohort_rows,
            cohort_columns,
            cohort_schema_sha256,
        ) = _parquet_envelope_at(root, temporary_name)
        derivations = tuple(
            OutputDerivation(
                output_column=column,
                sources=(
                    SourceColumnRef(
                        authority_sha256=verified.reference.sha256,
                        file=verified.authority.cohort_file,
                        column=column,
                        binding_sha256=binding_payload_sha256(
                            {column: parent_binding.columns[column]}
                        ),
                    ),
                ),
                transform_id="ordered_row_subset",
            )
            for column in parent_binding.columns
        )
        bound_parameters = {
            **dict(_thaw_json(producer_parameters)),
            "parent_authority_sha256": verified.reference.sha256,
            "selected_row_positions_sha256": _ordered_positions_sha256(positions),
            "selected_row_count": len(positions),
            "transform": "ordered_row_subset",
        }
        authority = MaterializedCohortAuthority(
            cohort_file=target_name,
            cohort_sha256=cohort_sha,
            cohort_size=cohort_size,
            cohort_rows=cohort_rows,
            cohort_columns=cohort_columns,
            cohort_schema_sha256=cohort_schema_sha256,
            identity_column=verified.authority.identity_column,
            row_identity_sha256=_row_identity_sha256_at(
                root,
                temporary_name,
                identity_column=verified.authority.identity_column,
            ),
            column_metadata=sidecar_ref,
            column_metadata_scope=MATERIALIZED_COHORT_SCOPE,
            file_metadata_payload_sha256=child_binding.metadata_payload_sha256,
            source_export_authority_sha256=(
                verified.authority.source_export_authority_sha256
            ),
            source_column_metadata=verified.authority.source_column_metadata,
            source_column_metadata_sha256=(
                verified.authority.source_column_metadata_sha256
            ),
            producer="analysis_cohort_ordered_subset",
            producer_implementation_sha256=producer_implementation_sha256,
            producer_parameters=bound_parameters,
            producer_parameters_sha256=canonical_parameters_sha256(bound_parameters),
            semantic_provenance=semantic_provenance,
            output_derivations=derivations,
            parent_authority_sha256=verified.reference.sha256,
        )
        authority_ref = _write_authority_at(root, authority)
        root.replace_temporary(temporary_name, target_name, require_absent=True)
        temporary_name = ""
        final_provenance = {
            **dict(_thaw_json(semantic_provenance)),
            "materialized_authority_required": True,
            "column_metadata": _descriptor(
                authority=authority_ref,
                sidecar=sidecar_ref,
                file_binding=child_binding,
            ),
        }
        _atomic_write_json_at(
            root,
            name=selector_name,
            payload=final_provenance,
        )
        return authority_ref
    finally:
        if descriptor >= 0:
            os.close(descriptor)
        if temporary_name:
            root.unlink(temporary_name, missing_ok=True)


__all__ = [
    "MATERIALIZED_COHORT_AUTHORITY_SCHEMA",
    "MATERIALIZED_COHORT_DESCRIPTOR_SCHEMA",
    "MaterializedCohortAuthority",
    "MaterializedCohortAuthorityRef",
    "MaterializedColumnMetadataCollector",
    "MaterializedMetadataError",
    "OutputDerivation",
    "SourceColumnRef",
    "VerifiedMaterializedCohortAuthority",
    "canonical_parameters_sha256",
    "implementation_bundle_sha256",
    "load_verified_materialized_cohort_authority",
    "materialized_provenance_path",
    "prepare_real_directory",
    "publish_ordered_subset_materialized_cohort",
    "read_verified_materialized_cohort_table",
    "stage_materialized_cohort_authority",
]
