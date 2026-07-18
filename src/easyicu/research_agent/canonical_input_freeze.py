"""Fail-closed archive assessment for Figure-2 canonical inputs.

The tracked v1 manifest records the exact legacy E2/E3/H2 files that were
available when the fresh Canonical9 coordinate was prepared.  It is an
*assessment authority*: every case is deliberately blocked and the typed
blockers explain what must be rematerialized or supplied by the benchmark
owner before a fresh run may start.

This module does not generate an Agent handoff.  A runnable handoff must be
created only after the existing materialized-cohort and materialized-trajectory
publishers have produced, and their host-owned loaders have reverified, the
complete selector -> authority -> metadata -> ancestry graph.  Treating one
JSON sidecar as that graph would be a false authority.

No database, exposure, outcome, cohort, method, or estimand is inferred here.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import stat
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from types import MappingProxyType
from typing import Annotated, Literal, Mapping, Optional, Sequence

import pyarrow.parquet as pq
from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    StrictBool,
    StrictInt,
    StrictStr,
    field_validator,
    model_validator,
)

CANONICAL_INPUT_FREEZE_SCHEMA_VERSION = "easyicu.figure2_canonical_input_freeze/1"
CANONICAL_INPUT_FREEZE_REF = "figure2_canonical9/input_freeze/20260718"

Sha256 = Annotated[StrictStr, Field(pattern=r"^[0-9a-f]{64}$")]
CaseId = Literal["e2", "e3", "h2"]
FileRole = Literal[
    "build_provenance",
    "cohort",
    "legacy_handoff",
    "selection_report",
    "trajectory",
    "trajectory_provenance",
]
FileFormat = Literal["json", "jsonl", "parquet"]
BlockerCode = Literal[
    "CONCEPT_DICTIONARY_AUTHORITY_UNRECORDED",
    "MISSING_BUILD_PROVENANCE",
    "MISSING_SELECTION_REPORT",
    "OWNER_DATABASE_REQUIRED",
    "OWNER_OPERATIONAL_EXPOSURE_REQUIRED",
    "PHYSICAL_ROW_COUNT_PROVENANCE_MISMATCH",
    "PHYSICAL_SCHEMA_PROVENANCE_MISMATCH",
    "RECORDED_COHORT_SEMANTIC_DIGEST_UNVERIFIED",
    "RECORDED_TRAJECTORY_SEMANTIC_DIGEST_UNVERIFIED",
    "TYPED_COHORT_AUTHORITY_MISSING",
    "TYPED_TRAJECTORY_AUTHORITY_MISSING",
]
BlockerResolution = Literal["benchmark_owner", "rematerialize"]

_EXPECTED_CASE_IDS = ("e2", "e3", "h2")
_EXPECTED_BENCHMARK_IDS: Mapping[str, str] = MappingProxyType(
    {
        "e2": "e2_lactate_mortality",
        "e3": "e3_kdigo_gradient",
        "h2": "h2_vasopressor_causal",
    }
)
_ROLE_FORMATS: Mapping[str, str] = MappingProxyType(
    {
        "build_provenance": "json",
        "cohort": "parquet",
        "legacy_handoff": "jsonl",
        "selection_report": "json",
        "trajectory": "parquet",
        "trajectory_provenance": "json",
    }
)
_OWNER_BLOCKERS = frozenset(
    {"OWNER_DATABASE_REQUIRED", "OWNER_OPERATIONAL_EXPOSURE_REQUIRED"}
)
_EVALUATOR_ONLY_FIELDS = frozenset(
    {
        "expected_direction",
        "expected_or_direction",
        "forbidden_claims",
        "gold_answer",
        "hazards",
        "reference_answer",
        "rubric",
    }
)
_MAX_MANIFEST_BYTES = 2 * 1024 * 1024
_MAX_JSON_MEMBER_BYTES = 8 * 1024 * 1024


class CanonicalInputFreezeError(RuntimeError):
    """The tracked manifest or its explicitly selected local bytes are invalid."""

    def __init__(
        self,
        message: str,
        *,
        code: str = "canonical_input_freeze_invalid",
        case_id: Optional[str] = None,
        member: Optional[str] = None,
    ) -> None:
        super().__init__(message)
        self.code = code
        self.case_id = case_id
        self.member = member


class _StrictFrozenModel(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)


class FrozenSubmissionProfile(_StrictFrozenModel):
    ref: StrictStr
    concept_dict_sha256: Sha256
    sofa2_dict_sha256: Sha256

    @field_validator("ref")
    @classmethod
    def _normalized_ref(cls, value: str) -> str:
        if not re.fullmatch(r"[a-z][a-z0-9_-]*/[0-9]{8}", value):
            raise ValueError("submission profile ref is not normalized")
        return value


class FrozenInputFile(_StrictFrozenModel):
    role: FileRole
    relative_path: StrictStr
    format: FileFormat
    sha256: Sha256
    size_bytes: StrictInt = Field(ge=0)
    row_count: StrictInt = Field(ge=0)
    column_count: Optional[StrictInt] = Field(default=None, ge=0)
    schema_sha256: Sha256

    @field_validator("relative_path")
    @classmethod
    def _portable_basename(cls, value: str) -> str:
        pure = PurePosixPath(value)
        if (
            not value
            or pure.is_absolute()
            or len(pure.parts) != 1
            or pure.name in {".", ".."}
            or "\\" in value
        ):
            raise ValueError("input member must be one portable path component")
        return value

    @model_validator(mode="after")
    def _coordinates_match_role(self) -> "FrozenInputFile":
        if self.format != _ROLE_FORMATS[self.role]:
            raise ValueError("input member format does not match its role")
        expected_suffix = {
            "json": ".json",
            "jsonl": ".jsonl",
            "parquet": ".parquet",
        }[self.format]
        if Path(self.relative_path).suffix.lower() != expected_suffix:
            raise ValueError("input member suffix does not match its format")
        if self.format == "parquet" and self.column_count is None:
            raise ValueError("parquet input requires column_count")
        if self.format != "parquet" and self.column_count is not None:
            raise ValueError("JSON input must not claim a column count")
        return self


class ProvenanceAssessment(_StrictFrozenModel):
    build_provenance_present: StrictBool
    selection_report_present: StrictBool
    concept_dictionary_authority_recorded: Literal[False]
    typed_cohort_authority_present: Literal[False]
    typed_trajectory_authority_present: Literal[False]
    physical_rows_match_provenance: Optional[StrictBool]
    physical_schema_matches_provenance: Optional[StrictBool]
    recorded_cohort_semantic_digest_present: StrictBool
    recorded_cohort_semantic_digest_reverified: Literal[False]
    recorded_trajectory_semantic_digest_present: StrictBool
    recorded_trajectory_semantic_digest_reverified: Literal[False]
    legacy_absolute_source_locator_present: StrictBool
    legacy_handoff_requires_sanitization: Literal[True]

    @model_validator(mode="after")
    def _coordinates_are_honest(self) -> "ProvenanceAssessment":
        comparisons = (
            self.physical_rows_match_provenance,
            self.physical_schema_matches_provenance,
        )
        if self.build_provenance_present and any(
            value is None for value in comparisons
        ):
            raise ValueError(
                "build provenance requires explicit physical row/schema comparisons"
            )
        if not self.build_provenance_present and any(
            value is not None for value in comparisons
        ):
            raise ValueError("physical comparisons require build provenance")
        if (
            self.recorded_cohort_semantic_digest_present
            and not self.build_provenance_present
        ):
            raise ValueError("cohort semantic digest requires build provenance")
        if self.recorded_trajectory_semantic_digest_present is False:
            # Absence is allowed; the case validator additionally binds presence
            # to a frozen trajectory-provenance member.
            return self
        return self


class InputBlocker(_StrictFrozenModel):
    code: BlockerCode
    resolution: BlockerResolution

    @model_validator(mode="after")
    def _resolution_matches_code(self) -> "InputBlocker":
        expected = (
            "benchmark_owner" if self.code in _OWNER_BLOCKERS else "rematerialize"
        )
        if self.resolution != expected:
            raise ValueError("blocker resolution does not match blocker type")
        return self


class CanonicalInputCase(_StrictFrozenModel):
    case_id: CaseId
    benchmark_item_id: StrictStr
    state: Literal["blocked"]
    blockers: tuple[InputBlocker, ...]
    files: tuple[FrozenInputFile, ...]
    provenance: ProvenanceAssessment

    @field_validator("benchmark_item_id")
    @classmethod
    def _normalized_item_id(cls, value: str) -> str:
        if not re.fullmatch(r"[a-z][a-z0-9_]{2,127}", value):
            raise ValueError("benchmark item id is not normalized")
        return value

    @model_validator(mode="after")
    def _blocked_archive_is_consistent(self) -> "CanonicalInputCase":
        if self.benchmark_item_id != _EXPECTED_BENCHMARK_IDS[self.case_id]:
            raise ValueError("case id does not match canonical benchmark item id")
        blocker_codes = tuple(item.code for item in self.blockers)
        if (
            not blocker_codes
            or blocker_codes != tuple(sorted(blocker_codes))
            or len(blocker_codes) != len(set(blocker_codes))
        ):
            raise ValueError("case blockers must be non-empty, sorted and unique")
        file_keys = tuple((item.role, item.relative_path) for item in self.files)
        if file_keys != tuple(sorted(file_keys)) or len(file_keys) != len(
            set(file_keys)
        ):
            raise ValueError("case files must be sorted and unique")
        roles = tuple(item.role for item in self.files)
        if len(roles) != len(set(roles)):
            raise ValueError("archive v1 permits one file per frozen role")
        if "cohort" not in roles or "legacy_handoff" not in roles:
            raise ValueError("each case requires cohort and legacy handoff files")
        if self.provenance.build_provenance_present != ("build_provenance" in roles):
            raise ValueError("build provenance flag does not match frozen files")
        if self.provenance.selection_report_present != ("selection_report" in roles):
            raise ValueError("selection report flag does not match frozen files")
        if self.provenance.recorded_trajectory_semantic_digest_present != (
            "trajectory_provenance" in roles
        ):
            raise ValueError("trajectory digest flag does not match frozen provenance")
        required: set[str] = {
            "CONCEPT_DICTIONARY_AUTHORITY_UNRECORDED",
            "TYPED_COHORT_AUTHORITY_MISSING",
        }
        if not self.provenance.build_provenance_present:
            required.add("MISSING_BUILD_PROVENANCE")
        if not self.provenance.selection_report_present:
            required.add("MISSING_SELECTION_REPORT")
        if self.provenance.physical_rows_match_provenance is False:
            required.add("PHYSICAL_ROW_COUNT_PROVENANCE_MISMATCH")
        if self.provenance.physical_schema_matches_provenance is False:
            required.add("PHYSICAL_SCHEMA_PROVENANCE_MISMATCH")
        if self.provenance.recorded_cohort_semantic_digest_present:
            required.add("RECORDED_COHORT_SEMANTIC_DIGEST_UNVERIFIED")
        if "trajectory" in roles:
            required.add("TYPED_TRAJECTORY_AUTHORITY_MISSING")
        if self.provenance.recorded_trajectory_semantic_digest_present:
            required.add("RECORDED_TRAJECTORY_SEMANTIC_DIGEST_UNVERIFIED")
        if not required.issubset(blocker_codes):
            missing = sorted(required - set(blocker_codes))
            raise ValueError(f"missing required blocker(s): {missing}")
        if self.case_id != "h2" and set(blocker_codes) & _OWNER_BLOCKERS:
            raise ValueError("owner-science blockers are authorized only for h2")
        if self.case_id == "h2":
            if "trajectory" not in roles:
                raise ValueError("h2 archive requires its trajectory input")
            if not _OWNER_BLOCKERS.issubset(blocker_codes):
                raise ValueError("h2 requires explicit owner science blockers")
        return self


class CanonicalInputFreezeManifest(_StrictFrozenModel):
    schema_version: Literal[CANONICAL_INPUT_FREEZE_SCHEMA_VERSION]
    manifest_ref: Literal[CANONICAL_INPUT_FREEZE_REF]
    submission_profile: FrozenSubmissionProfile
    cases: tuple[CanonicalInputCase, ...]

    @model_validator(mode="after")
    def _suite_is_exact(self) -> "CanonicalInputFreezeManifest":
        if tuple(item.case_id for item in self.cases) != _EXPECTED_CASE_IDS:
            raise ValueError("freeze manifest must contain ordered e2/e3/h2 cases")
        return self


@dataclass(frozen=True, slots=True)
class VerifiedLocalFile:
    frozen: FrozenInputFile
    path: Path


@dataclass(frozen=True, slots=True)
class VerifiedLocalCase:
    frozen: CanonicalInputCase
    files: Mapping[str, VerifiedLocalFile]


@dataclass(frozen=True, slots=True)
class VerifiedLocalInputFreeze:
    """Verified diagnostic view; paths are never authorised for Agent intake."""

    manifest: CanonicalInputFreezeManifest
    manifest_sha256: str
    cases: Mapping[str, VerifiedLocalCase]


def _reject_duplicate_pairs(pairs: Sequence[tuple[str, object]]) -> dict[str, object]:
    result: dict[str, object] = {}
    for key, value in pairs:
        if key in result:
            raise CanonicalInputFreezeError(
                f"duplicate JSON key: {key!r}", code="duplicate_json_key"
            )
        result[key] = value
    return result


def _reject_nonfinite(value: str) -> None:
    raise CanonicalInputFreezeError(
        f"non-finite JSON constant is forbidden: {value}",
        code="nonfinite_json_value",
    )


def _decode_json(raw: bytes, *, label: str) -> object:
    try:
        text = raw.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise CanonicalInputFreezeError(
            f"{label} is not UTF-8", code="invalid_utf8"
        ) from exc
    try:
        return json.loads(
            text,
            object_pairs_hook=_reject_duplicate_pairs,
            parse_constant=_reject_nonfinite,
        )
    except CanonicalInputFreezeError:
        raise
    except (TypeError, ValueError, json.JSONDecodeError) as exc:
        raise CanonicalInputFreezeError(
            f"{label} is not strict JSON: {exc}", code="invalid_json"
        ) from exc


def _canonical_json_bytes(value: object) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")


def _scan_public_payload(value: object) -> None:
    if isinstance(value, Mapping):
        for key, item in value.items():
            if str(key).lower() in _EVALUATOR_ONLY_FIELDS:
                raise CanonicalInputFreezeError(
                    f"public input freeze contains evaluator-only key {key!r}",
                    code="evaluator_oracle_in_public_manifest",
                )
            _scan_public_payload(item)
        return
    if isinstance(value, list):
        for item in value:
            _scan_public_payload(item)
        return
    if isinstance(value, str) and (
        value.startswith(("/", "~/", "file://"))
        or re.match(r"^[A-Za-z]:[\\/]", value)
        or "/Volumes/" in value
    ):
        raise CanonicalInputFreezeError(
            "tracked public input freeze contains an absolute path",
            code="absolute_path_in_public_manifest",
        )


def _read_manifest_once(
    path: Path,
) -> tuple[CanonicalInputFreezeManifest, bytes, str]:
    path = Path(path)
    flags = os.O_RDONLY
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    try:
        fd = os.open(path, flags)
    except OSError as exc:
        raise CanonicalInputFreezeError(
            f"cannot open freeze manifest: {exc}", code="manifest_unreadable"
        ) from exc
    try:
        before = os.fstat(fd)
        if not stat.S_ISREG(before.st_mode):
            raise CanonicalInputFreezeError(
                "freeze manifest is not a regular file", code="manifest_not_regular"
            )
        if before.st_size > _MAX_MANIFEST_BYTES:
            raise CanonicalInputFreezeError(
                "freeze manifest exceeds size limit", code="manifest_too_large"
            )
        with os.fdopen(os.dup(fd), "rb", closefd=True) as handle:
            raw = handle.read(_MAX_MANIFEST_BYTES + 1)
        after = os.fstat(fd)
        if (
            before.st_dev,
            before.st_ino,
            before.st_size,
            before.st_mtime_ns,
        ) != (after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns):
            raise CanonicalInputFreezeError(
                "freeze manifest changed while being read", code="manifest_mutated"
            )
    finally:
        os.close(fd)
    payload = _decode_json(raw, label="canonical input freeze manifest")
    if not isinstance(payload, Mapping):
        raise CanonicalInputFreezeError(
            "freeze manifest must be a JSON object", code="manifest_not_object"
        )
    _scan_public_payload(payload)
    try:
        manifest = CanonicalInputFreezeManifest.model_validate_json(raw, strict=True)
    except ValueError as exc:
        raise CanonicalInputFreezeError(
            f"freeze manifest schema is invalid: {exc}", code="manifest_schema_invalid"
        ) from exc
    canonical = _canonical_json_bytes(manifest.model_dump(mode="json")) + b"\n"
    if raw != canonical:
        raise CanonicalInputFreezeError(
            "freeze manifest is not canonical JSON", code="manifest_not_canonical"
        )
    return manifest, raw, hashlib.sha256(raw).hexdigest()


def load_canonical_input_freeze_manifest(path: Path) -> CanonicalInputFreezeManifest:
    """Load the immutable blocked-archive assessment."""

    manifest, _, _ = _read_manifest_once(path)
    return manifest


def canonical_input_freeze_manifest_sha256(path: Path) -> str:
    """Digest the exact bytes that passed validation (one anchored read)."""

    _, _, digest = _read_manifest_once(path)
    return digest


def _json_shape(value: object) -> object:
    if value is None:
        return "null"
    if isinstance(value, bool):
        return "boolean"
    if isinstance(value, int):
        return "integer"
    if isinstance(value, float):
        return "number"
    if isinstance(value, str):
        return "string"
    if isinstance(value, Mapping):
        return {"object": {str(key): _json_shape(item) for key, item in value.items()}}
    if isinstance(value, (list, tuple)):
        shapes = sorted({_canonical_json_bytes(_json_shape(item)) for item in value})
        return {"array": [json.loads(item.decode("utf-8")) for item in shapes]}
    raise CanonicalInputFreezeError(
        f"unsupported JSON type: {type(value).__name__}", code="unsupported_json_type"
    )


def _parquet_shape(parquet: pq.ParquetFile) -> object:
    schema = parquet.schema_arrow
    return {
        "fields": [
            {
                "name": field.name,
                "nullable": bool(field.nullable),
                "type": str(field.type),
            }
            for field in schema
        ],
        "num_columns": len(schema),
    }


def _schema_sha256(value: object) -> str:
    return hashlib.sha256(_canonical_json_bytes(value)).hexdigest()


def _verify_member(
    root: Path, frozen: FrozenInputFile, *, case_id: str
) -> VerifiedLocalFile:
    path = root / frozen.relative_path
    flags = os.O_RDONLY
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    try:
        fd = os.open(path, flags)
    except OSError as exc:
        raise CanonicalInputFreezeError(
            f"cannot open frozen input {frozen.relative_path}: {exc}",
            code="frozen_input_unreadable",
            case_id=case_id,
            member=frozen.relative_path,
        ) from exc
    try:
        before = os.fstat(fd)
        if not stat.S_ISREG(before.st_mode):
            raise CanonicalInputFreezeError(
                "frozen input is not a regular file",
                code="frozen_input_not_regular",
                case_id=case_id,
                member=frozen.relative_path,
            )
        digest = hashlib.sha256()
        with os.fdopen(os.dup(fd), "rb", closefd=True) as handle:
            while chunk := handle.read(1024 * 1024):
                digest.update(chunk)
        if digest.hexdigest() != frozen.sha256 or before.st_size != frozen.size_bytes:
            raise CanonicalInputFreezeError(
                "frozen input digest or size mismatch",
                code="frozen_input_digest_mismatch",
                case_id=case_id,
                member=frozen.relative_path,
            )
        os.lseek(fd, 0, os.SEEK_SET)
        with os.fdopen(os.dup(fd), "rb", closefd=True) as handle:
            if frozen.format == "parquet":
                parquet = pq.ParquetFile(handle)
                row_count = int(parquet.metadata.num_rows)
                column_count: Optional[int] = int(parquet.metadata.num_columns)
                schema_sha = _schema_sha256(_parquet_shape(parquet))
            else:
                if before.st_size > _MAX_JSON_MEMBER_BYTES:
                    raise CanonicalInputFreezeError(
                        "frozen JSON input exceeds size limit",
                        code="frozen_json_too_large",
                        case_id=case_id,
                        member=frozen.relative_path,
                    )
                raw = handle.read()
                if frozen.format == "json":
                    decoded = _decode_json(raw, label=frozen.relative_path)
                    row_count = 1
                    schema_sha = _schema_sha256(_json_shape(decoded))
                else:
                    records: list[object] = []
                    for number, line in enumerate(raw.splitlines(), start=1):
                        if not line.strip():
                            continue
                        record = _decode_json(
                            line, label=f"{frozen.relative_path} line {number}"
                        )
                        if not isinstance(record, Mapping):
                            raise CanonicalInputFreezeError(
                                "frozen JSONL row must be an object",
                                code="jsonl_row_not_object",
                                case_id=case_id,
                                member=frozen.relative_path,
                            )
                        records.append(record)
                    row_count = len(records)
                    schema_sha = _schema_sha256(_json_shape(records))
                column_count = None
        after = os.fstat(fd)
        if (
            before.st_dev,
            before.st_ino,
            before.st_size,
            before.st_mtime_ns,
        ) != (after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns):
            raise CanonicalInputFreezeError(
                "frozen input changed while being verified",
                code="frozen_input_mutated",
                case_id=case_id,
                member=frozen.relative_path,
            )
        if (
            row_count != frozen.row_count
            or column_count != frozen.column_count
            or schema_sha != frozen.schema_sha256
        ):
            raise CanonicalInputFreezeError(
                "frozen input row count or schema mismatch",
                code="frozen_input_schema_mismatch",
                case_id=case_id,
                member=frozen.relative_path,
            )
    finally:
        os.close(fd)
    return VerifiedLocalFile(frozen=frozen, path=path)


def verify_local_input_freeze(
    manifest_path: Path,
    *,
    local_roots: Mapping[str, Path],
    case_ids: Optional[Sequence[str]] = None,
) -> VerifiedLocalInputFreeze:
    """Verify archived bytes for diagnosis only; never authorise Agent intake."""

    manifest, _, manifest_sha = _read_manifest_once(manifest_path)
    selected = tuple(_EXPECTED_CASE_IDS if case_ids is None else case_ids)
    if not selected or len(selected) != len(set(selected)):
        raise CanonicalInputFreezeError(
            "selected case ids must be non-empty and unique",
            code="invalid_case_selection",
        )
    canonical_selected = tuple(item for item in _EXPECTED_CASE_IDS if item in selected)
    if selected != canonical_selected:
        raise CanonicalInputFreezeError(
            "selected case ids must follow manifest order",
            code="invalid_case_selection",
        )
    known = {item.case_id: item for item in manifest.cases}
    if any(item not in known for item in selected):
        raise CanonicalInputFreezeError(
            "selected case id is not frozen", code="unknown_case_id"
        )
    if set(local_roots) != set(selected):
        raise CanonicalInputFreezeError(
            "local root mapping must exactly match selected case ids",
            code="local_root_mapping_mismatch",
        )
    verified_cases: dict[str, VerifiedLocalCase] = {}
    for case_id in selected:
        root = Path(local_roots[case_id])
        try:
            if root.is_symlink() or not root.is_dir():
                raise CanonicalInputFreezeError(
                    "local case root must be a real directory",
                    code="local_root_invalid",
                    case_id=case_id,
                )
            root = root.resolve(strict=True)
        except OSError as exc:
            raise CanonicalInputFreezeError(
                f"cannot resolve local case root: {exc}",
                code="local_root_invalid",
                case_id=case_id,
            ) from exc
        case = known[case_id]
        files = {
            item.role: _verify_member(root, item, case_id=case_id)
            for item in case.files
        }
        verified_cases[case_id] = VerifiedLocalCase(
            frozen=case,
            files=MappingProxyType(files),
        )
    return VerifiedLocalInputFreeze(
        manifest=manifest,
        manifest_sha256=manifest_sha,
        cases=MappingProxyType(verified_cases),
    )


__all__ = [
    "CANONICAL_INPUT_FREEZE_REF",
    "CANONICAL_INPUT_FREEZE_SCHEMA_VERSION",
    "CanonicalInputCase",
    "CanonicalInputFreezeError",
    "CanonicalInputFreezeManifest",
    "FrozenInputFile",
    "FrozenSubmissionProfile",
    "InputBlocker",
    "ProvenanceAssessment",
    "VerifiedLocalInputFreeze",
    "canonical_input_freeze_manifest_sha256",
    "load_canonical_input_freeze_manifest",
    "verify_local_input_freeze",
]
