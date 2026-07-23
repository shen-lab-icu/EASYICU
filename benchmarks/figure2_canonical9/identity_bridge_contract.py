"""Fail-closed contract for a future Canonical9 patient-identity bridge.

The historical ``full6_20260717`` export contains stay-level data but not a
patient-identity relation.  A source owner may, after a separately approved
data-lane decision, produce a protected stay-to-patient mapping from the
original source snapshots.  This module describes how that *future handoff*
is bound to the exact historical export without ever reading the mapping or
patient data itself.

It is intentionally not imported by the launcher or ``realrun_authority``.
An identity bridge can at most become an input to a later native typed
materialization review.  It never makes a Canonical9 run launchable by itself.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import stat
from pathlib import Path
from typing import Annotated, Any, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

IDENTITY_BRIDGE_SCHEMA = "easyicu.figure2_identity_bridge/1"
IDENTITY_BRIDGE_REF = "figure2_canonical9/identity-bridge/20260722-v1"
_MAX_CONTRACT_BYTES = 256 * 1024

Sha256 = Annotated[str, Field(pattern=r"^[0-9a-f]{64}$")]
PositiveInt = Annotated[int, Field(ge=0)]
NonEmptyInt = Annotated[int, Field(gt=0)]

_SOURCE_KEYS: tuple[tuple[str, str, str], ...] = (
    ("mimic_iv", "stay_id", "subject_id"),
    ("mimic_iii", "icustay_id", "subject_id"),
    ("eicu", "patientunitstayid", "uniquepid"),
    ("amsterdamumcdb", "admissionid", "patientid"),
    ("hirid", "patientid", "patientid"),
    ("sicdb", "CaseID", "PatientID"),
)
_SOURCE_IDS = tuple(item[0] for item in _SOURCE_KEYS)
_SOURCE_KEY_BY_ID = {item[0]: item[1:] for item in _SOURCE_KEYS}


class IdentityBridgeContractError(ValueError):
    """The bridge contract is malformed or unsafe to consume."""


class _StrictFrozenModel(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)


class HistoricalExportIdentity(_StrictFrozenModel):
    """Content identity of the immutable development export, never a path."""

    export_label: Literal["full6_20260717"]
    export_manifest_sha256: Sha256
    export_content_sha256: Sha256


class MappingProjectionIdentity(_StrictFrozenModel):
    """Reference to a protected mapping without placing patient identifiers here."""

    artifact_sha256: Sha256
    artifact_size_bytes: NonEmptyInt
    source_snapshot_sha256: Sha256
    relation_schema_sha256: Sha256


class SourceIdentityMapping(_StrictFrozenModel):
    """Per-source semantics plus aggregate cardinality evidence.

    The mapping contents stay in the controlled host environment.  No source
    path, raw identifier, row value, or free-text clinical description is
    allowed in this document.
    """

    source_id: str
    stay_key: str
    patient_key: str
    mapping_semantics: Literal[
        "attested_icu_stay_to_patient", "attested_source_key_semantics"
    ]
    source_semantics_attestation_sha256: Sha256
    projection: MappingProjectionIdentity
    mapped_stay_count: PositiveInt
    unmapped_stay_count: PositiveInt
    duplicate_stay_count: PositiveInt
    max_stays_per_patient: PositiveInt

    @model_validator(mode="after")
    def _validate_observed_key_contract(self) -> "SourceIdentityMapping":
        expected = _SOURCE_KEY_BY_ID.get(self.source_id)
        if expected is None:
            raise ValueError("identity bridge has an unknown source_id")
        if (self.stay_key, self.patient_key) != expected:
            raise ValueError(
                "identity bridge source keys do not match the audited schema"
            )
        if self.source_id in {"hirid", "sicdb"}:
            if self.mapping_semantics != "attested_source_key_semantics":
                raise ValueError(
                    "HiRID and SICdb require explicit source-key semantic attestation"
                )
        elif self.mapping_semantics != "attested_icu_stay_to_patient":
            raise ValueError(
                "observed ICU sources require ICU-stay-to-patient attestation"
            )
        if self.mapped_stay_count == 0:
            raise ValueError("identity bridge mapping must cover at least one stay")
        if self.duplicate_stay_count != 0:
            raise ValueError("identity bridge cannot contain duplicate stay mappings")
        if self.max_stays_per_patient == 0:
            raise ValueError("identity bridge must report patient cardinality")
        return self


class DataLaneAuthorization(_StrictFrozenModel):
    """Owner-controlled authorization for the mapping-production lane only."""

    status: Literal["not_authorized", "authorized"]
    authorization_reference: str | None

    @model_validator(mode="after")
    def _validate_authorization_reference(self) -> "DataLaneAuthorization":
        if self.status == "authorized":
            if (
                not self.authorization_reference
                or not self.authorization_reference.strip()
            ):
                raise ValueError(
                    "authorized identity bridge needs an authorization reference"
                )
        elif self.authorization_reference is not None:
            raise ValueError(
                "unauthorized identity bridge cannot carry an authorization reference"
            )
        return self


class IdentityBridgeContract(_StrictFrozenModel):
    """A reviewable bridge descriptor, deliberately not a production authority."""

    schema_version: Literal["easyicu.figure2_identity_bridge/1"]
    bridge_ref: Literal["figure2_canonical9/identity-bridge/20260722-v1"]
    historical_export: HistoricalExportIdentity
    data_lane: DataLaneAuthorization
    source_mappings: tuple[SourceIdentityMapping, ...]

    @model_validator(mode="before")
    @classmethod
    def _convert_json_source_array(cls, raw: Any) -> Any:
        """JSON has arrays; the frozen in-memory contract has a tuple.

        This is the only structural conversion permitted before strict field
        validation.  It does not coerce values inside a source mapping.
        """

        if not isinstance(raw, dict):
            return raw
        source_mappings = raw.get("source_mappings")
        if not isinstance(source_mappings, list):
            return raw
        converted = dict(raw)
        converted["source_mappings"] = tuple(source_mappings)
        return converted

    @model_validator(mode="after")
    def _validate_exact_sources(self) -> "IdentityBridgeContract":
        sources = tuple(item.source_id for item in self.source_mappings)
        if sources != _SOURCE_IDS:
            raise ValueError("identity bridge must contain exact six-source order")
        projections = tuple(
            item.projection.artifact_sha256 for item in self.source_mappings
        )
        if len(projections) != len(set(projections)):
            raise ValueError(
                "each source needs an independently content-addressed mapping"
            )
        return self


class IdentityBridgeReadiness(_StrictFrozenModel):
    """Report eligibility for native input review, never an execution permit."""

    contract_sha256: Sha256
    data_lane_authorized: bool
    eligible_for_native_materialization_review: bool
    real_run_authorized: Literal[False]
    blockers: tuple[str, ...]


def _canonical_json_bytes(value: object) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")


def _read_contract_bytes(path: Path) -> bytes:
    descriptor: int | None = None
    try:
        if not path.is_absolute() or path.is_symlink():
            raise IdentityBridgeContractError(
                "identity bridge contract must be an absolute, non-symlink file"
            )
        try:
            descriptor = os.open(
                path,
                os.O_RDONLY
                | getattr(os, "O_CLOEXEC", 0)
                | getattr(os, "O_NOFOLLOW", 0),
            )
        except OSError as exc:
            raise IdentityBridgeContractError(
                "identity bridge contract cannot be opened safely"
            ) from exc
        info = os.fstat(descriptor)
        if not stat.S_ISREG(info.st_mode) or info.st_size > _MAX_CONTRACT_BYTES:
            raise IdentityBridgeContractError(
                "identity bridge contract must be a small regular file"
            )
        pieces: list[bytes] = []
        size = 0
        while block := os.read(descriptor, 64 * 1024):
            size += len(block)
            if size > _MAX_CONTRACT_BYTES:
                raise IdentityBridgeContractError(
                    "identity bridge contract exceeds the size limit while reading"
                )
            pieces.append(block)
        return b"".join(pieces)
    finally:
        if descriptor is not None:
            os.close(descriptor)


def _reject_duplicate_json_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise IdentityBridgeContractError(
                f"identity bridge contract has duplicate key {key!r}"
            )
        result[key] = value
    return result


def load_identity_bridge_contract(
    path: Path | str,
) -> tuple[IdentityBridgeContract, str]:
    """Load a canonical descriptor without reading its patient mapping artifacts."""

    source = Path(path)
    raw = _read_contract_bytes(source)
    try:
        decoded = json.loads(
            raw.decode("utf-8"), object_pairs_hook=_reject_duplicate_json_keys
        )
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise IdentityBridgeContractError(
            "identity bridge contract is not valid JSON"
        ) from exc
    if not isinstance(decoded, dict):
        raise IdentityBridgeContractError(
            "identity bridge contract must be canonical JSON"
        )
    try:
        canonical = _canonical_json_bytes(decoded) + b"\n"
    except ValueError as exc:
        raise IdentityBridgeContractError(
            "identity bridge contract cannot contain non-finite values"
        ) from exc
    if raw != canonical:
        raise IdentityBridgeContractError(
            "identity bridge contract must be canonical JSON"
        )
    try:
        contract = IdentityBridgeContract.model_validate(decoded, strict=True)
    except ValueError as exc:
        raise IdentityBridgeContractError(str(exc)) from exc
    return contract, hashlib.sha256(raw).hexdigest()


def assess_identity_bridge_contract(
    contract: IdentityBridgeContract, *, contract_sha256: str
) -> IdentityBridgeReadiness:
    """Assess only native-materialization handoff eligibility.

    A true result for ``eligible_for_native_materialization_review`` means that
    the owner may ask the typed materializer to review the controlled mapping.
    It has no launch effect: P4 still requires fresh typed cohort and trajectory
    authority, scientific case decisions, and a final operator freeze.
    """

    if not isinstance(contract_sha256, str) or not re.fullmatch(
        r"[0-9a-f]{64}", contract_sha256
    ):
        raise IdentityBridgeContractError("identity bridge contract digest is invalid")
    allowed = contract.data_lane.status == "authorized"
    blockers = () if allowed else ("OWNER_DATA_LANE_AUTHORIZATION_REQUIRED",)
    return IdentityBridgeReadiness(
        contract_sha256=contract_sha256,
        data_lane_authorized=allowed,
        eligible_for_native_materialization_review=allowed,
        real_run_authorized=False,
        blockers=blockers
        + (
            "NATIVE_TYPED_MATERIALIZATION_REQUIRED",
            "P4_PRODUCTION_INPUT_AUTHORITY_REQUIRED",
            "FINAL_OPERATOR_FREEZE_REQUIRED",
        ),
    )


__all__ = [
    "DataLaneAuthorization",
    "IDENTITY_BRIDGE_REF",
    "IDENTITY_BRIDGE_SCHEMA",
    "IdentityBridgeContract",
    "IdentityBridgeContractError",
    "IdentityBridgeReadiness",
    "SourceIdentityMapping",
    "assess_identity_bridge_contract",
    "load_identity_bridge_contract",
]
