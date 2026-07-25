"""Strict handoff contract for a future full0717 typed-materialization review.

The historical full0717 export is intentionally not upgraded in place.  This
module describes the *review handoff* that a data, transformation, and identity
owner must complete before an independently reviewed native typed
materialization can begin.  It does not read clinical data, bridge mappings,
or typed inventory contents; it only validates their immutable identities.

It is deliberately not imported by :mod:`realrun_authority`.  Even a fully
attested contract is not a production input authority and can never authorize
a Provider, Docker runner, or Canonical9 execution.
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

SOURCE_ATTESTATION_SCHEMA = "easyicu.figure2_source_attestation/1"
SOURCE_ATTESTATION_REF = "figure2_canonical9/full0717-source-attestation/20260722-v1"
_MAX_CONTRACT_BYTES = 256 * 1024

# This contract is intentionally a one-snapshot review handoff.  A different
# export or a rebuilt bridge needs a new review contract and cannot borrow this
# evidence merely by reusing the human-readable ``full6_20260717`` label.
FULL0717_EXPORT_CONTENT_SHA256 = (
    "2812dbf45f1308013147ea0f09cbdf7b93b39f31c01e15588e5923a800a3fcdf"
)
FULL0717_RUN_MANIFEST_SHA256 = (
    "3cd70b85daabf6470ea19ed315cf07b88d6fcc0737cdfa4980c4639d109e9ed2"
)
FULL0717_IDENTITY_BRIDGE_CONTRACT_SHA256 = (
    "4092104a40d2b22d80a93cf00323be0f3c048bfc3d9ea6bb002124361ecce794"
)

Sha256 = Annotated[str, Field(pattern=r"^[0-9a-f]{64}$")]
PositiveInt = Annotated[int, Field(gt=0)]
Reference = Annotated[
    str,
    Field(pattern=r"^[A-Za-z0-9][A-Za-z0-9._:/@-]{3,255}$"),
]

_SOURCE_IDS = (
    "mimic_iv",
    "mimic_iii",
    "eicu",
    "amsterdamumcdb",
    "hirid",
    "sicdb",
)


class SourceAttestationContractError(ValueError):
    """A source-attestation handoff is malformed, forged, or unsafe."""


class _StrictFrozenModel(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)


class HistoricalExportIdentity(_StrictFrozenModel):
    """Content identities for the immutable historical payload."""

    export_label: Literal["full6_20260717"]
    export_content_sha256: Sha256
    export_run_manifest_sha256: Sha256

    @model_validator(mode="after")
    def _reject_placeholder_digests(self) -> "HistoricalExportIdentity":
        if self.export_content_sha256 != FULL0717_EXPORT_CONTENT_SHA256:
            raise ValueError("historical export content digest does not bind full0717")
        if self.export_run_manifest_sha256 != FULL0717_RUN_MANIFEST_SHA256:
            raise ValueError("historical run manifest digest does not bind full0717")
        return self


class IdentityBridgeBinding(_StrictFrozenModel):
    """Reference to a separately verified host-only identity bridge."""

    contract_sha256: Sha256
    review_handoff_only: Literal[True]
    real_run_authorized: Literal[False]

    @model_validator(mode="after")
    def _reject_placeholder_digest(self) -> "IdentityBridgeBinding":
        if self.contract_sha256 != FULL0717_IDENTITY_BRIDGE_CONTRACT_SHA256:
            raise ValueError("identity bridge digest does not bind the approved bridge")
        return self


class SourceTypedInventoryAttestation(_StrictFrozenModel):
    """One source's source-attested typed inventory, kept outside this contract.

    The inventory itself is a protected, reviewed artifact.  Its content must
    bind file/column semantics to the exact historical export; this descriptor
    holds only its digest and related source evidence identities.
    """

    source_id: str
    source_snapshot_sha256: Sha256
    relation_schema_sha256: Sha256
    typed_column_inventory_sha256: Sha256
    source_semantics_attestation_sha256: Sha256
    inventory_record_count: PositiveInt

    @model_validator(mode="after")
    def _validate_source_and_digests(self) -> "SourceTypedInventoryAttestation":
        if self.source_id not in _SOURCE_IDS:
            raise ValueError("source attestation has an unknown source_id")
        values = (
            self.source_snapshot_sha256,
            self.relation_schema_sha256,
            self.typed_column_inventory_sha256,
            self.source_semantics_attestation_sha256,
        )
        if any(value == "0" * 64 for value in values):
            raise ValueError("source attestation digest cannot be a placeholder")
        return self


class ReviewAttestation(_StrictFrozenModel):
    """Non-clinical owner references for the native materialization review."""

    status: Literal["pending", "attested_for_native_materialization_review"]
    data_owner_reference: Reference | None
    transformation_owner_reference: Reference | None
    identity_owner_reference: Reference | None

    @model_validator(mode="after")
    def _validate_state(self) -> "ReviewAttestation":
        values = (
            self.data_owner_reference,
            self.transformation_owner_reference,
            self.identity_owner_reference,
        )
        if self.status == "pending":
            if any(value is not None for value in values):
                raise ValueError("pending source review cannot carry owner references")
        elif any(value is None for value in values):
            raise ValueError(
                "attested source review requires all three owner references"
            )
        return self


class SourceAttestationContract(_StrictFrozenModel):
    """One full0717 review handoff, never a real-run authority."""

    schema_version: Literal["easyicu.figure2_source_attestation/1"]
    attestation_ref: Literal[
        "figure2_canonical9/full0717-source-attestation/20260722-v1"
    ]
    historical_export: HistoricalExportIdentity
    identity_bridge: IdentityBridgeBinding
    review: ReviewAttestation
    source_attestations: tuple[SourceTypedInventoryAttestation, ...]

    @model_validator(mode="before")
    @classmethod
    def _convert_json_sources(cls, raw: Any) -> Any:
        if not isinstance(raw, dict):
            return raw
        values = raw.get("source_attestations")
        if not isinstance(values, list):
            return raw
        converted = dict(raw)
        converted["source_attestations"] = tuple(values)
        return converted

    @model_validator(mode="after")
    def _validate_review_scope(self) -> "SourceAttestationContract":
        if self.review.status == "pending":
            if self.source_attestations:
                raise ValueError(
                    "pending source review cannot present typed-inventory attestations"
                )
            return self
        source_ids = tuple(item.source_id for item in self.source_attestations)
        if source_ids != _SOURCE_IDS:
            raise ValueError("attested source review requires exact six-source order")
        inventory_digests = tuple(
            item.typed_column_inventory_sha256 for item in self.source_attestations
        )
        if len(inventory_digests) != len(set(inventory_digests)):
            raise ValueError("each source needs an independent typed column inventory")
        return self


class SourceAttestationReadiness(_StrictFrozenModel):
    """Readiness for review only; it intentionally cannot authorize a run."""

    contract_sha256: Sha256
    source_review_attested: bool
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
            raise SourceAttestationContractError(
                "source attestation contract must be an absolute, non-symlink file"
            )
        try:
            descriptor = os.open(
                path,
                os.O_RDONLY
                | getattr(os, "O_CLOEXEC", 0)
                | getattr(os, "O_NOFOLLOW", 0),
            )
        except OSError as exc:
            raise SourceAttestationContractError(
                "source attestation contract cannot be opened safely"
            ) from exc
        info = os.fstat(descriptor)
        if not stat.S_ISREG(info.st_mode) or info.st_size > _MAX_CONTRACT_BYTES:
            raise SourceAttestationContractError(
                "source attestation contract must be a small regular file"
            )
        pieces: list[bytes] = []
        size = 0
        while block := os.read(descriptor, 64 * 1024):
            size += len(block)
            if size > _MAX_CONTRACT_BYTES:
                raise SourceAttestationContractError(
                    "source attestation contract exceeds the size limit while reading"
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
            raise SourceAttestationContractError(
                f"source attestation contract has duplicate key {key!r}"
            )
        result[key] = value
    return result


def load_source_attestation_contract(
    path: Path | str,
) -> tuple[SourceAttestationContract, str]:
    """Load a canonical review descriptor without reading clinical artifacts."""

    raw = _read_contract_bytes(Path(path))
    try:
        decoded = json.loads(
            raw.decode("utf-8"), object_pairs_hook=_reject_duplicate_json_keys
        )
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise SourceAttestationContractError(
            "source attestation contract is not valid JSON"
        ) from exc
    if not isinstance(decoded, dict):
        raise SourceAttestationContractError(
            "source attestation contract must be canonical JSON"
        )
    try:
        canonical = _canonical_json_bytes(decoded) + b"\n"
    except ValueError as exc:
        raise SourceAttestationContractError(
            "source attestation contract cannot contain non-finite values"
        ) from exc
    if raw != canonical:
        raise SourceAttestationContractError(
            "source attestation contract must be canonical JSON"
        )
    try:
        contract = SourceAttestationContract.model_validate(decoded, strict=True)
    except ValueError as exc:
        raise SourceAttestationContractError(str(exc)) from exc
    return contract, hashlib.sha256(raw).hexdigest()


def assess_source_attestation_contract(
    contract: SourceAttestationContract, *, contract_sha256: str
) -> SourceAttestationReadiness:
    """Assess native-materialization review eligibility, never P4 authorization."""

    if not isinstance(contract_sha256, str) or not re.fullmatch(
        r"[0-9a-f]{64}", contract_sha256
    ):
        raise SourceAttestationContractError(
            "source attestation contract digest is invalid"
        )
    attested = contract.review.status == "attested_for_native_materialization_review"
    initial_blockers = (
        ()
        if attested
        else (
            "SOURCE_DATA_IDENTITY_ATTESTATION_REQUIRED",
            "TYPED_COLUMN_INVENTORY_REQUIRED",
        )
    )
    return SourceAttestationReadiness(
        contract_sha256=contract_sha256,
        source_review_attested=attested,
        eligible_for_native_materialization_review=attested,
        real_run_authorized=False,
        blockers=initial_blockers
        + (
            "NATIVE_TYPED_MATERIALIZATION_REQUIRED",
            "P4_PRODUCTION_INPUT_AUTHORITY_REQUIRED",
            "FINAL_OPERATOR_FREEZE_REQUIRED",
        ),
    )


__all__ = [
    "IdentityBridgeBinding",
    "FULL0717_EXPORT_CONTENT_SHA256",
    "FULL0717_IDENTITY_BRIDGE_CONTRACT_SHA256",
    "FULL0717_RUN_MANIFEST_SHA256",
    "ReviewAttestation",
    "SOURCE_ATTESTATION_REF",
    "SOURCE_ATTESTATION_SCHEMA",
    "SourceAttestationContract",
    "SourceAttestationContractError",
    "SourceAttestationReadiness",
    "SourceTypedInventoryAttestation",
    "assess_source_attestation_contract",
    "load_source_attestation_contract",
]
