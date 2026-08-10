"""Digest-bound E2/H2/H3 clinical-and-methods protocol authority.

This module verifies documentary review authority only.  It never creates an
attestation, infers reviewer approval, reads cohort data, launches Docker, or
calls a Provider.  A formal Canonical9 run may proceed only when the operator
supplies three already-reviewed KnowHow cards whose exact bytes and reviewable
content are frozen in one ordered authority document.
"""

from __future__ import annotations

import hashlib
import json
import os
import stat
from pathlib import Path
from typing import Annotated, Literal, Sequence

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from easyicu.research_agent.know_how.registry import (
    KnowHowCard,
    reviewable_card_content_sha256,
)

from .case_scientific_protocol import (
    case_protocol_content_sha256,
    load_case_scientific_protocol,
)

SCIENTIFIC_PROTOCOL_AUTHORITY_SCHEMA = (
    "easyicu.figure2_scientific_protocol_authority/2"
)
REQUIRED_SCIENTIFIC_PROTOCOLS: tuple[tuple[str, str], ...] = (
    ("e2_lactate_mortality", "early_peak_lactate_association"),
    ("h2_vasopressor_causal", "vasopressor_comparative_effectiveness"),
    ("h3_trajectory_clustering", "longitudinal_icu_phenotyping"),
)

_MAX_AUTHORITY_BYTES = 256 * 1024
_MAX_CARD_BYTES = 512 * 1024
_MAX_PROTOCOL_BYTES = 256 * 1024
_SHA256 = Annotated[str, Field(pattern=r"^[0-9a-f]{64}$")]


class ScientificProtocolAuthorityError(ValueError):
    """A scientific protocol authority or reviewed card is unsafe or invalid."""


class _StrictFrozenModel(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)


def _canonical_json_bytes(value: object) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")


def _reject_duplicate_pairs(
    pairs: Sequence[tuple[str, object]],
) -> dict[str, object]:
    decoded: dict[str, object] = {}
    for key, value in pairs:
        if key in decoded:
            raise ScientificProtocolAuthorityError(f"duplicate JSON key {key!r}")
        decoded[key] = value
    return decoded


def _reject_nonfinite(value: str) -> object:
    raise ScientificProtocolAuthorityError(f"non-finite JSON constant {value!r}")


def _read_regular_file(path: Path | str, *, byte_limit: int) -> tuple[Path, bytes]:
    candidate = Path(path).expanduser()
    if not candidate.is_absolute() or candidate.is_symlink():
        raise ScientificProtocolAuthorityError(
            "scientific protocol paths must be absolute and non-symlink"
        )
    fd: int | None = None
    try:
        fd = os.open(
            candidate,
            os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0),
        )
        metadata = os.fstat(fd)
        if not stat.S_ISREG(metadata.st_mode):
            raise ScientificProtocolAuthorityError(
                "scientific protocol input must be a regular file"
            )
        if metadata.st_size > byte_limit:
            raise ScientificProtocolAuthorityError(
                "scientific protocol input exceeds its size limit"
            )
        chunks: list[bytes] = []
        while chunk := os.read(fd, 64 * 1024):
            chunks.append(chunk)
        resolved = candidate.resolve(strict=True)
        return resolved, b"".join(chunks)
    except OSError as exc:
        raise ScientificProtocolAuthorityError(
            f"cannot read scientific protocol input: {candidate}"
        ) from exc
    finally:
        if fd is not None:
            os.close(fd)


def _strict_json_object(raw: bytes, *, label: str) -> dict[str, object]:
    try:
        decoded = json.loads(
            raw.decode("utf-8"),
            object_pairs_hook=_reject_duplicate_pairs,
            parse_constant=_reject_nonfinite,
        )
    except (UnicodeDecodeError, json.JSONDecodeError, TypeError, ValueError) as exc:
        raise ScientificProtocolAuthorityError(f"{label} is not strict JSON") from exc
    if not isinstance(decoded, dict):
        raise ScientificProtocolAuthorityError(f"{label} must be a JSON object")
    return decoded


class ScientificProtocolTaskBinding(_StrictFrozenModel):
    """One reviewed task pinned by card and case-protocol content."""

    task_id: str
    card_id: str
    card_version: str
    card_path: str
    card_file_sha256: _SHA256
    reviewed_content_sha256: _SHA256
    protocol_path: str
    protocol_file_sha256: _SHA256
    protocol_content_sha256: _SHA256

    @field_validator("card_path", "protocol_path")
    @classmethod
    def _absolute_review_path(cls, value: str) -> str:
        candidate = Path(str(value or "")).expanduser()
        if not candidate.is_absolute():
            raise ValueError("reviewed card and protocol paths must be absolute")
        return str(candidate)


class ScientificProtocolAuthority(_StrictFrozenModel):
    """Exact ordered E2/H2/H3 protocol bindings selected by the operator."""

    schema_version: Literal["easyicu.figure2_scientific_protocol_authority/2"]
    tasks: tuple[ScientificProtocolTaskBinding, ...]
    authority_digest: _SHA256

    @model_validator(mode="after")
    def _exact_protocols_and_digest(self) -> "ScientificProtocolAuthority":
        observed = tuple((task.task_id, task.card_id) for task in self.tasks)
        if observed != REQUIRED_SCIENTIFIC_PROTOCOLS:
            raise ValueError(
                "scientific protocol authority must bind exact ordered E2/H2/H3 cards"
            )
        body = self.model_dump(mode="json", exclude={"authority_digest"})
        actual = hashlib.sha256(_canonical_json_bytes(body)).hexdigest()
        if actual != self.authority_digest:
            raise ValueError("scientific protocol authority digest mismatch")
        return self

    @classmethod
    def build(
        cls,
        *,
        tasks: Sequence[ScientificProtocolTaskBinding],
    ) -> "ScientificProtocolAuthority":
        body = {
            "schema_version": SCIENTIFIC_PROTOCOL_AUTHORITY_SCHEMA,
            "tasks": [task.model_dump(mode="json") for task in tasks],
        }
        return cls(
            schema_version=SCIENTIFIC_PROTOCOL_AUTHORITY_SCHEMA,
            tasks=tuple(tasks),
            authority_digest=hashlib.sha256(_canonical_json_bytes(body)).hexdigest(),
        )


def _verify_reviewed_card(
    binding: ScientificProtocolTaskBinding,
) -> None:
    resolved, raw = _read_regular_file(binding.card_path, byte_limit=_MAX_CARD_BYTES)
    if str(resolved) != str(Path(binding.card_path)):
        raise ScientificProtocolAuthorityError(
            f"{binding.task_id} reviewed card path changed during resolution"
        )
    file_sha256 = hashlib.sha256(raw).hexdigest()
    if file_sha256 != binding.card_file_sha256:
        raise ScientificProtocolAuthorityError(
            f"{binding.task_id} reviewed card file digest mismatch"
        )
    payload = _strict_json_object(raw, label=f"{binding.task_id} reviewed card")
    try:
        card = KnowHowCard.model_validate(payload, strict=True)
    except Exception as exc:  # noqa: BLE001
        raise ScientificProtocolAuthorityError(
            f"{binding.task_id} reviewed card schema is invalid: {exc}"
        ) from exc
    if card.card_id != binding.card_id or card.version != binding.card_version:
        raise ScientificProtocolAuthorityError(
            f"{binding.task_id} reviewed card identity mismatch"
        )
    if card.trust_level not in {"built_in_reviewed", "project_reviewed"}:
        raise ScientificProtocolAuthorityError(
            f"{binding.task_id} reviewed card has no trusted review owner"
        )
    if card.review_status != "clinical_reviewed" or card.review_attestation is None:
        raise ScientificProtocolAuthorityError(
            f"{binding.task_id} lacks formal clinical-and-methods attestation"
        )
    attestation = card.review_attestation
    if not (attestation.clinical_reviewed and attestation.methods_reviewed):
        raise ScientificProtocolAuthorityError(
            f"{binding.task_id} lacks dual clinical-and-methods approval"
        )
    if attestation.card_version != card.version:
        raise ScientificProtocolAuthorityError(
            f"{binding.task_id} review attestation version mismatch"
        )
    content_sha256 = reviewable_card_content_sha256(payload)
    if (
        content_sha256 != attestation.reviewed_content_sha256
        or content_sha256 != binding.reviewed_content_sha256
    ):
        raise ScientificProtocolAuthorityError(
            f"{binding.task_id} review attestation content digest mismatch"
        )
    protocol_resolved, protocol_raw = _read_regular_file(
        binding.protocol_path,
        byte_limit=_MAX_PROTOCOL_BYTES,
    )
    if str(protocol_resolved) != str(Path(binding.protocol_path)):
        raise ScientificProtocolAuthorityError(
            f"{binding.task_id} reviewed case protocol path changed during resolution"
        )
    if hashlib.sha256(protocol_raw).hexdigest() != binding.protocol_file_sha256:
        raise ScientificProtocolAuthorityError(
            f"{binding.task_id} reviewed case protocol file digest mismatch"
        )
    try:
        protocol = load_case_scientific_protocol(
            protocol_resolved,
            expected_task_id=binding.task_id,
        )
    except Exception as exc:  # noqa: BLE001
        raise ScientificProtocolAuthorityError(
            f"{binding.task_id} reviewed case protocol is invalid: {exc}"
        ) from exc
    protocol_sha256 = case_protocol_content_sha256(protocol)
    if (
        protocol_sha256 != binding.protocol_content_sha256
        or protocol_sha256 != attestation.protocol_content_sha256
    ):
        raise ScientificProtocolAuthorityError(
            f"{binding.task_id} review attestation protocol digest mismatch"
        )


def load_verified_scientific_protocol_authority(
    path: Path | str,
) -> tuple[ScientificProtocolAuthority, str]:
    """Strict-load the authority and verify all three reviewed card files."""

    _resolved, raw = _read_regular_file(path, byte_limit=_MAX_AUTHORITY_BYTES)
    _strict_json_object(raw, label="scientific protocol authority")
    try:
        # Validate from JSON so an on-disk JSON array can satisfy the immutable
        # tuple field while all scalar and object types remain strict.
        authority = ScientificProtocolAuthority.model_validate_json(raw, strict=True)
    except Exception as exc:  # noqa: BLE001
        raise ScientificProtocolAuthorityError(
            f"scientific protocol authority schema is invalid: {exc}"
        ) from exc
    for task in authority.tasks:
        _verify_reviewed_card(task)
    return authority, hashlib.sha256(raw).hexdigest()


__all__ = [
    "REQUIRED_SCIENTIFIC_PROTOCOLS",
    "SCIENTIFIC_PROTOCOL_AUTHORITY_SCHEMA",
    "ScientificProtocolAuthority",
    "ScientificProtocolAuthorityError",
    "ScientificProtocolTaskBinding",
    "load_verified_scientific_protocol_authority",
]
