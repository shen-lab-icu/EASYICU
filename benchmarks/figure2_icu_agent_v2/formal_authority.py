"""Typed fail-closed authority owner for Figure 2 formal Provider calls.

The frozen launch contract contains the trusted Ed25519 public key.  A formal
call is eligible only when one signed atomic declaration binds the current
protocol digest, release identity, every receipt payload, and the exact call
coordinate.  The review candidate deliberately has no registered key, so it
cannot authorize a call.
"""

from __future__ import annotations

import base64
import binascii
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import re
from typing import Any, Mapping, NoReturn

from .design_errors import DesignContractError


PACKAGE_ROOT = Path(__file__).resolve().parent
PROTOCOL_PATH = PACKAGE_ROOT / "experiment_protocol_v2_1.json"
LAUNCH_CONTRACT_PATH = PACKAGE_ROOT / "formal_launch_contract_v1.json"
PREREGISTRATION_PLAN_PATH = PACKAGE_ROOT / "preregistration_plan_v1.json"

_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_SCOPE_RECEIPT_GROUPS = {
    "qualification12": ("qualification_preconditions",),
    "core_wp2_wp3": (
        "design",
        "data",
        "evaluation",
        "qualification",
        "runtime",
        "batch",
    ),
    "wp5_phase_a": (
        "design",
        "data",
        "evaluation",
        "qualification",
        "runtime",
        "batch",
        "wp5_phase_a",
    ),
    "wp5_phase_b_showcase": (
        "design",
        "data",
        "evaluation",
        "qualification",
        "runtime",
        "batch",
        "wp5_phase_a",
        "wp5_phase_b_showcase",
    ),
}


def _fail(reason_code: str, detail: str) -> NoReturn:
    raise DesignContractError(reason_code, detail)


def _load_json(path: Path) -> dict[str, Any]:
    def reject_duplicates(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                _fail("FORMAL_AUTHORITY_JSON_DUPLICATE_KEY", key)
            result[key] = value
        return result

    try:
        value = json.loads(
            path.read_text(encoding="utf-8"), object_pairs_hook=reject_duplicates
        )
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        _fail("FORMAL_AUTHORITY_CONTRACT_UNREADABLE", str(exc))
    if not isinstance(value, dict):
        _fail("FORMAL_AUTHORITY_CONTRACT_INVALID", str(path))
    return value


def _canonical_json_bytes(value: Any) -> bytes:
    try:
        return json.dumps(
            value,
            ensure_ascii=False,
            allow_nan=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    except (TypeError, ValueError) as exc:
        _fail("FORMAL_AUTHORITY_PAYLOAD_NOT_CANONICALIZABLE", str(exc))


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _require_mapping(value: Any, *, field: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping) or not all(
        isinstance(key, str) for key in value
    ):
        _fail("FORMAL_AUTHORITY_FIELD_INVALID", field)
    return value


def _require_exact_keys(
    value: Mapping[str, Any], *, field: str, expected: set[str]
) -> None:
    actual = set(value)
    if actual != expected:
        _fail(
            "FORMAL_AUTHORITY_FIELD_SET_INVALID",
            f"{field}: missing={sorted(expected - actual)!r}, extra={sorted(actual - expected)!r}",
        )


def _require_nonempty_string(value: Any, *, field: str) -> str:
    if not isinstance(value, str) or not value.strip():
        _fail("FORMAL_AUTHORITY_FIELD_INVALID", field)
    return value.strip()


def _require_sha256(value: Any, *, field: str) -> str:
    text = _require_nonempty_string(value, field=field)
    if not _SHA256_RE.fullmatch(text):
        _fail("FORMAL_AUTHORITY_DIGEST_INVALID", field)
    return text


def _expected_receipts(
    launch: Mapping[str, Any], scope: str
) -> dict[str, str]:
    groups = _SCOPE_RECEIPT_GROUPS.get(scope)
    if groups is None:
        _fail("FORMAL_AUTHORITY_SCOPE_INVALID", scope)
    required_receipts = _require_mapping(
        launch.get("required_receipts"), field="launch.required_receipts"
    )
    receipts: dict[str, str] = {}
    for group in groups:
        descriptions = required_receipts.get(group)
        if not isinstance(descriptions, list) or not descriptions:
            _fail("FORMAL_AUTHORITY_CONTRACT_INVALID", f"required_receipts.{group}")
        for index, description in enumerate(descriptions, start=1):
            if not isinstance(description, str) or not description.strip():
                _fail(
                    "FORMAL_AUTHORITY_CONTRACT_INVALID",
                    f"required_receipts.{group}:{index}",
                )
            receipts[f"{group}:{index:02d}"] = description
    return receipts


def _require_utc_timestamp(value: Any, *, field: str) -> str:
    text = _require_nonempty_string(value, field=field)
    try:
        parsed = datetime.fromisoformat(text.replace("Z", "+00:00"))
    except ValueError:
        _fail("FORMAL_AUTHORITY_TIMESTAMP_INVALID", field)
    if parsed.tzinfo is None or parsed.utcoffset() != timezone.utc.utcoffset(parsed):
        _fail("FORMAL_AUTHORITY_TIMESTAMP_INVALID", field)
    return text


def _validate_registration_details(
    details: Any,
    *,
    binding: Mapping[str, str],
    signer_id: str,
    public_key_text: str,
) -> None:
    registration = _require_mapping(details, field="registration_receipt.details")
    preregistration = _load_json(PREREGISTRATION_PLAN_PATH)
    required_fields = preregistration.get("required_receipt_fields")
    if not isinstance(required_fields, list) or not all(
        isinstance(field, str) and field for field in required_fields
    ):
        _fail("FORMAL_AUTHORITY_CONTRACT_INVALID", "required_receipt_fields")
    missing = sorted(set(required_fields) - set(registration))
    if missing:
        _fail("FORMAL_AUTHORITY_REGISTRATION_RECEIPT_INVALID", repr(missing))
    for field in (
        "registry_name",
        "immutable_registration_id",
        "embargo_or_public_status",
        "registrant_identity",
    ):
        _require_nonempty_string(registration[field], field=f"registration.{field}")
    _require_utc_timestamp(
        registration["registration_timestamp_utc"],
        field="registration.registration_timestamp_utc",
    )
    for field in (
        "package_sha256",
        "protocol_sha256",
        "validator_sha256",
        "validator_test_sha256",
        "formal_authority_sha256",
        "formal_authority_test_sha256",
    ):
        _require_sha256(registration[field], field=f"registration.{field}")
    if registration["protocol_sha256"] != binding["protocol_sha256"]:
        _fail("FORMAL_AUTHORITY_REGISTRATION_BINDING_MISMATCH", "protocol_sha256")
    if registration["design_commit"] != binding["design_commit"]:
        _fail("FORMAL_AUTHORITY_REGISTRATION_BINDING_MISMATCH", "design_commit")
    if registration["annotated_tag"] != binding["annotated_tag"]:
        _fail("FORMAL_AUTHORITY_REGISTRATION_BINDING_MISMATCH", "annotated_tag")
    if registration["trusted_authority_signer_identity"] != signer_id:
        _fail("FORMAL_AUTHORITY_REGISTRATION_BINDING_MISMATCH", "signer_identity")
    if registration["trusted_authority_ed25519_public_key_base64"] != public_key_text:
        _fail("FORMAL_AUTHORITY_REGISTRATION_BINDING_MISMATCH", "public_key")
    if registration["amendment_policy_acknowledged"] is not True:
        _fail(
            "FORMAL_AUTHORITY_REGISTRATION_RECEIPT_INVALID",
            "amendment_policy_acknowledged",
        )


def _validate_receipt_payload(
    receipt: Mapping[str, Any],
    *,
    receipt_id: str,
    description: str,
    binding: Mapping[str, str],
    signer_id: str,
    public_key_text: str,
) -> None:
    required = {
        "schema_version",
        "receipt_id",
        "status",
        "binding",
        "evidence_sha256",
        "issuer",
        "issued_at_utc",
    }
    missing = sorted(required - set(receipt))
    if missing:
        _fail("FORMAL_AUTHORITY_RECEIPT_SCHEMA_INVALID", f"{receipt_id}: {missing!r}")
    if receipt["schema_version"] != "easyicu.figure2_launch_receipt/1":
        _fail("FORMAL_AUTHORITY_RECEIPT_SCHEMA_INVALID", receipt_id)
    if receipt["receipt_id"] != receipt_id or receipt["status"] != "passed":
        _fail("FORMAL_AUTHORITY_RECEIPT_NOT_PASSING", receipt_id)
    if receipt["binding"] != binding:
        _fail("FORMAL_AUTHORITY_RECEIPT_BINDING_MISMATCH", receipt_id)
    _require_sha256(
        receipt["evidence_sha256"], field=f"receipt_payloads.{receipt_id}.evidence_sha256"
    )
    _require_nonempty_string(
        receipt["issuer"], field=f"receipt_payloads.{receipt_id}.issuer"
    )
    _require_utc_timestamp(
        receipt["issued_at_utc"], field=f"receipt_payloads.{receipt_id}.issued_at_utc"
    )
    if "external registration receipt" in description or (
        "external preregistration receipt" in description
    ):
        _validate_registration_details(
            receipt.get("details"),
            binding=binding,
            signer_id=signer_id,
            public_key_text=public_key_text,
        )


def _validate_coordinate(value: Any, *, expected: Mapping[str, Any]) -> None:
    coordinate = _require_mapping(value, field="authorized_call_coordinate")
    required = {"scope", "task_id", "arm", "call_id"}
    _require_exact_keys(
        coordinate, field="authorized_call_coordinate", expected=required
    )
    normalized = {
        key: _require_nonempty_string(coordinate.get(key), field=f"coordinate.{key}")
        for key in sorted(required)
    }
    expected_normalized = {
        key: _require_nonempty_string(expected.get(key), field=f"requested.{key}")
        for key in sorted(required)
    }
    if normalized != expected_normalized:
        _fail(
            "FORMAL_AUTHORITY_COORDINATE_MISMATCH",
            f"requested={expected_normalized!r}",
        )


def _verify_signature(
    declaration: Mapping[str, Any], *, signature_text: Any, public_key_text: str
) -> None:
    try:
        public_key_bytes = base64.b64decode(public_key_text, validate=True)
        signature = base64.b64decode(
            _require_nonempty_string(signature_text, field="signature"),
            validate=True,
        )
    except (binascii.Error, ValueError) as exc:
        _fail("FORMAL_AUTHORITY_SIGNATURE_ENCODING_INVALID", str(exc))
    if len(public_key_bytes) != 32 or len(signature) != 64:
        _fail("FORMAL_AUTHORITY_SIGNATURE_ENCODING_INVALID", "Ed25519 length")
    try:
        from cryptography.exceptions import InvalidSignature
        from cryptography.hazmat.primitives.asymmetric.ed25519 import (
            Ed25519PublicKey,
        )
    except ImportError:
        _fail("FORMAL_AUTHORITY_CRYPTO_UNAVAILABLE", "cryptography")
    try:
        Ed25519PublicKey.from_public_bytes(public_key_bytes).verify(
            signature, _canonical_json_bytes(declaration)
        )
    except InvalidSignature:
        _fail("FORMAL_AUTHORITY_SIGNATURE_INVALID", "atomic declaration")
    except ValueError as exc:
        _fail("FORMAL_AUTHORITY_PUBLIC_KEY_INVALID", str(exc))


def authorize_formal_provider_call(authority_payload: Mapping[str, Any]) -> dict[str, Any]:
    """Validate a signed atomic declaration for one exact formal call."""

    launch = _load_json(LAUNCH_CONTRACT_PATH)
    signature_contract = _require_mapping(
        launch.get("signature_verification"), field="launch.signature_verification"
    )
    if signature_contract.get("algorithm") != "Ed25519":
        _fail("FORMAL_AUTHORITY_ALGORITHM_INVALID", repr(signature_contract))
    signer_id = signature_contract.get("trusted_signer_id")
    public_key_text = signature_contract.get("trusted_public_key_base64")
    if not isinstance(signer_id, str) or not signer_id.strip() or not isinstance(
        public_key_text, str
    ) or not public_key_text.strip():
        reason = launch.get("current_authority", {}).get(
            "reason", "no trusted signer is registered"
        )
        _fail("FORMAL_AUTHORITY_SIGNER_NOT_REGISTERED", str(reason))

    payload = _require_mapping(authority_payload, field="authority_payload")
    _require_exact_keys(
        payload,
        field="authority_payload",
        expected={"receipts", "call_coordinate"},
    )
    requested_coordinate = _require_mapping(
        payload["call_coordinate"], field="call_coordinate"
    )
    envelope = _require_mapping(payload["receipts"], field="receipts")
    _require_exact_keys(
        envelope,
        field="receipts",
        expected={
            "atomic_declaration",
            "atomic_declaration_signature_base64",
            "receipt_payloads",
        },
    )
    declaration = _require_mapping(
        envelope["atomic_declaration"], field="atomic_declaration"
    )
    declaration_fields = {
        "schema_version",
        "signer_id",
        "scope",
        "protocol_sha256",
        "design_commit",
        "annotated_tag",
        "receipt_sha256",
        "authorized_call_coordinates",
    }
    _require_exact_keys(
        declaration,
        field="atomic_declaration",
        expected=declaration_fields,
    )
    if declaration["schema_version"] != "easyicu.figure2_atomic_declaration/1":
        _fail(
            "FORMAL_AUTHORITY_DECLARATION_SCHEMA_INVALID",
            repr(declaration["schema_version"]),
        )
    if declaration["signer_id"] != signer_id:
        _fail("FORMAL_AUTHORITY_SIGNER_MISMATCH", repr(declaration["signer_id"]))

    scope = _require_nonempty_string(declaration["scope"], field="scope")
    if requested_coordinate.get("scope") != scope:
        _fail("FORMAL_AUTHORITY_SCOPE_MISMATCH", scope)
    protocol_sha256 = _require_sha256(
        declaration["protocol_sha256"], field="protocol_sha256"
    )
    current_protocol_sha256 = _sha256_bytes(PROTOCOL_PATH.read_bytes())
    if protocol_sha256 != current_protocol_sha256:
        _fail(
            "FORMAL_AUTHORITY_PROTOCOL_DIGEST_MISMATCH",
            f"expected={current_protocol_sha256}, signed={protocol_sha256}",
        )
    design_commit = _require_nonempty_string(
        declaration["design_commit"], field="design_commit"
    )
    annotated_tag = _require_nonempty_string(
        declaration["annotated_tag"], field="annotated_tag"
    )

    coordinates = declaration["authorized_call_coordinates"]
    if not isinstance(coordinates, list) or not coordinates:
        _fail("FORMAL_AUTHORITY_FIELD_INVALID", "authorized_call_coordinates")
    canonical_coordinates = [_canonical_json_bytes(item) for item in coordinates]
    if len(canonical_coordinates) != len(set(canonical_coordinates)):
        _fail("FORMAL_AUTHORITY_COORDINATE_DUPLICATE", scope)
    if not any(
        _canonical_json_bytes(item) == _canonical_json_bytes(requested_coordinate)
        for item in coordinates
    ):
        _fail("FORMAL_AUTHORITY_COORDINATE_NOT_DECLARED", repr(requested_coordinate))
    for coordinate in coordinates:
        coordinate_mapping = _require_mapping(
            coordinate, field="authorized_call_coordinates[]"
        )
        _validate_coordinate(coordinate_mapping, expected=coordinate_mapping)
        if coordinate_mapping.get("scope") != scope:
            _fail("FORMAL_AUTHORITY_SCOPE_MISMATCH", repr(coordinate_mapping))
    _validate_coordinate(requested_coordinate, expected=requested_coordinate)

    expected_receipts = _expected_receipts(launch, scope)
    expected_receipt_ids = set(expected_receipts)
    signed_digests = _require_mapping(
        declaration["receipt_sha256"], field="receipt_sha256"
    )
    receipt_payloads = _require_mapping(
        envelope["receipt_payloads"], field="receipt_payloads"
    )
    if set(signed_digests) != expected_receipt_ids or set(receipt_payloads) != expected_receipt_ids:
        _fail(
            "FORMAL_AUTHORITY_RECEIPT_SET_MISMATCH",
            f"expected={sorted(expected_receipt_ids)!r}",
        )
    expected_binding = {
        "protocol_sha256": protocol_sha256,
        "design_commit": design_commit,
        "annotated_tag": annotated_tag,
    }
    for receipt_id in sorted(expected_receipt_ids):
        signed_digest = _require_sha256(
            signed_digests[receipt_id], field=f"receipt_sha256.{receipt_id}"
        )
        receipt = _require_mapping(
            receipt_payloads[receipt_id], field=f"receipt_payloads.{receipt_id}"
        )
        _validate_receipt_payload(
            receipt,
            receipt_id=receipt_id,
            description=expected_receipts[receipt_id],
            binding=expected_binding,
            signer_id=signer_id,
            public_key_text=public_key_text,
        )
        actual_digest = _sha256_bytes(_canonical_json_bytes(receipt))
        if actual_digest != signed_digest:
            _fail("FORMAL_AUTHORITY_RECEIPT_DIGEST_MISMATCH", receipt_id)

    _verify_signature(
        declaration,
        signature_text=envelope["atomic_declaration_signature_base64"],
        public_key_text=public_key_text,
    )
    return {
        "authorized": True,
        "scope": scope,
        "call_coordinate": dict(requested_coordinate),
        "protocol_sha256": protocol_sha256,
        "signer_id": signer_id,
        "receipt_count": len(expected_receipt_ids),
    }


__all__ = ["authorize_formal_provider_call"]
