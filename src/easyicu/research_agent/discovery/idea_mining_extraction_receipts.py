"""Content-bound checkpoints for batched literature-idea extraction.

The model is asked to emit JSON for a small group of source articles.  A
single malformed response must not make already parsed groups disappear or be
silently "repaired" into scientific content.  This leaf module persists the
verbatim response together with the exact request digest.  Only receipts whose
request, response, and receipt digests all verify are eligible for reuse.
"""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
from pathlib import Path
from typing import Any, Literal, Mapping, Sequence

from ..providers.protocol import LLMMessage

EXTRACTION_BATCH_RECEIPT_SCHEMA_VERSION = (
    "easyicu.idea_mining_extraction_batch_receipt/1"
)


class ExtractionBatchReceiptIntegrityError(RuntimeError):
    """Raised when a persisted extraction checkpoint fails verification."""


def _canonical_json(payload: Any) -> str:
    return json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )


def _sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def extraction_batch_request(
    *,
    source_snapshot_id: str,
    batch_index: int,
    citation_keys: Sequence[str],
    messages: Sequence[LLMMessage],
    max_tokens: int,
    temperature: float,
    provider_name: str,
) -> dict[str, Any]:
    """Return the canonical request coordinates used to bind a receipt."""

    return {
        "source_snapshot_id": str(source_snapshot_id),
        "batch_index": int(batch_index),
        "citation_keys": [str(key) for key in citation_keys],
        "messages": [
            {"role": str(message.role), "content": str(message.content)}
            for message in messages
        ],
        "max_tokens": int(max_tokens),
        "temperature": float(temperature),
        "provider_name": str(provider_name),
    }


def extraction_batch_request_sha256(request: Mapping[str, Any]) -> str:
    return _sha256_text(_canonical_json(dict(request)))


def persist_extraction_batch_receipt(
    receipt_dir: str | Path,
    *,
    request: Mapping[str, Any],
    raw_response: str,
    parse_status: Literal["parsed", "malformed"],
    parse_error: str | None = None,
) -> Path:
    """Persist one immutable, digest-verified provider response receipt."""

    root = Path(receipt_dir)
    root.mkdir(parents=True, exist_ok=True)
    request_sha256 = extraction_batch_request_sha256(request)
    response = str(raw_response)
    response_sha256 = _sha256_text(response)
    payload: dict[str, Any] = {
        "schema_version": EXTRACTION_BATCH_RECEIPT_SCHEMA_VERSION,
        "request": dict(request),
        "request_sha256": request_sha256,
        "raw_response": response,
        "response_sha256": response_sha256,
        "parse_status": str(parse_status),
        "parse_error": str(parse_error) if parse_error else None,
    }
    payload["receipt_sha256"] = _sha256_text(_canonical_json(payload))
    batch_index = int(request["batch_index"])
    path = root / (
        f"batch_{batch_index:03d}_{request_sha256[:16]}_"
        f"{parse_status}_{response_sha256[:16]}.json"
    )
    encoded = json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True)
    if path.exists():
        if path.read_text(encoding="utf-8") != encoded:
            raise ExtractionBatchReceiptIntegrityError(
                f"existing extraction receipt differs: {path}"
            )
        return path
    fd, temp_name = tempfile.mkstemp(prefix=f".{path.name}.", dir=root)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            handle.write(encoded)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temp_name, path)
    finally:
        try:
            Path(temp_name).unlink(missing_ok=True)
        except OSError:
            pass
    return path


def load_verified_parsed_extraction_response(
    receipt_dir: str | Path,
    *,
    request: Mapping[str, Any],
) -> str | None:
    """Load a parsed response for this exact request, or return ``None``.

    Malformed receipts are evidence but are never reusable.  A matching parsed
    receipt is reused only after strict schema and digest verification.  More
    than one distinct parsed response for one request is ambiguous and fails
    closed rather than choosing a scientifically convenient answer.
    """

    root = Path(receipt_dir)
    if not root.exists():
        return None
    request_sha256 = extraction_batch_request_sha256(request)
    batch_index = int(request["batch_index"])
    paths = sorted(
        root.glob(f"batch_{batch_index:03d}_{request_sha256[:16]}_parsed_*.json")
    )
    if not paths:
        return None
    responses: dict[str, str] = {}
    for path in paths:
        payload = _load_verified_receipt(path)
        if payload["parse_status"] != "parsed":
            raise ExtractionBatchReceiptIntegrityError(
                f"parsed receipt filename has non-parsed status: {path}"
            )
        if payload["request_sha256"] != request_sha256:
            raise ExtractionBatchReceiptIntegrityError(
                f"extraction request digest mismatch: {path}"
            )
        if payload["request"] != dict(request):
            raise ExtractionBatchReceiptIntegrityError(
                f"extraction request coordinates mismatch: {path}"
            )
        responses[payload["response_sha256"]] = payload["raw_response"]
    if len(responses) != 1:
        raise ExtractionBatchReceiptIntegrityError(
            "multiple distinct parsed responses exist for one extraction request"
        )
    return next(iter(responses.values()))


def _load_verified_receipt(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ExtractionBatchReceiptIntegrityError(
            f"cannot read extraction receipt: {path}"
        ) from exc
    expected_keys = {
        "schema_version",
        "request",
        "request_sha256",
        "raw_response",
        "response_sha256",
        "parse_status",
        "parse_error",
        "receipt_sha256",
    }
    if not isinstance(payload, dict) or set(payload) != expected_keys:
        raise ExtractionBatchReceiptIntegrityError(
            f"invalid extraction receipt fields: {path}"
        )
    if payload["schema_version"] != EXTRACTION_BATCH_RECEIPT_SCHEMA_VERSION:
        raise ExtractionBatchReceiptIntegrityError(
            f"unsupported extraction receipt schema: {path}"
        )
    receipt_sha256 = str(payload.pop("receipt_sha256"))
    if _sha256_text(_canonical_json(payload)) != receipt_sha256:
        raise ExtractionBatchReceiptIntegrityError(
            f"extraction receipt digest mismatch: {path}"
        )
    payload["receipt_sha256"] = receipt_sha256
    request = payload["request"]
    if not isinstance(request, dict):
        raise ExtractionBatchReceiptIntegrityError(
            f"extraction receipt request must be an object: {path}"
        )
    if extraction_batch_request_sha256(request) != payload["request_sha256"]:
        raise ExtractionBatchReceiptIntegrityError(
            f"extraction request digest mismatch: {path}"
        )
    response = payload["raw_response"]
    if not isinstance(response, str):
        raise ExtractionBatchReceiptIntegrityError(
            f"extraction response must be text: {path}"
        )
    if _sha256_text(response) != payload["response_sha256"]:
        raise ExtractionBatchReceiptIntegrityError(
            f"extraction response digest mismatch: {path}"
        )
    if payload["parse_status"] not in {"parsed", "malformed"}:
        raise ExtractionBatchReceiptIntegrityError(
            f"invalid extraction parse status: {path}"
        )
    return payload


__all__ = [
    "EXTRACTION_BATCH_RECEIPT_SCHEMA_VERSION",
    "ExtractionBatchReceiptIntegrityError",
    "extraction_batch_request",
    "extraction_batch_request_sha256",
    "load_verified_parsed_extraction_response",
    "persist_extraction_batch_receipt",
]
