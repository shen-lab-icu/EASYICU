"""Fail-closed two-host parity acceptance for Figure 2 formal execution."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import re
from typing import Any, Mapping, Sequence


CONTRACT_PATH = Path(__file__).with_name("execution_acceptance_contract_v1.json")
PROTOCOL_PATH = Path(__file__).with_name("experiment_protocol_v2_1.json")
_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_COMMIT_RE = re.compile(r"^[0-9a-f]{40}$")


class MultiHostAcceptanceError(ValueError):
    reason_code = "MULTI_HOST_ACCEPTANCE_INVALID"


def _load_json(path: Path) -> dict[str, Any]:
    def reject_duplicates(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        value: dict[str, Any] = {}
        for key, item in pairs:
            if key in value:
                raise MultiHostAcceptanceError(f"duplicate JSON key: {key}")
            value[key] = item
        return value

    try:
        value = json.loads(
            path.read_text(encoding="utf-8"),
            object_pairs_hook=reject_duplicates,
        )
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise MultiHostAcceptanceError(f"unreadable JSON: {path}") from exc
    if not isinstance(value, dict):
        raise MultiHostAcceptanceError(f"expected JSON object: {path}")
    return value


def _canonical_json(value: Mapping[str, Any]) -> bytes:
    try:
        return json.dumps(
            dict(value),
            ensure_ascii=False,
            allow_nan=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    except (TypeError, ValueError) as exc:
        raise MultiHostAcceptanceError("receipt is not canonicalizable") from exc


def _require_sha256(value: Any, field: str) -> str:
    if not isinstance(value, str) or not _SHA256_RE.fullmatch(value):
        raise MultiHostAcceptanceError(f"{field} must be a lowercase SHA-256")
    return value


def validate_two_host_preflight(
    receipts: Sequence[Mapping[str, Any]],
    *,
    expected_design_commit: str,
    expected_annotated_tag: str,
) -> dict[str, Any]:
    """Return a no-Provider GO receipt only for two parity-matched hosts."""

    contract = _load_json(CONTRACT_PATH)
    logical_sites = tuple(contract["logical_sites"])
    if logical_sites != ("server", "laptop") or len(receipts) != 2:
        raise MultiHostAcceptanceError("exactly server and laptop receipts are required")
    if not _COMMIT_RE.fullmatch(expected_design_commit):
        raise MultiHostAcceptanceError("expected design commit is invalid")
    if not isinstance(expected_annotated_tag, str) or not expected_annotated_tag.strip():
        raise MultiHostAcceptanceError("expected annotated tag is invalid")

    matched_fields = tuple(contract["matched_site_fields"])
    boolean_expectations = contract["site_preflight_boolean_expectations"]
    required = {
        "schema_version",
        "logical_site",
        "host_fingerprint_sha256",
        "clock_offset_ms",
        *matched_fields,
        *boolean_expectations,
    }
    by_site: dict[str, Mapping[str, Any]] = {}
    for receipt in receipts:
        if set(receipt) != required:
            raise MultiHostAcceptanceError(
                "site receipt fields do not match the frozen schema"
            )
        if receipt["schema_version"] != "easyicu.figure2_site_preflight/1":
            raise MultiHostAcceptanceError("site receipt schema is invalid")
        site = receipt["logical_site"]
        if site not in logical_sites or site in by_site:
            raise MultiHostAcceptanceError("logical site identity is invalid or duplicate")
        _require_sha256(receipt["host_fingerprint_sha256"], "host_fingerprint_sha256")
        if receipt["design_commit"] != expected_design_commit:
            raise MultiHostAcceptanceError(f"{site} design commit mismatch")
        if receipt["annotated_tag"] != expected_annotated_tag:
            raise MultiHostAcceptanceError(f"{site} annotated tag mismatch")
        clock_offset = receipt["clock_offset_ms"]
        if type(clock_offset) not in {int, float} or abs(clock_offset) > contract[
            "maximum_absolute_clock_offset_ms"
        ]:
            raise MultiHostAcceptanceError(f"{site} clock offset exceeds the limit")
        for field, expected in boolean_expectations.items():
            if receipt[field] is not expected:
                raise MultiHostAcceptanceError(f"{site} failed {field}")
        for field in (
            "package_lock_sha256",
            "provider_route_sha256",
            "sampling_policy_sha256",
            "runtime_budget_sha256",
            "network_policy_sha256",
            "input_manifest_set_sha256",
        ):
            _require_sha256(receipt[field], field)
        if not isinstance(receipt["container_image_digest"], str) or not re.fullmatch(
            r"sha256:[0-9a-f]{64}", receipt["container_image_digest"]
        ):
            raise MultiHostAcceptanceError("container image digest is invalid")
        if not isinstance(receipt["immutable_model_identifier"], str) or not receipt[
            "immutable_model_identifier"
        ].strip():
            raise MultiHostAcceptanceError("immutable model identifier is invalid")
        for field in ("cpu_limit", "memory_limit_bytes", "pids_limit"):
            if type(receipt[field]) not in {int, float} or receipt[field] <= 0:
                raise MultiHostAcceptanceError(f"{field} must be positive")
        by_site[site] = receipt

    if set(by_site) != set(logical_sites):
        raise MultiHostAcceptanceError("both logical sites must be present")
    if len({receipt["host_fingerprint_sha256"] for receipt in by_site.values()}) != 2:
        raise MultiHostAcceptanceError("server and laptop fingerprints must be distinct")
    reference = by_site[logical_sites[0]]
    mismatches = [
        field
        for field in matched_fields
        if any(receipt[field] != reference[field] for receipt in by_site.values())
    ]
    if mismatches:
        raise MultiHostAcceptanceError(
            "matched site fields differ: " + ", ".join(mismatches)
        )

    site_receipt_sha256 = {
        site: hashlib.sha256(_canonical_json(by_site[site])).hexdigest()
        for site in logical_sites
    }
    matched_runtime = {field: reference[field] for field in matched_fields}
    return {
        "schema_version": "easyicu.figure2_two_host_acceptance/1",
        "status": "passed",
        "provider_accessed": False,
        "protocol_sha256": hashlib.sha256(PROTOCOL_PATH.read_bytes()).hexdigest(),
        "contract_sha256": hashlib.sha256(CONTRACT_PATH.read_bytes()).hexdigest(),
        "logical_sites": list(logical_sites),
        "site_receipt_sha256": site_receipt_sha256,
        "matched_runtime_sha256": hashlib.sha256(
            _canonical_json(matched_runtime)
        ).hexdigest(),
    }


__all__ = ["MultiHostAcceptanceError", "validate_two_host_preflight"]
