"""Arm-neutral Figure 2 review-bundle normalization.

This owner performs only deterministic projection and identifier redaction.
It never repairs or recomputes scientific content.  Inputs that cannot be
normalized under the closed rules fail before a reviewer bundle is emitted.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import re
from typing import Any, Mapping


CANONICAL_FILES = (
    "01_plan.json",
    "02_cohort.json",
    "03_results.json",
    "04_diagnostics.json",
    "05_evidence_manifest.json",
    "06_report.md",
    "07_run_receipt.json",
)
_SCIENTIFIC_JSON_FILES = frozenset(CANONICAL_FILES[:5])
_RECEIPT_VISIBLE_FIELDS = (
    "terminal_status",
    "within_frozen_budget",
    "failure_category",
    "mandatory_artifact_presence",
)
_REDACTIONS = (
    ("easyicu_name", re.compile(r"\bEasyICU\b", re.IGNORECASE), "the producing workflow"),
    (
        "arm_label",
        re.compile(r"\b(?:easyicu_full|generic_code_agent)\b", re.IGNORECASE),
        "the producing workflow",
    ),
    (
        "repository_path",
        re.compile(r"(?:/Users/[^\s\]\[(){}<>'\"]+|\bsrc/easyicu/[^\s\]\[(){}<>'\"]*)"),
        "<redacted-path>",
    ),
)
_REJECTED_MARKERS = (
    re.compile(r"\b(?:FORMAL_PROVIDER|WP5_|SAFETY12_|EASYICU_)[A-Z0-9_]*\b"),
    re.compile(r"\bI\d{2}_[A-Z][A-Z0-9_]+\b"),
)


class ReviewBundleNormalizationError(ValueError):
    def __init__(self, reason_code: str, detail: str) -> None:
        self.reason_code = reason_code
        self.detail = detail
        super().__init__(f"{reason_code}: {detail}")


@dataclass(frozen=True)
class NormalizedReviewBundle:
    files: Mapping[str, bytes]
    pre_normalization_sha256: Mapping[str, str]
    post_normalization_sha256: Mapping[str, str]
    redaction_log: tuple[Mapping[str, str], ...]


def _sha256(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _load_json_object(raw: bytes, name: str) -> dict[str, Any]:
    def reject_constant(value: str) -> None:
        raise ReviewBundleNormalizationError(
            "REVIEW_BUNDLE_NONFINITE_JSON",
            f"{name}: {value}",
        )

    try:
        value = json.loads(raw.decode("utf-8"), parse_constant=reject_constant)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ReviewBundleNormalizationError(
            "REVIEW_BUNDLE_JSON_INVALID",
            name,
        ) from exc
    if not isinstance(value, dict):
        raise ReviewBundleNormalizationError(
            "REVIEW_BUNDLE_JSON_SHAPE_INVALID",
            name,
        )
    return value


def _redact_text(text: str, *, file_name: str, location: str) -> tuple[str, list[dict[str, str]]]:
    output = text
    findings: list[dict[str, str]] = []
    for rule, pattern, replacement in _REDACTIONS:
        output, count = pattern.subn(replacement, output)
        if count:
            findings.append(
                {
                    "file": file_name,
                    "location": location,
                    "rule": rule,
                    "replacement_count": str(count),
                }
            )
    for pattern in _REJECTED_MARKERS:
        if pattern.search(output):
            raise ReviewBundleNormalizationError(
                "REVIEW_BUNDLE_UNSAFE_MARKER",
                f"{file_name}:{location}",
            )
    return output, findings


def _normalize_value(
    value: Any,
    *,
    file_name: str,
    location: str,
) -> tuple[Any, list[dict[str, str]]]:
    if isinstance(value, str):
        return _redact_text(value, file_name=file_name, location=location)
    if isinstance(value, list):
        normalized: list[Any] = []
        findings: list[dict[str, str]] = []
        for index, item in enumerate(value):
            result, item_findings = _normalize_value(
                item,
                file_name=file_name,
                location=f"{location}/{index}",
            )
            normalized.append(result)
            findings.extend(item_findings)
        return normalized, findings
    if isinstance(value, dict):
        normalized_dict: dict[str, Any] = {}
        findings = []
        for key, item in value.items():
            if not isinstance(key, str):
                raise ReviewBundleNormalizationError(
                    "REVIEW_BUNDLE_JSON_KEY_INVALID",
                    f"{file_name}:{location}",
                )
            normalized_key, key_findings = _redact_text(
                key,
                file_name=file_name,
                location=f"{location}/<key>",
            )
            if normalized_key in normalized_dict:
                raise ReviewBundleNormalizationError(
                    "REVIEW_BUNDLE_REDACTION_KEY_COLLISION",
                    f"{file_name}:{location}/{normalized_key}",
                )
            normalized_item, item_findings = _normalize_value(
                item,
                file_name=file_name,
                location=f"{location}/{normalized_key}",
            )
            normalized_dict[normalized_key] = normalized_item
            findings.extend(key_findings)
            findings.extend(item_findings)
        return normalized_dict, findings
    return value, []


def _canonical_json(value: Mapping[str, Any]) -> bytes:
    return (
        json.dumps(
            dict(value),
            ensure_ascii=False,
            allow_nan=False,
            sort_keys=True,
            indent=2,
        )
        + "\n"
    ).encode("utf-8")


def normalize_review_bundle(source_dir: Path) -> NormalizedReviewBundle:
    """Return a deterministic reviewer projection without writing files."""

    source_dir = Path(source_dir).resolve()
    if not source_dir.is_dir():
        raise ReviewBundleNormalizationError(
            "REVIEW_BUNDLE_SOURCE_INVALID",
            str(source_dir),
        )
    entries = {entry.name: entry for entry in source_dir.iterdir()}
    if set(entries) != set(CANONICAL_FILES):
        raise ReviewBundleNormalizationError(
            "REVIEW_BUNDLE_FILE_SET_INVALID",
            repr(sorted(entries)),
        )

    normalized_files: dict[str, bytes] = {}
    pre_digests: dict[str, str] = {}
    post_digests: dict[str, str] = {}
    redactions: list[dict[str, str]] = []
    for name in CANONICAL_FILES:
        path = entries[name]
        if path.is_symlink() or not path.is_file():
            raise ReviewBundleNormalizationError(
                "REVIEW_BUNDLE_FILE_INVALID",
                name,
            )
        raw = path.read_bytes()
        pre_digests[name] = _sha256(raw)
        if name in _SCIENTIFIC_JSON_FILES:
            value = _load_json_object(raw, name)
            normalized, findings = _normalize_value(
                value,
                file_name=name,
                location="",
            )
            output = _canonical_json(normalized)
        elif name == "06_report.md":
            try:
                report = raw.decode("utf-8")
            except UnicodeDecodeError as exc:
                raise ReviewBundleNormalizationError(
                    "REVIEW_BUNDLE_REPORT_INVALID",
                    name,
                ) from exc
            normalized_report, findings = _redact_text(
                report,
                file_name=name,
                location="",
            )
            output = normalized_report.replace("\r\n", "\n").encode("utf-8")
        else:
            receipt = _load_json_object(raw, name)
            missing = [field for field in _RECEIPT_VISIBLE_FIELDS if field not in receipt]
            if missing:
                raise ReviewBundleNormalizationError(
                    "REVIEW_BUNDLE_RECEIPT_FIELD_MISSING",
                    repr(missing),
                )
            projected = {field: receipt[field] for field in _RECEIPT_VISIBLE_FIELDS}
            normalized, findings = _normalize_value(
                projected,
                file_name=name,
                location="",
            )
            output = _canonical_json(normalized)
        normalized_files[name] = output
        post_digests[name] = _sha256(output)
        redactions.extend(findings)

    return NormalizedReviewBundle(
        files=normalized_files,
        pre_normalization_sha256=pre_digests,
        post_normalization_sha256=post_digests,
        redaction_log=tuple(redactions),
    )


__all__ = [
    "CANONICAL_FILES",
    "NormalizedReviewBundle",
    "ReviewBundleNormalizationError",
    "normalize_review_bundle",
]
