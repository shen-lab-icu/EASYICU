"""Arm-neutral Figure 2 review-bundle normalization.

This owner performs only deterministic projection and identifier redaction.
It never repairs or recomputes scientific content.  Inputs that cannot be
normalized under the closed rules fail before a reviewer bundle is emitted.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from functools import lru_cache
import hashlib
import json
import math
from pathlib import Path
import re
from typing import Any, Mapping

from .review_bundle_semantics import CANONICAL_FILES


ACTION_SPACE_PATH = Path(__file__).resolve().parent / "action_space_v1.json"
_SCIENTIFIC_JSON_FILES = frozenset(CANONICAL_FILES[:5])
_RECEIPT_VISIBLE_FIELDS = (
    "terminal_status",
    "within_frozen_budget",
    "failure_category",
    "agent_asserted_mandatory_artifact_presence",
    "substantive_output_files",
)
_REDACTIONS = (
    ("easyicu_name", re.compile(r"\bEasyICU\b", re.IGNORECASE), "the producing workflow"),
    (
        "arm_label",
        re.compile(r"\b(?:easyicu_full|generic_code_agent)\b", re.IGNORECASE),
        "the producing workflow",
    ),
)
_STATIC_REJECTED_MARKERS = (
    re.compile(
        r"\b(?:FORMAL_|GENERIC_|REVIEW_BUNDLE_|WP5_|SAFETY12_|EASYICU_)"
        r"[A-Z0-9_]*\b"
    ),
    re.compile(r"\bI\d{2}_[A-Z][A-Z0-9_]+\b"),
)
_PATH_MARKER = re.compile(
    r"(?<![:\w])/(?:Users|home|workspace|workspaces|tmp|var/tmp|mnt|opt|app)/"
    r"[^\s\]\[(){}<>'\"]+"
    r"|\b(?:src/easyicu|benchmarks/figure2_icu_agent_v2)/[^\s\]\[(){}<>'\"]+"
)
_RESOURCE_FINGERPRINTS = (
    re.compile(
        r"\b(?:model[ _-]?turns?|provider[ _-]?calls?|tool[ _-]?calls?|"
        r"provider[ _-]?tokens?|billed[ _-]?cost|per[ _-]?tool[ _-]?latency|"
        r"stage[ _-]?shaped[ _-]?timing|execution[ _-]?site|logical[ _-]?site|"
        r"host[ _-]?fingerprint|machine[ _-]?name)\b",
        re.IGNORECASE,
    ),
    re.compile(
        r"\b\d+(?:\.\d+)?\s*(?:input|output|provider)?\s*tokens?\b",
        re.IGNORECASE,
    ),
)


class ReviewBundleNormalizationError(ValueError):
    def __init__(self, reason_code: str, detail: str) -> None:
        self.reason_code = reason_code
        self.detail = detail
        super().__init__(f"{reason_code}: {detail}")


@lru_cache(maxsize=1)
def _internal_markers() -> tuple[frozenset[str], frozenset[str]]:
    try:
        payload = json.loads(ACTION_SPACE_PATH.read_text(encoding="utf-8"))
        stages = payload["stages"]
        stage_ids = frozenset(stage["stage_id"] for stage in stages)
        reason_codes = frozenset(
            reason
            for stage in stages
            for reason in stage["failure_reason_codes"]
        )
    except (OSError, KeyError, TypeError, json.JSONDecodeError) as exc:
        raise ReviewBundleNormalizationError(
            "REVIEW_BUNDLE_MARKER_AUTHORITY_INVALID",
            str(ACTION_SPACE_PATH),
        ) from exc
    if len(stage_ids) != 11 or len(reason_codes) != 22:
        raise ReviewBundleNormalizationError(
            "REVIEW_BUNDLE_MARKER_AUTHORITY_INVALID",
            f"stages={len(stage_ids)}, reason_codes={len(reason_codes)}",
        )
    return stage_ids, reason_codes


@dataclass(frozen=True)
class NormalizedReviewBundle:
    files: Mapping[str, bytes]
    pre_normalization_sha256: Mapping[str, str]
    post_normalization_sha256: Mapping[str, str]
    redaction_log: tuple[Mapping[str, str], ...]


def _sha256(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _load_json_object(raw: bytes, name: str) -> dict[str, Any]:
    def reject_duplicates(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                raise ReviewBundleNormalizationError(
                    "REVIEW_BUNDLE_JSON_KEY_DUPLICATE",
                    f"{name}:{key}",
                )
            result[key] = value
        return result

    def reject_constant(value: str) -> None:
        raise ReviewBundleNormalizationError(
            "REVIEW_BUNDLE_NONFINITE_JSON",
            f"{name}: {value}",
        )

    try:
        value = json.loads(
            raw.decode("utf-8"),
            object_pairs_hook=reject_duplicates,
            parse_constant=reject_constant,
        )
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
    _reject_nonfinite_json(value, file_name=name, location="")
    return value


def _reject_nonfinite_json(value: Any, *, file_name: str, location: str) -> None:
    if isinstance(value, float) and not math.isfinite(value):
        raise ReviewBundleNormalizationError(
            "REVIEW_BUNDLE_NONFINITE_JSON",
            f"{file_name}:{location}",
        )
    if isinstance(value, list):
        for index, item in enumerate(value):
            _reject_nonfinite_json(
                item,
                file_name=file_name,
                location=f"{location}/{index}",
            )
    elif isinstance(value, dict):
        for key, item in value.items():
            _reject_nonfinite_json(
                item,
                file_name=file_name,
                location=f"{location}/{key}",
            )


def _redact_path(match: re.Match[str]) -> str:
    value = match.group(0)
    trailing = ""
    while value and value[-1] in ".,;:!?":
        trailing = value[-1] + trailing
        value = value[:-1]
    return "<redacted-path>" + trailing


def _contains_exact_marker(text: str, marker: str) -> bool:
    return bool(re.search(rf"(?<!\w){re.escape(marker)}(?!\w)", text))


def _reject_forbidden_markers(text: str, *, file_name: str, location: str) -> None:
    stage_ids, reason_codes = _internal_markers()
    for marker in sorted(stage_ids | reason_codes):
        if _contains_exact_marker(text, marker):
            raise ReviewBundleNormalizationError(
                "REVIEW_BUNDLE_UNSAFE_MARKER",
                f"{file_name}:{location}:{marker}",
            )
    for pattern in _STATIC_REJECTED_MARKERS:
        if pattern.search(text):
            raise ReviewBundleNormalizationError(
                "REVIEW_BUNDLE_UNSAFE_MARKER",
                f"{file_name}:{location}",
            )
    if any(pattern.search(text) for pattern in _RESOURCE_FINGERPRINTS):
        raise ReviewBundleNormalizationError(
            "REVIEW_BUNDLE_RESOURCE_FINGERPRINT",
            f"{file_name}:{location}",
        )


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
    output, path_count = _PATH_MARKER.subn(_redact_path, output)
    if path_count:
        findings.append(
            {
                "file": file_name,
                "location": location,
                "rule": "repository_path",
                "replacement_count": str(path_count),
            }
        )
    _reject_forbidden_markers(
        output,
        file_name=file_name,
        location=location,
    )
    return output, findings


def _numeric_leaf_multiset(value: Any) -> Counter[tuple[str, str]]:
    leaves: Counter[tuple[str, str]] = Counter()
    if isinstance(value, bool) or value is None:
        return leaves
    if isinstance(value, int):
        leaves[("int", str(value))] += 1
    elif isinstance(value, float):
        leaves[("float", repr(value))] += 1
    elif isinstance(value, list):
        for item in value:
            leaves.update(_numeric_leaf_multiset(item))
    elif isinstance(value, dict):
        for item in value.values():
            leaves.update(_numeric_leaf_multiset(item))
    return leaves


def _assert_numeric_content_preserved(
    before: Mapping[str, Any],
    after: Mapping[str, Any],
    *,
    file_name: str,
) -> None:
    if _numeric_leaf_multiset(before) != _numeric_leaf_multiset(after):
        raise ReviewBundleNormalizationError(
            "REVIEW_BUNDLE_NUMERIC_CONTENT_CHANGED",
            file_name,
        )


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
            _assert_numeric_content_preserved(value, normalized, file_name=name)
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
            artifact_presence = receipt[
                "agent_asserted_mandatory_artifact_presence"
            ]
            substantive_output_files = receipt["substantive_output_files"]
            if (
                not isinstance(receipt["terminal_status"], str)
                or not receipt["terminal_status"].strip()
                or not isinstance(receipt["within_frozen_budget"], bool)
                or (
                    receipt["failure_category"] is not None
                    and not isinstance(receipt["failure_category"], str)
                )
                or not isinstance(artifact_presence, dict)
                or not artifact_presence
                or any(
                    not isinstance(label, str)
                    or not label.strip()
                    or not isinstance(present, bool)
                    for label, present in artifact_presence.items()
                )
                or set(substantive_output_files)
                != {
                    "02_cohort.json",
                    "03_results.json",
                    "04_diagnostics.json",
                    "06_report.md",
                }
                or any(
                    not isinstance(present, bool)
                    for present in substantive_output_files.values()
                )
            ):
                raise ReviewBundleNormalizationError(
                    "REVIEW_BUNDLE_RECEIPT_FIELD_INVALID",
                    name,
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
