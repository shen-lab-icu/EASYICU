"""Project finalized EasyICU artifacts into the arm-neutral review bundle.

This adapter is intentionally mechanical: it validates shape, computes file
digests, and writes the shared seven-file contract.  It never repairs,
interprets, or recomputes scientific content.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path
from typing import Any, Mapping, Sequence

from .review_bundle_semantics import (
    CANONICAL_FILES,
    asserted_artifact_presence,
    normalize_artifact_inventory,
    substantive_file_flags,
)


class EasyICUReviewBundleError(ValueError):
    reason_code = "EASYICU_REVIEW_BUNDLE_INVALID"


@dataclass(frozen=True)
class EasyICUReviewMaterial:
    """Already-finalized scientific content supplied by the governed pipeline."""

    plan: Mapping[str, Any]
    cohort: Mapping[str, Any]
    results: Mapping[str, Any]
    diagnostics: Mapping[str, Any]
    report: str
    headline_evidence: Sequence[Mapping[str, Any]]
    artifact_inventory: Mapping[str, Any]


def _canonical_json(value: Mapping[str, Any]) -> bytes:
    try:
        text = json.dumps(
            dict(value),
            ensure_ascii=False,
            allow_nan=False,
            sort_keys=True,
            indent=2,
        )
    except (TypeError, ValueError) as exc:
        raise EasyICUReviewBundleError("review material is not finite JSON") from exc
    return (text + "\n").encode("utf-8")


def _write_new_file(root: Path, name: str, payload: bytes) -> None:
    path = root / name
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
    descriptor = os.open(path, flags, 0o600)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(payload)
    except BaseException:
        path.unlink(missing_ok=True)
        raise


def _prepare_destination(output_dir: Path) -> Path:
    destination = Path(output_dir)
    if destination.is_symlink():
        raise EasyICUReviewBundleError("review-bundle directory may not be a symlink")
    destination.mkdir(parents=True, exist_ok=True)
    if not destination.is_dir() or any(destination.iterdir()):
        raise EasyICUReviewBundleError("review-bundle directory must be empty")
    return destination


def write_easyicu_review_bundle(
    material: EasyICUReviewMaterial,
    *,
    output_dir: Path,
    mandatory_artifacts: Sequence[str],
    resource_receipt: Mapping[str, Any],
    terminal_status: str = "completed",
    failure_category: str | None = None,
) -> Path:
    """Write one immutable canonical bundle from finalized EasyICU material."""

    if terminal_status not in {"completed", "failed"}:
        raise EasyICUReviewBundleError("terminal_status must be completed or failed")
    if (terminal_status == "completed") != (failure_category is None):
        raise EasyICUReviewBundleError(
            "completed requires null failure_category; failed requires a category"
        )
    if not all(
        isinstance(value, Mapping)
        for value in (
            material.plan,
            material.cohort,
            material.results,
            material.diagnostics,
        )
    ):
        raise EasyICUReviewBundleError("scientific JSON artifacts must be objects")
    if not isinstance(material.report, str) or not material.report.strip():
        raise EasyICUReviewBundleError("report must be non-empty Markdown")
    if not isinstance(material.headline_evidence, Sequence) or isinstance(
        material.headline_evidence, (str, bytes)
    ):
        raise EasyICUReviewBundleError("headline_evidence must be a sequence")
    if not all(isinstance(item, Mapping) for item in material.headline_evidence):
        raise EasyICUReviewBundleError("headline_evidence entries must be objects")
    try:
        inventory = normalize_artifact_inventory(
            material.artifact_inventory,
            mandatory_artifacts,
        )
    except ValueError as exc:
        raise EasyICUReviewBundleError(str(exc)) from exc

    payloads = {
        "01_plan.json": _canonical_json(material.plan),
        "02_cohort.json": _canonical_json(material.cohort),
        "03_results.json": _canonical_json(material.results),
        "04_diagnostics.json": _canonical_json(material.diagnostics),
        "06_report.md": (material.report.rstrip() + "\n").encode("utf-8"),
    }
    manifest = {
        "harness_computed_file_digests": {
            name: hashlib.sha256(payload).hexdigest()
            for name, payload in payloads.items()
        },
        "agent_asserted_headline_evidence": [
            dict(item) for item in material.headline_evidence
        ],
        "agent_asserted_mandatory_artifact_inventory": inventory,
    }
    payloads["05_evidence_manifest.json"] = _canonical_json(manifest)

    receipt = {
        **dict(resource_receipt),
        "terminal_status": terminal_status,
        "within_frozen_budget": bool(
            resource_receipt.get("within_frozen_budget", False)
        ),
        "failure_category": failure_category,
        "agent_asserted_mandatory_artifact_presence": asserted_artifact_presence(
            inventory,
            plan=material.plan,
            cohort=material.cohort,
            results=material.results,
            diagnostics=material.diagnostics,
            report=material.report,
        ),
        "substantive_output_files": substantive_file_flags(
            plan=material.plan,
            cohort=material.cohort,
            results=material.results,
            diagnostics=material.diagnostics,
            report=material.report,
        ),
    }
    payloads["07_run_receipt.json"] = _canonical_json(receipt)

    destination = _prepare_destination(output_dir)
    try:
        for name in CANONICAL_FILES:
            _write_new_file(destination, name, payloads[name])
    except BaseException:
        for name in CANONICAL_FILES:
            (destination / name).unlink(missing_ok=True)
        raise
    return destination


__all__ = [
    "EasyICUReviewBundleError",
    "EasyICUReviewMaterial",
    "write_easyicu_review_bundle",
]
