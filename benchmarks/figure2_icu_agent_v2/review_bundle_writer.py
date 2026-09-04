"""Arm-neutral writer for the Figure 2 seven-file review bundle."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path
import shutil
import tempfile
from typing import Any, Mapping, Sequence

from .review_bundle_semantics import (
    CANONICAL_FILES,
    FailureCategory,
    ReviewResourceReceipt,
    SUBSTANTIVE_OUTPUT_FILES,
    TerminalOutcome,
    asserted_artifact_presence,
    normalize_artifact_inventory,
    substantive_file_flags,
)


class ReviewBundleWriteError(ValueError):
    """The canonical bundle could not be validated or committed."""

    reason_code = "REVIEW_BUNDLE_WRITE_INVALID"


@dataclass(frozen=True)
class ReviewBundleMaterial:
    """Final scientific material shared by both experiment arms."""

    plan: Mapping[str, Any]
    cohort: Mapping[str, Any]
    results: Mapping[str, Any]
    diagnostics: Mapping[str, Any]
    report: str
    headline_evidence: Sequence[Mapping[str, Any]]
    artifact_inventory: Mapping[str, Any]


def terminal_failure_material(
    *,
    plan: Mapping[str, Any],
    failure_category: FailureCategory,
    mandatory_artifacts: Sequence[str],
) -> ReviewBundleMaterial:
    """Build the single neutral failure projection used by both arms."""

    if not isinstance(failure_category, FailureCategory):
        raise TypeError("failure_category must be FailureCategory")
    category = failure_category.value
    unavailable = {"available": False, "failure_category": category}
    return ReviewBundleMaterial(
        plan=plan,
        cohort=unavailable,
        results=unavailable,
        diagnostics=unavailable,
        report=(
            "The task ended with the neutral terminal category "
            f"`{category}`."
        ),
        headline_evidence=(),
        artifact_inventory={label: [] for label in mandatory_artifacts},
    )


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
        raise ReviewBundleWriteError(
            "review material is not finite JSON"
        ) from exc
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


def _validate_destination(output_dir: Path) -> Path:
    destination = Path(output_dir)
    if destination.is_symlink():
        raise ReviewBundleWriteError("review-bundle directory may not be a symlink")
    if destination.exists() and (
        not destination.is_dir() or any(destination.iterdir())
    ):
        raise ReviewBundleWriteError("review-bundle directory must be empty")
    return destination


class _ReviewBundlePublication:
    """Reserve, stage, verify, and atomically install one bundle."""

    def __init__(self, output_dir: Path) -> None:
        self.destination = Path(output_dir)
        self._staging: Path | None = None
        self._lock_path: Path | None = None
        self._lock_descriptor: int | None = None

    def __enter__(self) -> _ReviewBundlePublication:
        _validate_destination(self.destination)
        parent = self.destination.parent
        parent.mkdir(parents=True, exist_ok=True)
        lock_path = parent / f".{self.destination.name}.review-bundle.lock"
        if lock_path.is_symlink():
            raise ReviewBundleWriteError("review-bundle lock may not be a symlink")
        try:
            descriptor = os.open(
                lock_path,
                os.O_WRONLY | os.O_CREAT | os.O_EXCL,
                0o600,
            )
        except FileExistsError as exc:
            raise ReviewBundleWriteError(
                "review-bundle destination is already reserved"
            ) from exc
        except OSError as exc:
            raise ReviewBundleWriteError(
                "review-bundle destination cannot be reserved"
            ) from exc
        self._lock_path = lock_path
        self._lock_descriptor = descriptor
        try:
            _validate_destination(self.destination)
            self._staging = Path(
                tempfile.mkdtemp(
                    prefix=f".{self.destination.name}.review-bundle-stage-",
                    dir=parent,
                )
            )
        except BaseException:
            self._release()
            raise
        return self

    def install(self, payloads: Mapping[str, bytes]) -> Path:
        if self._staging is None:
            raise RuntimeError("review-bundle publication is not reserved")
        for name in CANONICAL_FILES:
            _write_new_file(self._staging, name, payloads[name])
        staged_names = {entry.name for entry in self._staging.iterdir()}
        if staged_names != set(CANONICAL_FILES):
            raise ReviewBundleWriteError("staged review bundle is incomplete")
        for name in CANONICAL_FILES:
            if (self._staging / name).read_bytes() != payloads[name]:
                raise ReviewBundleWriteError(
                    f"staged review bundle payload changed: {name}"
                )
        _validate_destination(self.destination)
        try:
            os.replace(self._staging, self.destination)
            self._staging = None
            parent_descriptor = os.open(self.destination.parent, os.O_RDONLY)
            try:
                os.fsync(parent_descriptor)
            finally:
                os.close(parent_descriptor)
        except OSError as exc:
            raise ReviewBundleWriteError(
                "review-bundle destination cannot be installed atomically"
            ) from exc
        return self.destination

    def _release(self) -> None:
        if self._lock_descriptor is not None:
            os.close(self._lock_descriptor)
            self._lock_descriptor = None
        if self._lock_path is not None:
            self._lock_path.unlink(missing_ok=True)
            self._lock_path = None

    def __exit__(self, exc_type, exc, traceback) -> None:
        del exc_type, exc, traceback
        if self._staging is not None:
            shutil.rmtree(self._staging, ignore_errors=True)
            self._staging = None
        self._release()


class ReviewBundleWriter:
    """Prepared one-shot writer; construction validates before work begins."""

    def __init__(self, output_dir: Path) -> None:
        self.output_dir = _validate_destination(output_dir)

    def write(
        self,
        material: ReviewBundleMaterial,
        *,
        mandatory_artifacts: Sequence[str],
        resource_receipt: ReviewResourceReceipt,
        outcome: TerminalOutcome | None = None,
    ) -> Path:
        return _write_review_bundle(
            material,
            output_dir=self.output_dir,
            mandatory_artifacts=mandatory_artifacts,
            resource_receipt=resource_receipt,
            outcome=outcome,
        )


def _write_review_bundle(
    material: ReviewBundleMaterial,
    *,
    output_dir: Path,
    mandatory_artifacts: Sequence[str],
    resource_receipt: ReviewResourceReceipt,
    outcome: TerminalOutcome | None = None,
) -> Path:
    """Validate and commit one canonical bundle, rolling back partial writes."""

    if not isinstance(resource_receipt, ReviewResourceReceipt):
        raise ReviewBundleWriteError(
            "resource_receipt must be a ReviewResourceReceipt"
        )
    terminal_outcome = outcome or TerminalOutcome.completed()
    if not isinstance(terminal_outcome, TerminalOutcome):
        raise ReviewBundleWriteError("outcome must be a TerminalOutcome")
    if not all(
        isinstance(value, Mapping)
        for value in (
            material.plan,
            material.cohort,
            material.results,
            material.diagnostics,
        )
    ):
        raise ReviewBundleWriteError("scientific JSON artifacts must be objects")
    if not isinstance(material.report, str) or not material.report.strip():
        raise ReviewBundleWriteError("report must be non-empty Markdown")
    if not isinstance(material.headline_evidence, Sequence) or isinstance(
        material.headline_evidence, (str, bytes)
    ):
        raise ReviewBundleWriteError("headline_evidence must be a sequence")
    if not all(isinstance(item, Mapping) for item in material.headline_evidence):
        raise ReviewBundleWriteError("headline_evidence entries must be objects")

    if terminal_outcome.failure_category is not None:
        if set(material.artifact_inventory) != set(mandatory_artifacts) or any(
            references != [] for references in material.artifact_inventory.values()
        ):
            raise ReviewBundleWriteError(
                "failed bundle inventory must map every mandatory artifact to []"
            )
        inventory = {label: [] for label in mandatory_artifacts}
    else:
        try:
            inventory = normalize_artifact_inventory(
                material.artifact_inventory,
                mandatory_artifacts,
            )
        except ValueError as exc:
            raise ReviewBundleWriteError(str(exc)) from exc

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
        **resource_receipt.as_dict(),
        **terminal_outcome.receipt_fields(),
        "agent_asserted_mandatory_artifact_presence": (
            {label: False for label in mandatory_artifacts}
            if terminal_outcome.failure_category is not None
            else asserted_artifact_presence(
                inventory,
                plan=material.plan,
                cohort=material.cohort,
                results=material.results,
                diagnostics=material.diagnostics,
                report=material.report,
            )
        ),
        "substantive_output_files": (
            {name: False for name in SUBSTANTIVE_OUTPUT_FILES}
            if terminal_outcome.failure_category is not None
            else substantive_file_flags(
                plan=material.plan,
                cohort=material.cohort,
                results=material.results,
                diagnostics=material.diagnostics,
                report=material.report,
            )
        ),
    }
    payloads["07_run_receipt.json"] = _canonical_json(receipt)

    with _ReviewBundlePublication(output_dir) as publication:
        return publication.install(payloads)


def write_review_bundle(
    material: ReviewBundleMaterial,
    *,
    output_dir: Path,
    mandatory_artifacts: Sequence[str],
    resource_receipt: ReviewResourceReceipt,
    outcome: TerminalOutcome | None = None,
) -> Path:
    """Validate and commit one canonical bundle, rolling back partial writes."""

    return ReviewBundleWriter(output_dir).write(
        material,
        mandatory_artifacts=mandatory_artifacts,
        resource_receipt=resource_receipt,
        outcome=outcome,
    )


__all__ = [
    "ReviewBundleMaterial",
    "ReviewBundleWriter",
    "ReviewBundleWriteError",
    "terminal_failure_material",
    "write_review_bundle",
]
