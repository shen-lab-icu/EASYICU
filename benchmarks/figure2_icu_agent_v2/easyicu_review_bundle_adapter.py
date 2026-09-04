"""Project finalized EasyICU artifacts into the arm-neutral review bundle.

This adapter is intentionally mechanical: it validates shape, computes file
digests, and writes the shared seven-file contract.  It never repairs,
interprets, or recomputes scientific content.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping, Sequence

from easyicu.research_agent.schema import PipelineResult

from .review_bundle_writer import (
    ReviewBundleMaterial,
    ReviewBundleWriteError,
    write_review_bundle,
)


class EasyICUReviewBundleError(ValueError):
    reason_code = "EASYICU_REVIEW_BUNDLE_INVALID"


class EasyICUReviewMaterial(ReviewBundleMaterial):
    """Already-finalized scientific content supplied by the governed pipeline."""

    plan: Mapping[str, Any]
    cohort: Mapping[str, Any]
    results: Mapping[str, Any]
    diagnostics: Mapping[str, Any]
    report: str
    headline_evidence: Sequence[Mapping[str, Any]]
    artifact_inventory: Mapping[str, Any]

    @classmethod
    def from_pipeline_result(
        cls,
        result: PipelineResult,
        *,
        artifact_inventory: Mapping[str, Any],
    ) -> EasyICUReviewMaterial:
        """Read the fixed native final artifacts without a postrun editing seam."""

        if not isinstance(result, PipelineResult):
            raise EasyICUReviewBundleError(
                "a terminal PipelineResult is required for review projection"
            )
        run_dir = Path(result.workdir).resolve(strict=True)
        if run_dir.is_symlink() or not run_dir.is_dir():
            raise EasyICUReviewBundleError("pipeline workdir must be a real directory")

        def read_path(path_value: str, *, label: str, json_object: bool) -> Any:
            candidate = Path(path_value)
            if not candidate.is_absolute():
                candidate = run_dir / candidate
            if candidate.is_symlink():
                raise EasyICUReviewBundleError(f"{label} may not be a symlink")
            try:
                resolved = candidate.resolve(strict=True)
                resolved.relative_to(run_dir)
            except (OSError, ValueError) as exc:
                raise EasyICUReviewBundleError(
                    f"{label} must be a regular file inside the pipeline workdir"
                ) from exc
            if not resolved.is_file():
                raise EasyICUReviewBundleError(f"{label} must be a regular file")
            if not json_object:
                try:
                    return resolved.read_text(encoding="utf-8")
                except UnicodeError as exc:
                    raise EasyICUReviewBundleError(
                        f"{label} must be UTF-8 text"
                    ) from exc
            try:
                value = json.loads(resolved.read_text(encoding="utf-8"))
            except (UnicodeError, json.JSONDecodeError) as exc:
                raise EasyICUReviewBundleError(
                    f"{label} must be valid JSON"
                ) from exc
            if not isinstance(value, dict):
                raise EasyICUReviewBundleError(f"{label} must be a JSON object")
            return value

        plan = read_path(result.plan_path, label="plan", json_object=True)
        context = read_path(result.context_path, label="context", json_object=True)
        manifest = read_path(result.manifest_path, label="manifest", json_object=True)
        diagnostics = read_path(
            str(run_dir / "run_status.json"),
            label="run_status",
            json_object=True,
        )
        report = read_path(result.report_path, label="report", json_object=False)
        evidence = manifest.get("evidence")
        if not isinstance(evidence, list) or not all(
            isinstance(item, dict) for item in evidence
        ):
            raise EasyICUReviewBundleError(
                "final manifest evidence must be a list of objects"
            )
        return cls(
            plan=plan,
            cohort=context,
            results=manifest,
            diagnostics=diagnostics,
            report=report,
            headline_evidence=tuple(evidence),
            artifact_inventory=artifact_inventory,
        )


def write_easyicu_review_bundle(
    material: ReviewBundleMaterial,
    *,
    output_dir: Path,
    mandatory_artifacts: Sequence[str],
    resource_receipt: Mapping[str, Any],
    terminal_status: str = "completed",
    failure_category: str | None = None,
) -> Path:
    """Write one immutable canonical bundle from finalized EasyICU material."""
    try:
        return write_review_bundle(
            material,
            output_dir=output_dir,
            mandatory_artifacts=mandatory_artifacts,
            resource_receipt=resource_receipt,
            terminal_status=terminal_status,
            failure_category=failure_category,
        )
    except ReviewBundleWriteError as exc:
        raise EasyICUReviewBundleError(str(exc)) from exc


__all__ = [
    "EasyICUReviewBundleError",
    "EasyICUReviewMaterial",
    "write_easyicu_review_bundle",
]
