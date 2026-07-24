"""Validator-side comparison for the StepResultEnvelope migration.

This module is observational.  It compares the legacy inputs a validator would
read with a shadow envelope, but it does not return ``ValidationFinding`` and is
not imported by the live execution path.  A later migration can require an
exact comparison before switching one validator consumer.
"""

from __future__ import annotations

import hashlib
import json
from typing import Any, Literal, Mapping

from pydantic import BaseModel, ConfigDict, Field

from easyicu.research_agent.execution.result_envelope import (
    StepResultEnvelope,
    verify_step_result_envelope,
)


class _StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class ValidatorShadowMismatch(_StrictModel):
    code: Literal[
        "canonical_artifact_missing",
        "canonical_source_digest_mismatch",
        "canonical_status_mismatch",
        "canonical_unexpected_artifact",
        "envelope_digest_invalid",
        "normalization_error",
        "summary_not_mapping",
    ]
    detail: str = Field(min_length=1, max_length=500)


class ValidatorShadowComparison(_StrictModel):
    schema_version: Literal["easyicu.validator_shadow_comparison/1"] = (
        "easyicu.validator_shadow_comparison/1"
    )
    step_id: str
    exact_match: bool
    compared_product_ids: tuple[str, ...] = ()
    mismatches: tuple[ValidatorShadowMismatch, ...] = ()
    decision_effect: Literal["none"] = "none"


def _canonical_json_sha256(payload: Any) -> str:
    raw = (
        json.dumps(
            payload,
            ensure_ascii=False,
            allow_nan=False,
            separators=(",", ":"),
            sort_keys=True,
        )
        + "\n"
    ).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()


def compare_validator_shadow_inputs(
    *,
    step_summary: Any,
    envelope: StepResultEnvelope,
    current_status: str | None = None,
) -> ValidatorShadowComparison:
    """Compare legacy validator inputs to a shadow envelope without deciding."""

    mismatches: list[ValidatorShadowMismatch] = []
    if not verify_step_result_envelope(envelope):
        mismatches.append(
            ValidatorShadowMismatch(
                code="envelope_digest_invalid",
                detail="The shadow envelope content digest did not verify.",
            )
        )
    if not isinstance(step_summary, Mapping):
        mismatches.append(
            ValidatorShadowMismatch(
                code="summary_not_mapping",
                detail="The legacy step summary was not an object.",
            )
        )
        summary: Mapping[str, Any] = {}
    else:
        summary = step_summary
        try:
            summary_sha256 = _canonical_json_sha256(summary)
        except (TypeError, ValueError):
            summary_sha256 = None
        if summary_sha256 != envelope.source_summary_sha256:
            mismatches.append(
                ValidatorShadowMismatch(
                    code="canonical_source_digest_mismatch",
                    detail="The envelope was not compiled from these validator inputs.",
                )
            )
    if current_status is not None and current_status != envelope.status:
        mismatches.append(
            ValidatorShadowMismatch(
                code="canonical_status_mismatch",
                detail="The current ledger status differed from the envelope status.",
            )
        )

    output_files = summary.get("output_files")
    declared_ids = (
        {str(product_id) for product_id in output_files if isinstance(product_id, str)}
        if isinstance(output_files, Mapping)
        else set()
    )
    canonical_ids = {artifact.product_id for artifact in envelope.artifacts}
    for product_id in sorted(declared_ids - canonical_ids):
        mismatches.append(
            ValidatorShadowMismatch(
                code="canonical_artifact_missing",
                detail=f"Declared product {product_id!r} was absent from the envelope.",
            )
        )
    for product_id in sorted(canonical_ids - declared_ids):
        mismatches.append(
            ValidatorShadowMismatch(
                code="canonical_unexpected_artifact",
                detail=f"Envelope product {product_id!r} was not declared by the summary.",
            )
        )
    error_codes = sorted(
        {
            issue.code
            for issue in envelope.normalization_issues
            if issue.severity == "error"
        }
    )
    for code in error_codes:
        mismatches.append(
            ValidatorShadowMismatch(
                code="normalization_error",
                detail=f"Canonical normalization reported error {code!r}.",
            )
        )
    return ValidatorShadowComparison(
        step_id=envelope.step_id,
        exact_match=not mismatches,
        compared_product_ids=tuple(sorted(declared_ids | canonical_ids)),
        mismatches=tuple(mismatches),
    )


__all__ = [
    "ValidatorShadowComparison",
    "ValidatorShadowMismatch",
    "compare_validator_shadow_inputs",
]
