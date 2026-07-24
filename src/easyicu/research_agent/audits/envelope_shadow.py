"""Validator-side comparison for the StepResultEnvelope migration.

This module is observational.  It compares the legacy inputs a validator would
read with a shadow envelope, but it does not return ``ValidationFinding`` and is
not imported by the live execution path.  A later migration can require an
exact comparison before switching one validator consumer.
"""

from __future__ import annotations

import hashlib
import json
import re
from typing import Any, Literal, Mapping, Sequence

from pydantic import BaseModel, ConfigDict, Field

from easyicu.research_agent.execution.result_envelope import (
    StepResultEnvelope,
    verify_step_result_envelope,
)
from easyicu.research_agent.schema import AnalysisStep, ValidationFinding


class _StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class ValidatorShadowMismatch(_StrictModel):
    code: Literal[
        "canonical_artifact_missing",
        "canonical_envelope_missing",
        "canonical_source_digest_mismatch",
        "canonical_status_mismatch",
        "canonical_fraction_view_mismatch",
        "canonical_scalar_tree_invalid",
        "canonical_table_presence_mismatch",
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


class RegisteredOutputShadowComparison(_StrictModel):
    schema_version: Literal["easyicu.registered_output_shadow_comparison/1"] = (
        "easyicu.registered_output_shadow_comparison/1"
    )
    step_id: str
    exact_match: bool
    legacy_table_artifacts: tuple[str, ...] = ()
    canonical_table_artifacts: tuple[str, ...] = ()
    mismatches: tuple[ValidatorShadowMismatch, ...] = ()
    decision_effect: Literal["none"] = "none"


class FractionScaleShadowComparison(_StrictModel):
    schema_version: Literal["easyicu.fraction_scale_shadow_comparison/1"] = (
        "easyicu.fraction_scale_shadow_comparison/1"
    )
    step_id: str
    exact_match: bool
    legacy_finding_count: int = Field(ge=0)
    canonical_finding_count: int = Field(ge=0)
    legacy_findings_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    canonical_findings_sha256: str | None = Field(
        default=None,
        pattern=r"^[0-9a-f]{64}$",
    )
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


def compare_registered_output_shadow(
    *,
    step_id: str,
    step_summary: Any,
    current_status: str | None,
    legacy_table_artifacts: Sequence[str],
    envelope: StepResultEnvelope | None,
) -> RegisteredOutputShadowComparison:
    """Require exact source and table-presence agreement before dual-read."""

    if envelope is None:
        return RegisteredOutputShadowComparison(
            step_id=step_id,
            exact_match=False,
            legacy_table_artifacts=tuple(sorted(set(legacy_table_artifacts))),
            mismatches=(
                ValidatorShadowMismatch(
                    code="canonical_envelope_missing",
                    detail="No canonical envelope was available for the upstream step.",
                ),
            ),
        )
    base = compare_validator_shadow_inputs(
        step_summary=step_summary,
        envelope=envelope,
        current_status=current_status,
    )
    table_product_ids = {table.product_id for table in envelope.tables}
    canonical_table_artifacts = tuple(
        sorted(
            {
                artifact.relative_path
                for artifact in envelope.artifacts
                if artifact.kind == "table" or artifact.product_id in table_product_ids
            }
        )
    )
    mismatches = list(base.mismatches)
    if bool(legacy_table_artifacts) != bool(canonical_table_artifacts):
        mismatches.append(
            ValidatorShadowMismatch(
                code="canonical_table_presence_mismatch",
                detail=(
                    "Legacy and canonical registered-output views disagreed "
                    "about whether the upstream step produced a table."
                ),
            )
        )
    return RegisteredOutputShadowComparison(
        step_id=envelope.step_id,
        exact_match=not mismatches,
        legacy_table_artifacts=tuple(sorted(set(legacy_table_artifacts))),
        canonical_table_artifacts=canonical_table_artifacts,
        mismatches=tuple(mismatches),
    )


_SCALAR_PATH_TOKEN = re.compile(r"([^\.\[\]]+)|\[(\d+)\]")


def _scalar_path_tokens(field_path: str) -> tuple[str | int, ...] | None:
    tokens: list[str | int] = []
    for match in _SCALAR_PATH_TOKEN.finditer(field_path):
        raw_index = match.group(2)
        tokens.append(int(raw_index) if raw_index is not None else match.group(1))
    if not tokens:
        return None
    rendered = str(tokens[0])
    for token in tokens[1:]:
        rendered += f"[{token}]" if isinstance(token, int) else f".{token}"
    return tuple(tokens) if rendered == field_path else None


def _set_scalar_path(
    root: dict[str, Any],
    *,
    tokens: tuple[str | int, ...],
    value: Any,
) -> bool:
    current: Any = root
    for index, token in enumerate(tokens):
        final = index == len(tokens) - 1
        next_is_index = not final and isinstance(tokens[index + 1], int)
        if isinstance(token, str):
            if not isinstance(current, dict):
                return False
            if final:
                if token in current and current[token] != value:
                    return False
                current[token] = value
                return True
            child = current.get(token)
            expected_type = list if next_is_index else dict
            if child is None:
                child = expected_type()
                current[token] = child
            elif not isinstance(child, expected_type):
                return False
            current = child
            continue
        if not isinstance(current, list):
            return False
        while len(current) <= token:
            current.append(None)
        if final:
            if current[token] is not None and current[token] != value:
                return False
            current[token] = value
            return True
        child = current[token]
        expected_type = list if next_is_index else dict
        if child is None:
            child = expected_type()
            current[token] = child
        elif not isinstance(child, expected_type):
            return False
        current = child
    return False


def _fraction_view_summary(
    envelope: StepResultEnvelope,
) -> tuple[dict[str, Any], bool]:
    """Rebuild only the normalized scalar tree carried by the envelope."""

    summary: dict[str, Any] = {}
    for scalar in envelope.observed_scalars:
        tokens = _scalar_path_tokens(scalar.field_path)
        if tokens is None or not isinstance(tokens[0], str):
            return {}, False
        if not _set_scalar_path(summary, tokens=tokens, value=scalar.value):
            return {}, False
    return summary, True


def _finding_payload_sha256(findings: Sequence[ValidationFinding]) -> str:
    payloads = [finding.model_dump(mode="json") for finding in findings]
    payloads.sort(
        key=lambda payload: json.dumps(
            payload,
            ensure_ascii=False,
            allow_nan=False,
            separators=(",", ":"),
            sort_keys=True,
        )
    )
    return _canonical_json_sha256(payloads)


def compare_fraction_scale_shadow(
    *,
    step: AnalysisStep,
    step_summary: Any,
    current_status: str | None,
    envelope: StepResultEnvelope | None,
) -> FractionScaleShadowComparison:
    """Compare the legacy bounded-metric decision with envelope scalars."""

    from .validators import StepSummaryFractionValidator

    legacy_summary = step_summary if isinstance(step_summary, dict) else {}
    legacy_findings = StepSummaryFractionValidator().audit(
        step=step,
        step_summary=legacy_summary,
    )
    legacy_sha256 = _finding_payload_sha256(legacy_findings)
    if envelope is None:
        return FractionScaleShadowComparison(
            step_id=step.step_id,
            exact_match=False,
            legacy_finding_count=len(legacy_findings),
            canonical_finding_count=0,
            legacy_findings_sha256=legacy_sha256,
            mismatches=(
                ValidatorShadowMismatch(
                    code="canonical_envelope_missing",
                    detail="No canonical envelope was available for this step.",
                ),
            ),
        )

    base = compare_validator_shadow_inputs(
        step_summary=step_summary,
        envelope=envelope,
        current_status=current_status,
    )
    mismatches = list(base.mismatches)
    canonical_summary, tree_valid = _fraction_view_summary(envelope)
    canonical_findings: list[ValidationFinding] = []
    if tree_valid:
        canonical_findings = StepSummaryFractionValidator().audit(
            step=step,
            step_summary=canonical_summary,
        )
    else:
        mismatches.append(
            ValidatorShadowMismatch(
                code="canonical_scalar_tree_invalid",
                detail="Canonical observed scalar paths could not form one tree.",
            )
        )
    canonical_sha256 = _finding_payload_sha256(canonical_findings)
    if canonical_sha256 != legacy_sha256:
        mismatches.append(
            ValidatorShadowMismatch(
                code="canonical_fraction_view_mismatch",
                detail=(
                    "Legacy and canonical bounded fraction/percentage findings "
                    "were not byte-equivalent."
                ),
            )
        )
    return FractionScaleShadowComparison(
        step_id=envelope.step_id,
        exact_match=not mismatches,
        legacy_finding_count=len(legacy_findings),
        canonical_finding_count=len(canonical_findings),
        legacy_findings_sha256=legacy_sha256,
        canonical_findings_sha256=canonical_sha256,
        mismatches=tuple(mismatches),
    )


def fraction_scale_shadow_blocking_finding(
    *,
    validator_name: str,
    step_id: str,
    comparison: FractionScaleShadowComparison,
) -> ValidationFinding:
    """Render one fail-closed bounded-metric migration finding."""

    return ValidationFinding(
        validator=validator_name,
        severity="error",
        message=(
            "Canonical bounded fraction/percentage shadow could not safely "
            f"replace the legacy view for step {step_id}. Keep the legacy "
            "consumer active until source, digest, normalization, scalar-tree, "
            "and finding decisions agree exactly."
        ),
        detail={
            "step_id": step_id,
            "canonical_shadow_blocked": True,
            "mismatch_codes": sorted(
                {mismatch.code for mismatch in comparison.mismatches}
            ),
            "legacy_finding_count": comparison.legacy_finding_count,
            "canonical_finding_count": comparison.canonical_finding_count,
            "legacy_findings_sha256": comparison.legacy_findings_sha256,
            "canonical_findings_sha256": comparison.canonical_findings_sha256,
        },
    )


def registered_output_shadow_blocking_finding(
    *,
    validator_name: str,
    consumer_step_id: str,
    upstream_step: str,
    comparison: RegisteredOutputShadowComparison,
) -> ValidationFinding:
    """Render one fail-closed migration finding without changing legacy logic."""

    return ValidationFinding(
        validator=validator_name,
        severity="error",
        message=(
            f"Canonical registered-output shadow could not safely replace "
            f"the legacy view for upstream step {upstream_step}. Keep the "
            "legacy consumer active until the envelope source, digest, "
            "normalization, and table-presence decisions agree exactly."
        ),
        detail={
            "step_id": consumer_step_id,
            "upstream_step": upstream_step,
            "canonical_shadow_blocked": True,
            "mismatch_codes": sorted(
                {mismatch.code for mismatch in comparison.mismatches}
            ),
            "legacy_table_artifacts": list(comparison.legacy_table_artifacts),
            "canonical_table_artifacts": list(comparison.canonical_table_artifacts),
        },
    )


__all__ = [
    "FractionScaleShadowComparison",
    "RegisteredOutputShadowComparison",
    "ValidatorShadowComparison",
    "ValidatorShadowMismatch",
    "compare_fraction_scale_shadow",
    "compare_registered_output_shadow",
    "compare_validator_shadow_inputs",
    "fraction_scale_shadow_blocking_finding",
    "registered_output_shadow_blocking_finding",
]
