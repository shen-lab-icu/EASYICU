"""Validator-side comparison for the StepResultEnvelope migration.

This module is a pure comparator.  It compares the legacy inputs a validator
would read with a shadow envelope but does not itself return
``ValidationFinding``.  Fail-closed consumer adapters may use the comparison at
an explicitly sealed execution boundary.
"""

from __future__ import annotations

import hashlib
import json
from typing import Any, Literal, Mapping, Sequence

from pydantic import BaseModel, ConfigDict, Field

from easyicu.research_agent.contracts.result_envelope import (
    StepResultEnvelope,
    rebuild_observed_scalar_tree,
    verify_step_result_envelope,
)
from easyicu.research_agent.schema import AnalysisStep, ValidationFinding

_MAX_REPORTED_NORMALIZATION_ERRORS = 12
"""How many distinct normalization errors the blocking finding spells out.

Bounded because the message reaches a prompt, and stated rather than silently
truncated: a report that stops without saying it stopped reads as the whole
list.
"""


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


def canonical_registered_output_table_artifacts(
    envelope: StepResultEnvelope,
) -> tuple[str, ...]:
    """The canonical registered table artifacts a completed step declares.

    The M8 registered-output consumer reads table presence ONLY from this
    canonical view of a recovered :class:`StepResultEnvelope` -- never from a
    raw ``evidence_ids`` / ``output_files`` glob.  A table is present when an
    artifact is typed ``kind == "table"`` or belongs to a declared table
    product.
    """

    table_product_ids = {table.product_id for table in envelope.tables}
    return tuple(
        sorted(
            {
                artifact.relative_path
                for artifact in envelope.artifacts
                if artifact.kind == "table" or artifact.product_id in table_product_ids
            }
        )
    )


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
    # Per issue, not per code. Collapsing to a set of codes discarded the
    # product and the field path the normalizer had already computed, and a
    # real blocked step then read only "reported error 'invalid_registered_count'".
    # Finding the cause meant hand-scanning four emitted CSVs; the issue itself
    # said `row[4].n` of `table:cohort_input_reconciliation` all along.
    rendered: list[str] = []
    seen_details: set[str] = set()
    for issue in envelope.normalization_issues:
        if issue.severity != "error":
            continue
        where = " ".join(
            part
            for part in (
                f"in {issue.product_id}" if issue.product_id else "",
                f"at {issue.field_path}" if issue.field_path else "",
            )
            if part
        )
        detail = (
            f"Canonical normalization reported error {issue.code!r}"
            + (f" {where}" if where else "")
            + f": {issue.message}"
        )
        if detail in seen_details:
            continue
        seen_details.add(detail)
        rendered.append(detail)
    for detail in rendered[:_MAX_REPORTED_NORMALIZATION_ERRORS]:
        mismatches.append(
            ValidatorShadowMismatch(code="normalization_error", detail=detail)
        )
    withheld = len(rendered) - _MAX_REPORTED_NORMALIZATION_ERRORS
    if withheld > 0:
        mismatches.append(
            ValidatorShadowMismatch(
                code="normalization_error",
                detail=(
                    f"{withheld} further normalization error(s) were not listed; "
                    "read the canonical envelope for the rest."
                ),
            )
        )
    return ValidatorShadowComparison(
        step_id=envelope.step_id,
        exact_match=not mismatches,
        compared_product_ids=tuple(sorted(declared_ids | canonical_ids)),
        mismatches=tuple(mismatches),
    )


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
    legacy_findings: Sequence[ValidationFinding],
) -> FractionScaleShadowComparison:
    """Compare the legacy bounded-metric decision with envelope scalars."""

    from .validators import StepSummaryFractionValidator

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
    rebuilt = rebuild_observed_scalar_tree(envelope.observed_scalars)
    canonical_summary = rebuilt or {}
    tree_valid = rebuilt is not None
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


def _blocking_message_causes(comparison: FractionScaleShadowComparison) -> str:
    """The mismatch details, deduplicated, in the order they were recorded."""

    seen: set[str] = set()
    causes: list[str] = []
    for mismatch in comparison.mismatches:
        if mismatch.detail in seen:
            continue
        seen.add(mismatch.detail)
        causes.append(mismatch.detail)
    if not causes:
        return "No mismatch detail was recorded."
    return " ".join(causes)


def fraction_scale_shadow_blocking_finding(
    *,
    validator_name: str,
    step_id: str,
    comparison: FractionScaleShadowComparison,
) -> ValidationFinding:
    """Render one fail-closed bounded-metric migration finding.

    The cause leads the message and the migration boilerplate trails it. Only
    ``message`` reaches an LLM consumer -- ``detail`` is read by deterministic
    repairs but is projected away before any prompt (``_compact_findings``
    keeps validator / severity / message and clips the message to 240 chars
    from the tail). A message that opened with boilerplate therefore delivered
    a blocked step and no noun: the live E1 step ``06`` failure reported only
    that a shadow "could not safely replace the legacy view", never that the
    canonical normalizer had rejected a registered missingness partition.
    """

    return ValidationFinding(
        validator=validator_name,
        severity="error",
        message=(
            f"Bounded-metric shadow blocked step {step_id}. "
            f"{_blocking_message_causes(comparison)} "
            "Keep the legacy consumer active until source, digest, "
            "normalization, scalar-tree and finding decisions agree exactly."
        ),
        detail={
            "step_id": step_id,
            "canonical_shadow_blocked": True,
            "mismatch_codes": sorted(
                {mismatch.code for mismatch in comparison.mismatches}
            ),
            # The code alone is a label, not a diagnosis: ``normalization_error``
            # is raised for whichever inner error the canonical normalizer hit,
            # and that inner code lives only on ``mismatch.detail``. Dropping it
            # left a real blocked step with nothing to debug.
            "mismatch_details": [mismatch.detail for mismatch in comparison.mismatches],
            "legacy_finding_count": comparison.legacy_finding_count,
            "canonical_finding_count": comparison.canonical_finding_count,
            "legacy_findings_sha256": comparison.legacy_findings_sha256,
            "canonical_findings_sha256": comparison.canonical_findings_sha256,
        },
    )


__all__ = [
    "FractionScaleShadowComparison",
    "ValidatorShadowComparison",
    "ValidatorShadowMismatch",
    "canonical_registered_output_table_artifacts",
    "compare_fraction_scale_shadow",
    "compare_validator_shadow_inputs",
    "fraction_scale_shadow_blocking_finding",
]
