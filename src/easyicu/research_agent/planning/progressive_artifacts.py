"""Persist the Progressive Planner contract chain as governed evidence.

This module owns the boundary between planning contracts and EvidenceStore.  It
does not plan, compile, execute, or infer authority from mutable pipeline state.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Protocol, Sequence

from ..canonical_json import canonical_sha256
from .progressive_contract import (
    ProgressivePlanCompileReceipt,
    ProgressivePlanOutline,
    ProgressivePlanSkeleton,
    ProgressiveStepMaterialization,
)


_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")


class ProgressivePlanningArtifactError(ValueError):
    """A planning artifact chain is incomplete or authority-inconsistent."""

    def __init__(self, reason_code: str, message: str) -> None:
        super().__init__(f"{reason_code}: {message}")
        self.reason_code = reason_code


class ProgressiveEvidenceRegistrar(Protocol):
    """Small EvidenceStore surface required by this owner module."""

    def get(self, evidence_id_or_alias: str) -> object | None: ...

    def register_file(self, **kwargs: Any) -> object: ...


@dataclass(frozen=True)
class ProgressivePlanningArtifactPaths:
    """Paths for the persisted outline-to-compile authority chain."""

    outline: Path
    materializations: Path
    skeleton: Path
    compile_receipt: Path


def _authority_digest(value: object, *, field: str) -> str | None:
    if value is None:
        return None
    digest = str(value).strip().lower()
    if not _SHA256_RE.fullmatch(digest):
        raise ProgressivePlanningArtifactError(
            "progressive_schema_authority_invalid",
            f"{field} must be null or a lowercase SHA-256 digest",
        )
    return digest


def _register_once(
    evidence: ProgressiveEvidenceRegistrar,
    *,
    evidence_id: str,
    description: str,
    source_path: Path,
    inputs: Sequence[str],
    producer: str,
    generation_mode: str,
    prompt_pack_version: str,
) -> None:
    if evidence.get(evidence_id) is not None:
        return
    evidence.register_file(
        kind="log",
        description=description,
        source_path=source_path,
        evidence_id=evidence_id,
        inputs=list(inputs),
        producer=producer,
        generation_mode=generation_mode,
        prompt_pack_version=prompt_pack_version,
    )


def persist_progressive_planning_artifacts(
    *,
    run_dir: Path,
    evidence: ProgressiveEvidenceRegistrar,
    outline: ProgressivePlanOutline,
    materializations: Sequence[ProgressiveStepMaterialization],
    skeleton: ProgressivePlanSkeleton,
    compile_receipt: ProgressivePlanCompileReceipt,
    prompt_metrics: Mapping[str, Any],
    prompt_pack_version: str,
) -> ProgressivePlanningArtifactPaths:
    """Write and register one complete outline/materialize/compile evidence chain."""
    if not materializations:
        raise ProgressivePlanningArtifactError(
            "progressive_materialization_ledger_empty",
            "at least one current-step materialization is required",
        )

    observed_outline_sha256 = canonical_sha256(outline.model_dump(mode="json"))
    if prompt_metrics.get("outline_sha256") != observed_outline_sha256:
        raise ProgressivePlanningArtifactError(
            "progressive_outline_identity_mismatch",
            "prompt metrics do not identify the persisted outline",
        )
    if prompt_metrics.get("final_skeleton_sha256") != compile_receipt.skeleton_sha256:
        raise ProgressivePlanningArtifactError(
            "progressive_skeleton_identity_mismatch",
            "prompt metrics and compile receipt identify different skeletons",
        )

    raw_step_schema_digests = prompt_metrics.get(
        "step_materialization_schema_sha256"
    )
    if not isinstance(raw_step_schema_digests, list) or len(
        raw_step_schema_digests
    ) != len(materializations):
        raise ProgressivePlanningArtifactError(
            "progressive_step_schema_authority_count_mismatch",
            "one schema authority entry is required per materialized step",
        )
    step_schema_digests = [
        _authority_digest(value, field=f"step_schema[{index}]")
        for index, value in enumerate(raw_step_schema_digests)
    ]
    outline_schema_digest = _authority_digest(
        prompt_metrics.get("structured_output_authority_sha256"),
        field="outline_schema",
    )

    outline_path = run_dir / "progressive_plan_outline.json"
    outline_path.write_text(outline.model_dump_json(indent=2), encoding="utf-8")
    _register_once(
        evidence,
        evidence_id="progressive_plan_outline",
        description=(
            "Retrieval-informed high-level scientific outline before any "
            "executable step detail."
        ),
        source_path=outline_path,
        inputs=("research_context",),
        producer="progressive_planner",
        generation_mode="llm",
        prompt_pack_version=prompt_pack_version,
    )

    materializations_path = run_dir / "progressive_step_materializations.json"
    materializations_path.write_text(
        json.dumps(
            {
                "schema_version": (
                    "easyicu.progressive_step_materialization_ledger/1"
                ),
                "outline_sha256": observed_outline_sha256,
                "outline_structured_output_authority_sha256": (
                    outline_schema_digest
                ),
                "materializations": [
                    {
                        "step_id": item.step.step_id,
                        "structured_output_authority_sha256": (
                            step_schema_digests[index]
                        ),
                        "materialization": item.model_dump(mode="json"),
                    }
                    for index, item in enumerate(materializations)
                ],
            },
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    _register_once(
        evidence,
        evidence_id="progressive_step_materializations",
        description=(
            "Ordered current-step strict materializations with their run-bound "
            "schema authority digests."
        ),
        source_path=materializations_path,
        inputs=("progressive_plan_outline", "research_context"),
        producer="progressive_planner",
        generation_mode="llm",
        prompt_pack_version=prompt_pack_version,
    )

    skeleton_path = run_dir / "progressive_plan_skeleton.json"
    skeleton_path.write_text(skeleton.model_dump_json(indent=2), encoding="utf-8")
    _register_once(
        evidence,
        evidence_id="progressive_plan_skeleton",
        description=(
            "Host-assembled scientific skeleton produced from coordinate-bound "
            "step materializations."
        ),
        source_path=skeleton_path,
        inputs=(
            "progressive_plan_outline",
            "progressive_step_materializations",
            "research_context",
        ),
        producer="progressive_planner_compiler",
        generation_mode="deterministic_skill",
        prompt_pack_version=prompt_pack_version,
    )

    receipt_path = run_dir / "progressive_plan_compile_receipt.json"
    receipt_path.write_text(
        compile_receipt.model_dump_json(indent=2),
        encoding="utf-8",
    )
    _register_once(
        evidence,
        evidence_id="progressive_plan_compile_receipt",
        description=(
            "Host-derived immutable-prefix and AnalysisPlan compilation receipt "
            "for Progressive Planner v2."
        ),
        source_path=receipt_path,
        inputs=("progressive_plan_skeleton", "research_context"),
        producer="progressive_plan_compiler",
        generation_mode="deterministic_skill",
        prompt_pack_version=prompt_pack_version,
    )
    return ProgressivePlanningArtifactPaths(
        outline=outline_path,
        materializations=materializations_path,
        skeleton=skeleton_path,
        compile_receipt=receipt_path,
    )


__all__ = [
    "ProgressivePlanningArtifactError",
    "ProgressivePlanningArtifactPaths",
    "persist_progressive_planning_artifacts",
]
