"""Persist the Progressive Planner contract chain as governed evidence.

This module owns the boundary between planning contracts and EvidenceStore.  It
does not plan, compile, execute, or infer authority from mutable pipeline state.
"""

from __future__ import annotations

import json
import hashlib
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal, Mapping, Protocol, Sequence

from pydantic import BaseModel, ConfigDict, Field, model_validator

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


class ProgressivePlanningStepAuthority(BaseModel):
    """One materialized step and the exact schema that governed its response."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    step_id: str
    materialization_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    structured_output_authority_sha256: str | None = Field(
        default=None,
        pattern=r"^[0-9a-f]{64}$",
    )


class ProgressivePlanningAuthority(BaseModel):
    """Content-addressed root joining progressive planning to normalized Plan."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: Literal["easyicu.progressive_planning_authority/1"] = (
        "easyicu.progressive_planning_authority/1"
    )
    planner_strategy: Literal["progressive_v2"] = "progressive_v2"
    outline_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    outline_structured_output_authority_sha256: str | None = Field(
        default=None,
        pattern=r"^[0-9a-f]{64}$",
    )
    ordered_steps: tuple[ProgressivePlanningStepAuthority, ...]
    compiled_skeleton_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    compiled_analysis_plan_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    normalized_plan_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    normalized_plan_authority_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    outline_artifact_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    materialization_ledger_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    skeleton_artifact_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    compile_receipt_artifact_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    planner_prompt_metrics_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    analysis_plan_artifact_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    normalized_plan_artifact_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    strict_transport_bound: bool
    authority_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")

    @model_validator(mode="after")
    def _verify_authority(self) -> "ProgressivePlanningAuthority":
        if not self.ordered_steps:
            raise ValueError("progressive planning authority has no steps")
        step_ids = [item.step_id for item in self.ordered_steps]
        if len(step_ids) != len(set(step_ids)):
            raise ValueError("progressive planning authority repeats a step id")
        observed_strict = bool(
            self.outline_structured_output_authority_sha256
            and all(
                item.structured_output_authority_sha256
                for item in self.ordered_steps
            )
        )
        if self.strict_transport_bound is not observed_strict:
            raise ValueError("progressive strict transport projection mismatch")
        unsigned = self.model_dump(mode="json", exclude={"authority_sha256"})
        if canonical_sha256(unsigned) != self.authority_sha256:
            raise ValueError("progressive planning authority digest mismatch")
        return self


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


def _record_sha256(record: object) -> str | None:
    raw = (
        record.get("sha256")
        if isinstance(record, Mapping)
        else getattr(record, "sha256", None)
    )
    digest = str(raw or "").strip().lower()
    return digest if _SHA256_RE.fullmatch(digest) else None


def _verified_source_bytes(
    *,
    run_dir: Path,
    evidence: ProgressiveEvidenceRegistrar,
    evidence_id: str,
    filename: str,
) -> tuple[bytes, str]:
    record = evidence.get(evidence_id)
    expected_sha256 = _record_sha256(record) if record is not None else None
    path = Path(run_dir) / filename
    if expected_sha256 is None:
        raise ProgressivePlanningArtifactError(
            "progressive_source_evidence_missing",
            f"{evidence_id} has no registered SHA-256 authority",
        )
    if path.is_symlink() or not path.is_file():
        raise ProgressivePlanningArtifactError(
            "progressive_source_artifact_missing",
            f"{filename} is absent or not a regular file",
        )
    content = path.read_bytes()
    observed_sha256 = hashlib.sha256(content).hexdigest()
    if observed_sha256 != expected_sha256:
        raise ProgressivePlanningArtifactError(
            "progressive_source_artifact_digest_mismatch",
            f"{filename} differs from EvidenceStore authority",
        )
    return content, observed_sha256


def _write_and_register_once(
    evidence: ProgressiveEvidenceRegistrar,
    *,
    content: str,
    evidence_id: str,
    description: str,
    source_path: Path,
    inputs: Sequence[str],
    producer: str,
    generation_mode: str,
    prompt_pack_version: str,
) -> None:
    content_sha256 = hashlib.sha256(content.encode("utf-8")).hexdigest()
    existing = evidence.get(evidence_id)
    if existing is not None:
        existing_sha256 = _record_sha256(existing)
        if existing_sha256 != content_sha256:
            raise ProgressivePlanningArtifactError(
                "progressive_existing_evidence_identity_mismatch",
                f"{evidence_id} already identifies different content",
            )
        source_path.write_text(content, encoding="utf-8")
        return
    source_path.write_text(content, encoding="utf-8")
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
    _write_and_register_once(
        evidence,
        content=outline.model_dump_json(indent=2),
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
    materializations_content = (
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
        + "\n"
    )
    _write_and_register_once(
        evidence,
        content=materializations_content,
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
    _write_and_register_once(
        evidence,
        content=skeleton.model_dump_json(indent=2),
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
    _write_and_register_once(
        evidence,
        content=compile_receipt.model_dump_json(indent=2),
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


def persist_progressive_planning_authority(
    *,
    run_dir: Path,
    evidence: ProgressiveEvidenceRegistrar,
    proposed_plan_sha256: str,
    normalized_plan_sha256: str,
    normalized_plan_authority_sha256: str,
    normalized_plan_evidence_id: str,
    normalized_plan_filename: str,
    prompt_pack_version: str,
) -> ProgressivePlanningAuthority:
    """Verify the on-disk chain and bind it to one normalized plan authority."""

    sources = {
        "outline": _verified_source_bytes(
            run_dir=run_dir,
            evidence=evidence,
            evidence_id="progressive_plan_outline",
            filename="progressive_plan_outline.json",
        ),
        "materializations": _verified_source_bytes(
            run_dir=run_dir,
            evidence=evidence,
            evidence_id="progressive_step_materializations",
            filename="progressive_step_materializations.json",
        ),
        "skeleton": _verified_source_bytes(
            run_dir=run_dir,
            evidence=evidence,
            evidence_id="progressive_plan_skeleton",
            filename="progressive_plan_skeleton.json",
        ),
        "compile_receipt": _verified_source_bytes(
            run_dir=run_dir,
            evidence=evidence,
            evidence_id="progressive_plan_compile_receipt",
            filename="progressive_plan_compile_receipt.json",
        ),
        "prompt_metrics": _verified_source_bytes(
            run_dir=run_dir,
            evidence=evidence,
            evidence_id="planner_prompt_metrics",
            filename="planner_prompt_metrics.json",
        ),
        "analysis_plan": _verified_source_bytes(
            run_dir=run_dir,
            evidence=evidence,
            evidence_id="analysis_plan",
            filename="analysis_plan.json",
        ),
        "normalized_plan": _verified_source_bytes(
            run_dir=run_dir,
            evidence=evidence,
            evidence_id=normalized_plan_evidence_id,
            filename=normalized_plan_filename,
        ),
    }
    try:
        outline = ProgressivePlanOutline.model_validate_json(sources["outline"][0])
        ledger = json.loads(sources["materializations"][0])
        skeleton = ProgressivePlanSkeleton.model_validate_json(
            sources["skeleton"][0]
        )
        receipt = ProgressivePlanCompileReceipt.model_validate_json(
            sources["compile_receipt"][0]
        )
        prompt_metrics = json.loads(sources["prompt_metrics"][0])
    except Exception as exc:
        raise ProgressivePlanningArtifactError(
            "progressive_authority_source_invalid",
            str(exc),
        ) from exc
    if not isinstance(ledger, dict) or not isinstance(prompt_metrics, dict):
        raise ProgressivePlanningArtifactError(
            "progressive_authority_source_invalid",
            "materialization ledger and prompt metrics must be objects",
        )
    entries = ledger.get("materializations")
    if not isinstance(entries, list) or not entries:
        raise ProgressivePlanningArtifactError(
            "progressive_materialization_ledger_invalid",
            "materialization ledger must contain ordered steps",
        )
    try:
        materializations = [
            ProgressiveStepMaterialization.model_validate(item["materialization"])
            for item in entries
            if isinstance(item, dict)
        ]
    except Exception as exc:
        raise ProgressivePlanningArtifactError(
            "progressive_materialization_ledger_invalid",
            str(exc),
        ) from exc
    if len(materializations) != len(entries):
        raise ProgressivePlanningArtifactError(
            "progressive_materialization_ledger_invalid",
            "every ledger row must be one materialization object",
        )
    outline_step_ids = [item.step_id for item in outline.steps]
    materialized_step_ids = [item.step.step_id for item in materializations]
    skeleton_step_ids = [item.step_id for item in skeleton.steps]
    ledger_step_ids = [str(item.get("step_id") or "") for item in entries]
    if not (
        outline_step_ids
        == ledger_step_ids
        == materialized_step_ids
        == skeleton_step_ids
    ):
        raise ProgressivePlanningArtifactError(
            "progressive_step_order_identity_mismatch",
            "outline, ledger, materializations, and skeleton disagree on step order",
        )
    outline_sha256 = canonical_sha256(outline.model_dump(mode="json"))
    if (
        ledger.get("outline_sha256") != outline_sha256
        or prompt_metrics.get("outline_sha256") != outline_sha256
    ):
        raise ProgressivePlanningArtifactError(
            "progressive_outline_identity_mismatch",
            "authority sources disagree on the outline digest",
        )
    skeleton_sha256 = canonical_sha256(skeleton.model_dump(mode="json"))
    if (
        skeleton_sha256 != receipt.skeleton_sha256
        or prompt_metrics.get("final_skeleton_sha256") != skeleton_sha256
    ):
        raise ProgressivePlanningArtifactError(
            "progressive_skeleton_identity_mismatch",
            "authority sources disagree on the compiled skeleton digest",
        )
    if receipt.analysis_plan_sha256 != proposed_plan_sha256:
        raise ProgressivePlanningArtifactError(
            "progressive_compiled_plan_identity_mismatch",
            "compiler receipt does not identify the normalized lineage proposal",
        )
    outline_schema_sha256 = _authority_digest(
        ledger.get("outline_structured_output_authority_sha256"),
        field="outline_schema",
    )
    if outline_schema_sha256 != _authority_digest(
        prompt_metrics.get("structured_output_authority_sha256"),
        field="prompt_metrics.outline_schema",
    ):
        raise ProgressivePlanningArtifactError(
            "progressive_schema_authority_mismatch",
            "outline schema authority differs between ledger and prompt metrics",
        )
    metrics_step_schema = prompt_metrics.get("step_materialization_schema_sha256")
    if not isinstance(metrics_step_schema, list) or len(metrics_step_schema) != len(
        entries
    ):
        raise ProgressivePlanningArtifactError(
            "progressive_step_schema_authority_count_mismatch",
            "prompt metrics do not contain one schema digest per step",
        )
    ordered_steps: list[ProgressivePlanningStepAuthority] = []
    for index, (entry, materialization) in enumerate(
        zip(entries, materializations, strict=True)
    ):
        entry_schema_sha256 = _authority_digest(
            entry.get("structured_output_authority_sha256"),
            field=f"ledger.step_schema[{index}]",
        )
        if entry_schema_sha256 != _authority_digest(
            metrics_step_schema[index],
            field=f"prompt_metrics.step_schema[{index}]",
        ):
            raise ProgressivePlanningArtifactError(
                "progressive_schema_authority_mismatch",
                f"step schema authority differs at index {index}",
            )
        ordered_steps.append(
            ProgressivePlanningStepAuthority(
                step_id=materialization.step.step_id,
                materialization_sha256=canonical_sha256(
                    materialization.model_dump(mode="json")
                ),
                structured_output_authority_sha256=entry_schema_sha256,
            )
        )
    body: dict[str, Any] = {
        "schema_version": "easyicu.progressive_planning_authority/1",
        "planner_strategy": "progressive_v2",
        "outline_sha256": outline_sha256,
        "outline_structured_output_authority_sha256": outline_schema_sha256,
        "ordered_steps": [item.model_dump(mode="json") for item in ordered_steps],
        "compiled_skeleton_sha256": skeleton_sha256,
        "compiled_analysis_plan_sha256": receipt.analysis_plan_sha256,
        "normalized_plan_sha256": str(normalized_plan_sha256),
        "normalized_plan_authority_sha256": str(
            normalized_plan_authority_sha256
        ),
        "outline_artifact_sha256": sources["outline"][1],
        "materialization_ledger_sha256": sources["materializations"][1],
        "skeleton_artifact_sha256": sources["skeleton"][1],
        "compile_receipt_artifact_sha256": sources["compile_receipt"][1],
        "planner_prompt_metrics_sha256": sources["prompt_metrics"][1],
        "analysis_plan_artifact_sha256": sources["analysis_plan"][1],
        "normalized_plan_artifact_sha256": sources["normalized_plan"][1],
        "strict_transport_bound": bool(
            outline_schema_sha256
            and all(
                item.structured_output_authority_sha256 for item in ordered_steps
            )
        ),
    }
    body["authority_sha256"] = canonical_sha256(body)
    authority = ProgressivePlanningAuthority.model_validate(body)
    _write_and_register_once(
        evidence,
        content=authority.model_dump_json(indent=2),
        evidence_id="progressive_planning_authority",
        description=(
            "Content-addressed Progressive Planner outline, per-step schema, "
            "compile, and normalized-plan authority root."
        ),
        source_path=Path(run_dir) / "progressive_planning_authority.json",
        inputs=(
            "progressive_plan_outline",
            "progressive_step_materializations",
            "progressive_plan_skeleton",
            "progressive_plan_compile_receipt",
            "planner_prompt_metrics",
            "analysis_plan",
            normalized_plan_evidence_id,
        ),
        producer="progressive_planning_authority",
        generation_mode="deterministic_skill",
        prompt_pack_version=prompt_pack_version,
    )
    return authority


__all__ = [
    "ProgressivePlanningArtifactError",
    "ProgressivePlanningArtifactPaths",
    "ProgressivePlanningAuthority",
    "ProgressivePlanningStepAuthority",
    "persist_progressive_planning_artifacts",
    "persist_progressive_planning_authority",
]
