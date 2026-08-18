"""Orchestrate one Progressive Planner call and optional Dev checkpoint replay."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Mapping

from ..agents.progressive_planner import ProgressivePlannerAgent
from ..authority.evidence_store import sha256_of_file
from ..planning.progressive_artifacts import (
    ProgressiveEvidenceRegistrar,
    ProgressivePlannerCheckpointRecorder,
    ProgressiveResumePersistenceReceipt,
    load_progressive_planner_checkpoint_chain,
    persist_progressive_compile_failure_replay,
    persist_progressive_planning_artifacts,
)
from ..planning.preplan_know_how import PlannerKnowHowBinding
from ..schema import AnalysisPlan, ResearchContext, ValidationFinding


@dataclass(frozen=True)
class ProgressivePlannerRunResult:
    """Normalized plan plus the provenance-visible generation mode."""

    plan: AnalysisPlan
    generation_mode: str
    prompt_metrics: Mapping[str, Any]


def _resume_finding(
    *,
    receipt: ProgressiveResumePersistenceReceipt,
    terminal_artifact_sha256: str,
) -> ValidationFinding:
    return ValidationFinding(
        validator="progressive_planner_resume",
        severity="warning",
        message=(
            "Development-only Progressive Planner prefix was dependency-verified "
            "and recompiled by the current host."
        ),
        detail={
            "reason_code": "progressive_development_checkpoint_resumed",
            "development_only": True,
            "source_terminal_artifact_sha256": terminal_artifact_sha256,
            "source_checkpoint_sha256": receipt.source_checkpoint_sha256,
            "source_sequence": receipt.source_sequence,
            "reused_materialization_count": receipt.reused_materialization_count,
            "new_checkpoint_count": receipt.new_checkpoint_count,
        },
    )


def run_progressive_planner(
    *,
    planner: ProgressivePlannerAgent,
    context: ResearchContext,
    run_dir: Path,
    evidence: ProgressiveEvidenceRegistrar,
    prompt_pack_version: str,
    resume_checkpoint_path: str | Path | None,
    resume_checkpoint_sha256: str | None,
    cohort_path: Path,
    llm_signature: str,
    planner_kwargs: Mapping[str, Any],
    know_how_binding: PlannerKnowHowBinding,
    planning_contract_context: str,
    finding_sink: Callable[[ValidationFinding], None],
) -> ProgressivePlannerRunResult:
    """Run Progressive v2 while importing only a host-validated Dev chain."""

    source_chain = (
        load_progressive_planner_checkpoint_chain(
            last_checkpoint_path=Path(resume_checkpoint_path).expanduser(),
            expected_artifact_sha256=str(resume_checkpoint_sha256 or ""),
        )
        if resume_checkpoint_path is not None
        else ()
    )
    recorder = ProgressivePlannerCheckpointRecorder(
        run_dir=run_dir,
        evidence=evidence,
        prompt_pack_version=prompt_pack_version,
        source_chain=source_chain,
    )
    reserved = {
        "checkpoint_callback",
        "resume_checkpoint",
        "resume_dependency_context",
    }
    overlap = sorted(reserved & set(planner_kwargs))
    if overlap:
        raise ValueError(
            "progressive orchestration owns replay kwargs: " + ", ".join(overlap)
        )

    try:
        plan = planner.run(
            context,
            **dict(planner_kwargs),
            checkpoint_callback=recorder.record,
            resume_checkpoint=source_chain[-1] if source_chain else None,
            resume_dependency_context={
                "cohort_file_sha256": sha256_of_file(cohort_path),
                "llm_signature": llm_signature,
                "prompt_version": prompt_pack_version,
            },
        )
    except Exception:
        planner.capture_efficiency_metrics()
        if source_chain and planner.last_resume_validated:
            receipt = recorder.persist_validated_resume()
            finding_sink(
                _resume_finding(
                    receipt=receipt,
                    terminal_artifact_sha256=str(resume_checkpoint_sha256),
                )
            )
        if (
            planner.last_compile_failure_attempts
            and recorder.latest_checkpoint is not None
        ):
            persist_progressive_compile_failure_replay(
                run_dir=run_dir,
                evidence=evidence,
                attempts=planner.last_compile_failure_attempts,
                prefix_checkpoint=recorder.latest_checkpoint,
                prompt_pack_version=prompt_pack_version,
            )
        raise

    if source_chain:
        if not planner.last_resume_validated:
            raise RuntimeError(
                "Progressive Planner returned without validating its development "
                "resume checkpoint."
            )
        receipt = recorder.persist_validated_resume()
        finding_sink(
            _resume_finding(
                receipt=receipt,
                terminal_artifact_sha256=str(resume_checkpoint_sha256),
            )
        )
    prompt_metrics = know_how_binding.prompt_metrics(
        planner,
        context,
        planning_contract_context=planning_contract_context,
    )
    persist_progressive_planner_output(
        planner=planner,
        run_dir=run_dir,
        evidence=evidence,
        prompt_metrics=prompt_metrics,
        prompt_pack_version=prompt_pack_version,
    )
    return ProgressivePlannerRunResult(
        plan=plan,
        generation_mode=(
            "llm_progressive_v2_dev_resume"
            if source_chain
            else "llm_progressive_v2"
        ),
        prompt_metrics=prompt_metrics,
    )


def persist_progressive_planner_output(
    *,
    planner: ProgressivePlannerAgent,
    run_dir: Path,
    evidence: ProgressiveEvidenceRegistrar,
    prompt_metrics: Mapping[str, Any],
    prompt_pack_version: str,
) -> None:
    """Persist the complete outline-to-compiler chain from one Planner run."""

    if (
        planner.last_outline is None
        or planner.last_foundation is None
        or not planner.last_materializations
        or planner.last_skeleton is None
        or planner.last_compile_receipt is None
    ):
        raise RuntimeError(
            "Progressive Planner returned without its outline, foundation, step "
            "materializations, skeleton, or compile receipt"
        )
    persist_progressive_planning_artifacts(
        run_dir=run_dir,
        evidence=evidence,
        outline=planner.last_outline,
        foundation=planner.last_foundation,
        materializations=planner.last_materializations,
        skeleton=planner.last_skeleton,
        compile_receipt=planner.last_compile_receipt,
        prompt_metrics=prompt_metrics,
        prompt_pack_version=prompt_pack_version,
    )


__all__ = [
    "ProgressivePlannerRunResult",
    "persist_progressive_planner_output",
    "run_progressive_planner",
]
