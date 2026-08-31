"""Orchestrate one Progressive Planner call and optional Dev checkpoint replay."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Mapping, Optional

from ..agents.progressive_planner import ProgressivePlannerAgent
from ..authority.evidence_store import sha256_of_file
from ..planning.progressive_artifacts import (
    ProgressiveCompileReplayAttempt,
    ProgressiveEvidenceRegistrar,
    ProgressivePlannerCheckpointRecorder,
    ProgressiveResumePersistenceReceipt,
    load_progressive_planner_checkpoint_chain,
    persist_progressive_compile_failure_replay,
    persist_progressive_design_canary_receipt,
    persist_progressive_planning_artifacts,
)
from ..planning.progressive_contract import (
    ProgressiveFoundationMaterialization,
    ProgressivePlanCompileReceipt,
    ProgressivePlanOutline,
    ProgressivePlanSkeleton,
    ProgressivePlannerCheckpoint,
    ProgressiveStepMaterialization,
)
from ..planning.preplan_know_how import PlannerKnowHowBinding
from ..planning import literature_design_authority as _literature_design
from ..schema import AnalysisPlan, ResearchContext, ValidationFinding
from .workflow import PlannerDesignCanaryComplete


@dataclass(frozen=True)
class ProgressivePlannerRunFacts:
    """Success and failure facts captured together from one attempt."""

    prompt_metrics: Mapping[str, Any]
    compile_receipt: Optional[ProgressivePlanCompileReceipt]
    outline: Optional[ProgressivePlanOutline]
    foundation: Optional[ProgressiveFoundationMaterialization]
    materializations: tuple[ProgressiveStepMaterialization, ...]
    compile_failure_attempts: tuple[ProgressiveCompileReplayAttempt, ...]
    skeleton: Optional[ProgressivePlanSkeleton]
    resume_validated: bool
    dropped_plan_keys: Mapping[str, tuple[str, ...]]

    @property
    def complete_for_persistence(self) -> bool:
        return bool(
            self.outline is not None
            and self.foundation is not None
            and self.materializations
            and self.skeleton is not None
            and self.compile_receipt is not None
        )


def snapshot_progressive_planner_run(
    planner: ProgressivePlannerAgent,
) -> ProgressivePlannerRunFacts:
    """Capture every success or failure fact without temporal caller reads."""

    return ProgressivePlannerRunFacts(
        prompt_metrics=dict(planner.last_prompt_metrics),
        compile_receipt=planner.last_compile_receipt,
        outline=planner.last_outline,
        foundation=planner.last_foundation,
        materializations=tuple(planner.last_materializations),
        compile_failure_attempts=tuple(planner.last_compile_failure_attempts),
        skeleton=planner.last_skeleton,
        resume_validated=bool(planner.last_resume_validated),
        dropped_plan_keys={
            str(key): tuple(str(value) for value in values)
            for key, values in planner.last_dropped_plan_keys.items()
        },
    )


@dataclass(frozen=True)
class ProgressivePlannerRunResult:
    """Plan and every provenance fact captured from the same attempt."""

    plan: AnalysisPlan
    generation_mode: str
    prompt_metrics: Mapping[str, Any]
    facts: ProgressivePlannerRunFacts


@dataclass(frozen=True)
class ProgressiveDesignCanaryDraft:
    """Validated design outline before any executable-plan materialization."""

    outline: ProgressivePlanOutline
    checkpoint: ProgressivePlannerCheckpoint
    generation_mode: str
    prompt_metrics: Mapping[str, Any]


def finalize_progressive_design_canary(
    draft: ProgressiveDesignCanaryDraft,
    run_id: str,
    run_dir: Path,
    evidence: ProgressiveEvidenceRegistrar,
    cost_meter: Any,
    provider_hard_stop: Any,
    prompt_pack_version: str,
    emit_progress: Callable[..., None],
) -> PlannerDesignCanaryComplete:
    """Persist and project one non-executable design-canary terminal result."""

    emit_progress(
        "plan",
        "Research-design canary completed at the validated outline boundary.",
        run_id=run_id,
        total_steps=0,
    )
    hard_stop_accounting = None
    accounting_summary = getattr(provider_hard_stop, "accounting_summary", None)
    if callable(accounting_summary):
        hard_stop_accounting = accounting_summary()
    cost_summary = (
        cost_meter.summary(hard_stop_accounting=hard_stop_accounting)
        if cost_meter is not None
        else {}
    )
    receipt, receipt_path, receipt_sha256 = (
        persist_progressive_design_canary_receipt(
            run_dir=run_dir,
            evidence=evidence,
            checkpoint=draft.checkpoint,
            prompt_metrics=draft.prompt_metrics,
            cost_summary=cost_summary,
            prompt_pack_version=prompt_pack_version,
        )
    )
    return PlannerDesignCanaryComplete(
        run_id=run_id,
        run_dir=str(run_dir),
        receipt_path=str(receipt_path),
        receipt_sha256=receipt_sha256,
        candidate_design_count=receipt.candidate_design_count,
        rejected_design_count=len(receipt.rejected_design_ids),
        selected_literature_dimension_count=(
            receipt.selected_literature_dimension_count
        ),
        provider_calls=int(receipt.planner_efficiency.get("calls") or 0),
        reported_tokens=int(
            receipt.planner_efficiency.get("reported_tokens") or 0
        ),
        estimated_cost_usd=(
            float(cost_summary["total_cost_usd"])
            if cost_summary.get("total_cost_usd") is not None
            else None
        ),
    )


def run_pipeline_progressive_planner(
    *,
    planner: ProgressivePlannerAgent,
    context: ResearchContext,
    run_dir: Path,
    evidence: ProgressiveEvidenceRegistrar,
    prompt_pack_version: str,
    resume_checkpoint_path: str | Path | None,
    resume_checkpoint_sha256: str | None,
    stop_after_outline: bool,
    cohort_path: Path,
    llm_signature: str,
    planner_kwargs: Mapping[str, Any],
    preplan_literature: Any,
    required_primary_cohort_selection_mode: str | None,
    know_how_binding: PlannerKnowHowBinding,
    planning_contract_context: str,
    finding_sink: Callable[[ValidationFinding], None],
) -> ProgressivePlannerRunResult | ProgressiveDesignCanaryDraft:
    """Bind pipeline-owned inputs to the narrow Progressive Planner contract."""

    kwargs = dict(planner_kwargs)
    kwargs |= _literature_design.progressive_literature_design_kwargs(
        preplan_literature
    )
    kwargs["required_primary_cohort_selection_mode"] = (
        required_primary_cohort_selection_mode
    )
    return run_progressive_planner(
        planner=planner,
        context=context,
        run_dir=run_dir,
        evidence=evidence,
        prompt_pack_version=prompt_pack_version,
        resume_checkpoint_path=resume_checkpoint_path,
        resume_checkpoint_sha256=resume_checkpoint_sha256,
        cohort_path=cohort_path,
        llm_signature=llm_signature,
        planner_kwargs=kwargs,
        know_how_binding=know_how_binding,
        planning_contract_context=planning_contract_context,
        finding_sink=finding_sink,
        stop_after_outline=stop_after_outline,
    )


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
    stop_after_outline: bool = False,
) -> ProgressivePlannerRunResult | ProgressiveDesignCanaryDraft:
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
        generated = planner.run(
            context,
            **dict(planner_kwargs),
            checkpoint_callback=recorder.record,
            resume_checkpoint=source_chain[-1] if source_chain else None,
            resume_dependency_context={
                "cohort_file_sha256": sha256_of_file(cohort_path),
                "llm_signature": llm_signature,
                "prompt_version": prompt_pack_version,
            },
            stop_after_outline=stop_after_outline,
        )
    except BaseException:
        planner.capture_efficiency_metrics()
        facts = snapshot_progressive_planner_run(planner)
        if source_chain and facts.resume_validated:
            receipt = recorder.persist_validated_resume()
            finding_sink(
                _resume_finding(
                    receipt=receipt,
                    terminal_artifact_sha256=str(resume_checkpoint_sha256),
                )
            )
        if facts.compile_failure_attempts and recorder.latest_checkpoint is not None:
            persist_progressive_compile_failure_replay(
                run_dir=run_dir,
                evidence=evidence,
                attempts=facts.compile_failure_attempts,
                prefix_checkpoint=recorder.latest_checkpoint,
                prompt_pack_version=prompt_pack_version,
            )
        raise

    facts = snapshot_progressive_planner_run(planner)
    if source_chain:
        if not facts.resume_validated:
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
    if stop_after_outline:
        if not isinstance(generated, ProgressivePlanOutline):
            raise RuntimeError(
                "Progressive outline-only canary returned an executable plan"
            )
        checkpoint = recorder.latest_checkpoint
        if checkpoint is None or checkpoint.stage != "outline":
            raise RuntimeError(
                "Progressive outline-only canary has no validated outline checkpoint"
            )
        return ProgressiveDesignCanaryDraft(
            outline=generated,
            checkpoint=checkpoint,
            generation_mode=(
                "llm_progressive_v2_design_canary_dev_resume"
                if source_chain
                else "llm_progressive_v2_design_canary"
            ),
            prompt_metrics=prompt_metrics,
        )
    if not isinstance(generated, AnalysisPlan):
        raise RuntimeError("Progressive Planner returned no executable AnalysisPlan")
    persist_progressive_planner_output(
        facts=facts,
        run_dir=run_dir,
        evidence=evidence,
        prompt_metrics=prompt_metrics,
        prompt_pack_version=prompt_pack_version,
    )
    return ProgressivePlannerRunResult(
        plan=generated,
        generation_mode=(
            "llm_progressive_v2_dev_resume"
            if source_chain
            else "llm_progressive_v2"
        ),
        prompt_metrics=prompt_metrics,
        facts=facts,
    )


def persist_progressive_planner_output(
    *,
    facts: ProgressivePlannerRunFacts,
    run_dir: Path,
    evidence: ProgressiveEvidenceRegistrar,
    prompt_metrics: Mapping[str, Any],
    prompt_pack_version: str,
) -> None:
    """Persist the complete outline-to-compiler chain from one Planner run."""

    if not facts.complete_for_persistence:
        raise RuntimeError(
            "Progressive Planner returned without its outline, foundation, step "
            "materializations, skeleton, or compile receipt"
        )
    persist_progressive_planning_artifacts(
        run_dir=run_dir,
        evidence=evidence,
        outline=facts.outline,
        foundation=facts.foundation,
        materializations=facts.materializations,
        skeleton=facts.skeleton,
        compile_receipt=facts.compile_receipt,
        prompt_metrics=prompt_metrics,
        prompt_pack_version=prompt_pack_version,
    )


__all__ = [
    "finalize_progressive_design_canary",
    "ProgressiveDesignCanaryDraft",
    "ProgressivePlannerRunResult",
    "persist_progressive_planner_output",
    "run_progressive_planner",
    "run_pipeline_progressive_planner",
]
