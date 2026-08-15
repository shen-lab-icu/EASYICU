"""Rehydrate a digest-bound Plan-to-review handoff after process restart.

This owner contains the recovery mechanics.  ``ResearchAgentPipeline`` remains
the execution host and supplies the phase invokers; recovery itself must never
invoke Planner or infer missing checkpoint state.
"""

from __future__ import annotations

import json
import os
import re
import tempfile
from copy import deepcopy
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Dict, Mapping, Optional, Sequence

from ..authority.evidence_store import (
    EvidenceEnforcementError,
    EvidenceStore,
    sha256_of_file,
)
from ..authority.run_input import (
    RUN_INPUT_CAPSULE_EVIDENCE_ID,
    build_environment_identity,
    load_verified_run_input_capsule,
)
from ..authority.declared_levels import bind_step_declared_levels
from ..authority.table_one_binding import bind_table_one_execution_spec
from ..authority.runtime_artifacts import AuditLogger
from ..authority.plan_lifecycle import approve_normalized_plan_for_execution
from ..canonical_json import canonical_sha256
from ..intake.materialized_trajectory import (
    MaterializedTrajectoryAuthorityRef,
    StagedTrajectoryBinding,
    VerifiedLegacyTrajectoryCapsuleReceipt,
)
from ..literature import LiteratureBundle
from ..research_context.typed import parse_research_context_json
from ..schema import AnalysisPlan, ResearchContext, ValidationFinding
from ..contracts.runtime import _WritePhaseResult
from ..skills import get_skill
from .human_review_checkpoint import (
    HumanReviewCheckpointError,
    HumanReviewCheckpointPhaseUncertain,
    checkpoint_path,
    load_checkpoint,
    write_checkpoint,
)
from .progress import ResumableProgressChannel
from .profiles import is_paper_facing_profile
from .workflow import HumanReviewDecision, HumanReviewPending, build_pipeline_workflow


def _atomic_write_json(path: Path, payload: Mapping[str, Any]) -> None:
    """Publish review evidence durably before registering its digest."""

    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    temporary = Path(temporary_name)
    try:
        os.fchmod(descriptor, 0o600)
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
        directory_fd = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
    finally:
        temporary.unlink(missing_ok=True)


def persist_human_review_records(
    records: Sequence[Mapping[str, Any]],
    *,
    run_id: str,
    run_dir: Path,
    evidence: EvidenceStore,
    submission_profile_name: str,
) -> None:
    """Persist authenticated decisions and a rejection terminal receipt."""

    if not records:
        return
    decision_records = [dict(item) for item in records]
    if is_paper_facing_profile(submission_profile_name):
        unauthenticated = [
            str(record.get("review_id") or "<unknown>")
            for record in decision_records
            if record.get("reviewer_identity_source") != "authenticated"
        ]
        if unauthenticated:
            raise RuntimeError(
                "human review under submission profile "
                f"{submission_profile_name!r} requires an authenticated reviewer "
                "identity; a client-claimed reviewer is diagnostic-only "
                f"(unauthenticated: {', '.join(unauthenticated)})"
            )
    decisions_path = run_dir / "human_review_decisions.json"
    _atomic_write_json(
        decisions_path,
        {
            "schema": "easyicu.human_review_decisions/1",
            "run_id": run_id,
            "decisions": decision_records,
        },
    )
    evidence.register_file(
        kind="log",
        description=(
            "Operator decisions for the human-review interrupts raised by this "
            "run, with server-stamped decision time and authority digest."
        ),
        source_path=decisions_path,
        evidence_id="human_review_decisions",
        producer="pipeline",
        generation_mode="human_confirmed",
    )
    rejected = [
        str(record.get("review_id") or "<unknown>")
        for record in decision_records
        if record.get("decision") == "rejected"
    ]
    if not rejected:
        return
    run_status_path = run_dir / "run_status.json"
    _atomic_write_json(
        run_status_path,
        {
            "schema_version": "easyicu.run_status/2",
            "run_id": run_id,
            "status": "human_review_rejected",
            "strict_fail_closed": True,
            "terminal_reason": "operator_rejected",
            "rejected_review_ids": rejected,
            "gates": {
                "human_review_approved": False,
                "execution_complete": False,
                "manuscript_ready": False,
                "publication_ready": False,
                "publication_artifacts_ready": False,
                "execution_paper_eligible": False,
                "paper_authorized": False,
            },
            "canonical_outputs": {
                "human_review_decisions": "human_review_decisions.json",
                "run_status": "run_status.json",
            },
        },
    )
    evidence.register_file(
        kind="log",
        description="Fail-closed terminal status for an operator-rejected pause.",
        source_path=run_status_path,
        evidence_id="run_status",
        aliases=["run_status"],
        producer="pipeline",
        generation_mode="system",
    )


def bind_checkpoint_decision_payloads(
    checkpoint_commit: Dict[str, Any],
    *,
    requests: Sequence[Any],
    payload: Sequence[Mapping[str, Any]],
) -> None:
    """Order decision payloads by the signed request set and bind their digest."""

    by_review_id = {
        str(item.get("review_id") or ""): dict(item) for item in payload
    }
    ordered = [by_review_id.get(str(request.review_id), {}) for request in requests]
    checkpoint_commit["decision_payloads"] = ordered
    checkpoint_commit["decision_sha256"] = canonical_sha256(ordered)


def recover_checkpoint_decisions_from_evidence(
    checkpoint_file: Path,
    *,
    run_dir: Path,
) -> Any:
    """Converge the historical recorder-before-checkpoint crash window."""

    checkpoint = load_checkpoint(checkpoint_file, require_pending=False)
    if checkpoint.state != "pending" or checkpoint.approved_decisions_sha256:
        return checkpoint
    decisions_path = run_dir / "human_review_decisions.json"
    if not decisions_path.is_file() or decisions_path.is_symlink():
        return checkpoint
    try:
        envelope = json.loads(decisions_path.read_text(encoding="utf-8"))
        records = envelope["decisions"]
        if (
            envelope.get("schema") != "easyicu.human_review_decisions/1"
            or envelope.get("run_id") != checkpoint.run_id
            or not isinstance(records, list)
        ):
            raise ValueError("invalid decision evidence envelope")
        requests = {request.review_id: request for request in checkpoint.requests}
        payloads = []
        for raw_record in records:
            if not isinstance(raw_record, Mapping):
                raise ValueError("invalid decision record")
            record = dict(raw_record)
            request = requests.get(str(record.get("review_id") or ""))
            if request is None:
                raise ValueError("decision record does not bind a paused request")
            payload = HumanReviewDecision(
                review_id=request.review_id,
                authority_sha256=str(record.get("authority_sha256") or ""),
                decision=str(record.get("decision") or ""),
                reviewer=str(record.get("claimed_reviewer") or ""),
                decided_at=str(record.get("claimed_decided_at") or ""),
                note=str(record.get("note") or ""),
            ).model_dump(mode="json")
            if (
                record.get("request_sha256")
                != canonical_sha256(request.model_dump(mode="json"))
                or record.get("decision_sha256") != canonical_sha256(payload)
            ):
                raise ValueError("decision record digest mismatch")
            payloads.append(payload)
        if len(payloads) != len(checkpoint.requests):
            raise ValueError("decision evidence does not cover the request set")
        by_id = {item["review_id"]: item for item in payloads}
        ordered = [by_id[request.review_id] for request in checkpoint.requests]
        by_record_id = {str(item["review_id"]): dict(item) for item in records}
        ordered_records = [
            by_record_id[request.review_id] for request in checkpoint.requests
        ]
    except Exception as exc:
        raise HumanReviewCheckpointError(
            "existing human-review decision evidence cannot be recovered"
        ) from exc
    recovered = checkpoint.decision_recorded(
        decisions=ordered,
        decision_records=ordered_records,
        decision_sha256=canonical_sha256(ordered),
    )
    write_checkpoint(checkpoint_file, recovered)
    return recovered


def prepare_human_review_decision(
    *,
    checkpoint_file: Path,
    decision_payloads: Sequence[Mapping[str, Any]],
    decision_records: Sequence[Mapping[str, Any]],
    decision_sha256: str,
) -> None:
    """Stage exact decisions before any separately persisted review evidence."""

    payloads = [dict(item) for item in decision_payloads]
    records = [dict(item) for item in decision_records]
    if not re.fullmatch(r"[0-9a-f]{64}", str(decision_sha256)) or not payloads:
        raise HumanReviewCheckpointError(
            "durable review decision set is unavailable or not digest bound"
        )
    selected = load_checkpoint(checkpoint_file, require_pending=False)
    write_checkpoint(
        checkpoint_file,
        selected.decision_recorded(
            decisions=payloads,
            decision_records=records,
            decision_sha256=str(decision_sha256),
        ),
    )


def commit_human_review_decision(
    *,
    checkpoint_file: Path,
    run_dir: Path,
    evidence: EvidenceStore,
    plan_revision: int,
    decision_payloads: Sequence[Mapping[str, Any]],
    decision_records: Sequence[Mapping[str, Any]],
    decision_sha256: str,
) -> None:
    """Commit staged evidence and authorize an approved plan before Execute."""

    payloads = [dict(item) for item in decision_payloads]
    records = [dict(item) for item in decision_records]
    if not re.fullmatch(r"[0-9a-f]{64}", str(decision_sha256)):
        raise HumanReviewCheckpointError(
            "durable review decision set is not digest bound"
        )
    if not payloads:
        raise HumanReviewCheckpointError(
            "durable review decision payloads are unavailable"
        )
    selected = load_checkpoint(checkpoint_file, require_pending=False)
    if (
        selected.consumed_decision_sha256 != str(decision_sha256)
        or list(selected.approved_decisions) != payloads
        or list(selected.approved_decision_records) != records
    ):
        raise HumanReviewCheckpointError(
            "restored decision does not match the durable decision set"
        )
    if selected.state == "pending":
        selected = selected.decision_committed()
        write_checkpoint(checkpoint_file, selected)
    if selected.state == "rejected":
        return
    if selected.state in {"approved_pending_execution", "executing"}:
        approve_normalized_plan_for_execution(
            run_dir=run_dir,
            evidence=evidence,
            revision=int(plan_revision),
            review_requests=selected.requests,
            decision_set_sha256=str(decision_sha256),
        )
        return
    raise HumanReviewCheckpointError(
        f"checkpoint state {selected.state!r} is not resumable"
    )


def mark_human_review_execution_started(checkpoint_file: Path) -> None:
    """Durably bind the Execute start before the first analysis side effect."""

    selected = load_checkpoint(checkpoint_file, require_pending=False)
    write_checkpoint(checkpoint_file, selected.execution_started())


def mark_human_review_execution_phase(
    checkpoint_file: Path,
    phase: str,
) -> None:
    """Fail-closed marker around Write and Finalize side-effect boundaries."""

    if phase not in {"write_in_progress", "finalize_in_progress"}:
        raise ValueError(f"unsupported human-review execution phase: {phase!r}")
    selected = load_checkpoint(checkpoint_file, require_pending=False)
    write_checkpoint(
        checkpoint_file,
        selected.execution_phase_started(phase),  # type: ignore[arg-type]
    )


def fail_human_review_checkpoint(checkpoint_commit: Mapping[str, Any]) -> None:
    """Terminalise a resumable durable handoff without hiding its last state."""

    path = checkpoint_commit.get("path")
    if not path:
        return
    selected = load_checkpoint(Path(str(path)), require_pending=False)
    if selected.state in {"write_in_progress", "finalize_in_progress"}:
        # The exception proves failure, but not whether the paid/irreversible
        # side effect happened before it was raised. Preserve the explicit
        # phase so restart cannot mistake this for a safely replayable failure.
        return
    if selected.state not in {
        "pending",
        "approved_pending_execution",
        "executing",
        "consumed",
    }:
        return
    decision_sha256 = (
        str(checkpoint_commit.get("decision_sha256") or "") or None
        if selected.state != "pending"
        else None
    )
    write_checkpoint(
        Path(str(path)),
        selected.transitioned(
            "failed",
            decision_sha256=decision_sha256,
        ),
    )


def restore_durable_human_review_pause(
    pipeline: Any,
    *,
    run_id: str,
    progress_callback: Optional[Callable[[Dict[str, Any]], None]],
    plan_result_factory: Callable[..., Any],
    load_resume_state: Callable[[Path], Optional[Dict[str, Any]]],
    rejection_only: bool = False,
) -> Dict[str, Any]:
    """Restore one verified pause without repeating Plan or provider calls."""

    run_dir = (Path(pipeline.workdir) / str(run_id)).resolve()
    checkpoint_file = checkpoint_path(run_dir)
    checkpoint = recover_checkpoint_decisions_from_evidence(
        checkpoint_file,
        run_dir=run_dir,
    )
    if checkpoint.state in {"write_in_progress", "finalize_in_progress"}:
        raise HumanReviewCheckpointPhaseUncertain(
            "durable human-review recovery is fail-closed at explicit checkpoint "
            f"phase {checkpoint.state!r}; paid or irreversible side effects may "
            "already have occurred and will not be replayed automatically"
        )
    if checkpoint.state in {"rejected", "consumed", "completed", "failed"}:
        raise HumanReviewCheckpointError(
            f"durable human-review checkpoint is already {checkpoint.state}"
        )
    if checkpoint.run_id != str(run_id) or checkpoint.thread_id != str(run_id):
        raise HumanReviewCheckpointError(
            "durable human-review checkpoint belongs to a different run"
        )
    if (
        not rejection_only
        and checkpoint.pipeline_config_sha256 != pipeline._config.canonical_digest()
    ):
        raise HumanReviewCheckpointError(
            "pipeline configuration changed after human review was requested"
        )
    if rejection_only and checkpoint.state != "pending":
        raise HumanReviewCheckpointError(
            f"checkpoint state {checkpoint.state!r} cannot accept a new rejection"
        )
    if not rejection_only:
        llm = pipeline._llm
        if llm is None:
            raise HumanReviewCheckpointError(
                "durable human-review resume requires the configured provider"
            )
        llm_signature = pipeline._llm_signature(llm)
        if canonical_sha256(llm_signature) != checkpoint.llm_signature_sha256:
            raise HumanReviewCheckpointError(
                "model provider identity changed after human review was requested"
            )
        if build_environment_identity(llm_signature=llm_signature) != (
            checkpoint.environment_identity
        ):
            raise HumanReviewCheckpointError(
                "execution environment changed after human review was requested"
            )
        activation_sha256 = canonical_sha256(
            pipeline._capability_runtime.activation.model_dump(mode="json")
            if pipeline._capability_runtime.activation is not None
            else None
        )
        if activation_sha256 != checkpoint.capability_activation_sha256:
            raise HumanReviewCheckpointError(
                "capability activation changed after human review was requested"
            )
        pipeline._approved_capability_resources = (
            pipeline._capability_runtime.approved_resources
        )

    execution = dict(checkpoint.execution_coordinates)
    scientific_identity = dict(execution.get("scientific_identity") or {})
    target_outcome = execution.get("target_outcome")
    cohort_path = run_dir
    runtime_capabilities: Sequence[str] = ()
    if not rejection_only:
        authority = load_verified_run_input_capsule(
            run_dir=run_dir,
            scientific_identity=scientific_identity,
        )
        capsule_record = authority.evidence_records.get(RUN_INPUT_CAPSULE_EVIDENCE_ID)
        if (
            not isinstance(capsule_record, Mapping)
            or str(capsule_record.get("sha256") or "")
            != checkpoint.run_input_capsule_sha256
        ):
            raise HumanReviewCheckpointError(
                "run-input capsule no longer matches the reviewed execution"
            )
        cohort_path = run_dir / authority.capsule.cohort_relative_path
        runtime_capabilities = pipeline._preflight_execution_runtime(
            run_dir=run_dir,
            cohort_path=cohort_path,
            target_outcome=str(target_outcome) if target_outcome else None,
            cap_provider_timeout=False,
        )
        if tuple(runtime_capabilities) != checkpoint.runtime_capabilities:
            raise HumanReviewCheckpointError(
                "runtime capability set changed after human review was requested"
            )
        if canonical_sha256(pipeline._validated_runtime_bundle) != (
            checkpoint.runtime_bundle_sha256
        ):
            raise HumanReviewCheckpointError(
                "runtime capability bundle changed after human review was requested"
            )

    handoff = dict(checkpoint.plan_handoff)
    try:
        context = ResearchContext.model_validate(handoff["context"])
        agent_context = ResearchContext.model_validate(handoff["agent_context"])
        findings = [
            ValidationFinding.model_validate(item)
            for item in list(handoff.get("findings") or ())
        ]
        plan = AnalysisPlan.model_validate(handoff["plan"])
        preplan_literature = (
            LiteratureBundle.model_validate(handoff["preplan_literature"])
            if handoff.get("preplan_literature") is not None
            else None
        )
        started_at = datetime.fromisoformat(str(handoff["started_at"]))
    except Exception as exc:
        raise HumanReviewCheckpointError(
            "durable human-review Plan handoff is invalid"
        ) from exc
    context_path = pipeline._checkpoint_run_path(
        run_dir=run_dir, relative_path=handoff.get("context_path")
    )
    plan_path = pipeline._checkpoint_run_path(
        run_dir=run_dir, relative_path=handoff.get("plan_path")
    )
    assert context_path is not None and plan_path is not None
    try:
        if parse_research_context_json(context_path.read_text(encoding="utf-8")) != context:
            raise ValueError("context differs")
        if AnalysisPlan.model_validate_json(plan_path.read_text(encoding="utf-8")) != plan:
            raise ValueError("plan differs")
    except Exception as exc:
        raise HumanReviewCheckpointError(
            "reviewed context or plan artifact changed after the pause"
        ) from exc
    for step in plan.steps:
        bind_table_one_execution_spec(step, agent_context)
        bind_step_declared_levels(step, agent_context)

    if rejection_only:
        def role_resolver(_role: str) -> Any:
            raise RuntimeError("a rejection-only restore cannot invoke the provider")

        cost_meter = None
        repro_envelope = None
    else:
        role_resolver, cost_meter, repro_envelope = pipeline._restore_role_handoff(
            run_id=str(run_id),
            run_dir=run_dir,
            repro_payload=(
                handoff.get("repro_envelope")
                if isinstance(handoff.get("repro_envelope"), Mapping)
                else None
            ),
        )
    evidence = EvidenceStore(
        run_dir, enforcement_mode=pipeline._evidence_enforcement_mode
    )
    plan_result = plan_result_factory(
        context=context,
        agent_context=agent_context,
        context_path=context_path,
        evidence=evidence,
        findings=findings,
        plan=plan,
        plan_path=plan_path,
        llm_signature=str(handoff.get("llm_signature") or ""),
        used_mock_llm=bool(handoff.get("used_mock_llm")),
        prompt_version=str(handoff.get("prompt_version") or ""),
        prompt_files=dict(handoff.get("prompt_files") or {}),
        role_resolver=role_resolver,
        cost_meter=cost_meter,
        repro_envelope=repro_envelope,
        started_at=started_at,
        resume_state=(
            load_resume_state(run_dir) if checkpoint.state == "executing" else None
        ),
        allowed_literature_citation_keys=tuple(
            str(item)
            for item in list(handoff.get("allowed_literature_citation_keys") or ())
        ),
        direct_comparator_literature_keys=tuple(
            str(item)
            for item in list(handoff.get("direct_comparator_literature_keys") or ())
        ),
        preplan_literature=preplan_literature,
    )

    trajectory_binding = None
    trajectory_payload = execution.get("trajectory_binding")
    if not rejection_only and isinstance(trajectory_payload, Mapping):
        trajectory_path = pipeline._checkpoint_run_path(
            run_dir=run_dir, relative_path=trajectory_payload.get("path")
        )
        assert trajectory_path is not None
        expected_sha256 = str(trajectory_payload.get("sha256") or "")
        expected_size = int(trajectory_payload.get("size") or 0)
        if (
            sha256_of_file(trajectory_path) != expected_sha256
            or trajectory_path.stat().st_size != expected_size
        ):
            raise HumanReviewCheckpointError(
                "reviewed trajectory artifact changed after the pause"
            )
        raw_ref = trajectory_payload.get("authority_ref")
        raw_legacy = trajectory_payload.get("legacy_capsule_receipt")
        trajectory_binding = StagedTrajectoryBinding(
            path=trajectory_path,
            sha256=expected_sha256,
            size=expected_size,
            authority_ref=(
                MaterializedTrajectoryAuthorityRef.from_dict(raw_ref)
                if isinstance(raw_ref, Mapping)
                else None
            ),
            legacy_capsule_receipt=(
                VerifiedLegacyTrajectoryCapsuleReceipt(**dict(raw_legacy))
                if isinstance(raw_legacy, Mapping)
                else None
            ),
        )

    progress_channel = ResumableProgressChannel(progress_callback)
    audit_logger = AuditLogger(run_dir / "audit_log.jsonl")
    progress_channel.bind_audit_logger(audit_logger)
    emit_progress = progress_channel.emit
    skill_key = str(execution.get("skill_key") or "").strip()
    skill_obj = get_skill(skill_key) if skill_key and not rejection_only else None
    stop_after_step_id = execution.get("stop_after_step_id")
    stop_after_analysis = bool(execution.get("stop_after_analysis"))
    manuscript_authors = tuple(
        str(item) for item in list(execution.get("manuscript_authors") or ())
    )
    experiment_spec_path = (
        pipeline._checkpoint_run_path(
            run_dir=run_dir,
            relative_path=execution.get("experiment_spec_path"),
            required=False,
        )
        if not rejection_only
        else None
    )
    cache_key = execution.get("cache_key")

    def execute_invoker(restored_plan: Any):
        return pipeline._run_execute_phase(
            plan_result=restored_plan,
            cohort_path=cohort_path,
            trajectory_binding=trajectory_binding,
            run_dir=run_dir,
            run_id=str(run_id),
            skill_obj=skill_obj,
            notes=execution.get("notes"),
            emit_progress=emit_progress,
            resume_from_step_id=None,
            stop_after_step_id=str(stop_after_step_id) if stop_after_step_id else None,
        )

    def write_invoker(restored_plan: Any, execute_result: Any):
        try:
            return pipeline._run_write_phase(
                plan_result=restored_plan,
                execute_result=execute_result,
                run_dir=run_dir,
                run_id=str(run_id),
                stop_after_analysis=stop_after_analysis,
                manuscript_title=(
                    str(execution.get("manuscript_title"))
                    if execution.get("manuscript_title")
                    else None
                ),
                manuscript_authors=manuscript_authors,
                run_language=str(execution.get("run_language") or "en"),
                emit_progress=emit_progress,
                force_writer_probe=bool(execution.get("force_writer_probe")),
            )
        except EvidenceEnforcementError as exc:
            validator = (
                "manuscript_numeric_auditor"
                if "untraced" in getattr(exc, "detail", {})
                else "evidence_bound_writer"
            )
            restored_plan.findings.append(
                ValidationFinding(
                    validator=validator,
                    severity="error",
                    message=(
                        "STRICT evidence enforcement blocked manuscript "
                        f"generation: {exc}"
                    ),
                    detail=getattr(exc, "detail", {}) or None,
                )
            )
            bound_path = run_dir / "manuscript_scaffold_bound.md"
            bound_path.write_text(
                "# Manuscript scaffold not generated\n\n"
                "STRICT evidence enforcement failed before final binding.\n\n"
                f"Error: {exc}\n",
                encoding="utf-8",
            )
            emit_progress(
                "writer",
                "STRICT evidence enforcement blocked manuscript generation.",
                status="error",
                run_id=str(run_id),
            )
            return _WritePhaseResult(literature=None, bound_path=bound_path)

    def finalise_invoker(restored_plan: Any, execute_result: Any, write_result: Any):
        return pipeline._finalise_success(
            plan_result=restored_plan,
            execute_result=execute_result,
            write_result=write_result,
            run_id=str(run_id),
            run_dir=run_dir,
            cohort_path=cohort_path,
            notes=execution.get("notes"),
            database=str(execution.get("database") or ""),
            target_outcome=str(target_outcome) if target_outcome else None,
            stop_after_analysis=stop_after_analysis,
            cache_key=str(cache_key) if cache_key else None,
            scientific_identity=scientific_identity,
            experiment_spec_path=experiment_spec_path,
            audit_logger=audit_logger,
            emit_progress=emit_progress,
        )

    checkpoint_commit: Dict[str, Any] = {
        "path": str(checkpoint_file),
        "decision_sha256": checkpoint.consumed_decision_sha256,
        "decision_payloads": list(checkpoint.approved_decisions),
    }

    def prepare_human_review_execution(
        decision_records: Sequence[Mapping[str, Any]],
    ) -> None:
        prepare_human_review_decision(
            checkpoint_file=checkpoint_file,
            decision_payloads=checkpoint_commit.get("decision_payloads") or (),
            decision_records=decision_records,
            decision_sha256=str(checkpoint_commit.get("decision_sha256") or ""),
        )

    def commit_human_review_execution(
        decision_records: Sequence[Mapping[str, Any]],
    ) -> None:
        commit_human_review_decision(
            checkpoint_file=checkpoint_file,
            run_dir=run_dir,
            evidence=evidence,
            plan_revision=plan_result.plan.revision,
            decision_payloads=checkpoint_commit.get("decision_payloads") or (),
            decision_records=decision_records,
            decision_sha256=str(checkpoint_commit.get("decision_sha256") or ""),
        )

    def commit_human_review_execution_start() -> None:
        mark_human_review_execution_started(checkpoint_file)

    def commit_human_review_write_start() -> None:
        mark_human_review_execution_phase(checkpoint_file, "write_in_progress")

    def commit_human_review_finalize_start() -> None:
        mark_human_review_execution_phase(checkpoint_file, "finalize_in_progress")

    workflow = build_pipeline_workflow(
        plan_invoker=lambda: (_ for _ in ()).throw(
            RuntimeError("restored workflow must not invoke Planner")
        ),
        execute_invoker=execute_invoker,
        write_invoker=write_invoker,
        finalise_invoker=finalise_invoker,
        human_review_recorder=lambda records: persist_human_review_records(
            records,
            run_id=str(run_id),
            run_dir=run_dir,
            evidence=evidence,
            submission_profile_name=pipeline._submission_profile_name,
        ),
        human_review_decision_prepare=prepare_human_review_execution,
        human_review_execution_commit=commit_human_review_execution,
        human_review_execution_start=commit_human_review_execution_start,
        human_review_write_start=commit_human_review_write_start,
        human_review_finalize_start=commit_human_review_finalize_start,
        reviewer_identity_resolver=(
            getattr(pipeline._human_review_gate, "reviewer_identity_resolver", None)
            if pipeline._human_review_gate is not None
            else None
        ),
    )
    workflow.restore_paused(
        plan_result=plan_result,
        requests=checkpoint.requests,
        decision_payloads=checkpoint.approved_decisions,
        decision_records=checkpoint.approved_decision_records,
    )
    pending = HumanReviewPending(
        run_id=str(run_id),
        thread_id=str(run_id),
        run_dir=str(run_dir),
        requests=checkpoint.requests,
        resume_scope="durable_checkpoint",
        resume_pid=None,
    )
    state = {
        "workflow": workflow,
        "pending": pending,
        "runtime_capabilities": tuple(runtime_capabilities),
        "runtime_bundle": (
            deepcopy(pipeline._validated_runtime_bundle)
            if not rejection_only
            else None
        ),
        "progress_sink": progress_channel,
        "checkpoint_commit": checkpoint_commit,
    }
    pipeline._pending_human_review = state
    return state
