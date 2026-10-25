"""Biomni-style governed one-shot entry point.

Biomni's ``A1().go(prompt)`` is a useful usability benchmark, but EasyICU
deliberately does not copy its "full system privileges" execution model.
This module provides the same one-call ergonomics while keeping every
EasyICU invariant: the returned run stays fail-closed, analysis-only, and
never gains manuscript/publication authority through this facade.
"""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Optional, Sequence, Union

from .canonical_json import sha256_file
from .orchestration.profiles import PUBLIC_DEMO_ACCEPTANCE
from .orchestration.workflow import HumanReviewDecision, HumanReviewPending
from .pipeline import ResearchAgentPipeline
from .providers.clients import RecordingLLMClient, ReplayLLMClient
from .providers.mocks import MockLLMClient
from .providers.protocol import RecordedApproval, ReplayCassette, ReplayError

__all__ = ["go", "replicate", "resume_human_review"]


def go(
    question: str,
    *,
    cohort: Union[str, Path, Any],
    llm: Optional[Any] = None,
    workdir: Optional[Union[str, Path]] = None,
    database: str = "miiv",
    target_outcome: Optional[str] = None,
    primary_exposure: Optional[str] = None,
    inclusion_criteria: Optional[list[str]] = None,
    exclusion_criteria: Optional[list[str]] = None,
    time_windows: Optional[list[Any]] = None,
    notes: Optional[str] = None,
    manuscript_language: Optional[str] = None,
    stop_after_analysis: bool = True,
    progress_callback: Optional[Any] = None,
    **run_kwargs: Any,
) -> Any:
    """Run one research question through the governed Plan → Execute → Write path.

    This is a convenience wrapper over :class:`ResearchAgentPipeline`. It
    intentionally does not expose publication authority, formal-mode toggles,
    or provider-budget overrides: those stay on the explicit pipeline/CLI
    surface. ``llm`` defaults to the deterministic offline mock; pass a real
    provider client only when the surrounding process has already satisfied
    the opt-in gate.
    """
    client = llm if llm is not None else MockLLMClient()
    pipeline = ResearchAgentPipeline(
        workdir=workdir,
        llm=client,
    )
    return pipeline.run(
        question=question,
        cohort=cohort,
        database=database,
        target_outcome=target_outcome,
        primary_exposure=primary_exposure,
        inclusion_criteria=inclusion_criteria,
        exclusion_criteria=exclusion_criteria,
        time_windows=time_windows,
        notes=notes,
        manuscript_language=manuscript_language,
        stop_after_analysis=stop_after_analysis,
        progress_callback=progress_callback,
        **run_kwargs,
    )


def replicate(
    *,
    cohorts: dict[str, Union[str, Path, Any]],
    llm: Optional[Any] = None,
    workdir: Optional[Union[str, Path]] = None,
    question: Optional[str] = None,
    target_outcome: Optional[str] = None,
    cohort_name_prefix: str = "cohort",
    stop_after_analysis: bool = True,
    notes: Optional[str] = None,
    manuscript_language: Optional[str] = None,
    **replicate_kwargs: Any,
) -> Any:
    """Run the same question across cohorts/databases, facade-style.

    Mirrors :meth:`ResearchAgentPipeline.replicate` with the same fail-closed
    default client (offline mock unless ``llm`` is supplied by a caller that
    already satisfied the opt-in gate).
    """
    client = llm if llm is not None else MockLLMClient()
    pipeline = ResearchAgentPipeline(workdir=workdir, llm=client)
    return pipeline.replicate(
        cohorts=cohorts,
        question=question,
        target_outcome=target_outcome,
        cohort_name_prefix=cohort_name_prefix,
        stop_after_analysis=stop_after_analysis,
        notes=notes,
        manuscript_language=manuscript_language,
        **replicate_kwargs,
    )


def resume_human_review(
    decisions: list[Any],
    *,
    llm: Optional[Any] = None,
    workdir: Optional[Union[str, Path]] = None,
    run_id: Optional[str] = None,
    progress_callback: Optional[Any] = None,
) -> Any:
    """Answer a durable human-review pause through the same thin surface."""
    client = llm if llm is not None else MockLLMClient()
    pipeline = ResearchAgentPipeline(workdir=workdir, llm=client)
    return pipeline.resume_human_review(
        decisions,
        run_id=run_id,
        progress_callback=progress_callback,
    )


def verified_review_coordinates(pending, request) -> tuple[str, str]:
    """Verify the fresh complete plan packet and every evidence digest it binds."""
    from .authority.evidence_store import EvidenceStore
    from .authority.plan_review import PlanReviewAuthority

    root = Path(pending.run_dir)
    authority = PlanReviewAuthority.model_validate(request.payload["plan_review_authority"])
    records = {record.evidence_id: record.sha256 for record in EvidenceStore(root=root).verified_records()}
    if not authority.evidence_sha256 or any(records.get(key) != value for key, value in authority.evidence_sha256.items()):
        raise ReplayError("model_replay_review_evidence_mismatch")
    capsule_bytes = (root / "run_input_capsule.json").read_bytes()
    if authority.execution is None or hashlib.sha256(capsule_bytes).hexdigest() != authority.execution.run_input_capsule_sha256:
        raise ReplayError("model_replay_review_evidence_mismatch")
    capsule = json.loads(capsule_bytes)
    # The local capsule and the complete local evidence set are checked again
    # by resume_human_review. Portable approval binds the actual cohort bytes,
    # not a timestamp, work directory or an old run's runtime authority.
    cohort_sha = sha256_file(root / "cohort.parquet")
    if cohort_sha != capsule["cohort_sha256"]:
        raise ReplayError("model_replay_review_evidence_mismatch")
    return authority.plan_sha256, cohort_sha


def replay_review_decisions(pending, approvals: Sequence[RecordedApproval]):
    decisions = []
    for request in pending.requests:
        approval = next((item for item in approvals if item.review_id == request.review_id
                         and item.authority_sha256 == request.authority_sha256), None)
        if approval is None and getattr(request, "payload", {}).get("reason") == "operator_plan_approval_required":
            plan_sha, cohort_sha = verified_review_coordinates(pending, request)
            approval = next((item for item in approvals if item.acceptance_reuse_authorized
                and item.plan_sha256 == plan_sha and item.cohort_sha256 == cohort_sha), None)
        if approval is None:
            raise ReplayError("model_replay_review_missing", request_sha256=request.authority_sha256)
        decisions.append(HumanReviewDecision(
            review_id=request.review_id, authority_sha256=request.authority_sha256,
            decision=approval.decision, reviewer="recorded public-demo acceptance approval",
            decided_at=datetime.now(timezone.utc).isoformat(),
            note="Recorded acceptance-only grant; current complete plan and input evidence verified.",
            population_criterion_acceptances=approval.population_criterion_acceptances,
        ))
    return decisions


def verify_first_result(run_dir: Path) -> int:
    """Require verified primary outputs sealed by the normal execution owner."""
    from .authority.evidence_store import EvidenceStore
    from .authority.plan_input_closure import resolve_registered_plan_authority
    from .authority.step_runtime import load_explicit_executed_success_step_capsule
    from .gates.fallback_generation import primary_records_for_readiness
    from .schema import AnalysisPlan

    try:
        manifest = json.loads((run_dir / "manifest.json").read_text(encoding="utf-8"))
        gates = manifest["readiness"]
        if any(gates.get(key) is not True for key in ("execution_complete", "analysis_validated", "plan_authority_verified")):
            raise ReplayError("model_replay_result_unverified")
        if gates.get("paper_authorized") is not False:
            raise ReplayError("model_replay_result_unverified")
        plan_path = (run_dir / manifest["plan_path"]).resolve()
        plan_path.relative_to(run_dir.resolve())
        plan = AnalysisPlan.model_validate_json(plan_path.read_text(encoding="utf-8"))
        evidence = EvidenceStore(root=run_dir)
        records = evidence.verified_records()
        resolve_registered_plan_authority(run_dir=run_dir, evidence=evidence, plan=plan, plan_path=plan_path)
        primary_ids = {step.step_id for step in plan.steps if step.planned_analysis_role == "primary"}
        outputs = set()
        for row in primary_records_for_readiness(manifest["per_step_records"], plan):
            if row.get("step_id") not in primary_ids or not row.get("step_authority_capsule_ref"):
                continue
            executed = load_explicit_executed_success_step_capsule(run_dir, step_id=row["step_id"], record=row)
            if executed is not None and executed.capsule.execution is not None:
                outputs.update(output.content.sha256 for output in executed.capsule.execution.outputs)
        verified = [record for record in records if record.kind in {"statistic", "table"} and record.sha256 in outputs]
        if not verified:
            raise ReplayError("model_replay_result_unverified")
        return len(verified)
    except ReplayError:
        raise
    except (OSError, ValueError, KeyError, TypeError, RuntimeError) as exc:
        raise ReplayError("model_replay_result_unverified") from exc


def run_public_demo(*, export_path: Path, database: str, work_dir: Path, client,
                    question: str, primary_exposure: str, target_outcome: str,
                    covariates: Sequence[str], review: Callable):
    """The same materializer, profile and pipeline for recording and playback."""
    from .cohort.materializer import materialize_to_parquet
    from .orchestration.config import PipelineConfig
    from .orchestration.services import PipelineServices
    from .pipeline import ResearchAgentPipeline
    from .providers.clients import require_replay_client_scope
    from .providers.client_trust import require_run_clearance

    if type(client) not in {ReplayLLMClient, RecordingLLMClient}:
        raise ReplayError("model_replay_client_required")
    require_replay_client_scope(client)
    require_run_clearance(database, None, client)
    if export_path.resolve() != client._scope.export_path:
        raise ReplayError("model_replay_source_mismatch")
    if (work_dir / "runs").exists():
        raise ReplayError("model_replay_workdir_not_empty")
    work_dir.mkdir(parents=True, exist_ok=True)
    paths = materialize_to_parquet(work_dir / "materialized", data_path=export_path,
        database=database, static_concepts=tuple(dict.fromkeys((primary_exposure, *covariates))),
        feature_concepts=(), outcome_concepts=(target_outcome,))
    config = PipelineConfig(workdir=work_dir / "runs", runner_kind="host", host_runner_authorized=True,
        development_diagnostic=True, **PUBLIC_DEMO_ACCEPTANCE.pipeline_options())
    pipeline = ResearchAgentPipeline(config=config, services=PipelineServices(llm=client))
    outcome = pipeline.run(question=question, cohort=paths["parquet"], database=database,
        primary_exposure=primary_exposure, target_outcome=target_outcome)
    while isinstance(outcome, HumanReviewPending):
        decisions = review(outcome)
        outcome = pipeline.resume_human_review(decisions, run_id=outcome.run_id)
    run_dir = Path(outcome.manifest_path).parent
    count = verify_first_result(run_dir)
    return {"status": "passed", "reason_code": "public_demo_replay_result_verified",
            "evidence_bound_result": True, "verified_primary_artifacts": count,
            "reportable": False, "publishable": False, "analysis_executed": True}


def replay_public_demo(*, cassette_path: Path, export_path: Path, database: str, work_dir: Path,
                       question: str, primary_exposure: str, target_outcome: str,
                       covariates: Sequence[str]):
    from easyicu.webserver.demo_source_storage import public_demo_replay_scope

    cassette = ReplayCassette.read(cassette_path)
    with public_demo_replay_scope(export_path=export_path, database=database, profile_ref=PUBLIC_DEMO_ACCEPTANCE.ref):
        client = ReplayLLMClient(cassette)
        receipt = client.receipt()
        work_dir.mkdir(parents=True, exist_ok=True)
        # Persist provenance before the first model request, including failures.
        (work_dir / "model_replay_receipt.json").write_text(json.dumps(receipt, indent=2) + "\n", encoding="utf-8")
        try:
            result = run_public_demo(export_path=export_path, database=database, work_dir=work_dir,
                client=client, question=question, primary_exposure=primary_exposure, target_outcome=target_outcome,
                covariates=covariates, review=lambda pending: replay_review_decisions(pending, cassette.approvals))
        except ReplayError as exc:
            result = {"status": "blocked", "reason_code": exc.reason_code, "evidence_bound_result": False,
                      "missing_request_sha256": exc.request_sha256, "analysis_executed": False}
        except Exception:
            # Aggregate receipts retain replay provenance even when a later
            # owner fails; raw diagnostics stay in the private run directory.
            result = {"status": "blocked", "reason_code": "model_replay_pipeline_failed",
                      "evidence_bound_result": False, "analysis_executed": False}
        return {**receipt, **result}


def record_review(pending, recorder: RecordingLLMClient, *, ask: Callable[[str], str] = input):
    """An explicit operator grant for this exact plan/input in acceptance only."""
    decisions = []
    for request in pending.requests:
        if request.payload.get("reason") != "operator_plan_approval_required":
            raise ReplayError("model_replay_scientific_review_required")
        plan_sha, cohort_sha = verified_review_coordinates(pending, request)
        print(json.dumps(request.model_dump(mode="json"), indent=2, ensure_ascii=False))
        answer = ask("Approve this complete plan and cohort for public-demo acceptance replay only? [approve/reject]: ")
        if answer.strip().lower() != "approve":
            raise ReplayError("model_replay_recording_not_approved")
        grant = RecordedApproval(review_id=request.review_id, authority_sha256=request.authority_sha256,
            decision="approved", acceptance_reuse_authorized=True, plan_sha256=plan_sha, cohort_sha256=cohort_sha)
        recorder.approvals[grant.review_id] = grant
        decisions.append(HumanReviewDecision(review_id=request.review_id, authority_sha256=request.authority_sha256,
            decision="approved", reviewer="public-demo acceptance recorder operator",
            decided_at=datetime.now(timezone.utc).isoformat(), note="Explicit acceptance replay approval recorded."))
    return decisions
