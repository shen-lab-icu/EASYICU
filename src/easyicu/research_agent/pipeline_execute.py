"""Execute phase of the research-agent pipeline.

Implements the probe → per-step analysis loop with optional replanning
and final figure visual-QA. Extracted from
:class:`ResearchAgentPipeline._run_execute_phase` (which is now a thin
delegate) so:

* the 1500-line execute loop reads as its own module;
* the planning / writing phases in :mod:`pipeline` don't have to scroll
  past it;
* a future graph-style runner (LangGraph or similar) has a single
  free-function entry point to wrap, rather than a method buried in a
  god-object.

The function is intentionally a free function, not a class. All state
that the execute phase mutates (``runtime_state``, ``per_step_records``,
``probe_summary``, ``findings``, ``plan``) is local to one call; nothing
needs to survive across calls. The pipeline instance is passed in only
as a *read-only collaborator* — execute-phase reads several ``_enable_*``
flags and calls ``pipeline._build_runner(...)``, but never mutates
pipeline state. The audit on 2026-05-15 confirmed zero ``self.* = ...``
writes inside the original method body.
"""

from __future__ import annotations

import json
import logging
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable, Dict, List, Optional, Sequence

from .agents import (
    AnalyzerAgent,
    ClinicalSemanticsAgent,
    CoderAgent,
    CriticAgent,
    DataExtractionAgent,
    ReplannerAgent,
    RuntimeSupervisor,
    StatisticalAnalysisAgent,
    VisualizationAgent,
)
from .audits.validators import (
    ClinicalConstraintValidator,
    ConceptUsageAuditor,
    LLMConceptAuditor,
    StatisticalGuard,
    StatisticalValidator,
)
from .code_repair import _deterministic_runner_repair, _deterministic_summary_repair
from .cohort_schema import assert_cohort_definition_locked
from .contracts import ValidationFinding, _ExecutePhaseResult, _PlanPhaseResult
from .estimators import fit_robustness_rows_from_records
from .llm import MockLLMClient
from .pipeline import (
    _build_probe_summary,
    _clear_output_dir,
    _has_figure_exports,
    _promote_prior_publication_bundle,
    _promote_sibling_figure_exports,
    _render_prediction_publication_bundle_from_prior_outputs,
    _semantic_aliases_for,
)
from .plan_utils import (
    _parent_step_id_for_figure_step,
    _preserve_figure_steps_after_replan,
    _step_contract_findings,
    _step_contract_repair_guidance,
    _step_expects_figure,
)
from .schema import AnalysisPlan, AnalysisStep, EvidenceRef
from .robustness_panel import (
    assert_robustness_specs_locked,
    build_robustness_panel_from_records,
    robustness_specs_for_execution,
    write_robustness_panel,
)
from .repair_registry import InvariantStatus, RepairLedger, RepairObservedState
from .scalar_utils import _expected_numeric_annotations_for_step
from .side_findings import SideFinding
from .skills import ClinicalSkill
from .summary_repair import salvage_step_summary
from .visual_qa import VLMVisualQAAdapter, VisualQAAuditor

logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    from .pipeline import ResearchAgentPipeline


def _is_cosmetic_visual_finding(finding: ValidationFinding) -> bool:
    """Return true only for deterministic layout errors safe to demote."""

    if finding.severity != "error" or finding.validator != "visual_qa":
        return False
    message = (finding.message or "").lower()
    return (
        "overlapping text elements" in message
        and "spacing" in message
    )


def _demote_cosmetic_visual_findings(
    findings: Sequence[ValidationFinding],
) -> tuple[List[ValidationFinding], List[ValidationFinding]]:
    """Demote cosmetic visual errors and return remaining hard errors."""

    demoted: List[ValidationFinding] = []
    for finding in findings:
        if _is_cosmetic_visual_finding(finding):
            demoted.append(finding.model_copy(update={"severity": "warning"}))
        else:
            demoted.append(finding)
    blocking_errors = [f for f in demoted if f.severity == "error"]
    return demoted, blocking_errors


def run_execute_phase(
    pipeline: "ResearchAgentPipeline",
    *,
    plan_result: _PlanPhaseResult,
    cohort_path: Path,
    run_dir: Path,
    run_id: str,
    skill_obj: Optional[ClinicalSkill],
    notes: Optional[str],
    emit_progress: Callable[..., None],
) -> _ExecutePhaseResult:
    """Execute probe + per-step analysis loop, with optional replanning."""
    context = plan_result.context
    agent_context = plan_result.agent_context
    evidence = plan_result.evidence
    findings = plan_result.findings
    plan = plan_result.plan
    plan_path = plan_result.plan_path
    role_resolver = plan_result.role_resolver
    llm_signature = plan_result.llm_signature
    prompt_version = plan_result.prompt_version
    prompt_files = plan_result.prompt_files
    assert_cohort_definition_locked(run_dir=run_dir, plan=plan)
    assert_robustness_specs_locked(run_dir=run_dir, plan=plan)

    coder = CoderAgent(role_resolver("coder"))
    analyzer = AnalyzerAgent(role_resolver("analyzer"))
    supervisor = RuntimeSupervisor(
        clinical_semantics=ClinicalSemanticsAgent(),
        data_extraction=DataExtractionAgent(),
        statistical_analysis=StatisticalAnalysisAgent(),
        visualization=VisualizationAgent(),
        critic=CriticAgent(role_resolver("analyzer")),
    )
    runner = pipeline._build_runner(
        run_dir=run_dir,
        cohort_path=cohort_path,
        target_outcome=context.target_outcome,
    )
    usage_auditor = ConceptUsageAuditor()
    from .audits.patterns import AnalysisPatternAuditor

    pattern_auditor = AnalysisPatternAuditor()
    stat_validator = StatisticalValidator()
    clinical_validator = ClinicalConstraintValidator()
    statistical_guard = StatisticalGuard()
    runtime_state = supervisor.bootstrap_state(run_id=run_id, context=context)
    repair_ledger = RepairLedger(run_dir / "repairs_applied.json")
    repair_ledger_lock = threading.Lock()

    per_step_records: List[Dict[str, Any]] = []
    probe_summary: Dict[str, Any] = {}
    resumed_step_ids: set = set()
    if plan_result.resume_state is not None:
        try:
            prior_records = [
                rec
                for rec in (
                    plan_result.resume_state.get("per_step_records", []) or []
                )
                if isinstance(rec, dict) and rec.get("step_id")
            ]
            prior_ok_step_ids = {
                rec["step_id"] for rec in prior_records if rec.get("status") == "ok"
            }
            for rec in plan_result.resume_state.get("per_step_records", []) or []:
                if (
                    isinstance(rec, dict)
                    and rec.get("status") == "ok"
                    and rec.get("step_id")
                ):
                    per_step_records.append(rec)
                    resumed_step_ids.add(rec["step_id"])
            for f in plan_result.resume_state.get("findings", []) or []:
                try:
                    finding = ValidationFinding.model_validate(f)
                except Exception:
                    continue
                if finding.validator == "cohort_auditor":
                    continue
                if finding.validator == "runner":
                    msg = finding.message or ""
                    if any(step_id in msg for step_id in prior_ok_step_ids):
                        continue
                findings.append(finding)
            if resumed_step_ids:
                print(
                    f"[research_agent] resume: skipping {len(resumed_step_ids)} "
                    f"already-completed step(s) — {sorted(resumed_step_ids)}"
                )
            for rec in per_step_records:
                if rec.get("step_id") == "00_probe" and isinstance(
                    rec.get("step_summary"), dict
                ):
                    probe_summary = rec["step_summary"]
        except Exception:
            resumed_step_ids = set()

    def _flush_partial_manifest(extra: Optional[Dict[str, Any]] = None) -> None:
        payload: Dict[str, Any] = {
            "schema_version": "easyicu.research_manifest_partial/1",
            "run_id": run_id,
            "research_question": context.research_question,
            "started_at": plan_result.started_at.isoformat(),
            "context_path": str(plan_result.context_path.relative_to(run_dir)),
            "plan_path": str(plan_path.relative_to(run_dir)),
            "evidence": [r.model_dump(mode="json") for r in evidence.records()],
            "findings": [f.model_dump(mode="json") for f in findings],
            "per_step_records": per_step_records,
            "llm_signature": llm_signature,
            "used_mock_llm": plan_result.used_mock_llm,
            "prompt_pack_version": prompt_version,
            "prompt_pack_files": prompt_files,
            "notes": notes,
            "runtime_state": runtime_state.model_dump(mode="json"),
            "repair_ledger_path": str(repair_ledger.path.relative_to(run_dir)),
            "repairs_applied": [
                record.__dict__ for record in repair_ledger.records
            ],
        }
        if extra:
            payload.update(extra)
        (run_dir / "manifest_partial.json").write_text(
            json.dumps(payload, indent=2, ensure_ascii=False, default=str),
            encoding="utf-8",
        )

    runtime_packets = {
        "clinical_semantics_resolution": runtime_state.semantics,
        "data_extraction_request": runtime_state.extraction_request,
        "data_extraction_result": runtime_state.extraction_result,
    }
    for alias, packet in runtime_packets.items():
        if packet is None or evidence.get(alias) is not None:
            continue
        evidence.register_json(
            kind="log",
            description=f"Typed runtime packet: {alias}.",
            payload=packet.model_dump(mode="json"),
            filename=f"{alias}.json",
            evidence_id=alias,
            aliases=[alias],
            producer="runtime_supervisor",
            generation_mode="system",
            prompt_pack_version=prompt_version,
            metadata={"run_id": run_id},
        )

    _flush_partial_manifest()

    def _register_plan_revision(
        revised_plan: AnalysisPlan,
        *,
        reason: str,
    ) -> Path:
        revision_path = (
            run_dir / f"analysis_plan_revision_{revised_plan.revision}.json"
        )
        revision_path.write_text(
            revised_plan.model_dump_json(indent=2),
            encoding="utf-8",
        )
        base_id = f"analysis_plan_revision_{revised_plan.revision}"
        try:
            evidence.register_file(
                kind="log",
                description=f"Revised analysis plan (reason={reason}).",
                source_path=revision_path,
                evidence_id=base_id,
                producer="replanner",
                generation_mode="llm",
                prompt_pack_version=prompt_version,
                metadata={"reason": reason, "llm_signature": llm_signature},
            )
        except ValueError:
            # Resume + replan can legitimately re-emit the same revision number
            # with different content (the replanner is non-deterministic across
            # runs), which collides with the prior run's
            # ``analysis_plan_revision_N`` id. Keep both by versioning the id
            # with a content digest instead of crashing the resumed run. The
            # global evidence-id collision guard stays intact for every other
            # artefact.
            import hashlib

            digest = hashlib.sha256(
                revision_path.read_bytes()
            ).hexdigest()[:8]
            evidence.register_file(
                kind="log",
                description=(
                    f"Revised analysis plan (reason={reason}; "
                    f"resume re-revision)."
                ),
                source_path=revision_path,
                evidence_id=f"{base_id}_{digest}",
                producer="replanner",
                generation_mode="llm",
                prompt_pack_version=prompt_version,
                metadata={
                    "reason": reason,
                    "llm_signature": llm_signature,
                    "resume_reregistration": True,
                },
            )
        return revision_path

    def _maybe_replan(
        *,
        current_plan: AnalysisPlan,
        reason: str,
        probe_summary_payload: Optional[Dict[str, Any]] = None,
        completed_records: Optional[Sequence[Dict[str, Any]]] = None,
    ) -> AnalysisPlan:
        nonlocal plan_path
        if not pipeline._enable_replanning or skill_obj is not None:
            return current_plan
        replanner = ReplannerAgent(role_resolver("planner"))
        try:
            revised = replanner.run(
                context=agent_context,
                current_plan=current_plan,
                probe_summary=probe_summary_payload,
                completed_step_records=completed_records,
            )
        except Exception as exc:
            findings.append(
                ValidationFinding(
                    validator="replanner",
                    severity="warning",
                    message=f"Replanner failed; keeping existing plan: {exc}",
                    detail={"reason": reason},
                )
            )
            return current_plan
        # Guard against the replanner silently dropping figure-producing
        # steps; task contracts (e.g. EasyICU experiment runner) still
        # require those artefacts regardless of the LLM's revised framing.
        revised, preservation_findings = _preserve_figure_steps_after_replan(
            current=current_plan,
            revised=revised,
        )
        if preservation_findings:
            findings.extend(preservation_findings)

        # C1 (pilot 20260515 fix): cap total plan size after a replan.
        # The pilot saw the replanner grow a simple SOFA-2 association
        # to 30 steps with 13 revisions and never converge. The cap
        # truncates excess late-stage steps and forces the replanner
        # to revise existing steps in place on later passes. Cap of 0
        # disables the guard for backward compatibility.
        cap = pipeline._max_total_steps
        if cap > 0 and len(revised.steps) > cap:
            dropped = [s.step_id for s in revised.steps[cap:]]
            revised = revised.model_copy(update={"steps": list(revised.steps[:cap])})
            findings.append(
                ValidationFinding(
                    validator="replanner",
                    severity="warning",
                    message=(
                        f"Replanner produced {len(dropped) + cap} steps; "
                        f"truncated to max_total_steps={cap}. Dropped: "
                        f"{', '.join(dropped[:6])}"
                        + (" ..." if len(dropped) > 6 else "")
                    ),
                    detail={"dropped_step_ids": dropped, "cap": cap},
                )
            )

        if revised.model_dump(mode="json") == current_plan.model_dump(mode="json"):
            return current_plan
        plan_path = _register_plan_revision(revised, reason=reason)
        plan_result.plan_path = plan_path
        findings.append(
            ValidationFinding(
                validator="replanner",
                severity="info",
                message=f"Plan revised after {reason}.",
                detail={
                    "from_revision": current_plan.revision,
                    "to_revision": revised.revision,
                },
            )
        )
        return revised

    probe_step_id = "00_probe"
    if pipeline._enable_probe_step and probe_step_id not in resumed_step_ids:
        probe_summary, probe_files = _build_probe_summary(
            context=context,
            cohort_path=cohort_path,
            out_dir=run_dir / "steps" / probe_step_id / "outputs",
        )
        probe_evidence_ids: List[str] = []
        for probe_file in probe_files:
            kind = "statistic" if probe_file.name.endswith(".json") else "table"
            aliases = [probe_step_id]
            if probe_file.name == "probe_summary.json":
                aliases.extend(
                    [
                        "probe_summary",
                        "cohort_probe",
                    ]
                )
            rec = evidence.register_file(
                kind=kind,
                description=f"Probe artefact {probe_file.name}.",
                source_path=probe_file,
                produced_by_step=probe_step_id,
                producer="pipeline",
                generation_mode="deterministic_probe",
                aliases=aliases,
            )
            probe_evidence_ids.append(rec.evidence_id)
        probe_record = {
            "step_id": probe_step_id,
            "intent": "Probe distributions, missingness, and obvious anomalies before execution.",
            "status": "ok",
            "generation_mode": "deterministic_probe",
            "step_summary": probe_summary,
            "evidence_ids": probe_evidence_ids,
        }
        per_step_records.append(probe_record)
        _flush_partial_manifest()
        plan = _maybe_replan(
            current_plan=plan,
            reason="probe_summary",
            probe_summary_payload=probe_summary,
            completed_records=[probe_record],
        )

    shared_lock = threading.Lock()
    step_order = {s.step_id: i for i, s in enumerate(plan.steps)}
    total_steps = len(plan.steps)

    def _record_repair(
        *,
        repair_id: str,
        step_id: str,
        trigger: Dict[str, Any],
        transformation: str,
        before_code: Optional[str] = None,
        after_code: Optional[str] = None,
        selection_rule: Optional[str] = None,
        before_state: Optional[RepairObservedState] = None,
        after_state: Optional[RepairObservedState] = None,
    ) -> None:
        try:
            with repair_ledger_lock:
                provenance = repair_ledger.append_application(
                    repair_id=repair_id,
                    step_id=step_id,
                    trigger=trigger,
                    transformation=transformation,
                    model_id=llm_signature,
                    before_text=before_code,
                    after_text=after_code,
                    selection_rule=selection_rule,
                    before_state=before_state,
                    after_state=after_state,
                )
            # P1: a runtime invariant that was actually checked and failed is a
            # non-blocking warning in soft mode; P2 will escalate this to a
            # fail-closed block for STRUCTURAL / CONTRACT_FILL repairs.
            if provenance.invariant_status == InvariantStatus.VERIFIED_FAIL.value:
                findings.append(
                    ValidationFinding(
                        validator="repair_invariant",
                        severity="warning",
                        message=(
                            f"Repair {repair_id} violated declared invariant(s) "
                            f"{list(provenance.invariant_failures)} on step {step_id}."
                        ),
                        detail={
                            "repair_id": repair_id,
                            "step_id": step_id,
                            "repair_class": provenance.repair_class,
                            "invariant_failures": list(provenance.invariant_failures),
                        },
                    )
                )
        except Exception as exc:
            findings.append(
                ValidationFinding(
                    validator="repair_ledger",
                    severity="warning",
                    message=(
                        f"Could not record repair provenance for {repair_id}: {exc}"
                    ),
                    detail={"repair_id": repair_id, "step_id": step_id},
                )
            )

    def _script_generation_mode(
        *,
        repair_attempts: int,
        fallback_used: bool,
        runner_repair_name: Optional[str] = None,
    ) -> str:
        if fallback_used:
            return "fallback"
        if repair_attempts > 0:
            return "repaired"
        if runner_repair_name:
            return "runner_repaired"
        return "llm"

    def _finding_severity(
        findings_for_step: Sequence[ValidationFinding],
    ) -> Optional[str]:
        if any(f.severity == "error" for f in findings_for_step):
            return "error"
        if any(f.severity == "warning" for f in findings_for_step):
            return "warning"
        if any(f.severity == "info" for f in findings_for_step):
            return "info"
        return None

    def _propagate_findings_to_evidence(
        evidence_ids: Sequence[str],
        findings_for_step: Sequence[ValidationFinding],
        *,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        severity = _finding_severity(findings_for_step)
        messages = [
            f.message
            for f in findings_for_step
            if f.severity in {"warning", "error"}
        ]
        for evidence_id in evidence_ids:
            evidence.update_record(
                evidence_id,
                finding_severity=severity,
                finding_messages=messages,
                metadata=metadata,
            )

    def _evidence_refs_for_names(names: Sequence[str]) -> List[EvidenceRef]:
        refs: List[EvidenceRef] = []
        seen: set[str] = set()
        for name in names:
            rec = evidence.get(str(name))
            if rec is None or rec.evidence_id in seen:
                continue
            refs.append(
                EvidenceRef(
                    evidence_id=rec.evidence_id,
                    kind=rec.kind,
                    description=rec.description,
                    relative_path=rec.relative_path,
                )
            )
            seen.add(rec.evidence_id)
        return refs

    def _validator_messages(
        *finding_groups: Sequence[ValidationFinding],
    ) -> List[str]:
        messages: List[str] = []
        for group in finding_groups:
            for finding in group:
                if finding.message:
                    messages.append(finding.message)
        return messages

    def _failed_dependency_record(step: AnalysisStep) -> Optional[Dict[str, Any]]:
        parent_step_id = _parent_step_id_for_figure_step(step)
        if parent_step_id is None:
            return None
        with shared_lock:
            records = list(per_step_records)
        for record in records:
            if record.get("step_id") != parent_step_id:
                continue
            if str(record.get("status") or "").lower() == "ok":
                return None
            return record
        return None

    def _execute_one_step(step: AnalysisStep) -> Dict[str, Any]:
        nonlocal runtime_state
        step_record: Dict[str, Any] = {
            "step_id": step.step_id,
            "intent": step.intent,
        }
        step_current = step_order.get(step.step_id, 0) + 1
        dependency_record = _failed_dependency_record(step)
        if dependency_record is not None:
            parent_step_id = str(dependency_record.get("step_id") or "")
            step_record.update(
                {
                    "status": "skipped_dependency_failed",
                    "dependency_step_id": parent_step_id,
                    "diagnostic_only": True,
                    "generation_mode": "system",
                }
            )
            with shared_lock:
                findings.append(
                    ValidationFinding(
                        validator="dependency_gate",
                        severity="warning",
                        message=(
                            f"Skipped downstream figure step {step.step_id} because "
                            f"required analysis step {parent_step_id} did not pass."
                        ),
                        detail={
                            "step_id": step.step_id,
                            "dependency_step_id": parent_step_id,
                            "dependency_status": dependency_record.get("status"),
                            "diagnostic_only": True,
                        },
                    )
                )
                per_step_records.append(step_record)
                _flush_partial_manifest()
            emit_progress(
                "step",
                f"Skipped {step.step_id}; required step {parent_step_id} failed.",
                status="skipped",
                run_id=run_id,
                step_id=step.step_id,
                current_step=step_current,
                total_steps=total_steps,
            )
            return step_record
        emit_progress(
            "step",
            f"Step {step_current}/{total_steps} started: {step.step_id}.",
            run_id=run_id,
            step_id=step.step_id,
            current_step=step_current,
            total_steps=total_steps,
        )
        existing_refs = _evidence_refs_for_names(step.inputs)
        local_runtime_state = supervisor.prepare_step_state(
            state=runtime_state,
            context=context,
            step=step,
            evidence_refs=existing_refs,
        )
        step_record["analysis_request"] = (
            local_runtime_state.analysis_request.model_dump(mode="json")
            if local_runtime_state.analysis_request is not None
            else None
        )
        step_record["visualization_request"] = (
            local_runtime_state.visualization_request.model_dump(mode="json")
            if local_runtime_state.visualization_request is not None
            else None
        )
        step_record["semantics_family"] = local_runtime_state.analysis_family

        try:
            emit_progress(
                "coder",
                f"Generating analysis script for {step.step_id}.",
                run_id=run_id,
                step_id=step.step_id,
                current_step=step_current,
                total_steps=total_steps,
            )
            code = coder.run(context=agent_context, step=step)
        except Exception as exc:
            with shared_lock:
                findings.append(
                    ValidationFinding(
                        validator="coder",
                        severity="error",
                        message=f"Coder agent failed for step {step.step_id}: {exc}",
                    )
                )
                step_record["status"] = "coder_failed"
                per_step_records.append(step_record)
                _flush_partial_manifest()
            emit_progress(
                "coder",
                f"Coder failed for {step.step_id}.",
                status="error",
                run_id=run_id,
                step_id=step.step_id,
                current_step=step_current,
                total_steps=total_steps,
            )
            return step_record
        deterministic_fallback_used = False

        def _deterministic_fallback_code(reason: str) -> Optional[str]:
            nonlocal deterministic_fallback_used
            if (
                deterministic_fallback_used
                or not pipeline._enable_deterministic_code_fallback
            ):
                return None
            deterministic_fallback_used = True
            plan_result.used_mock_llm = True
            step_record["deterministic_code_fallback"] = reason
            emit_progress(
                "coder",
                f"Using deterministic fallback script for {step.step_id}.",
                run_id=run_id,
                step_id=step.step_id,
                current_step=step_current,
                total_steps=total_steps,
                fallback_reason=reason,
            )
            fallback_coder = CoderAgent(MockLLMClient(context=agent_context))
            return fallback_coder.run(context=agent_context, step=step)

        concept_repair_attempts = 0
        concept_audit_error_count = 0
        while True:
            usage_findings = usage_auditor.audit(
                context=context,
                script_text=code,
                step=step,
            )
            # O-generic: analysis-pattern auditor (clustering /
            # prediction / survival footguns). Runs alongside the
            # concept-usage auditor so both sets of findings are
            # merged before the error-gate decision.
            usage_findings.extend(
                pattern_auditor.audit(
                    context=context,
                    script_text=code,
                    step=step,
                )
            )
            if pipeline._enable_llm_concept_audit:
                llm_audit_client = (
                    pipeline._llm_concept_auditor_client or role_resolver("analyzer")
                )
                if llm_audit_client is not None:
                    usage_findings.extend(
                        LLMConceptAuditor(llm_audit_client).audit(
                            context=context,
                            script_text=code,
                            step=step,
                        )
                    )
            step_record["usage_findings"] = [f.model_dump() for f in usage_findings]
            concept_audit_error_count += sum(
                1
                for f in usage_findings
                if f.validator == usage_auditor.name and f.severity == "error"
            )
            step_record["concept_audit_error_count"] = concept_audit_error_count
            step_record["concept_repair_attempts"] = concept_repair_attempts
            if not any(f.severity == "error" for f in usage_findings):
                with shared_lock:
                    findings.extend(usage_findings)
                break

            if concept_repair_attempts >= pipeline._max_code_repair_attempts:
                fallback_code = _deterministic_fallback_code("concept_audit")
                if fallback_code is not None:
                    # Surface the pattern/concept findings that
                    # forced the fallback; otherwise the manifest
                    # silently drops the original ICU rule
                    # violations that the LLM emitted. We dedupe by
                    # message so repeated retries don't spam.
                    with shared_lock:
                        seen_msgs = {f.message for f in findings}
                        for f in usage_findings:
                            if f.message in seen_msgs:
                                continue
                            # Demote ``error`` severity to
                            # ``warning`` because the run is
                            # continuing on the deterministic
                            # fallback; reviewer still sees the
                            # original violation in the manifest.
                            if f.severity == "error":
                                f = f.model_copy(update={
                                    "severity": "warning",
                                    "message": (
                                        "[surfaced after fallback] "
                                        + f.message
                                    ),
                                })
                            findings.append(f)
                    code = fallback_code
                    continue
                step_record["status"] = "blocked_by_concept_audit"
                with shared_lock:
                    findings.extend(usage_findings)
                    per_step_records.append(step_record)
                    _flush_partial_manifest()
                emit_progress(
                    "audit",
                    f"Concept audit blocked {step.step_id}.",
                    status="error",
                    run_id=run_id,
                    step_id=step.step_id,
                    current_step=step_current,
                    total_steps=total_steps,
                )
                return step_record

            concept_repair_attempts += 1
            step_record["concept_repair_attempts"] = concept_repair_attempts
            emit_progress(
                "coder",
                f"Repairing concept-audit violation for {step.step_id}.",
                run_id=run_id,
                step_id=step.step_id,
                current_step=step_current,
                total_steps=total_steps,
                repair_attempts=concept_repair_attempts,
            )
            audit_log = "\n".join(
                f"{f.severity.upper()}: {f.message}" for f in usage_findings
            )
            try:
                code = coder.repair(
                    context=agent_context,
                    step=step,
                    code=code,
                    run_log=(
                        "Static concept audit blocked this script before "
                        "execution. Fix all ICU-rule violations.\n\n" + audit_log
                    ),
                    attempt=concept_repair_attempts,
                )
            except Exception as exc:
                fallback_code = _deterministic_fallback_code(
                    "concept_repair_failed"
                )
                if fallback_code is not None:
                    code = fallback_code
                    continue
                with shared_lock:
                    findings.extend(usage_findings)
                    findings.append(
                        ValidationFinding(
                            validator="coder",
                            severity="error",
                            message=(
                                f"Coder repair failed after concept audit for "
                                f"step {step.step_id}: {exc}"
                            ),
                        )
                    )
                    step_record["status"] = "repair_failed"
                    per_step_records.append(step_record)
                    _flush_partial_manifest()
                emit_progress(
                    "coder",
                    f"Concept-audit repair failed for {step.step_id}.",
                    status="error",
                    run_id=run_id,
                    step_id=step.step_id,
                    current_step=step_current,
                    total_steps=total_steps,
                )
                return step_record

        repair_attempts = 0
        runner_repair_name: Optional[str] = None
        while True:
            run_label = "repaired script" if repair_attempts else "generated script"
            emit_progress(
                "runner",
                f"Running {run_label} for {step.step_id}.",
                run_id=run_id,
                step_id=step.step_id,
                current_step=step_current,
                total_steps=total_steps,
                repair_attempts=repair_attempts,
            )
            run_result = runner.run(step_id=step.step_id, code=code)
            step_record["returncode"] = run_result.returncode
            step_record["timed_out"] = run_result.timed_out
            step_record["requested_network_policy"] = run_result.requested_network_policy
            step_record["effective_isolation"] = run_result.effective_isolation
            step_record["isolation_degraded"] = run_result.isolation_degraded
            if run_result.isolation_degradation_reason:
                step_record["isolation_degradation_reason"] = (
                    run_result.isolation_degradation_reason
                )
            step_record["code_repair_attempts"] = repair_attempts

            script_description = (
                f"Generated analysis script for step {step.step_id}."
                if repair_attempts == 0
                else f"Repaired analysis script for step {step.step_id} (attempt {repair_attempts})."
            )
            script_record = evidence.register_file(
                kind="code",
                description=script_description,
                source_path=run_result.script_path,
                produced_by_step=step.step_id,
                producer="coder",
                generation_mode=_script_generation_mode(
                    repair_attempts=repair_attempts,
                    fallback_used=deterministic_fallback_used,
                    runner_repair_name=runner_repair_name,
                ),
                prompt_pack_version=prompt_version,
                metadata={
                    "repair_attempts": repair_attempts,
                    "fallback_reason": step_record.get(
                        "deterministic_code_fallback"
                    ),
                    "runner_repair": runner_repair_name,
                    "llm_signature": llm_signature,
                },
            )
            log_path = run_result.cwd / "run.log"
            if log_path.exists():
                evidence.register_file(
                    kind="log",
                    description=f"stdout/stderr log for step {step.step_id}.",
                    source_path=log_path,
                    produced_by_step=step.step_id,
                    script_evidence_id=script_record.evidence_id,
                    producer="runner",
                    generation_mode=_script_generation_mode(
                        repair_attempts=repair_attempts,
                        fallback_used=deterministic_fallback_used,
                        runner_repair_name=runner_repair_name,
                    ),
                    metadata={
                        "repair_attempts": repair_attempts,
                        "fallback_reason": step_record.get(
                            "deterministic_code_fallback"
                        ),
                        "runner_repair": runner_repair_name,
                    },
                )

            if run_result.succeeded:
                # Step-summary salvage reshapes the source from which numbers are
                # registered, so each salvage is recorded in the repair ledger
                # (ENG-REPAIR1 P1.5). The salvage decision lives in
                # salvage_step_summary() so it is unit-testable end-to-end; here
                # we only record what it did.
                salvage_outcome = salvage_step_summary(run_result, step=step)
                if salvage_outcome is not None:
                    if salvage_outcome.reset_artefacts:
                        run_result.artefacts = sorted(
                            p for p in run_result.out_dir.iterdir() if p.is_file()
                        )
                    _record_repair(
                        repair_id=salvage_outcome.repair_id,
                        step_id=step.step_id,
                        trigger={
                            "source": "summary_salvage",
                            "reason": salvage_outcome.trigger_reason,
                        },
                        transformation=salvage_outcome.transformation,
                        selection_rule=salvage_outcome.selection_rule,
                    )
                if not run_result.artefacts:
                    fallback_code = _deterministic_fallback_code("no_artefacts")
                    if fallback_code is not None:
                        code = fallback_code
                        _clear_output_dir(run_result.out_dir)
                        continue
                visual_step_summary: Dict[str, Any] = {}
                visual_summary_path = run_result.out_dir / "step_summary.json"
                if visual_summary_path.exists():
                    try:
                        vloaded = json.loads(
                            visual_summary_path.read_text(encoding="utf-8")
                        )
                    except Exception:
                        vloaded = None
                    if isinstance(vloaded, dict):
                        visual_step_summary = vloaded
                    else:
                        visual_step_summary = {"raw": vloaded}
                step_figures = [
                    art
                    for art in run_result.artefacts
                    if art.suffix.lower() in {".png", ".svg", ".tiff", ".tif"}
                ]
                if pipeline._enable_visual_qa and step_figures:
                    expected_numeric = _expected_numeric_annotations_for_step(
                        step=step,
                        step_summary=visual_step_summary,
                    )
                    numeric_expectations = (
                        {
                            str(path): expected_numeric
                            for path in step_figures
                            if path.suffix.lower() == ".svg"
                        }
                        if expected_numeric
                        else None
                    )
                    visual_findings = VisualQAAuditor().audit_with_expected(
                        figure_paths=step_figures,
                        expected_numeric_by_path=numeric_expectations,
                    )
                    step_record["visual_findings"] = [
                        f.model_dump() for f in visual_findings
                    ]
                    visual_errors = [
                        f for f in visual_findings if f.severity == "error"
                    ]
                    if visual_errors:
                        if repair_attempts >= pipeline._max_code_repair_attempts:
                            fallback_code = _deterministic_fallback_code(
                                "visual_qa"
                            )
                            if fallback_code is not None:
                                code = fallback_code
                                _clear_output_dir(run_result.out_dir)
                                continue
                            demoted_findings, blocking_visual_errors = (
                                _demote_cosmetic_visual_findings(visual_findings)
                            )
                            with shared_lock:
                                findings.extend(demoted_findings)
                            step_record["visual_qa_demoted"] = any(
                                original.severity == "error"
                                and demoted.severity == "warning"
                                for original, demoted in zip(
                                    visual_findings, demoted_findings
                                )
                            )
                            if blocking_visual_errors:
                                step_record["status"] = "execution_failed"
                                with shared_lock:
                                    per_step_records.append(step_record)
                                    _flush_partial_manifest()
                                emit_progress(
                                    "visual_qa",
                                    (
                                        f"Visual QA blocked {step.step_id} after "
                                        f"{repair_attempts} repair attempts."
                                    ),
                                    status="error",
                                    run_id=run_id,
                                    step_id=step.step_id,
                                    current_step=step_current,
                                    total_steps=total_steps,
                                )
                                return step_record
                            emit_progress(
                                "visual_qa",
                                (
                                    f"Cosmetic visual QA findings demoted to warning "
                                    f"for {step.step_id} after "
                                    f"{repair_attempts} repair attempts."
                                ),
                                status="warning",
                                run_id=run_id,
                                step_id=step.step_id,
                                current_step=step_current,
                                total_steps=total_steps,
                            )
                            # Fall through to contract checks and evidence
                            # registration only when every remaining visual
                            # error was a deterministic layout/cosmetic issue.
                        else:
                            repair_attempts += 1
                            step_record["code_repair_attempts"] = repair_attempts
                            emit_progress(
                                "visual_qa",
                                f"Repairing figure layout for {step.step_id}.",
                                run_id=run_id,
                                step_id=step.step_id,
                                current_step=step_current,
                                total_steps=total_steps,
                                repair_attempts=repair_attempts,
                            )
                            qa_log = "\n".join(
                                f"{f.severity.upper()}: {f.message}"
                                for f in visual_findings
                            )
                            try:
                                code = coder.repair(
                                    context=agent_context,
                                    step=step,
                                    code=code,
                                    run_log=(
                                        "Visual QA rejected one or more figure outputs "
                                        "before evidence registration. Fix the figure "
                                        "layout, preserve all tables/statistics, save PNG "
                                        "and editable SVG with the same stem, include "
                                        "publication figure exports when requested, and rerun.\n\n"
                                        + qa_log
                                    ),
                                    attempt=repair_attempts,
                                )
                                _clear_output_dir(run_result.out_dir)
                                continue
                            except Exception as exc:
                                fallback_code = _deterministic_fallback_code(
                                    "visual_qa_repair_failed"
                                )
                                if fallback_code is not None:
                                    code = fallback_code
                                    _clear_output_dir(run_result.out_dir)
                                    continue
                                with shared_lock:
                                    findings.extend(visual_findings)
                                    findings.append(
                                        ValidationFinding(
                                            validator="coder",
                                            severity="error",
                                            message=(
                                                f"Coder repair failed after visual QA "
                                                f"for step {step.step_id}: {exc}"
                                            ),
                                        )
                                    )
                                    step_record["status"] = "repair_failed"
                                    per_step_records.append(step_record)
                                    _flush_partial_manifest()
                                emit_progress(
                                    "visual_qa",
                                    f"Visual QA repair failed for {step.step_id}.",
                                    status="error",
                                    run_id=run_id,
                                    step_id=step.step_id,
                                    current_step=step_current,
                                    total_steps=total_steps,
                                )
                                return step_record
                with shared_lock:
                    completed_records_snapshot = list(per_step_records)
                early_contract_findings = _step_contract_findings(
                    step=step,
                    step_summary=visual_step_summary,
                    completed_step_records=completed_records_snapshot,
                )
                early_contract_errors = [
                    f for f in early_contract_findings if f.severity == "error"
                ]
                if early_contract_errors and repair_attempts < pipeline._max_code_repair_attempts:
                    repair_attempts += 1
                    step_record["code_repair_attempts"] = repair_attempts
                    emit_progress(
                        "coder",
                        f"Repairing contract violation for {step.step_id}.",
                        run_id=run_id,
                        step_id=step.step_id,
                        current_step=step_current,
                        total_steps=total_steps,
                        repair_attempts=repair_attempts,
                    )
                    contract_log = "\n".join(
                        f"{f.severity.upper()}: {f.message}"
                        for f in early_contract_findings
                        if f.message
                    )
                    repair_guidance = _step_contract_repair_guidance(
                        step=step,
                        step_summary=visual_step_summary,
                        code=code,
                    )
                    try:
                        code = coder.repair(
                            context=agent_context,
                            step=step,
                            code=code,
                            run_log=(
                                "The script executed but failed the machine-readable "
                                "step contract. Revise the analysis code; do not change "
                                "the research question. Ensure required primary metrics "
                                "are computed and written to step_summary.json with "
                                "explicit numeric keys or nested statistic fields.\n\n"
                                "STEP SUMMARY:\n"
                                + json.dumps(
                                    visual_step_summary,
                                    indent=2,
                                    ensure_ascii=False,
                                    default=str,
                                )
                                + "\n\nCONTRACT FINDINGS:\n"
                                + contract_log
                                + "\n\nREPAIR GUIDANCE:\n"
                                + repair_guidance
                            ),
                            attempt=repair_attempts,
                        )
                        _clear_output_dir(run_result.out_dir)
                        continue
                    except Exception as exc:
                        fallback_code = _deterministic_fallback_code(
                            "contract_repair_failed"
                        )
                        if fallback_code is not None:
                            code = fallback_code
                            _clear_output_dir(run_result.out_dir)
                            continue
                        with shared_lock:
                            findings.extend(early_contract_findings)
                            findings.append(
                                ValidationFinding(
                                    validator="coder",
                                    severity="error",
                                    message=(
                                        f"Coder repair failed after contract check "
                                        f"for step {step.step_id}: {exc}"
                                    ),
                                )
                            )
                            step_record["status"] = "repair_failed"
                            per_step_records.append(step_record)
                            _flush_partial_manifest()
                        emit_progress(
                            "coder",
                            f"Contract repair failed for {step.step_id}.",
                            status="error",
                            run_id=run_id,
                            step_id=step.step_id,
                            current_step=step_current,
                            total_steps=total_steps,
                        )
                        return step_record
                if pipeline._enable_deterministic_runner_repair:
                    before_repair_code = code
                    summary_repair = _deterministic_summary_repair(
                        code=code,
                        step_summary=visual_step_summary,
                        previous_repair=runner_repair_name,
                        analysis_family=local_runtime_state.analysis_family,
                    )
                else:
                    summary_repair = None
                if summary_repair is not None:
                    runner_repair_name, code = summary_repair
                    step_record["runner_repair"] = runner_repair_name
                    _record_repair(
                        repair_id=runner_repair_name,
                        step_id=step.step_id,
                        trigger={
                            "source": "deterministic_summary_repair",
                            "step_summary_keys": sorted(
                                str(key) for key in visual_step_summary.keys()
                            ),
                        },
                        transformation="Deterministic repair after step_summary contract inspection.",
                        before_code=before_repair_code,
                        after_code=code,
                    )
                    emit_progress(
                        "runner_repair",
                        f"Applied deterministic summary repair for {step.step_id}: {runner_repair_name}.",
                        run_id=run_id,
                        step_id=step.step_id,
                        current_step=step_current,
                        total_steps=total_steps,
                    )
                    _clear_output_dir(run_result.out_dir)
                    continue
                break

            if log_path.exists():
                run_log = log_path.read_text(encoding="utf-8", errors="replace")
            else:
                run_log = (
                    (run_result.stdout or "") + "\n" + (run_result.stderr or "")
                )
            if pipeline._enable_deterministic_runner_repair:
                before_repair_code = code
                plugin_repair = pipeline._case_plugin_registry.repair_code(
                    context=context,
                    step=step,
                    code=code,
                    run_log=run_log,
                )
                if (
                    plugin_repair is not None
                    and plugin_repair[0] != runner_repair_name
                ):
                    runner_repair = plugin_repair
                else:
                    runner_repair = _deterministic_runner_repair(
                        code=code,
                        run_log=run_log,
                        previous_repair=runner_repair_name,
                        analysis_family=local_runtime_state.analysis_family,
                    )
            else:
                runner_repair = None
            if runner_repair is not None:
                runner_repair_name, code = runner_repair
                step_record["runner_repair"] = runner_repair_name
                _record_repair(
                    repair_id=runner_repair_name,
                    step_id=step.step_id,
                    trigger={
                        "source": "deterministic_runner_repair",
                        "run_log_tail": run_log[-1200:],
                    },
                    transformation="Deterministic repair after runner failure.",
                    before_code=before_repair_code,
                    after_code=code,
                )
                emit_progress(
                    "runner_repair",
                    f"Applied deterministic runner repair for {step.step_id}: {runner_repair_name}.",
                    run_id=run_id,
                    step_id=step.step_id,
                    current_step=step_current,
                    total_steps=total_steps,
                )
                _clear_output_dir(run_result.out_dir)
                continue

            if repair_attempts >= pipeline._max_code_repair_attempts:
                fallback_code = _deterministic_fallback_code("execution_failure")
                if fallback_code is not None:
                    code = fallback_code
                    _clear_output_dir(run_result.out_dir)
                    continue
                with shared_lock:
                    findings.append(
                        ValidationFinding(
                            validator="runner",
                            severity="error",
                            message=(
                                f"Step {step.step_id} "
                                f"{'timed out' if run_result.timed_out else 'failed'} "
                                f"with returncode {run_result.returncode}."
                            ),
                        )
                    )
                    step_record["status"] = "execution_failed"
                    per_step_records.append(step_record)
                    _flush_partial_manifest()
                emit_progress(
                    "runner",
                    f"Execution failed for {step.step_id}.",
                    status="error",
                    run_id=run_id,
                    step_id=step.step_id,
                    current_step=step_current,
                    total_steps=total_steps,
                )
                return step_record

            repair_attempts += 1
            step_record["code_repair_attempts"] = repair_attempts
            emit_progress(
                "coder",
                f"Repairing failed script for {step.step_id}.",
                run_id=run_id,
                step_id=step.step_id,
                current_step=step_current,
                total_steps=total_steps,
                repair_attempts=repair_attempts,
            )
            try:
                code = coder.repair(
                    context=agent_context,
                    step=step,
                    code=code,
                    run_log=run_log,
                    attempt=repair_attempts,
                )
                _clear_output_dir(run_result.out_dir)
            except Exception as exc:
                # 🔧 2026-05-16: distinguish transient LLM/parse failures from
                # exhausted budget. JSON-parse errors after the OpenAIClient
                # retry chain already exhausted its own backoff still bubble up
                # here; treat them as one used repair attempt and loop instead
                # of immediately bailing out. Only fall through to the
                # deterministic fallback / repair_failed branch when we've
                # genuinely used up max_code_repair_attempts.
                _msg = str(exc).lower()
                _is_transient = (
                    isinstance(exc, json.JSONDecodeError)
                    or "expecting value" in _msg
                    or ("json" in _msg and "decode" in _msg)
                    or "503" in _msg
                    or "rate" in _msg
                )
                if (
                    _is_transient
                    and repair_attempts < pipeline._max_code_repair_attempts
                ):
                    emit_progress(
                        "coder",
                        f"Transient repair failure for {step.step_id} "
                        f"(attempt {repair_attempts}): {type(exc).__name__}; retrying.",
                        run_id=run_id,
                        step_id=step.step_id,
                        current_step=step_current,
                        total_steps=total_steps,
                        repair_attempts=repair_attempts,
                    )
                    # The retained `code` is unchanged → next loop iteration
                    # will re-run the same script, fail the same way, then
                    # come back here for repair attempt N+1 with the same
                    # traceback in run_log. That gives the LLM another shot
                    # at producing parseable output.
                    continue

                fallback_code = _deterministic_fallback_code("repair_failed")
                if fallback_code is not None:
                    code = fallback_code
                    _clear_output_dir(run_result.out_dir)
                    continue
                with shared_lock:
                    findings.append(
                        ValidationFinding(
                            validator="coder",
                            severity="error",
                            message=f"Coder repair failed for step {step.step_id}: {exc}",
                        )
                    )
                    step_record["status"] = "repair_failed"
                    per_step_records.append(step_record)
                    _flush_partial_manifest()
                emit_progress(
                    "coder",
                    f"Repair failed for {step.step_id}.",
                    status="error",
                    run_id=run_id,
                    step_id=step.step_id,
                    current_step=step_current,
                    total_steps=total_steps,
                )
                return step_record

        publication_step = (
            step.method == "publication_figure_generation"
            or "publication_figure" in (step.step_id or "").lower()
            or "publication figure" in (step.intent or "").lower()
            or "publication-ready figure" in (step.intent or "").lower()
            or any(
                str(item).startswith("figure:publication_figure")
                for item in step.expected_outputs
            )
        )
        figure_role = (
            "publication_figure"
            if publication_step
            else "analysis_figure"
            if _step_expects_figure(step)
            else None
        )
        if publication_step and not _has_figure_exports(run_result.out_dir):
            promoted = _promote_sibling_figure_exports(out_dir=run_result.out_dir)
            if promoted is not None:
                runner_repair_name = promoted
                step_record["runner_repair"] = promoted
                _record_repair(
                    repair_id=promoted,
                    step_id=step.step_id,
                    trigger={"source": "publication_figure_sibling_promotion"},
                    transformation="Promoted sibling figure exports into canonical outputs directory.",
                )
            else:
                promoted = _promote_prior_publication_bundle(
                    run_dir=run_dir,
                    current_step_id=step.step_id,
                    out_dir=run_result.out_dir,
                )
                if promoted is not None:
                    runner_repair_name = promoted
                    step_record["runner_repair"] = promoted
                    _record_repair(
                        repair_id=promoted,
                        step_id=step.step_id,
                        trigger={"source": "publication_figure_prior_bundle_promotion"},
                        transformation="Promoted prior publication figure bundle into current outputs directory.",
                    )
                else:
                    rescued = _render_prediction_publication_bundle_from_prior_outputs(
                        run_dir=run_dir,
                        current_step_id=step.step_id,
                        out_dir=run_result.out_dir,
                    )
                    if rescued is not None:
                        runner_repair_name = rescued
                        step_record["runner_repair"] = rescued
                        _record_repair(
                            repair_id=rescued,
                            step_id=step.step_id,
                            trigger={"source": "prediction_publication_bundle_rescue"},
                            transformation=(
                                "Rendered deterministic publication figure bundle "
                                "from prior prediction outputs."
                            ),
                        )

        run_result.artefacts = sorted(
            p for p in run_result.out_dir.iterdir() if p.is_file()
        )

        if publication_step and not _has_figure_exports(run_result.out_dir):
            with shared_lock:
                findings.append(
                    ValidationFinding(
                        validator="publication_figure_outputs",
                        severity="error",
                        message=(
                            f"Step {step.step_id} completed without any publication-figure exports."
                        ),
                    )
                )
                step_record["status"] = "execution_failed"
                per_step_records.append(step_record)
                _flush_partial_manifest()
            emit_progress(
                "runner",
                f"Publication figure missing for {step.step_id}.",
                status="error",
                run_id=run_id,
                step_id=step.step_id,
                current_step=step_current,
                total_steps=total_steps,
            )
            return step_record

        evidence_ids_for_step: List[str] = [script_record.evidence_id]
        step_summary_record_id: Optional[str] = None
        for art in run_result.artefacts:
            step_aliases = _semantic_aliases_for(step, art)
            generation_mode = _script_generation_mode(
                repair_attempts=repair_attempts,
                fallback_used=deterministic_fallback_used,
                runner_repair_name=runner_repair_name,
            )
            if art.name == "step_summary.json":
                rec = evidence.register_file(
                    kind="statistic",
                    description=f"Machine-readable summary for step {step.step_id}.",
                    source_path=art,
                    produced_by_step=step.step_id,
                    script_evidence_id=script_record.evidence_id,
                    aliases=step_aliases,
                    producer="runner",
                    generation_mode=generation_mode,
                    metadata={
                        "script_evidence_id": script_record.evidence_id,
                        "figure_role": figure_role or "analysis_figure",
                        "diagnostic_only": False,
                    },
                )
                step_summary_record_id = rec.evidence_id
            elif art.suffix.lower() in {".csv", ".tsv", ".parquet", ".feather"}:
                rec = evidence.register_file(
                    kind="table",
                    description=f"Table {art.stem} from step {step.step_id}.",
                    source_path=art,
                    produced_by_step=step.step_id,
                    script_evidence_id=script_record.evidence_id,
                    aliases=step_aliases,
                    producer="runner",
                    generation_mode=generation_mode,
                    metadata={"script_evidence_id": script_record.evidence_id},
                )
            elif art.suffix.lower() in {
                ".png",
                ".svg",
                ".pdf",
                ".tiff",
                ".tif",
                ".pptx",
            }:
                rec = evidence.register_file(
                    kind="figure",
                    description=f"Figure {art.stem} from step {step.step_id}.",
                    source_path=art,
                    produced_by_step=step.step_id,
                    script_evidence_id=script_record.evidence_id,
                    aliases=step_aliases,
                    producer="runner",
                    generation_mode=generation_mode,
                    metadata={
                        "script_evidence_id": script_record.evidence_id,
                        "figure_role": figure_role or "analysis_figure",
                        "diagnostic_only": False,
                    },
                )
            else:
                rec = evidence.register_file(
                    kind="log",
                    description=f"Auxiliary artefact {art.name}.",
                    source_path=art,
                    produced_by_step=step.step_id,
                    script_evidence_id=script_record.evidence_id,
                    aliases=step_aliases,
                    producer="runner",
                    generation_mode=generation_mode,
                    metadata={"script_evidence_id": script_record.evidence_id},
                )
            evidence_ids_for_step.append(rec.evidence_id)

        step_summary: Dict[str, Any] = {}
        ssj = run_result.out_dir / "step_summary.json"
        if ssj.exists():
            try:
                loaded = json.loads(ssj.read_text(encoding="utf-8"))
            except Exception:
                loaded = None
            if isinstance(loaded, dict):
                step_summary = loaded
            else:
                # The coder emitted a non-dict JSON (bare string /
                # list / number). Keep it accessible but coerce to
                # a dict so every downstream consumer that calls
                # ``.get(...)`` still works.
                step_summary = {"raw": loaded}
        if step_summary_record_id is not None:
            step_record["step_summary_evidence_id"] = step_summary_record_id
        if step_summary and step_summary_record_id is not None:
            # Value-level provenance (A-track): every numeric leaf in the
            # step's summary is registered as a NumericClaim so the
            # manuscript binder can reverse-link numbers in prose to the
            # exact field of the exact step output that produced them.
            try:
                cap = pipeline._max_numeric_claims_per_step
                evidence.register_step_summary_numerics(
                    step_id=step.step_id,
                    evidence_id=step_summary_record_id,
                    summary=step_summary,
                    max_leaves=cap if cap > 0 else None,
                )
            except Exception as exc:
                logger.warning(
                    "Failed to register numeric claims for step %s: %s",
                    step.step_id, exc,
                )
            # Phase-1 derived-claim hook (Commit 2). After every leaf
            # is registered, evaluate any ``derived_claims`` the coder
            # declared in step_summary. Sources must resolve to claims
            # that ALREADY exist in the registry, so this runs second.
            # Errors surface as ``derived_claim_error`` findings rather
            # than aborting — a bad formula should not kill the step.
            try:
                _, derived_errors = evidence.register_step_derived_claims(
                    step_id=step.step_id,
                    evidence_id=step_summary_record_id,
                    summary=step_summary,
                )
                for err in derived_errors:
                    findings.append(
                        ValidationFinding(
                            validator="derived_claim",
                            severity="warning",
                            message=(
                                f"derived_claims entry {err['name']!r} for step "
                                f"{step.step_id} was rejected: {err['message']}"
                            ),
                            detail={
                                "step_id": step.step_id,
                                "claim_name": err["name"],
                                "reason": err["message"],
                            },
                        )
                    )
            except Exception as exc:
                logger.warning(
                    "Failed to register derived claims for step %s: %s",
                    step.step_id, exc,
                )
        stat_findings = stat_validator.audit(
            context=context,
            cohort_path=cohort_path,
            step=step,
            out_dir=run_result.out_dir,
            step_summary=step_summary,
        )
        clinical_findings = clinical_validator.audit(
            context=context,
            step=step,
            out_dir=run_result.out_dir,
            step_summary=step_summary,
        )
        guard_findings = statistical_guard.audit(
            context=context,
            cohort_path=cohort_path,
            step=step,
            out_dir=run_result.out_dir,
            step_summary=step_summary,
        )
        with shared_lock:
            completed_records_snapshot = list(per_step_records)
        contract_findings = _step_contract_findings(
            step=step,
            step_summary=step_summary,
            completed_step_records=completed_records_snapshot,
        )
        with shared_lock:
            findings.extend(stat_findings)
            findings.extend(clinical_findings)
            findings.extend(guard_findings)
            findings.extend(contract_findings)
        step_record["stat_findings"] = [f.model_dump() for f in stat_findings]
        step_record["clinical_findings"] = [
            f.model_dump() for f in clinical_findings
        ]
        step_record["guard_findings"] = [f.model_dump() for f in guard_findings]
        step_record["contract_findings"] = [
            f.model_dump() for f in contract_findings
        ]
        step_record["generation_mode"] = _script_generation_mode(
            repair_attempts=repair_attempts,
            fallback_used=deterministic_fallback_used,
            runner_repair_name=runner_repair_name,
        )
        raw_side_findings = step_summary.get("side_findings")
        if isinstance(raw_side_findings, list):
            side_findings = []
            for idx, raw in enumerate(raw_side_findings):
                if not isinstance(raw, dict):
                    continue
                payload = dict(raw)
                payload.setdefault("step_id", step.step_id)
                payload.setdefault("finding_id", f"{step.step_id}_side_{idx + 1}")
                side_findings.append(SideFinding.from_dict(payload).to_dict())
            if side_findings:
                step_record["side_findings"] = side_findings
        step_record["step_summary"] = step_summary
        evidence_refs_for_step = _evidence_refs_for_names(evidence_ids_for_step)
        validator_messages = _validator_messages(
            usage_findings,
            stat_findings,
            clinical_findings,
            guard_findings,
            contract_findings,
        )
        local_runtime_state = supervisor.critique_step(
            state=local_runtime_state,
            step_summary=step_summary,
            evidence_refs=evidence_refs_for_step,
            findings=validator_messages,
        )
        critique = local_runtime_state.critique
        if critique is not None:
            critique_path = run_result.out_dir / "critique_report.json"
            critique_path.write_text(
                critique.model_dump_json(indent=2),
                encoding="utf-8",
            )
            critique_record = evidence.register_file(
                kind="log",
                description=f"Structured critique report for step {step.step_id}.",
                source_path=critique_path,
                produced_by_step=step.step_id,
                script_evidence_id=script_record.evidence_id,
                aliases=[f"{step.step_id}_critique"],
                producer="critic",
                generation_mode="system",
                metadata={"script_evidence_id": script_record.evidence_id},
            )
            evidence_ids_for_step.append(critique_record.evidence_id)
            step_record["critique_report"] = critique.model_dump(mode="json")
            if critique.status in {"needs_revision", "blocked"}:
                with shared_lock:
                    findings.append(
                        ValidationFinding(
                            validator="critic_agent",
                            severity=(
                                "warning"
                                if critique.status == "needs_revision"
                                else "error"
                            ),
                            message=(
                                f"CriticAgent marked {step.step_id} as {critique.status}: "
                                + "; ".join(
                                    critique.concerns
                                    or critique.suggested_repairs
                                    or ["review required"]
                                )
                            ),
                            evidence_ids=[critique_record.evidence_id],
                        )
                    )

        try:
            interpretation = analyzer.run(
                context=agent_context,
                step=step,
                step_summary=step_summary,
                evidence_ids=evidence_ids_for_step,
            )
        except Exception as exc:
            interpretation = f"(analyzer failed: {exc})"
        interp_record = evidence.register_text(
            kind="log",
            description=f"Analyzer interpretation for step {step.step_id}.",
            text=interpretation,
            filename=f"interpretation_{step.step_id}.md",
            produced_by_step=step.step_id,
            script_evidence_id=script_record.evidence_id,
            producer="analyzer",
            generation_mode="llm",
            prompt_pack_version=prompt_version,
        )
        step_record["interpretation_evidence_id"] = interp_record.evidence_id
        evidence_ids_for_step.append(interp_record.evidence_id)
        step_record["evidence_ids"] = list(dict.fromkeys(evidence_ids_for_step))
        _propagate_findings_to_evidence(
            evidence_ids_for_step,
            usage_findings
            + stat_findings
            + clinical_findings
            + guard_findings
            + contract_findings,
            metadata={
                "step_id": step.step_id,
                "generation_mode": step_record["generation_mode"],
            },
        )
        with shared_lock:
            runtime_state = local_runtime_state
        has_contract_error = any(
            finding.severity == "error" for finding in contract_findings
        )
        step_record["status"] = "contract_failed" if has_contract_error else "ok"
        with shared_lock:
            per_step_records.append(step_record)
            _flush_partial_manifest()
        emit_progress(
            "step",
            (
                f"Step {step_current}/{total_steps} failed contract checks: "
                f"{step.step_id}."
                if has_contract_error
                else f"Step {step_current}/{total_steps} complete: {step.step_id}."
            ),
            status="error" if has_contract_error else "complete",
            run_id=run_id,
            step_id=step.step_id,
            current_step=step_current,
            total_steps=total_steps,
        )
        return step_record

    steps_to_run = [s for s in plan.steps if s.step_id not in resumed_step_ids]
    for skipped_step_id in sorted(resumed_step_ids):
        emit_progress(
            "resume",
            f"Skipped completed step from prior run: {skipped_step_id}.",
            status="complete",
            run_id=run_id,
            step_id=skipped_step_id,
        )
    if pipeline._enable_replanning and pipeline._max_concurrent_steps > 1:
        findings.append(
            ValidationFinding(
                validator="replanner",
                severity="info",
                message=(
                    "Replanning is enabled, so step execution was forced to sequential "
                    "mode to preserve run-internal plan revisions."
                ),
            )
        )

    if (
        pipeline._max_concurrent_steps <= 1
        or len(steps_to_run) <= 1
        or pipeline._enable_replanning
    ):
        executed_step_ids = set(resumed_step_ids)
        remaining_steps = [
            s for s in plan.steps if s.step_id not in executed_step_ids
        ]
        while remaining_steps:
            step = remaining_steps.pop(0)
            record = _execute_one_step(step)
            executed_step_ids.add(step.step_id)
            if (
                pipeline._enable_replanning
                and record.get("status") == "ok"
                and remaining_steps
            ):
                plan = _maybe_replan(
                    current_plan=plan,
                    reason=step.step_id,
                    probe_summary_payload=probe_summary,
                    completed_records=per_step_records,
                )
                step_order.clear()
                step_order.update({s.step_id: i for i, s in enumerate(plan.steps)})
                remaining_steps = [
                    s for s in plan.steps if s.step_id not in executed_step_ids
                ]
                total_steps = len(plan.steps)
    else:
        workers = min(pipeline._max_concurrent_steps, len(steps_to_run))
        with ThreadPoolExecutor(
            max_workers=workers, thread_name_prefix="ra_step"
        ) as ex:
            futures = [ex.submit(_execute_one_step, s) for s in steps_to_run]
            for fut in as_completed(futures):
                exc = fut.exception()
                if exc is not None:
                    with shared_lock:
                        findings.append(
                            ValidationFinding(
                                validator="step_executor",
                                severity="error",
                            message=f"Worker raised an unhandled exception: {exc!r}",
                        )
                    )

    try:
        robustness_specs = robustness_specs_for_execution(run_dir=run_dir, plan=plan)
        if robustness_specs and not list(getattr(plan, "robustness_specs", []) or []):
            findings.append(
                ValidationFinding(
                    validator="robustness_panel",
                    severity="warning",
                    message=(
                        "Recovered robustness_specs from the plan-time lock because "
                        "the active replanned AnalysisPlan no longer carried them."
                    ),
                )
            )
        adapter_rows, adapter_warnings = fit_robustness_rows_from_records(
            specs=robustness_specs,
            per_step_records=per_step_records,
            primary_cohort=getattr(plan, "cohort", None),
            cohort_path=cohort_path,
            context=context,
        )
        for warning in adapter_warnings:
            findings.append(
                ValidationFinding(
                    validator="robustness_estimator",
                    severity="warning",
                    message=warning,
                )
            )
        robustness_panel = build_robustness_panel_from_records(
            specs=robustness_specs,
            per_step_records=per_step_records,
            adapter_rows=adapter_rows,
        )
        write_robustness_panel(
            run_dir=run_dir,
            panel=robustness_panel,
            evidence=evidence,
            prompt_pack_version=prompt_version,
        )
        _flush_partial_manifest(
            {
                "robustness_panel_path": "robustness_panel.json",
                "robustness_n_variants": robustness_panel.n_variants,
                "robustness_range_low": robustness_panel.range_low,
                "robustness_range_high": robustness_panel.range_high,
            }
        )
    except Exception as exc:
        findings.append(
            ValidationFinding(
                validator="robustness_panel",
                severity="warning",
                message=f"Robustness panel artifact could not be built: {exc}",
            )
        )

    if pipeline._enable_visual_qa:
        emit_progress(
            "visual_qa",
            "Auditing generated figures.",
            run_id=run_id,
        )
        fig_paths = [
            run_dir / r.relative_path
            for r in evidence.records()
            if r.kind == "figure"
        ]
        vlm_adapter = pipeline._visual_qa_adapter
        if vlm_adapter is None and pipeline._enable_vlm_visual_qa:
            client = pipeline._vlm_client or role_resolver("analyzer")
            if client is not None:
                vlm_adapter = VLMVisualQAAdapter(client)
        final_visual_findings = VisualQAAuditor(
            vlm_adapter=vlm_adapter
        ).audit(figure_paths=fig_paths)
        demoted_final_findings, _ = _demote_cosmetic_visual_findings(
            final_visual_findings
        )
        findings += demoted_final_findings

    plan_result.plan = plan
    plan_result.plan_path = plan_path
    return _ExecutePhaseResult(
        plan=plan,
        per_step_records=per_step_records,
        probe_summary=probe_summary,
        runtime_state=runtime_state,
        flush_partial_manifest=_flush_partial_manifest,
    )
