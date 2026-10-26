"""Durable append-run lifecycle using the existing plan-review authorities."""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
import json
import re
import uuid

from easyicu.research_agent.authority.evidence_store import EvidenceStore
from easyicu.research_agent.authority.plan_review import ReviewExecutionAuthority
from easyicu.research_agent.planning.family_spec.baseline_association_template import ResultAppendRuntimeAuthority
from easyicu.research_agent.authority.runtime_artifacts import verified_run_evidence_path
from easyicu.research_agent.canonical_json import canonical_sha256
from easyicu.research_agent.execution.kernel_identity import build_execution_kernel_identity
from easyicu.research_agent.execution.runners.baseline_association_executor import append_input_binding, execute_result_append
from easyicu.research_agent.orchestration.human_review_checkpoint import HumanReviewCheckpoint, load_checkpoint, write_checkpoint
from easyicu.research_agent.orchestration.services import PipelineServices
from easyicu.research_agent.orchestration.workflow import (
    HumanReviewDecision, HumanReviewPending, _human_review_decision_record, human_review_requests_for_plan,
)
from easyicu.research_agent.schema import AnalysisPlan
from easyicu.webserver import state_paths, study_contexts
from easyicu.webserver.agent_review_recovery import _locked, _write
from easyicu.webserver.research_append_source import AppendSource, load_append_source, read_json
from easyicu.webserver.research_plan_revision import PlanRevisionGuard


@dataclass(frozen=True)
class AppendRun:
    wrapper: Path
    run_dir: Path
    plan: AnalysisPlan
    authority: ResultAppendRuntimeAuthority
    record: dict
    checkpoint: HumanReviewCheckpoint

    @property
    def pending(self):
        return HumanReviewPending(run_id=self.checkpoint.run_id, thread_id=self.checkpoint.run_id,
                                  run_dir=str(self.run_dir), requests=self.checkpoint.requests,
                                  resume_scope="durable_checkpoint", resume_pid=None)


def _index(run_id):
    if not re.fullmatch(r"append-[0-9a-f]{32}", str(run_id)):
        return None
    return state_paths.state_root() / "result-appends" / (run_id + ".json")


def _environment():
    return {"execution_kernel": build_execution_kernel_identity(Path(__file__).resolve().parents[1]).to_dict()}


def _guard(source, plan, kind):
    return PipelineServices(plan_revision_guard=PlanRevisionGuard.for_append(
        plan=source.plan.model_dump(mode="json"), review=source.review,
        expected_plan=plan.model_dump(mode="json"), action=kind, append_run=True,
    ))


def propose_append_run(*, source: AppendSource, kind, wrapper: Path, study, registered_source_changed, predecessor=None):
    source = source.verify()
    if study["id"] != source.seal["study_id"]:
        raise ValueError("An append must stay in the source study.")
    wrapper = Path(wrapper)
    if wrapper.parent.resolve() != source.wrapper.parent.resolve() or wrapper.exists():
        raise ValueError("An append needs a new run directory in the same study.")
    authority = ResultAppendRuntimeAuthority.create(
        kind=kind, source_run_id=source.seal["run_id"], source_run_sha256=source.digest,
        source_plan=source.plan, context=source.context, baseline=source.baseline,
    )
    plan = authority.compilation().plan
    _guard(source, plan, kind).validate_plan_revision(plan=plan, context=source.context)
    run_id = "append-" + uuid.uuid4().hex
    run_dir = wrapper / "pipeline" / run_id
    run_dir.mkdir(parents=True, exist_ok=False)
    evidence = EvidenceStore(run_dir)
    needed = {value for step in plan.steps for value in step.inputs if ":" in value}
    produced = {value for step in plan.steps for value in step.expected_outputs}
    inputs = {}
    for key in sorted(needed - produced):
        original = source.seal["products"].get(key)
        path = verified_run_evidence_path(source.run_dir, original) if original else None
        if path is None:
            raise ValueError("The source lacks a required sealed append product.")
        adopted = evidence.register_file(
            kind=original["kind"], description="Read-only source evidence: " + key, source_path=path,
            evidence_id="source_" + original["evidence_id"],
            produced_by_step=original.get("produced_by_step"),
            metadata={"source_run_id": source.seal["run_id"], "source_run_sha256": source.digest,
                      "source_evidence_id": original["evidence_id"], "source_evidence_sha256": original["sha256"]},
        )
        if adopted.sha256 != original["sha256"]:
            raise ValueError("The adopted evidence differs from the source seal.")
        inputs[key] = append_input_binding(key, adopted, run_dir / adopted.relative_path)
    from .research_append_history import adopt_result_tables, adopt_result_figures

    prior_tables = adopt_result_tables(source, predecessor, evidence)
    prior_figures = adopt_result_figures(source, predecessor, evidence)
    notice = source.earlier_export_notice(changed=registered_source_changed)
    record = dict(schema_version="easyicu.result_append_run/1", run_id=run_id, study=study,
                  source_wrapper=str(source.wrapper), source_run_id=source.seal["run_id"], source_run_sha256=source.digest,
                  kind=kind, inputs=inputs, source_notice=notice, predecessor=predecessor,
                  source_data=source.seal["data_source"], prior_tables=prior_tables, prior_figures=prior_figures,
                  authority=authority.model_dump(mode="json"),
                  plan=plan.model_dump(mode="json"), environment=_environment())
    record["record_sha256"] = canonical_sha256(record)
    _write(wrapper / "result_append_run.json", record)
    _write(run_dir / "plan.json", record["plan"])
    # This is a bounded append review of the unchanged scientific coordinates
    # and the typed owner's exact executable roster, not a new study score.
    review = source.review.model_copy(deep=True, update={
        "plan_sha256": canonical_sha256(plan.model_dump(mode="json")), "top_journal_candidate": False,
        "status": "analysis_only", "generated_at": datetime.now(timezone.utc).isoformat(),
        "facts": {"append_runtime_sha256": authority.execution_contract_sha256,
                  "source_review_sha256": canonical_sha256(source.review.model_dump(mode="json"))},
    })
    for name, payload in (("scientific_plan_review", review.model_dump(mode="json")), ("append_input", record)):
        artifact = evidence.register_json(kind="log", description=name, payload=payload, filename=name + ".json", evidence_id=name)
        (run_dir / (name + ".json")).write_bytes((run_dir / artifact.relative_path).read_bytes())
    execution = ReviewExecutionAuthority(
        pipeline_config_sha256=record["record_sha256"], capability_activation_sha256=authority.execution_contract_sha256,
        run_input_capsule_sha256=canonical_sha256(inputs),
    )
    requests = human_review_requests_for_plan(findings=[], plan=plan, evidence=evidence,
                                              execution_authority=execution, require_plan_review=True)
    checkpoint = HumanReviewCheckpoint.create(
        run_id=run_id, pipeline_config_sha256=execution.pipeline_config_sha256, environment_identity=record["environment"],
        llm_signature_sha256=canonical_sha256("deterministic_result_append"), run_input_capsule_sha256=canonical_sha256(inputs),
        capability_activation_sha256=authority.execution_contract_sha256, runtime_capabilities=("result_append",),
        runtime_bundle=None, requests=requests, plan_handoff={"plan": record["plan"]},
        execution_coordinates={"source_run_id": source.seal["run_id"], "source_run_sha256": source.digest},
    )
    write_checkpoint(run_dir / "human_review_checkpoint.json", checkpoint)
    child = AppendRun(wrapper, run_dir, plan, authority, record, checkpoint)
    from .research_append_projection import publish_append_projection

    publish_append_projection(child)
    _write(_index(run_id), {"wrapper": str(wrapper), "study_id": study["id"]})
    return child


def load_append_run(wrapper: Path):
    wrapper = Path(wrapper)
    record = read_json(wrapper / "result_append_run.json")
    if record["record_sha256"] != canonical_sha256({k: v for k, v in record.items() if k != "record_sha256"}):
        raise ValueError("Append record changed.")
    if _index(record["run_id"]) is None:
        raise ValueError("Invalid append run id.")
    run_dir = wrapper / "pipeline" / record["run_id"]
    checkpoint_path = run_dir / "human_review_checkpoint.json"
    checkpoint = HumanReviewCheckpoint.model_validate(read_json(checkpoint_path))
    # The deadline limits outstanding review authority, not the lifetime of
    # completed evidence. Lineage admission verifies the completion seal too.
    if checkpoint.state != "completed":
        checkpoint = load_checkpoint(checkpoint_path, require_pending=False)
    authority = ResultAppendRuntimeAuthority.model_validate_json(json.dumps(record["authority"]))
    plan = AnalysisPlan.model_validate(record["plan"])
    authority.validate_plan(plan)
    if checkpoint.pipeline_config_sha256 != record["record_sha256"] or checkpoint.plan_handoff["plan"] != record["plan"]:
        raise ValueError("Append checkpoint changed.")
    return AppendRun(wrapper, run_dir, plan, authority, record, checkpoint)


def append_run_for_id(run_id):
    path = _index(run_id)
    if path is None or not path.exists():
        return None
    return load_append_run(Path(read_json(path)["wrapper"]))


def approve_append_run(wrapper, *, decision, reviewer, source=None, registered_source_changed=False, note="", job=None, current_study=None, require_recorded_decision=False, record_only=False):
    wrapper = Path(wrapper)
    with _locked(wrapper / "append.lock"):
        child = load_append_run(wrapper)
        checkpoint = child.checkpoint
        if checkpoint.state not in {"pending", "approved_pending_execution"}:
            raise ValueError("This append approval has already been consumed.")
        from .research_append_control import append_progress, check_cancel, stop_append
        from .research_append_history import verify_predecessor

        if require_recorded_decision:
            decisions = checkpoint.approved_decisions
            if (decision != "approved" or not decisions or not checkpoint.approved_decision_records
                    or any(d["decision"] != "approved" for d in decisions)
                    or any(d.get("reviewer_identity_source") != "authenticated" for d in checkpoint.approved_decision_records)):
                raise ValueError("A recorded human approval is required to recover this addition.")
            reviewer = decisions[0]["reviewer"]
        try:
            check_cancel(job)
            source = (source or load_append_source(Path(child.record["source_wrapper"]))).verify()
            if source.digest != child.authority.source_run_sha256 or source.seal["study_id"] != child.record["study"]["id"]:
                raise ValueError("Append source identity changed.")
            if child.record["environment"] != _environment():
                raise ValueError("The append execution owner changed; review a fresh addition.")
            _guard(source, child.plan, child.authority.kind).validate_plan_revision(plan=child.plan, context=source.context)
            evidence = EvidenceStore(child.run_dir)
            for request in checkpoint.requests:
                for evidence_id, digest in request.payload["plan_review_authority"]["evidence_sha256"].items():
                    record = evidence.get(evidence_id)
                    if record is None or record.sha256 != digest or verified_run_evidence_path(child.run_dir, record) is None:
                        raise ValueError("Review-bound append evidence changed.")
            verify_predecessor(child)
            if current_study is not None:
                from .research_append_source import registered_source_changed as source_changed

                registered_source_changed = source_changed(source, current_study)
        except Exception as exc:
            stop_append(child, code=getattr(exc, "code", "result_append_source_changed"))
            raise
        path = child.run_dir / "human_review_checkpoint.json"
        if checkpoint.state == "pending":
            decisions = ([HumanReviewDecision.model_validate(item) for item in checkpoint.approved_decisions]
                         if checkpoint.approved_decisions else [HumanReviewDecision(review_id=r.review_id, authority_sha256=r.authority_sha256,
                                             decision=decision, reviewer=reviewer, decided_at=datetime.now(timezone.utc).isoformat(), note=note)
                         for r in checkpoint.requests])
            if any(item.decision != decision or item.reviewer != reviewer for item in decisions):
                raise ValueError("The append already records a different approval.")
            wire = [item.model_dump(mode="json") for item in decisions]
            records = list(checkpoint.approved_decision_records) or [
                _human_review_decision_record(request=r, decision=d, reviewer_identity=reviewer)
                for r, d in zip(checkpoint.requests, decisions)]
            checkpoint = checkpoint.decision_recorded(decisions=wire, decision_records=records,
                                                       decision_sha256=canonical_sha256(wire))
            write_checkpoint(path, checkpoint)
            evidence.register_text(kind="log", description="Human append approval", filename="human_review.json",
                                   evidence_id="append_approval", text=json.dumps(records, sort_keys=True, ensure_ascii=False))
            checkpoint = checkpoint.decision_committed()
            write_checkpoint(path, checkpoint)
        if checkpoint.state == "rejected":
            from .research_append_projection import publish_append_projection

            publish_append_projection(load_append_run(wrapper))
            return {"execution_complete": False, "rejected": True}
        if decision != "approved":
            raise ValueError("An approved append cannot be changed by a later decision.")
        if record_only:
            # Submission commits the human decision before activating its
            # durable job. Only that job may start the approved step roster.
            return {"execution_complete": False, "approval_recorded": True}
        write_checkpoint(path, checkpoint.execution_started())
        try:
            steps = execute_result_append(run_dir=child.run_dir, plan=child.plan, authority=child.authority,
                                           inputs=child.record["inputs"], evidence=evidence,
                                           progress=append_progress(job))
            check_cancel(job)
            source.verify()
            verify_predecessor(child)
            result = dict(execution_complete=True, steps=steps, source_notice=source.earlier_export_notice(changed=registered_source_changed))
            _write(child.run_dir / "append_result.json", result)
            write_checkpoint(path, checkpoint.execution_started().transitioned("completed"))
        except Exception as exc:
            stop_append(load_append_run(wrapper), code=getattr(exc, "code", "result_append_execution_failed"))
            raise
        from .research_append_projection import publish_append_projection

        finished = load_append_run(wrapper)
        publish_append_projection(finished, result=result)
        from .research_append_history import seal_append_completion

        seal_append_completion(finished)
        return result


def pending_append_review(run_id):
    child = append_run_for_id(run_id)
    if child is None or child.checkpoint.state not in {"pending", "approved_pending_execution"}:
        return None
    # Refresh has no execution authority, but it must retire a known-bad card.
    with _locked(child.wrapper / "append.lock"):
        child = load_append_run(child.wrapper)
        if child.checkpoint.state not in {"pending", "approved_pending_execution"}:
            return None
        try:
            from .research_append_history import verify_predecessor

            source = load_append_source(Path(child.record["source_wrapper"]))
            if source.digest != child.authority.source_run_sha256:
                raise ValueError("Append source identity changed.")
            verify_predecessor(child)
            EvidenceStore(child.run_dir).verified_records()
        except (ValueError, OSError, KeyError):
            from .research_append_control import stop_append

            stop_append(child)
            return None
    return dict(run_id=run_id, study_id=child.record["study"]["id"], resumable_here=True, resume_scope="durable_checkpoint", is_append_run=True,
                scientific_configuration_sha256=study_contexts.scientific_configuration_sha256(child.record["study"]),
                scientific_plan_review=read_json(child.run_dir / "scientific_plan_review.json"),
                plan_approval_allowed=True, provider="", credential_source="scientific_provider", budget_mode="full_reviewed",
                research_input_state="prepared", requests=[dict(review_id=r.review_id, kind=r.kind, summary=r.summary,
                                                               authority_sha256=r.authority_sha256, approval_allowed=True,
                                                               reason_code="operator_plan_approval_required") for r in child.checkpoint.requests])


def resume_append_review(*, run_id, study_context_id, decision, reviewer, note, current_study, job=None, require_recorded_decision=False):
    from .research_append_projection import publish_append_projection

    child = append_run_for_id(run_id)
    if child is None or child.record["study"]["id"] != study_context_id or not current_study or current_study["id"] != study_context_id:
        raise ValueError("Append approval belongs to another study.")
    result = approve_append_run(child.wrapper, decision=decision, reviewer=reviewer, note=note, job=job,
                                current_study=current_study, require_recorded_decision=require_recorded_decision)
    return publish_append_projection(load_append_run(child.wrapper), result=result)
