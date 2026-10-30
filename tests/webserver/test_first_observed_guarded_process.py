"""The first-observed worker uses real host receipts in a separate process.

Only the six-column synthetic data source is stubbed; this is not an eICU
Demo result, a real user approval, or research execution.
"""
import os
from pathlib import Path
import subprocess
import sys


_SOURCE_STUB = r'''
import os
if os.getenv("EASYICU_TEST_FIRST_OBSERVED_SOURCE") == "1":
    from pathlib import Path
    from types import SimpleNamespace
    import pandas as pd
    from easyicu.research_agent.acquisition.eicu_first_observed_candidate import derive_first_observed_candidate
    from easyicu.research_agent.research_context.export_selection import SourceIdentity
    from easyicu.webserver import first_observed_preparation as owner
    root = Path(os.environ["EASYICU_TEST_FIRST_OBSERVED_ROOT"])
    identity = SourceIdentity(database="eicu_demo", registration_id="synthetic", archive_sha256="a" * 64)
    owner.sources.load_registry = lambda: {"sources": [{"path": str(root / "synthetic-export")}]}
    owner.sources.registered_source_identity = lambda _: identity
    owner.resolve_raw_source_identity_binding = lambda **_: SimpleNamespace(family="eicu",
        table_paths={"patient": str(root / "patient.parquet")}, export_manifest_sha256="b" * 64,
        table_sha256={"patient": "c" * 64})
    owner.verified_public_demo_raw = lambda _: SimpleNamespace(database="eicu_demo", archive_sha256="a" * 64)
    def derive(_):
        frame = pd.DataFrame({"patientunitstayid": [1, 2, 3], "uniquepid": ["p1", "p2", "p2"],
            "patienthealthsystemstayid": [11, 22, 23], "unitvisitnumber": [1, 1, 1],
            "unittype": ["MICU", "MICU", "MICU"], "age": ["65", "70", "70"]})
        candidate = derive_first_observed_candidate(frame)
        proof = {"official_release_independently_verified": True, "patient_table_sha256": "c" * 64,
            "export_manifest_sha256": "b" * 64, "complete_patient_source": {
                "schema_version": "easyicu.eicu_demo_complete_patient_source/1",
                "all_source_rows_compared": True, "archive_sha256": "a" * 64}}
        return SimpleNamespace(frame=candidate.frame, receipt={**candidate.receipt, "source_binding": proof})
    owner.derive_from_verified_eicu_demo = derive
'''


_PROCESS = r'''
import hashlib, json, os, sys, time
from pathlib import Path
from types import SimpleNamespace
from easyicu.research_agent.authority.evidence_store import EvidenceStore
from easyicu.research_agent.authority.plan_review import PlanReviewAuthority
from easyicu.research_agent.authority.raw_preparation import RawPreparationCheckpoint
from easyicu.research_agent.canonical_json import canonical_sha256
from easyicu.research_agent.orchestration.human_review_checkpoint import HumanReviewCheckpoint, write_checkpoint
from easyicu.research_agent.orchestration.workflow import HumanReviewRequest
from easyicu.research_agent.planning.population_compile import compile_population
from easyicu.research_agent.planning.population_spec import PopulationSpec
from easyicu.research_agent.planning.progressive_contract import ProgressivePlanSkeleton
from easyicu.research_agent.planning.scientific_review import PlanScientificReview
from easyicu.research_agent.schema import AnalysisPlan
from easyicu.webserver import agent_review_recovery, first_observed_preparation as producer, jobs, raw_preparation_durable as durable, study_contexts, settings
from easyicu.webserver.host_action_contracts import HostActionRequest, HostActionTag, RawPreparationReviewDecision, decision_key, host_action_id
from easyicu.webserver.pi_copilot import service
from easyicu.webserver.pi_copilot.service import PiCopilotService
from easyicu.webserver.raw_preparation_decisions import RawPreparationHostBinding, record_host_review
from tests.research_agent.planning.progressive_planner_fixtures import _payload
from tests.research_agent.planning.test_a_population_spec_compiles_to_the_cohort_it_states import _ctx
from tests.webserver.copilot.pi_copilot_contract_fixtures import FakeGateway

root = Path(sys.argv[1]); run_id = "run_fixture"; project = "project_fixture"
settings.update_settings({"ai_enabled": True})
copilot = PiCopilotService(gateway=FakeGateway()); service._SERVICE = copilot
created = copilot.create_session(project_id=project, external_llm_opt_in=True)["session"]
session, study_id = created["session_id"], created["binding"]["study_context_id"]
study = study_contexts.get_context(study_id)
study_sha = study_contexts.scientific_configuration_sha256(study)
wrapper = root / "wrapper"; run = wrapper / "pipeline" / run_id; run.mkdir(parents=True)
identity = {"database": "eicu_demo", "registration_id": "synthetic", "archive_sha256": "a" * 64}
context = _ctx(constraints={"source_identity": identity}).model_copy(update={
    "research_question": "Adult first observed ICU in the only visible hospitalization."})
spec = PopulationSpec.model_validate({"criteria": [{"id": "c1", "kind": "first_observed_icu_single_hospital_adult",
    "role": "include", "source": "question", "quote": context.research_question}]})
compiled = compile_population(spec, context)
assert compiled.criteria[0].disposition == "requires_extraction"
plan = AnalysisPlan(research_question=context.research_question, steps=[], cohort=compiled.cohort_definition())
plan_sha = canonical_sha256(plan.model_dump(mode="json"))
review = PlanScientificReview(status="changes_required", approval_allowed=False, top_journal_candidate=False,
    score=0, dimension_scores={}, context_sha256=canonical_sha256(context.model_dump(mode="json")),
    plan_sha256=plan_sha, literature_sha256="d" * 64, figure_strategy_sha256="e" * 64,
    generated_at="2026-10-30T00:00:00Z")
record = EvidenceStore(run).register_text(kind="log", description="synthetic review", filename="review.json",
    text=review.model_dump_json(), evidence_id="scientific_plan_review")
files = {"scientific_plan_review": record.sha256}
def write(key, filename, value):
    data = json.dumps(value, ensure_ascii=False, indent=2).encode()
    (run / filename).write_bytes(data); files[key] = hashlib.sha256(data).hexdigest()
sk = _payload(); sk["cohort"]["population_spec"] = spec.model_dump(mode="json")
skeleton = ProgressivePlanSkeleton.model_validate(sk).model_dump(mode="json")
sk_sha = canonical_sha256(skeleton)
write("progressive_plan_skeleton", "progressive_plan_skeleton.json", skeleton)
write("research_context", "research_context.json", context.model_dump(mode="json"))
write("progressive_plan_compile_receipt", "progressive_plan_compile_receipt.json", {
    "schema_version": "easyicu.progressive_plan_compile_receipt/1", "owner": "easyicu.planning.progressive_compiler_v1",
    "compiler_version": "1", "skeleton_sha256": sk_sha, "analysis_plan_sha256": "1" * 64, "compiled_steps": []})
body = {"schema_version": "easyicu.progressive_planning_authority/2", "planner_strategy": "progressive_v2",
    "outline_sha256": "1"*64, "outline_structured_output_authority_sha256": None,
    "foundation_sha256": "1"*64, "foundation_structured_output_authority_sha256": None,
    "ordered_steps": [{"step_id": "step1", "materialization_sha256": "1"*64, "structured_output_authority_sha256": None}],
    "compiled_skeleton_sha256": sk_sha, "compiled_analysis_plan_sha256": "1"*64,
    "normalized_plan_sha256": plan_sha, "normalized_plan_authority_sha256": "1"*64,
    "outline_artifact_sha256": "1"*64, "foundation_artifact_sha256": "1"*64, "materialization_ledger_sha256": "1"*64,
    "skeleton_artifact_sha256": files["progressive_plan_skeleton"],
    "compile_receipt_artifact_sha256": files["progressive_plan_compile_receipt"],
    "planner_prompt_metrics_sha256": "1"*64, "analysis_plan_artifact_sha256": "1"*64,
    "normalized_plan_artifact_sha256": "1"*64, "strict_transport_bound": False}
body["authority_sha256"] = canonical_sha256(body)
write("progressive_planning_authority", "progressive_planning_authority.json", body)
authority = PlanReviewAuthority.create(plan=plan, evidence_sha256=files)
original_payload = {"plan_review_authority": authority.model_dump(mode="json")}
original_request = HumanReviewRequest.create(kind="scientific_stop", summary="Synthetic plan review",
    payload=original_payload, authority_sha256=canonical_sha256(original_payload))
hc = HumanReviewCheckpoint.create(run_id=run_id, pipeline_config_sha256="1"*64, environment_identity={},
    llm_signature_sha256="2"*64, run_input_capsule_sha256="3"*64, capability_activation_sha256="4"*64,
    runtime_capabilities=(), runtime_bundle=None, requests=(original_request,), plan_handoff={}, execution_coordinates={})
write_checkpoint(run / "human_review_checkpoint.json", hc)
recovery = agent_review_recovery.WebReviewRecoveryRecord.create(run_id=run_id, wrapper_dir=str(wrapper), study=study,
    scientific_configuration_sha256=study_sha, provider_meta={}, provider_public={}, credential_source="pi_verified",
    budget_mode="planner_canary", prepared_package_binding=None, pipeline_config={}, pipeline_config_sha256="1"*64,
    acquisition_projection={}, hard_stop_ledger_path=str(root / "ledger"), hard_stop_task_id="job",
    hard_stop_declaration_sha256="5"*64, created_at=time.time())
agent_review_recovery.put_record(recovery)
scope = producer.proposed_first_observed_scope(run_dir=run, request=original_request)
payload = {"plan_review_authority": authority.model_dump(mode="json"), producer._SCOPE_KEY: scope}
request = HumanReviewRequest.create(kind="scientific_stop", summary=producer._SUMMARY, payload=payload,
    authority_sha256=canonical_sha256(payload))
pending = RawPreparationCheckpoint.pending(plan_sha256=plan_sha, request=request)
choice = RawPreparationReviewDecision(run_id=run_id, scientific_configuration_sha256=study_sha,
    checkpoint_sha256=pending.checkpoint_sha256, scope_sha256=canonical_sha256(scope),
    review_authority_sha256=request.authority_sha256, nonce="f"*64)
action = HostActionRequest(session_id=session, project_id=project, action_code="review_raw_preparation", decision=choice)
key = decision_key(choice, review_decision="approved")
job = jobs.MANAGER.reserve("raw-preparation-review", job_id="job")
tag = HostActionTag(session_id=session, project_id=project, study_context_id=study_id,
    action_code=action.action_code, decision_key=key, action_id=host_action_id(session, action.action_code, key))
assert job.tag_host_action(tag)
copilot.record_job_host_action(session, project_id=project, action_code=action.action_code, action_key=key, child_job_id="job")
approved = record_host_review(host_action=action, pending=pending, decision="approved", study_context_id=study_id,
    job_id="job", run_id=run_id, scientific_configuration_sha256=study_sha)
binding = RawPreparationHostBinding(project_id=project, session_id=session, study_context_id=study_id, run_id=run_id, nonce=choice.nonce)
durable.admit_preparation_intent(job_id="job", checkpoint=approved, binding=binding, scope=scope)
output = durable.recover_preparation_with_sealed_snapshot(job_id="job", job=job, timeout=30)
assert output.is_file()
import pyarrow.parquet as pq
from easyicu.research_agent.acquisition.eicu_first_observed_candidate import FLAG_COLUMN, REASON_COLUMN
rows = pq.read_table(output).to_pydict()
assert rows["stay_id"] == [1, 2, 3] and rows[FLAG_COLUMN] == [True, False, False]
assert rows[REASON_COLUMN][1:] == ["multiple_observed_hospitalizations"] * 2
assert durable.recover_preparation_with_sealed_snapshot(job_id="job", job=job, timeout=30) == output
assert producer.resolve_prepared_first_observed_coordinate(run_dir=run, checkpoint=approved, binding=binding, scope=scope).path == output
print("synthetic first-observed guarded preparation and recovery passed")
'''


def test_independent_first_observed_worker_with_real_host_state(tmp_path):
    (tmp_path / "sitecustomize.py").write_text(_SOURCE_STUB)
    env = os.environ.copy()
    env.update(EASYICU_HOME=str(tmp_path / "home"), EASYICU_TEST_FIRST_OBSERVED_SOURCE="1",
               EASYICU_TEST_FIRST_OBSERVED_ROOT=str(tmp_path),
               PYTHONPATH=str(tmp_path) + os.pathsep + env.get("PYTHONPATH", ""))
    result = subprocess.run([sys.executable, "-c", _PROCESS, str(tmp_path)], env=env,
                            capture_output=True, text=True, timeout=90)
    assert result.returncode == 0, result.stderr
    assert "guarded preparation and recovery passed" in result.stdout
