"""A real host verifier, never serialized context, supplies population evidence."""
from __future__ import annotations

import os
import subprocess
import sys

from easyicu.research_agent.intake.export_package import open_export_package
from tests.support.typed_export import typed_export
from tests.webserver.test_first_observed_guarded_process import _PROCESS, _SOURCE_STUB


def test_host_scope_verifies_the_cohort_in_each_population_compiler(tmp_path):
    export = typed_export(tmp_path / "synthetic-export", database="eicu")
    with open_export_package(export) as package:
        manifest = package.manifest_sha256
    (tmp_path / "sitecustomize.py").write_text(_SOURCE_STUB.replace('"b" * 64', repr(manifest)))
    env = os.environ.copy()
    env.update(EASYICU_HOME=str(tmp_path / "home"), EASYICU_TEST_FIRST_OBSERVED_SOURCE="1",
               EASYICU_TEST_FIRST_OBSERVED_ROOT=str(tmp_path),
               PYTHONPATH=str(tmp_path) + os.pathsep + env.get("PYTHONPATH", ""))
    process = _PROCESS.replace(
        'review_authority_sha256=request.authority_sha256, nonce="f"*64)',
        'review_authority_sha256=request.authority_sha256, nonce=canonical_sha256({'
        '"run_id": run_id, "study_context_id": study_id, "checkpoint_sha256": pending.checkpoint_sha256}))',
    )
    assert process != _PROCESS
    script = process + r'''
import asyncio
from concurrent.futures import ThreadPoolExecutor
from contextvars import copy_context
from easyicu.research_agent.acquisition.foundation import acquire_universe_for_question
from easyicu.research_agent.planning.population_compile import compile_population
from easyicu.research_agent.planning.population_shadow import population_cohort_audit
from easyicu.research_agent.planning.progressive_compiler import stated_population
from easyicu.research_agent.research_context.builder import build_research_context
from easyicu.webserver.first_observed_run_guard import host_first_observed_review_scope
(run / 'raw_preparation_checkpoint.json').write_text(approved.model_dump_json())
acquired = acquire_universe_for_question(export_dir=root/'synthetic-export', question=context.research_question,
    llm=None, output_dir=root/'materialized', target_outcome='death', outcome_concepts=('death',),
    required_feature_concepts=('lact',), static_concepts=('age',), database='eicu',
    concept_selection_authority='host_exact', first_observed_icu_coordinate=resolved)
assert not acquired.blocked
actual = build_research_context(research_question=context.research_question, cohort=acquired.universe_path,
    cohort_name='synthetic', database='eicu', target_outcome='death', id_columns=('stay_id',),
    outcome_columns=('death',), user_preferences={'data_constraints': json.dumps({'source_identity': identity})})
def disposition(ctx=actual):
    return compile_population(spec, ctx).criteria[0].disposition
assert disposition() == 'requires_extraction'
# Client-provided context cannot create the host scope or an applied proof.
forged = actual.model_copy(update={'cohort': actual.cohort.model_copy(update={'provenance': {
    **actual.cohort.provenance, 'first_observed_icu_restriction': {'verified': True}}})})
assert disposition(forged) == 'requires_extraction'
with host_first_observed_review_scope(study_context_id=study_id):
    result = compile_population(spec, actual)
    assert result.criteria[0].disposition == 'applied_by_source'
    assert result.criteria[0].proof.kind == 'host_first_observed_icu_single_hospital_adult'
    sk = _payload(); sk['cohort']['population_spec'] = spec.model_dump(mode='json')
    sk = ProgressivePlanSkeleton.model_validate(sk)
    evaluated = AnalysisPlan(research_question=actual.research_question, steps=[], cohort=result.cohort_definition())
    assert stated_population(sk.cohort, context=actual, plan=evaluated).criteria[0].applied
    audit = population_cohort_audit(context=actual, plan=evaluated, cohort=sk.cohort)
    assert audit['blocking'] == [] and audit['plan_applies_compiled']
    copied = copy_context()
    with ThreadPoolExecutor(max_workers=1) as pool:
        assert pool.submit(disposition).result() == 'requires_extraction'
        assert pool.submit(copied.run, disposition).result() == 'applied_by_source'
    with host_first_observed_review_scope(study_context_id='wrong-study'):
        assert disposition() == 'requires_extraction'
    assert disposition() == 'applied_by_source'
    receipt_path = root/'home'/'.easyicu'/'private-preparation-decisions'/(choice.nonce+'.json')
    original = receipt_path.read_bytes(); changed = json.loads(original); changed['signature']='0'*64
    receipt_path.write_text(json.dumps(changed))
    assert disposition() == 'requires_extraction'
    receipt_path.write_bytes(original)
    assert disposition() == 'applied_by_source'
assert disposition() == 'requires_extraction'
# Even an explicitly copied async/thread context is revoked on scope exit.
assert copied.run(disposition) == 'requires_extraction'
async def task(owner):
    with host_first_observed_review_scope(study_context_id=owner):
        await asyncio.sleep(0)
        return disposition()
async def check_tasks():
    return await asyncio.gather(task(study_id), task('wrong-study'))
assert asyncio.run(check_tasks()) == ['applied_by_source', 'requires_extraction']
# Recreate the actual V3 review artifacts and exercise the real recompile and
# final guard after the pipeline scope has closed (synthetic data source only).
from easyicu.research_agent.authority.plan_review import ReviewExecutionAuthority
from easyicu.research_agent.authority.run_input import build_scientific_identity, seal_run_input_capsule
from easyicu.research_agent.intake.materialized_metadata import load_verified_materialized_cohort_authority, stage_materialized_cohort_authority
from easyicu.research_agent.intake.materialized_trajectory import StagedTrajectoryBinding, load_verified_materialized_trajectory_authority, stage_materialized_trajectory_authority
from easyicu.webserver.first_observed_run_guard import verify_first_observed_execution_inputs
from easyicu.webserver.pi_copilot.population_spec_review import reviewed_population_spec
run2 = root/'reviewed_run'; run2.mkdir()
original_cohort = load_verified_materialized_cohort_authority(acquired.universe_path)
original_traj = load_verified_materialized_trajectory_authority(acquired.trajectory_path, expected_universe_authority=original_cohort.reference)
cohort, traj = run2/'cohort.parquet', run2/'cohort_trajectory.parquet'
staged = stage_materialized_cohort_authority(acquired.universe_path, cohort, expected_source_authority=original_cohort.reference, producer_implementation_sha256='f'*64)
staged_traj = stage_materialized_trajectory_authority(acquired.trajectory_path, traj, source_universe_path=acquired.universe_path, target_universe_path=cohort, expected_source_authority=original_traj.reference, expected_target_universe_authority=staged.reference, producer_implementation_sha256='f'*64)
preferences = {'data_constraints': json.dumps({'source_identity': identity})}
ctx = build_research_context(research_question=context.research_question, cohort=cohort, cohort_name='synthetic', database='eicu', target_outcome='death', id_columns=('stay_id',), outcome_columns=('death',), user_preferences=preferences, trajectory_binding=StagedTrajectoryBinding(path=traj, sha256=staged_traj.authority.trajectory_sha256, size=staged_traj.authority.trajectory_size, authority_ref=staged_traj.reference))
with host_first_observed_review_scope(study_context_id=study_id):
    real_plan = AnalysisPlan(research_question=ctx.research_question, steps=[], cohort=compile_population(spec, ctx).cohort_definition())
plan_digest = canonical_sha256(real_plan.model_dump(mode='json'))
new_files = {}
def write2(key, filename, data):
    raw = json.dumps(data, ensure_ascii=False, indent=2).encode()
    (run2/filename).write_bytes(raw); new_files[key] = hashlib.sha256(raw).hexdigest()
write2('research_context', 'research_context.json', ctx.model_dump(mode='json'))
write2('progressive_plan_skeleton', 'progressive_plan_skeleton.json', skeleton)
write2('progressive_plan_compile_receipt', 'progressive_plan_compile_receipt.json', {'schema_version': 'easyicu.progressive_plan_compile_receipt/1', 'owner': 'easyicu.planning.progressive_compiler_v1', 'compiler_version': '1', 'skeleton_sha256': sk_sha, 'analysis_plan_sha256': plan_digest, 'compiled_steps': []})
newbody = dict(body); newbody.pop('authority_sha256')
newbody.update(compiled_analysis_plan_sha256=plan_digest, normalized_plan_sha256=plan_digest, skeleton_artifact_sha256=new_files['progressive_plan_skeleton'], compile_receipt_artifact_sha256=new_files['progressive_plan_compile_receipt'])
newbody['authority_sha256'] = canonical_sha256(newbody)
write2('progressive_planning_authority', 'progressive_planning_authority.json', newbody)
evidence = EvidenceStore(run2)
evidence.register_file(kind='log', description='Synthetic context', source_path=run2/'research_context.json', evidence_id='research_context', producer='pipeline', generation_mode='system')
identity_run = build_scientific_identity(cohort=acquired.universe_path, question=ctx.research_question, cohort_name='synthetic', database='eicu', target_outcome='death', primary_exposure=None, cross_database_validation=None, inclusion_criteria=None, exclusion_criteria=None, id_columns=('stay_id',), time_columns=None, outcome_columns=('death',), time_windows=None, concept_descriptions=None, user_preferences=preferences, notes=None, skill_key=None, experiment_spec=None, source_files=None, disable_icu_context=False, materialized_cohort_authority_ref=original_cohort.reference.to_dict(), trajectory_path=acquired.trajectory_path, materialized_trajectory_authority_ref=original_traj.reference.to_dict())
capsule = seal_run_input_capsule(run_dir=run2, evidence=evidence, scientific_identity=identity_run, initial_environment={'llm_signature': 'synthetic'}, context_path=run2/'research_context.json', cohort_path=cohort, experiment_spec_path=None)
assert capsule.schema_version.endswith('/3')
capsule_sha = hashlib.sha256((run2/'run_input_capsule.json').read_bytes()).hexdigest()
new_authority = PlanReviewAuthority.create(plan=real_plan, evidence_sha256=new_files, execution=ReviewExecutionAuthority(pipeline_config_sha256='1'*64, capability_activation_sha256='2'*64, run_input_capsule_sha256=capsule_sha))
request_payload = {'plan_review_authority': new_authority.model_dump(mode='json')}
new_request = HumanReviewRequest.create(kind='scientific_stop', summary='Synthetic review', payload=request_payload, authority_sha256=canonical_sha256(request_payload))
checkpoint = HumanReviewCheckpoint.create(run_id='reviewed_run', pipeline_config_sha256='1'*64, environment_identity={}, llm_signature_sha256='3'*64, run_input_capsule_sha256=capsule_sha, capability_activation_sha256='2'*64, runtime_capabilities=(), runtime_bundle=None, requests=(new_request,), plan_handoff={}, execution_coordinates={})
write_checkpoint(run2/'human_review_checkpoint.json', checkpoint)
new_pending = SimpleNamespace(run_id='reviewed_run', run_dir=str(run2), requests=[new_request])
assert reviewed_population_spec(run_dir=run2, pending=new_pending) is None
projection = reviewed_population_spec(run_dir=run2, pending=new_pending, study_context_id=study_id)
assert projection and projection['criteria'][0]['disposition'] == 'applied_by_source'
assert reviewed_population_spec(run_dir=run2, pending=new_pending, study_context_id='wrong-study') is None
assert verify_first_observed_execution_inputs(pending=new_pending, study_context_id=study_id)
assert disposition() == 'requires_extraction'
original = receipt_path.read_bytes(); changed = json.loads(original); changed['signature'] = '0'*64
receipt_path.write_text(json.dumps(changed))
assert reviewed_population_spec(run_dir=run2, pending=new_pending, study_context_id=study_id) is None
try:
    verify_first_observed_execution_inputs(pending=new_pending, study_context_id=study_id)
except ValueError:
    pass
else:
    raise AssertionError('tampered host proof accepted')
receipt_path.write_bytes(original)
# Enter the actual resume owner. Stub only the unrelated scientific-policy
# predicates; population projection and the final guard remain real.
from easyicu.webserver import agent_pipeline_runs as web_runs
entry = SimpleNamespace(study={'id': study_id}, pipeline=SimpleNamespace(), budget_mode='normal', pending=new_pending, prepared_package_binding=None)
original_get = web_runs._PENDING_REVIEWS.get
web_runs._PENDING_REVIEWS.get = lambda key: entry if key == 'reviewed_run' else original_get(key)
web_runs._plan_has_complete_reviewable_recommendation = lambda plan: True
web_runs._load_pending_scientific_review = lambda *args: {'approval_allowed': True}
def resume_code():
    try:
        web_runs.resume_research_pipeline(run_id='reviewed_run', study_context_id=study_id, decision='approved', reviewer='synthetic', note='', job=None)
    except web_runs.ResearchPipelineRunError as exc:
        return exc.code
    raise AssertionError('unexpected execution')
code = resume_code(); assert code == 'research_pipeline_package_binding_missing', code
receipt_path.write_text(json.dumps(changed))
code = resume_code(); assert code == 'research_pipeline_first_observed_input_binding_unavailable', code
receipt_path.write_bytes(original)
print('host population review scope verified')
'''
    result = subprocess.run([sys.executable, "-c", script, str(tmp_path)], env=env,
                            capture_output=True, text=True, timeout=120)
    assert result.returncode == 0, result.stderr
    assert "host population review scope verified" in result.stdout
